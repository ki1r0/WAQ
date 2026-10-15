from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from safetensors.torch import safe_open
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.utils.data import DataLoader, DistributedSampler, default_collate

from wan_va.configs import VA_CONFIGS
from wan_va.utils.logging import init_logger

from .algorithms import (
    iql_losses,
    mc_flow_losses,
    mc_return_losses,
    sample_mc_flow_target,
)
from .canonical_cache import (
    AsyncCudaPrefetcher,
    BoundedGroupPrefetchLoader,
    CANONICAL_CACHE_FORMAT,
    CANONICAL_CACHE_SCHEMA_VERSION,
    CANONICAL_FEATURE_KEYS,
    EpisodeAwarePrefetchLoader,
    EpisodeLocalBatchSampler,
    GlobalBalancedEpisodeBatchSampler,
    Phase3CanonicalEpisodeCacheDataset,
    Phase3FederatedCanonicalEpisodeCacheDataset,
    phase3_canonical_collate,
    phase3_canonical_episode_cache_file,
    TransitionRangePrefetchLoader,
)
from .checkpoint import load_critic_checkpoint, save_critic_checkpoint
from .config import CriticTrainingConfig
from .critics import (
    build_critic_bundle,
    copy_value_language_parameters_to_target,
    initialize_language_critic_parameters,
    is_language_critic_parameter,
)
from .packed_cache import (
    CudaPrefetcher,
    Phase3PackedCacheDataset,
    ShardLocalBatchSampler,
    phase3_packed_collate,
)
from .transitions import ChunkTransitionDataset
from .wandb_logging import WandbRunLogger, critic_wandb_payload


PHASE3_CACHE_SCHEMA_VERSION = 2
PHASE3_QUERY_FEATURE_KEYS = (
    "value_context_tokens",
    "value_context_mask",
    "q_context_tokens",
    "q_context_mask",
    "target_value_context_tokens",
    "target_value_context_mask",
)
PHASE3_MLP_FEATURE_KEYS = (
    "previous_video_tokens",
    "action_tokens",
    "current_real_video_tokens",
)


def _provenance_sha256(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _snapshot_critic_provenance(config: CriticTrainingConfig, output_dir: Path) -> None:
    """Copy the small, resolved training/data definitions beside checkpoints."""
    provenance_dir = output_dir / "provenance"
    provenance_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(name: str, data: bytes, source_path: str | None) -> dict[str, Any]:
        destination = provenance_dir / name
        digest = _provenance_sha256(data)
        if destination.exists():
            if not destination.is_file() or _provenance_sha256(destination.read_bytes()) != digest:
                raise RuntimeError(f"refusing to replace different provenance snapshot: {destination}")
        else:
            temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                temporary.write_bytes(data)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        return {
            "source_path": source_path,
            "snapshot_path": name,
            "bytes": len(data),
            "sha256": digest,
        }

    records = {
        "resolved_training_config.json": snapshot(
            "resolved_training_config.json",
            (json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n").encode(),
            None,
        )
    }
    catalog_path: Path | None = None
    for attribute, snapshot_name in (
        ("dataset_config", "dataset_config.json"),
        ("phase3_federated_cache_catalog", "federated_catalog.json"),
        (
            "phase3_federated_cache_artifacts_manifest",
            "federated_cache_artifacts_manifest.json",
        ),
        ("phase3_language_sidecar_catalog", "language_sidecar_manifest.json"),
        ("phase3_episode_success_overlay", "episode_success_overlay.json"),
    ):
        configured_path = getattr(config, attribute)
        if configured_path is None:
            continue
        path = Path(configured_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        records[snapshot_name] = snapshot(snapshot_name, path.read_bytes(), str(path))
        if attribute == "phase3_federated_cache_catalog":
            catalog_path = path
    if catalog_path is not None:
        manifest = catalog_path.parent / "immutable_manifest.json"
        if manifest.is_file():
            records["immutable_manifest.json"] = snapshot(
                "immutable_manifest.json", manifest.read_bytes(), str(manifest)
            )

    bundle = {
        "schema_version": 1,
        "kind": "critic_training_provenance",
        "output_dir": str(output_dir),
        "snapshots": records,
    }
    bundle_path = provenance_dir / "bundle.json"
    serialized = json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    if bundle_path.exists():
        if json.loads(bundle_path.read_text()) != bundle:
            raise RuntimeError(f"refusing to replace different provenance bundle: {bundle_path}")
    else:
        temporary = bundle_path.with_name(f".{bundle_path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(serialized)
            os.replace(temporary, bundle_path)
        finally:
            temporary.unlink(missing_ok=True)


def _init_runtime() -> tuple[int, int, int, torch.device]:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise RuntimeError("Critic training requires CUDA")
    print(f"[critic][rank{rank}] setting CUDA device {local_rank}", flush=True)
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        print(f"[critic][rank{rank}] initializing NCCL process group", flush=True)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        print(f"[critic][rank{rank}] NCCL process group ready", flush=True)
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _transformer_path(base_config: Any, critic_config: CriticTrainingConfig) -> str:
    if critic_config.transformer_path:
        return critic_config.transformer_path
    resume_from = getattr(base_config, "resume_from", None)
    if resume_from:
        return str(Path(resume_from) / "transformer")
    return str(
        Path(base_config.wan22_pretrained_model_name_or_path) / "transformer"
    )


def _rank_log(rank: int, message: str) -> None:
    print(f"[phase3][rank{rank}] {message}", flush=True)


def _load_transformer_model(
    transformer_path: str,
    *,
    torch_dtype: torch.dtype,
    torch_device: str | torch.device,
    **kwargs: Any,
) -> torch.nn.Module:
    from wan_va.modules.model import WanTransformer3DModel

    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs,
    )
    return model.to(torch_device)


def _load_vae_model(
    vae_path: str | Path,
    *,
    torch_dtype: torch.dtype,
    torch_device: str | torch.device,
) -> torch.nn.Module:
    from diffusers import AutoencoderKLWan

    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def _load_transformer_rank0_broadcast(
    *,
    transformer_path: str,
    rank: int,
    device: torch.device,
) -> torch.nn.Module:
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        delay = float(os.getenv("PHASE3_TRANSFORMER_IMPORT_STAGGER_SECONDS", "5"))
        if delay > 0:
            sleep_seconds = delay * int(rank)
            if sleep_seconds > 0:
                _rank_log(
                    rank,
                    f"staggering transformer class import by {sleep_seconds:.1f}s",
                )
                time.sleep(sleep_seconds)
    _rank_log(rank, "importing transformer class")
    from wan_va.modules.model import WanTransformer3DModel
    _rank_log(rank, "imported transformer class")

    if not distributed or dist.get_world_size() == 1:
        return _load_transformer_model(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
        )

    if dist.get_backend() != "nccl":
        raise RuntimeError(
            "Phase 3 rank-0 transformer broadcast currently requires NCCL"
        )

    if rank == 0:
        _rank_log(rank, "rank0 loading transformer checkpoint for broadcast")
        transformer = _load_transformer_model(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device="cpu",
        )
    else:
        _rank_log(rank, "instantiating transformer from config for broadcast")
        config, unused_kwargs = WanTransformer3DModel.load_config(
            transformer_path,
            return_unused_kwargs=True,
        )
        transformer = WanTransformer3DModel.from_config(config, **unused_kwargs)
        transformer.to(torch.float32)

    dist.barrier()
    state_dict = transformer.state_dict()
    for name, tensor in state_dict.items():
        if rank == 0:
            payload = tensor.detach().to(device)
        else:
            payload = torch.empty_like(tensor, device=device)
        dist.broadcast(payload, src=0)
        if rank != 0:
            tensor.copy_(payload.cpu())
        del payload
    torch.cuda.empty_cache()
    dist.barrier()
    _rank_log(rank, "transformer checkpoint broadcast complete")
    return transformer


def _prediction_outputs(outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, tuple) or len(outputs) < 2:
        raise TypeError("Transformer train_mode output must contain video/action predictions")
    return outputs[0], outputs[1]


def _zero_timestep_stream(
    tensor: torch.Tensor,
    text_emb: torch.Tensor,
    patch_size: tuple[int, int, int],
    action_mode: bool,
    timestep: float,
    condition_tensor: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    from wan_va.utils.utils import get_mesh_id

    batch, _, frames, height, width = tensor.shape
    if condition_tensor is None:
        condition_tensor = tensor
    if condition_tensor.shape != tensor.shape:
        raise ValueError(
            "condition_tensor must match tensor shape; got "
            f"{tuple(condition_tensor.shape)} vs {tuple(tensor.shape)}"
        )
    patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else patch_size
    grid_id = get_mesh_id(
        frames // patch_f,
        height // patch_h,
        width // patch_w,
        t=1 if action_mode else 0,
        action=action_mode,
    ).to(tensor.device)
    timesteps = torch.full(
        (batch, frames),
        float(timestep) * 1000.0,
        device=tensor.device,
    )
    return {
        "timesteps": timesteps,
        "cond_timesteps": torch.zeros_like(timesteps),
        "noisy_latents": tensor,
        "latent": condition_tensor,
        "text_emb": text_emb,
        "grid_id": grid_id.unsqueeze(0).repeat(batch, 1, 1),
    }


def _masked_mean_distance(
    predicted: torch.Tensor,
    actual: torch.Tensor,
    mask: torch.Tensor,
    available: torch.Tensor | None,
) -> torch.Tensor:
    predicted = F.normalize(predicted.float(), dim=-1)
    actual = F.normalize(actual.float(), dim=-1)
    distance = 1.0 - (predicted * actual).sum(dim=-1)
    per_frame = distance.mean(dim=(2, 3))
    valid = mask.to(device=per_frame.device, dtype=per_frame.dtype)
    if available is not None:
        valid = valid * available.to(valid.device, valid.dtype).view(-1, 1)
    return (per_frame * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


def phase3_episode_cache_file(
    cache_dir: str | Path,
    episode_key: tuple[int, int],
) -> Path:
    dataset_id, episode_index = (int(episode_key[0]), int(episode_key[1]))
    return (
        Path(cache_dir)
        / f"dataset-{dataset_id:06d}"
        / f"shard-{episode_index // 1000:06d}"
        / f"episode_d{dataset_id:06d}_ep{episode_index:09d}.pt"
    )


class Phase3CriticTrainer:
    """Train critics on generated-video-conditioned real-action features."""

    def __init__(
        self,
        config: CriticTrainingConfig,
        rank: int,
        local_rank: int,
        world_size: int,
        device: torch.device,
        build_critic: bool = True,
        create_outputs: bool = True,
        dataset_init_workers: int = 128,
        load_jepa: bool = True,
    ) -> None:
        if (
            config.training_distribution
            != "predicted_video_conditioned_real_action"
        ):
            raise ValueError(
                "train_critic_phase3.py requires "
                "training_distribution="
                "'predicted_video_conditioned_real_action'"
            )
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device
        self.step = 0
        self.load_jepa = bool(load_jepa)

        self.base_config = deepcopy(VA_CONFIGS[config.base_config_name])
        self.base_config.cfg_prob = 0.0
        self.base_config.jepa_loss_enabled = True
        self.base_config.batch_size = config.batch_size
        if config.infer_latent_chunk_size != int(self.base_config.frame_chunk_size):
            raise ValueError(
                "infer_latent_chunk_size must match base-config frame_chunk_size"
            )

        self.transformer = None
        self.video_scheduler = None
        self.action_scheduler = None
        self.vae = None
        self.jepa_encoder = None
        if config.phase3_use_packed_cache:
            self._init_packed_cache_loader()
        elif (
            config.phase3_cache_format == "federated_canonical_episode"
            and config.phase3_federated_cache_catalog
            and config.phase3_require_precomputed_cache
        ):
            self._init_federated_canonical_cache_loader()
        elif (
            config.phase3_cache_format == "canonical_episode"
            and config.phase3_precomputed_cache_dir
            and config.phase3_require_precomputed_cache
        ):
            self._init_canonical_cache_loader()
        else:
            self._init_online_or_episode_cache_loader(dataset_init_workers)
        self.reward_term_stats: dict[str, dict[str, float]] = {}
        self.final_reward_stats: dict[str, float] | None = None
        self.goal_jepa_potential_stats: dict[str, float] | None = None
        self._goal_jepa_potential_next_by_transition_idx: dict[int, float] = {}
        self._reward_rows_cache: list[dict[str, Any]] | None = None
        if (
            self._has_normalized_reward_terms()
            or self._has_final_reward_normalization()
            or self._has_goal_jepa_potential_shaping()
        ):
            rows = self._reward_rows()
            if self._has_goal_jepa_potential_shaping():
                self._fit_goal_jepa_potential_stats(rows)
            if self._has_normalized_reward_terms():
                self._fit_reward_term_stats(rows)
            if self._has_final_reward_normalization():
                self._fit_final_reward_stats(rows)
        self.mc_return_by_transition_idx: dict[int, float] = {}
        self.mc_return_mean = 0.0
        self.mc_return_std = 1.0
        if config.rl_objective in {"mc_flow", "mc_return"}:
            self._init_mc_returns()

        self.bundle = None
        self.optimizer = None
        self.transfer_initialization: dict[str, Any] | None = None
        self._inherited_critic_trainable: bool | None = None
        if build_critic:
            self.bundle = build_critic_bundle(
                critic_type=config.critic_type,
                feature_dim=config.feature_dim,
                hidden_dim=config.hidden_dim,
                num_layers=config.num_layers,
                transformer_hidden_dim=config.critic_transformer_hidden_dim,
                transformer_num_layers=config.critic_transformer_layer_count,
                transformer_num_heads=config.critic_transformer_num_heads,
                transformer_ffn_dim=config.critic_transformer_ffn_dim,
                transformer_dropout=config.critic_transformer_dropout,
                text_embed_dim=config.critic_text_embed_dim,
                num_query_tokens=config.critic_num_query_tokens,
                num_q_heads=config.num_q_heads,
                with_value=True,
            ).to(device=device, dtype=torch.float32)
            if config.initialize_from:
                self._initialize_from_parent_checkpoint(config.initialize_from)
            elif config.resume_from:
                self._restore_transfer_metadata(config.resume_from)
            self.optimizer = torch.optim.AdamW(
                self._critic_optimizer_groups(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
            self._set_optimizer_lr(self.step)
            if config.resume_from:
                self.step = load_critic_checkpoint(
                    config.resume_from,
                    self.bundle,
                    self.optimizer,
                    expected_manifest={
                        "algorithm": config.algorithm,
                        "rl_objective": config.rl_objective,
                        "critic_type": config.critic_type,
                        "feature_dim": config.feature_dim,
                        "infer_latent_chunk_size": config.infer_latent_chunk_size,
                        "feature_layers": list(config.feature_layers),
                        "critic_feature_layers": list(
                            config.critic_feature_layers
                        ),
                        "feature_aggregation": config.feature_aggregation,
                        "feature_normalization": config.feature_normalization,
                        "training_distribution": config.training_distribution,
                        "phase3_video_prediction_context": (
                            config.phase3_video_prediction_context
                        ),
                        "reward_source": config.reward_source,
                        "gradient_accumulation_steps": (
                            config.gradient_accumulation_steps
                        ),
                        "critic_num_query_tokens": config.critic_num_query_tokens,
                        "critic_transformer_hidden_dim": (
                            config.critic_transformer_hidden_dim
                        ),
                        "critic_transformer_num_layers": (
                            config.critic_transformer_layer_count
                        ),
                        "critic_transformer_num_heads": (
                            config.critic_transformer_num_heads
                        ),
                        "critic_transformer_ffn_dim": (
                            config.critic_transformer_ffn_dim
                        ),
                        "critic_transformer_dropout": (
                            config.critic_transformer_dropout
                        ),
                        "critic_text_embed_dim": config.critic_text_embed_dim,
                        "num_q_heads": config.num_q_heads,
                        "q_ensemble_reduce": config.q_ensemble_reduce,
                        "q_loss_type": config.q_loss_type,
                        "q_huber_delta": config.q_huber_delta,
                        "scalar_flow_loss": config.scalar_flow_loss,
                        "scalar_flow_huber_delta": (
                            config.scalar_flow_huber_delta
                        ),
                        "value_video_feature_timestep_range": (
                            config.value_video_feature_timestep_range
                        ),
                        "value_action_feature_timestep_range": (
                            config.value_action_feature_timestep_range
                        ),
                        "q_video_feature_timestep_range": (
                            config.q_video_feature_timestep_range
                        ),
                        "q_action_feature_timestep_range": (
                            config.q_action_feature_timestep_range
                        ),
                        **(
                            {
                                "phase3_federated_cache_catalog_sha256": (
                                    self.dataset.catalog_sha256
                                )
                            }
                            if isinstance(
                                self.dataset,
                                Phase3FederatedCanonicalEpisodeCacheDataset,
                            )
                            else {}
                        ),
                    },
                )
            self._update_inherited_critic_trainability(force=True)
            self._set_optimizer_lr(self.step)

        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        if create_outputs and rank == 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (self.checkpoint_dir / "config.json").write_text(
                json.dumps(config.to_dict(), indent=2) + "\n"
            )
            _snapshot_critic_provenance(config, self.output_dir)
        self.log_path = self.checkpoint_dir / "loss.jsonl"
        self.wandb_logger = WandbRunLogger(
            config,
            rank=rank,
            run_name_prefix="phase3-critic",
        )
        _rank_log(rank, "trainer initialization complete")

    def _dataloader_worker_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.phase3_pin_memory,
            "persistent_workers": (
                self.config.phase3_persistent_workers
                and self.config.num_workers > 0
            ),
        }
        if (
            self.config.num_workers > 0
            and self.config.phase3_prefetch_factor is not None
        ):
            kwargs["prefetch_factor"] = self.config.phase3_prefetch_factor
        return kwargs

    def _has_reward_term(self, name: str) -> bool:
        return any(term["name"] == name for term in self.config.reward_terms)

    def _has_normalized_reward_terms(self) -> bool:
        return any(bool(term.get("normalize", False)) for term in self.config.reward_terms)

    def _has_final_reward_normalization(self) -> bool:
        return self.config.final_reward_normalization == "scale_neg1_pos1"

    def _reward_terms_need_goal_jepa_distance(self) -> bool:
        return any(
            term["name"]
            in {"goal_jepa_progress", "goal_jepa_potential_shaping"}
            for term in self.config.reward_terms
        )

    def _has_goal_jepa_potential_shaping(self) -> bool:
        return self._has_reward_term("goal_jepa_potential_shaping")

    def _reward_terms_need_episode_metadata(self) -> bool:
        return any(
            term["name"]
            in {
                "terminal_success",
                "terminal_outcome_cost",
                "successful_episode_progress",
                "goal_jepa_progress",
            }
            for term in self.config.reward_terms
        )

    def _require_reward_metadata_available(self, *, source: str) -> None:
        if self._reward_terms_need_episode_metadata():
            raise ValueError(
                "Configured reward_terms require episode_success and "
                "episode_chunk_count metadata, but it is missing from "
                f"{source}. Regenerate the Phase 3 episode cache and then "
                "repack it before using terminal_success, terminal_outcome_cost, "
                "successful_episode_progress, or goal_jepa_progress rewards."
            )

    def _require_goal_distance_available(self, *, source: str) -> None:
        if self._reward_terms_need_goal_jepa_distance():
            raise ValueError(
                "Configured reward_terms require goal_jepa_distance and "
                "episode_goal_jepa_distance_max metadata, but it is missing "
                f"from {source}. Regenerate the Phase 3 episode cache with "
                "a version that computes goal-distance rewards."
            )

    @staticmethod
    def _progress_reward_tensor(
        rl_chunk_idx: torch.Tensor,
        episode_chunk_count: torch.Tensor,
        episode_success: torch.Tensor,
    ) -> torch.Tensor:
        chunk_count = episode_chunk_count.float().clamp_min(1.0)
        progress = (rl_chunk_idx.float() + 1.0) / chunk_count
        return torch.where(episode_success.bool(), progress, torch.zeros_like(progress))

    @staticmethod
    def _progress_reward_row(row: dict[str, Any]) -> float:
        if not bool(row.get("episode_success", False)):
            return 0.0
        chunk_count = int(row.get("episode_chunk_count", 0))
        if chunk_count <= 0:
            raise ValueError(
                "successful_episode_progress requires positive "
                "episode_chunk_count for successful episodes"
            )
        return float(int(row["rl_chunk_idx"]) + 1) / float(chunk_count)

    def _goal_jepa_progress_reward_tensor(
        self,
        goal_distance: torch.Tensor,
        goal_distance_max: torch.Tensor,
        rl_chunk_idx: torch.Tensor,
        episode_chunk_count: torch.Tensor,
    ) -> torch.Tensor:
        terminal_span = (episode_chunk_count.float() - 1.0).clamp_min(1.0)
        progress = (rl_chunk_idx.float() / terminal_span).clamp(0.0, 1.0)
        normalized_distance = goal_distance.float() / goal_distance_max.float().clamp_min(1e-6)
        modulation = torch.exp(
            -float(self.config.goal_jepa_progress_lambda) * progress
        ) * (1.0 - progress).clamp_min(0.0).pow(
            float(self.config.goal_jepa_progress_beta)
        )
        reward = 1.0 - normalized_distance * modulation
        terminal = rl_chunk_idx.long() >= (episode_chunk_count.long() - 1).clamp_min(0)
        return torch.where(terminal, torch.ones_like(reward), reward)

    def _goal_jepa_progress_reward_row(self, row: dict[str, Any]) -> float:
        chunk_count = int(row.get("episode_chunk_count", 0))
        if chunk_count <= 1:
            return 1.0
        goal_distance = float(row["goal_jepa_distance"])
        goal_distance_max = max(float(row["episode_goal_jepa_distance_max"]), 1e-6)
        progress = min(
            1.0,
            max(0.0, float(int(row["rl_chunk_idx"])) / float(chunk_count - 1)),
        )
        if int(row["rl_chunk_idx"]) >= chunk_count - 1:
            return 1.0
        modulation = math.exp(
            -float(self.config.goal_jepa_progress_lambda) * progress
        ) * ((1.0 - progress) ** float(self.config.goal_jepa_progress_beta))
        return 1.0 - (goal_distance / goal_distance_max) * modulation

    def _fit_goal_jepa_potential_stats(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("Cannot fit goal-potential rewards from an empty cache")
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in rows:
            key = (int(row["dataset_id"]), int(row["episode_index"]))
            grouped.setdefault(key, []).append(row)

        next_by_transition_idx: dict[int, float] = {}
        distances: list[float] = []
        for key, episode_rows in grouped.items():
            episode_rows.sort(key=lambda row: int(row["rl_chunk_idx"]))
            for index, row in enumerate(episode_rows):
                distance = float(row["goal_jepa_distance"])
                distances.append(distance)
                if bool(row["done"]):
                    next_distance = 0.0
                elif index + 1 < len(episode_rows):
                    next_distance = float(episode_rows[index + 1]["goal_jepa_distance"])
                else:
                    raise ValueError(
                        "Goal-potential cache has a nonterminal final chunk for "
                        f"episode {key}"
                    )
                next_by_transition_idx[int(row["transition_idx"])] = next_distance

        values = torch.tensor(distances, dtype=torch.float32)
        lower, upper = torch.quantile(
            values,
            torch.tensor(
                [
                    float(self.config.goal_jepa_potential_quantile_low),
                    float(self.config.goal_jepa_potential_quantile_high),
                ],
                dtype=torch.float32,
            ),
        ).tolist()
        if upper - lower < 1e-6:
            upper = lower + 1e-6
        self.goal_jepa_potential_stats = {
            "lower": float(lower),
            "upper": float(upper),
            "quantile_low": float(self.config.goal_jepa_potential_quantile_low),
            "quantile_high": float(self.config.goal_jepa_potential_quantile_high),
        }
        self._goal_jepa_potential_next_by_transition_idx = next_by_transition_idx
        _rank_log(
            self.rank,
            "goal-potential calibration ready: "
            f"q{self.config.goal_jepa_potential_quantile_low:.2f}={lower:.6f} "
            f"q{self.config.goal_jepa_potential_quantile_high:.2f}={upper:.6f} "
            f"transitions={len(next_by_transition_idx)}",
        )

    def _goal_jepa_potential_reward_tensor(
        self,
        goal_distance: torch.Tensor,
        next_goal_distance: torch.Tensor,
    ) -> torch.Tensor:
        if self.goal_jepa_potential_stats is None:
            raise RuntimeError("Goal-potential calibration stats are not fitted")
        lower = float(self.goal_jepa_potential_stats["lower"])
        upper = float(self.goal_jepa_potential_stats["upper"])
        scale = max(upper - lower, 1e-6)
        current = ((goal_distance.float() - lower) / scale).clamp(0.0, 1.0)
        following = ((next_goal_distance.float() - lower) / scale).clamp(0.0, 1.0)
        return current - float(self.config.gamma) * following

    def _goal_jepa_potential_reward_row(self, row: dict[str, Any]) -> float:
        transition_idx = int(row["transition_idx"])
        try:
            next_distance = self._goal_jepa_potential_next_by_transition_idx[transition_idx]
        except KeyError as exc:
            raise KeyError(
                "Missing goal-potential successor distance for transition "
                f"{transition_idx}"
            ) from exc
        value = self._goal_jepa_potential_reward_tensor(
            torch.tensor([float(row["goal_jepa_distance"])]),
            torch.tensor([float(next_distance)]),
        )
        return float(value.item())

    def _raw_reward_term_row(self, row: dict[str, Any], name: str) -> float:
        if name == "negative_predicted_actual_jepa_distance":
            return -float(row["jepa_distance"])
        if name == "terminal_success":
            return float(bool(row.get("episode_success", False)) and bool(row["done"]))
        if name == "terminal_outcome_cost":
            if not bool(row["done"]):
                return -1.0
            return 0.0 if bool(row.get("episode_success", False)) else -1000.0
        if name == "successful_episode_progress":
            return self._progress_reward_row(row)
        if name == "goal_jepa_progress":
            return self._goal_jepa_progress_reward_row(row)
        if name == "goal_jepa_potential_shaping":
            return self._goal_jepa_potential_reward_row(row)
        raise KeyError(f"Unknown reward term: {name}")

    @staticmethod
    def _reward_normalization_mode(term: dict[str, Any]) -> str:
        value = term.get("normalize", False)
        if value is True:
            return "z_score"
        if value in {False, None}:
            return "none"
        if value == "z_score":
            return "z_score"
        if value == "scale_neg1_pos1":
            return "scale_neg1_pos1"
        raise ValueError(f"Unsupported reward normalization mode: {value!r}")

    @staticmethod
    def _apply_reward_normalization(
        value: Any,
        stats: dict[str, float],
    ) -> Any:
        mode = stats["mode"]
        if mode == "z_score":
            return (value - stats["mean"]) / stats["std"]
        if mode == "scale_neg1_pos1":
            scale = stats["max"] - stats["min"]
            if scale < 1e-6:
                return value * 0.0
            return 2.0 * (value - stats["min"]) / scale - 1.0
        if mode == "none":
            return value
        raise ValueError(f"Unsupported reward normalization mode: {mode!r}")

    def _apply_final_reward_normalization(self, value: Any) -> Any:
        if not self._has_final_reward_normalization():
            return value
        if self.final_reward_stats is None:
            raise RuntimeError("final reward normalization stats are not fitted")
        scale = self.final_reward_stats["max"] - self.final_reward_stats["min"]
        if scale < 1e-6:
            return value * 0.0
        return 2.0 * (value - self.final_reward_stats["min"]) / scale - 1.0

    def _compose_reward_row(
        self,
        row: dict[str, Any],
        *,
        apply_final_normalization: bool = True,
    ) -> float:
        reward = 0.0
        for term in self.config.reward_terms:
            name = str(term["name"])
            value = self._raw_reward_term_row(row, name)
            if self._reward_normalization_mode(term) != "none":
                stats = self.reward_term_stats[name]
                value = self._apply_reward_normalization(value, stats)
            reward += float(term["weight"]) * value
        if apply_final_normalization:
            reward = self._apply_final_reward_normalization(reward)
        return reward

    def _fit_reward_term_stats(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("Cannot normalize rewards from an empty cache")
        for term in self.config.reward_terms:
            mode = self._reward_normalization_mode(term)
            if mode == "none":
                continue
            name = str(term["name"])
            values = torch.tensor(
                [self._raw_reward_term_row(row, name) for row in rows],
                dtype=torch.float32,
            )
            if mode == "z_score":
                std = float(values.std(unbiased=False).clamp_min(1e-6))
                self.reward_term_stats[name] = {
                    "mode": mode,
                    "mean": float(values.mean()),
                    "std": std,
                }
                detail = (
                    f"mean={self.reward_term_stats[name]['mean']:.6f} "
                    f"std={std:.6f}"
                )
            elif mode == "scale_neg1_pos1":
                minimum = float(values.min())
                maximum = float(values.max())
                self.reward_term_stats[name] = {
                    "mode": mode,
                    "min": minimum,
                    "max": maximum,
                }
                detail = f"min={minimum:.6f} max={maximum:.6f}"
            else:
                raise ValueError(f"Unsupported reward normalization mode: {mode}")
            _rank_log(
                self.rank,
                "reward normalization ready: "
                f"{name} mode={mode} {detail}",
            )

    def _fit_final_reward_stats(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("Cannot normalize final rewards from an empty cache")
        values = torch.tensor(
            [
                self._compose_reward_row(
                    row,
                    apply_final_normalization=False,
                )
                for row in rows
            ],
            dtype=torch.float32,
        )
        minimum = float(values.min())
        maximum = float(values.max())
        self.final_reward_stats = {
            "mode": "scale_neg1_pos1",
            "min": minimum,
            "max": maximum,
        }
        _rank_log(
            self.rank,
            "final reward normalization ready: "
            f"mode=scale_neg1_pos1 min={minimum:.6f} max={maximum:.6f}",
        )

    def _compose_reward_batch(
        self,
        distance: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        reward = torch.zeros_like(distance, dtype=torch.float32, device=self.device)
        term_values = self._reward_term_batch_values(distance, batch)
        for term in self.config.reward_terms:
            name = str(term["name"])
            value = term_values[name]
            if self._reward_normalization_mode(term) != "none":
                stats = self.reward_term_stats[name]
                value = self._apply_reward_normalization(value, stats)
            reward = reward + float(term["weight"]) * value
        reward = self._apply_final_reward_normalization(reward)
        return reward

    def _reward_term_batch_values(
        self,
        distance: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        values: dict[str, torch.Tensor] = {}
        for term in self.config.reward_terms:
            name = str(term["name"])
            if name in values:
                continue
            if name == "negative_predicted_actual_jepa_distance":
                value = -distance
            elif name == "terminal_success":
                value = (
                    batch["episode_success"].to(self.device, non_blocking=True).bool()
                    & batch["done"].to(self.device, non_blocking=True).bool()
                ).float().view_as(distance)
            elif name == "terminal_outcome_cost":
                done = batch["done"].to(
                    self.device,
                    non_blocking=True,
                ).bool().view_as(distance)
                success = batch["episode_success"].to(
                    self.device,
                    non_blocking=True,
                ).bool().view_as(distance)
                value = torch.where(
                    done,
                    torch.where(
                        success,
                        torch.zeros_like(distance, dtype=torch.float32),
                        torch.full_like(distance, -1000.0, dtype=torch.float32),
                    ),
                    torch.full_like(distance, -1.0, dtype=torch.float32),
                )
            elif name == "successful_episode_progress":
                value = self._progress_reward_tensor(
                    batch["rl_chunk_idx"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                    batch["episode_chunk_count"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                    batch["episode_success"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                )
            elif name == "goal_jepa_progress":
                if "goal_jepa_distance" not in batch:
                    self._require_goal_distance_available(source="current batch")
                value = self._goal_jepa_progress_reward_tensor(
                    batch["goal_jepa_distance"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                    batch["episode_goal_jepa_distance_max"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                    batch["rl_chunk_idx"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                    batch["episode_chunk_count"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                )
            elif name == "goal_jepa_potential_shaping":
                if "goal_jepa_distance" not in batch:
                    self._require_goal_distance_available(source="current batch")
                transition_indices = batch["transition_idx"].detach().cpu().view(-1).tolist()
                try:
                    next_goal_distance = torch.tensor(
                        [
                            self._goal_jepa_potential_next_by_transition_idx[
                                int(transition_idx)
                            ]
                            for transition_idx in transition_indices
                        ],
                        dtype=torch.float32,
                        device=self.device,
                    ).view_as(distance)
                except KeyError as exc:
                    raise KeyError(
                        "Goal-potential successor lookup is missing a batch transition"
                    ) from exc
                value = self._goal_jepa_potential_reward_tensor(
                    batch["goal_jepa_distance"].to(
                        self.device,
                        non_blocking=True,
                    ).view_as(distance),
                    next_goal_distance,
                )
            else:
                raise KeyError(f"Unknown reward term: {name}")
            values[name] = value
        return values

    def _reward_term_batch_metrics(
        self,
        distance: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        values = self._reward_term_batch_values(distance, batch)
        metrics: dict[str, float] = {}
        for term in self.config.reward_terms:
            name = str(term["name"])
            raw_value = values[name].detach().float()
            normalized_value = values[name]
            if self._reward_normalization_mode(term) != "none":
                normalized_value = self._apply_reward_normalization(
                    normalized_value,
                    self.reward_term_stats[name],
                )
            weighted_value = normalized_value.detach().float() * float(term["weight"])
            metrics[f"reward_term/{name}_raw_mean"] = float(raw_value.mean().cpu())
            metrics[f"reward_term/{name}_weighted_mean"] = float(
                weighted_value.mean().cpu()
            )
        return metrics

    def _chunk_discount_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        done = batch["done"].to(self.device, non_blocking=True).bool()
        return torch.where(
            done,
            torch.zeros_like(done, dtype=torch.float32),
            torch.full_like(done, float(self.config.gamma), dtype=torch.float32),
        )

    @staticmethod
    def _discounted_returns_by_episode(
        rows: list[dict[str, Any]],
    ) -> dict[int, float]:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in rows:
            key = (int(row["dataset_id"]), int(row["episode_index"]))
            grouped.setdefault(key, []).append(row)

        returns: dict[int, float] = {}
        for episode_rows in grouped.values():
            ordered = sorted(episode_rows, key=lambda item: int(item["rl_chunk_idx"]))
            running = 0.0
            for row in reversed(ordered):
                reward = float(row["reward"])
                discount = float(row["discount"])
                running = reward + discount * running
                returns[int(row["transition_idx"])] = running
        return returns

    def _packed_cache_reward_rows(self) -> list[dict[str, Any]]:
        assert isinstance(self.dataset, Phase3PackedCacheDataset)
        rows: list[dict[str, Any]] = []
        for shard_id, shard_path in sorted(self.dataset.shards.items()):
            with safe_open(shard_path, framework="pt", device="cpu") as tensors:
                transition_idx = tensors.get_tensor("transition_idx")
                tensor_keys = set(tensors.keys())
                has_episode_metadata = (
                    bool(self.dataset.manifest.get("has_episode_metadata", False))
                    and "episode_success" in tensor_keys
                    and "episode_chunk_count" in tensor_keys
                )
                has_goal_distance = (
                    "goal_jepa_distance" in tensor_keys
                    and "episode_goal_jepa_distance_max" in tensor_keys
                )
                if (
                    self._reward_terms_need_episode_metadata()
                    and not has_episode_metadata
                ):
                    self._require_reward_metadata_available(
                        source="packed cache"
                    )
                if (
                    self._reward_terms_need_goal_jepa_distance()
                    and not has_goal_distance
                ):
                    self._require_goal_distance_available(source="packed cache")
                jepa_distance = tensors.get_tensor("jepa_distance").float()
                if has_goal_distance:
                    goal_jepa_distance = tensors.get_tensor("goal_jepa_distance").float()
                    goal_jepa_distance_max = tensors.get_tensor(
                        "episode_goal_jepa_distance_max"
                    ).float()
                else:
                    goal_jepa_distance = torch.zeros_like(jepa_distance)
                    goal_jepa_distance_max = torch.ones_like(jepa_distance)
                cached_reward = tensors.get_tensor("cached_reward").float()
                dataset_id = tensors.get_tensor("dataset_id")
                episode_index = tensors.get_tensor("episode_index")
                rl_chunk_idx = tensors.get_tensor("rl_chunk_idx")
                done = tensors.get_tensor("done")
                if has_episode_metadata:
                    episode_success = tensors.get_tensor("episode_success")
                    episode_chunk_count = tensors.get_tensor("episode_chunk_count")
                else:
                    episode_success = torch.zeros_like(done, dtype=torch.bool)
                    episode_chunk_count = torch.zeros_like(rl_chunk_idx)
                count = int(transition_idx.shape[0])
                for row_id in range(count):
                    done_value = bool(done[row_id])
                    rows.append(
                        {
                            "transition_idx": int(transition_idx[row_id]),
                            "jepa_distance": float(jepa_distance[row_id]),
                            "goal_jepa_distance": float(goal_jepa_distance[row_id]),
                            "episode_goal_jepa_distance_max": float(
                                goal_jepa_distance_max[row_id]
                            ),
                            "cached_reward": float(cached_reward[row_id]),
                            "discount": (
                                0.0
                                if done_value
                                else float(self.config.gamma)
                            ),
                            "dataset_id": int(dataset_id[row_id]),
                            "episode_index": int(episode_index[row_id]),
                            "rl_chunk_idx": int(rl_chunk_idx[row_id]),
                            "done": done_value,
                            "episode_success": bool(episode_success[row_id]),
                            "episode_chunk_count": int(episode_chunk_count[row_id]),
                            "_has_episode_metadata": has_episode_metadata,
                        }
                    )
        return rows

    def _canonical_cache_reward_rows(self) -> list[dict[str, Any]]:
        assert isinstance(self.dataset, Phase3CanonicalEpisodeCacheDataset)
        rows: list[dict[str, Any]] = []
        for path in self.dataset._episode_files():
            with safe_open(path, framework="pt", device="cpu") as tensors:
                tensor_keys = set(tensors.keys())
                has_episode_metadata = (
                    "episode_success" in tensor_keys
                    and "episode_chunk_count" in tensor_keys
                )
                has_goal_distance = (
                    "goal_jepa_distance" in tensor_keys
                    and "episode_goal_jepa_distance_max" in tensor_keys
                )
                if (
                    self._reward_terms_need_episode_metadata()
                    and not has_episode_metadata
                ):
                    self._require_reward_metadata_available(
                        source="canonical episode cache"
                    )
                if (
                    self._reward_terms_need_goal_jepa_distance()
                    and not has_goal_distance
                ):
                    self._require_goal_distance_available(
                        source="canonical episode cache"
                    )
                transition_idx = tensors.get_tensor("transition_idx")
                jepa_distance = tensors.get_tensor("jepa_distance").float()
                if has_goal_distance:
                    goal_jepa_distance = tensors.get_tensor("goal_jepa_distance").float()
                    goal_jepa_distance_max = tensors.get_tensor(
                        "episode_goal_jepa_distance_max"
                    ).float()
                else:
                    goal_jepa_distance = torch.zeros_like(jepa_distance)
                    goal_jepa_distance_max = torch.ones_like(jepa_distance)
                cached_reward = tensors.get_tensor("cached_reward").float()
                discount = tensors.get_tensor("discount")
                dataset_id = tensors.get_tensor("dataset_id")
                episode_index = tensors.get_tensor("episode_index")
                rl_chunk_idx = tensors.get_tensor("rl_chunk_idx")
                done = tensors.get_tensor("done")
                if has_episode_metadata:
                    episode_success = tensors.get_tensor("episode_success")
                    episode_chunk_count = tensors.get_tensor("episode_chunk_count")
                else:
                    episode_success = torch.zeros_like(done, dtype=torch.bool)
                    episode_chunk_count = torch.zeros_like(rl_chunk_idx)
                for row in range(int(transition_idx.shape[0])):
                    rows.append(
                        {
                            "transition_idx": int(transition_idx[row]),
                            "jepa_distance": float(jepa_distance[row]),
                            "goal_jepa_distance": float(goal_jepa_distance[row]),
                            "episode_goal_jepa_distance_max": float(
                                goal_jepa_distance_max[row]
                            ),
                            "cached_reward": float(cached_reward[row]),
                            "discount": (
                                0.0
                                if bool(done[row])
                                else float(self.config.gamma)
                            ),
                            "dataset_id": int(dataset_id[row]),
                            "episode_index": int(episode_index[row]),
                            "rl_chunk_idx": int(rl_chunk_idx[row]),
                            "done": bool(done[row]),
                            "episode_success": bool(episode_success[row]),
                            "episode_chunk_count": int(episode_chunk_count[row]),
                            "_has_episode_metadata": has_episode_metadata,
                        }
                    )
        return rows

    def _federated_canonical_cache_reward_rows(self) -> list[dict[str, Any]]:
        assert isinstance(
            self.dataset,
            Phase3FederatedCanonicalEpisodeCacheDataset,
        )
        metadata = self.dataset.federated_reward_metadata
        if (
            self._reward_terms_need_episode_metadata()
            and not bool(metadata["has_episode_metadata"].all())
        ):
            self._require_reward_metadata_available(
                source="federated canonical episode cache"
            )
        if (
            self._reward_terms_need_goal_jepa_distance()
            and not bool(metadata["has_goal_jepa_distance"].all())
        ):
            self._require_goal_distance_available(
                source="federated canonical episode cache"
            )
        rows: list[dict[str, Any]] = []
        transition_idx = metadata["transition_idx"]
        jepa_distance = metadata["jepa_distance"]
        goal_jepa_distance = metadata["goal_jepa_distance"]
        goal_jepa_distance_max = metadata["episode_goal_jepa_distance_max"]
        cached_reward = metadata["cached_reward"]
        dataset_id = metadata["dataset_id"]
        episode_index = metadata["episode_index"]
        rl_chunk_idx = metadata["rl_chunk_idx"]
        done = metadata["done"]
        episode_success = metadata["episode_success"]
        episode_chunk_count = metadata["episode_chunk_count"]
        has_episode_metadata = metadata["has_episode_metadata"]
        for row in range(int(transition_idx.shape[0])):
            rows.append(
                {
                    "transition_idx": int(transition_idx[row]),
                    "jepa_distance": float(jepa_distance[row]),
                    "goal_jepa_distance": float(goal_jepa_distance[row]),
                    "episode_goal_jepa_distance_max": float(
                        goal_jepa_distance_max[row]
                    ),
                    "cached_reward": float(cached_reward[row]),
                    "discount": (
                        0.0 if bool(done[row]) else float(self.config.gamma)
                    ),
                    "dataset_id": int(dataset_id[row]),
                    "episode_index": int(episode_index[row]),
                    "rl_chunk_idx": int(rl_chunk_idx[row]),
                    "done": bool(done[row]),
                    "episode_success": bool(episode_success[row]),
                    "episode_chunk_count": int(episode_chunk_count[row]),
                    "_has_episode_metadata": bool(has_episode_metadata[row]),
                }
            )
        return rows

    def _episode_cache_reward_rows(self) -> list[dict[str, Any]]:
        if not self.config.phase3_precomputed_cache_dir:
            if self._reward_terms_need_goal_jepa_distance():
                self._require_goal_distance_available(source="online reward rows")
            if self.config.reward_source == "negative_predicted_actual_jepa_distance":
                raise ValueError(
                    "MC-flow phase-3 JEPA rewards require a precomputed "
                    "episode cache or packed cache"
                )
            return [
                {
                    "transition_idx": index,
                    "jepa_distance": -float(record.reward)
                    / max(float(self.config.jepa_reward_weight), 1e-12),
                    "cached_reward": record.reward,
                    "discount": record.discount,
                    "dataset_id": record.episode_key[0],
                    "episode_index": record.episode_key[1],
                    "rl_chunk_idx": record.rl_chunk_idx,
                    "done": record.done,
                    "episode_success": record.episode_success,
                    "episode_chunk_count": record.episode_chunk_count,
                    "_has_episode_metadata": True,
                }
                for index, record in enumerate(self.dataset.records)
            ]

        cache_dir = Path(self.config.phase3_precomputed_cache_dir)
        files = sorted(cache_dir.glob("dataset-*/shard-*/episode_d*_ep*.pt"))
        if not files:
            raise FileNotFoundError(
                f"No Phase 3 episode cache files found under {cache_dir}"
            )
        rows: list[dict[str, Any]] = []
        for path in files:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not self._cache_signature_matches(payload.get("signature")):
                raise ValueError(
                    "Phase 3 cache signature mismatch while building MC "
                    f"returns: {path}"
                )
            episode_key = payload["episode_key"]
            dataset_id = int(episode_key[0])
            episode_index = int(episode_key[1])
            for chunk in payload["chunks"]:
                transition_idx = int(chunk["transition_idx"])
                record = self.dataset.records[transition_idx]
                has_episode_metadata = (
                    "episode_success" in chunk
                    and "episode_chunk_count" in chunk
                )
                if (
                    self._reward_terms_need_episode_metadata()
                    and not has_episode_metadata
                ):
                    self._require_reward_metadata_available(
                        source=f"episode cache {path}"
                    )
                has_goal_distance = (
                    "goal_jepa_distance" in chunk
                    and "episode_goal_jepa_distance_max" in chunk
                )
                if (
                    self._reward_terms_need_goal_jepa_distance()
                    and not has_goal_distance
                ):
                    self._require_goal_distance_available(
                        source=f"episode cache {path}"
                    )
                rows.append(
                    {
                        "transition_idx": transition_idx,
                        "jepa_distance": float(chunk["jepa_distance"]),
                        "goal_jepa_distance": float(
                            chunk.get("goal_jepa_distance", 0.0)
                        ),
                        "episode_goal_jepa_distance_max": float(
                            chunk.get("episode_goal_jepa_distance_max", 1.0)
                        ),
                        "cached_reward": float(chunk.get("reward", 0.0)),
                        "discount": float(record.discount),
                        "dataset_id": dataset_id,
                        "episode_index": episode_index,
                        "rl_chunk_idx": int(chunk["rl_chunk_idx"]),
                        "done": bool(chunk.get("done", record.done)),
                        "episode_success": bool(
                            chunk.get("episode_success", record.episode_success)
                        ),
                        "episode_chunk_count": int(
                            chunk.get(
                                "episode_chunk_count",
                                record.episode_chunk_count,
                            )
                        ),
                        "_has_episode_metadata": (
                            has_episode_metadata
                            or bool(record.episode_chunk_count)
                        ),
                    }
                )
        return rows

    def _reward_rows(self) -> list[dict[str, Any]]:
        if self._reward_rows_cache is None:
            if self.config.phase3_use_packed_cache:
                self._reward_rows_cache = self._packed_cache_reward_rows()
            elif isinstance(
                self.dataset,
                Phase3FederatedCanonicalEpisodeCacheDataset,
            ):
                self._reward_rows_cache = (
                    self._federated_canonical_cache_reward_rows()
                )
            elif isinstance(self.dataset, Phase3CanonicalEpisodeCacheDataset):
                self._reward_rows_cache = self._canonical_cache_reward_rows()
            else:
                self._reward_rows_cache = self._episode_cache_reward_rows()
        return self._reward_rows_cache

    def _init_mc_returns(self) -> None:
        _rank_log(self.rank, "building normalized MC return targets")
        rows = [dict(row, reward=self._compose_reward_row(row)) for row in self._reward_rows()]
        returns = self._discounted_returns_by_episode(rows)
        if not returns:
            raise ValueError("No MC-flow returns were built")
        values = torch.tensor(list(returns.values()), dtype=torch.float32)
        self.mc_return_by_transition_idx = returns
        self.mc_return_mean = float(values.mean())
        self.mc_return_std = float(values.std(unbiased=False).clamp_min(1e-6))
        _rank_log(
            self.rank,
            "MC returns ready: "
            f"count={len(returns)} mean={self.mc_return_mean:.6f} "
            f"std={self.mc_return_std:.6f}",
        )

    def _mc_return_targets(self, batch: dict[str, Any]) -> torch.Tensor:
        transition_indices = [
            int(value) for value in batch["transition_idx"].detach().cpu().view(-1)
        ]
        missing = [
            index
            for index in transition_indices
            if index not in self.mc_return_by_transition_idx
        ]
        if missing:
            raise KeyError(
                "Missing MC return targets for transition indices: "
                f"{missing[:10]}"
            )
        target = torch.tensor(
            [self.mc_return_by_transition_idx[index] for index in transition_indices],
            device=self.device,
            dtype=torch.float32,
        )
        target = (target - self.mc_return_mean) / self.mc_return_std
        if self.config.mc_return_clip is not None:
            clip_min, clip_max = self.config.mc_return_clip
            target = target.clamp(float(clip_min), float(clip_max))
        return target

    def _init_packed_cache_loader(self) -> None:
        assert self.config.phase3_packed_cache_dir is not None
        if self.config.phase3_exclude_last_n_chunks:
            raise ValueError(
                "phase3_exclude_last_n_chunks is not supported for packed "
                "transition shards; use canonical_episode cache training"
            )
        _rank_log(self.rank, "building packed Phase 3 cache dataset")
        self.dataset = Phase3PackedCacheDataset(
            self.config.phase3_packed_cache_dir,
            critic_feature_layers=self.config.critic_feature_layers,
        )
        self.cached_feature_layers = tuple(self.dataset.cached_feature_layers)
        actual_signature = self.dataset.manifest.get("source_signature")
        if not self._cache_signature_matches(actual_signature):
            raise ValueError(
                "Packed Phase 3 cache signature mismatch; regenerate the "
                "packed cache from the episode cache for this config"
            )
        if float(self.dataset.manifest.get("discount_gamma", -1.0)) != float(
            self.config.gamma
        ):
            raise ValueError(
                "Packed Phase 3 cache was generated with a different gamma; "
                "regenerate the packed cache"
            )
        if int(self.dataset.manifest.get("action_per_frame", -1)) != int(
            self.base_config.action_per_frame
        ):
            raise ValueError(
                "Packed Phase 3 cache action_per_frame does not match the "
                "base LingBot config"
            )
        if tuple(self.dataset.feature_keys) != self._cache_feature_keys():
            raise ValueError(
                "Packed Phase 3 cache feature keys do not match this critic: "
                f"{self.dataset.feature_keys} != {self._cache_feature_keys()}"
            )

        loader_kwargs = self._dataloader_worker_kwargs()
        if self.config.phase3_cache_batching == "shard_local":
            self.sampler = ShardLocalBatchSampler(
                self.dataset,
                batch_size=self.config.batch_size,
                rank=self.rank,
                world_size=self.world_size,
                seed=self.config.seed,
            )
            loader = DataLoader(
                self.dataset,
                batch_sampler=self.sampler,
                collate_fn=phase3_packed_collate,
                **loader_kwargs,
            )
        else:
            self.sampler = (
                DistributedSampler(
                    self.dataset,
                    num_replicas=self.world_size,
                    rank=self.rank,
                    shuffle=True,
                    seed=self.config.seed,
                )
                if self.world_size > 1
                else None
            )
            loader = DataLoader(
                self.dataset,
                batch_size=self.config.batch_size,
                shuffle=self.sampler is None,
                sampler=self.sampler,
                collate_fn=phase3_packed_collate,
                **loader_kwargs,
            )
        if self.config.phase3_cuda_prefetch:
            self.loader = CudaPrefetcher(loader, self.device)
        else:
            self.loader = loader
        _rank_log(
            self.rank,
            "packed cache ready: "
            f"transitions={len(self.dataset)} "
            f"cached_layers={list(self.cached_feature_layers)} "
            f"critic_layers={list(self.config.critic_feature_layers)}",
        )

    def _init_federated_canonical_cache_loader(self) -> None:
        assert self.config.phase3_federated_cache_catalog is not None
        _rank_log(
            self.rank,
            "building federated canonical Phase 3 episode cache dataset",
        )
        self.dataset = Phase3FederatedCanonicalEpisodeCacheDataset(
            self.config.phase3_federated_cache_catalog,
            critic_feature_layers=self.config.critic_feature_layers,
            critic_type=self.config.critic_type,
            infer_latent_chunk_size=self.config.infer_latent_chunk_size,
            gamma=self.config.gamma,
            lru_size=self.config.phase3_episode_cache_lru_size,
            exclude_last_n_chunks=self.config.phase3_exclude_last_n_chunks,
            episode_success_overlay=self.config.phase3_episode_success_overlay,
            language_sidecar_catalog=self.config.phase3_language_sidecar_catalog,
            artifacts_manifest=self.config.phase3_federated_cache_artifacts_manifest,
            window_size=self.config.window_size,
        )
        catalog_sources = {
            int(raw_source["source_id"]): raw_source
            for raw_source in self.dataset.catalog["sources"]
        }
        for source in self.dataset.sources:
            source_contract = catalog_sources[source.source_id]
            source_base_config_name = source_contract.get(
                "base_config_name", self.config.base_config_name
            )
            if not isinstance(source_base_config_name, str) or not source_base_config_name:
                raise ValueError(
                    "Federated source base_config_name must be a non-empty string "
                    f"for source_id={source.source_id}"
                )
            expected_signature = self._canonical_cache_signature(
                cached_feature_layers=source.cached_feature_layers,
            )
            # Dataset roots legitimately differ in a federated run. The catalog
            # pins each source manifest and names its base config; all payload-
            # producing settings must still match this training contract.
            expected_signature["base_config_name"] = source_base_config_name
            if source.manifest.get("signature") != expected_signature:
                raise ValueError(
                    "Federated canonical cache signature mismatch for "
                    f"source_id={source.source_id} task={source.task_name}; "
                    "regenerate or remove the incompatible source"
                )
        self.cached_feature_layers = tuple(self.config.critic_feature_layers)
        sampler_type = (
            GlobalBalancedEpisodeBatchSampler
            if self.config.phase3_global_balanced_batching
            else EpisodeLocalBatchSampler
        )
        self.sampler = sampler_type(
            self.dataset,
            batch_size=self.config.batch_size,
            episodes_per_batch=self.config.phase3_episodes_per_batch,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.config.seed,
            drop_last=self.config.phase3_drop_last,
            context_length_aware=(
                self.config.phase3_context_length_aware_batching
            ),
            preserve_episode_group_order=(
                self.config.phase3_preserve_episode_group_order
            ),
            source_balanced=(
                self.config.phase3_federated_source_balanced_sampling
            ),
        )
        if self.config.phase3_cache_residency != "on_demand":
            loader = BoundedGroupPrefetchLoader(
                self.dataset,
                self.sampler,
                group_size=self.config.phase3_cache_group_size,
                group_prefetch_depth=(
                    self.config.phase3_cache_group_prefetch_depth
                ),
                max_resident_bytes=int(
                    self.config.phase3_cache_max_resident_gib * 1024**3
                ),
                load_workers=self.config.phase3_cache_load_workers,
                batch_prefetch_depth=self.config.phase3_batch_prefetch_depth,
                batch_workers=max(1, self.config.num_workers),
                pin_memory=self.config.phase3_pin_memory,
                full=self.config.phase3_cache_residency == "full",
                warmup=self.config.phase3_cache_warmup,
            )
        else:
            loader = DataLoader(
                self.dataset,
                batch_sampler=self.sampler,
                collate_fn=phase3_canonical_collate,
                **self._dataloader_worker_kwargs(),
            )
        if (
            self.config.phase3_cache_residency == "on_demand"
            and self.config.phase3_loader_mode == "transition_range"
        ):
            loader = TransitionRangePrefetchLoader(
                self.dataset,
                self.sampler,
                transition_prefetch_max_bytes=int(
                    self.config.phase3_transition_max_inflight_gib * 1024**3
                ),
                transition_prefetch_workers=(
                    self.config.phase3_transition_prefetch_workers
                ),
                transition_prefetch_lookahead=(
                    self.config.phase3_transition_prefetch_lookahead
                ),
                transition_batch_workers=(
                    self.config.phase3_transition_batch_workers
                ),
                batch_prefetch_depth=self.config.phase3_batch_prefetch_depth,
                max_active_episodes=(
                    self.config.phase3_transition_max_active_episodes
                ),
                pin_memory=self.config.phase3_pin_memory,
            )
        elif (
            self.config.phase3_cache_residency == "on_demand"
            and self.config.phase3_episode_aware_prefetch
        ):
            loader = EpisodeAwarePrefetchLoader(
                self.dataset,
                self.sampler,
                episode_cache_max_bytes=int(
                    self.config.phase3_episode_cache_max_gib * 1024**3
                ),
                episode_load_workers=(
                    self.config.phase3_episode_prefetch_workers
                ),
                episode_lookahead_batches=(
                    self.config.phase3_episode_prefetch_lookahead_batches
                ),
                batch_prefetch_depth=self.config.phase3_batch_prefetch_depth,
                batch_workers=max(1, self.config.num_workers),
                pin_memory=self.config.phase3_pin_memory,
            )
        if self.config.phase3_cuda_prefetch:
            if (
                self.config.phase3_cache_residency == "on_demand"
                and self.config.phase3_loader_mode == "transition_range"
            ):
                self.loader = AsyncCudaPrefetcher(
                    loader,
                    self.device,
                    depth=self.config.phase3_cuda_prefetch_depth,
                )
            elif (
                self.config.phase3_cache_residency == "on_demand"
                and self.config.phase3_episode_aware_prefetch
            ):
                self.loader = AsyncCudaPrefetcher(
                    loader,
                    self.device,
                    depth=self.config.phase3_cuda_prefetch_depth,
                )
            elif (
                self.config.phase3_cache_residency != "on_demand"
                and self.config.phase3_cuda_prefetch_depth > 1
            ):
                # Preserve the legacy single-stage behavior at depth 1 while
                # allowing bounded/full residency loaders to overlap CPU batch
                # materialization, H2D transfer, and critic computation.
                self.loader = AsyncCudaPrefetcher(
                    loader,
                    self.device,
                    depth=self.config.phase3_cuda_prefetch_depth,
                )
            else:
                self.loader = CudaPrefetcher(loader, self.device)
        else:
            self.loader = loader
        _rank_log(
            self.rank,
            "federated canonical cache ready: "
            f"sources={len(self.dataset.sources)} "
            f"transitions={len(self.dataset)} "
            f"episodes={len(self.dataset.positions_by_episode)} "
            f"catalog_sha256={self.dataset.catalog_sha256} "
            f"critic_layers={list(self.config.critic_feature_layers)} "
            f"episodes_per_batch={self.config.phase3_episodes_per_batch} "
            f"drop_last={self.config.phase3_drop_last} "
            "context_length_aware="
            f"{self.config.phase3_context_length_aware_batching} "
            "global_balanced_batching="
            f"{self.config.phase3_global_balanced_batching} "
            "source_balanced_sampling="
            f"{self.config.phase3_federated_source_balanced_sampling} "
            "preserve_episode_group_order="
            f"{self.config.phase3_preserve_episode_group_order} "
            "episode_aware_prefetch="
            f"{self.config.phase3_episode_aware_prefetch}",
        )

    def _init_canonical_cache_loader(self) -> None:
        assert self.config.phase3_precomputed_cache_dir is not None
        _rank_log(self.rank, "building canonical Phase 3 episode cache dataset")
        cache_dir = Path(self.config.phase3_precomputed_cache_dir)
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing canonical Phase 3 cache manifest: {manifest_path}"
            )
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        self.cached_feature_layers = tuple(
            int(layer) for layer in manifest["cached_feature_layers"]
        )
        if not self._canonical_cache_signature_matches(manifest.get("signature")):
            raise ValueError(
                "Canonical Phase 3 cache signature mismatch; regenerate the "
                "episode cache for this config"
            )
        self.dataset = Phase3CanonicalEpisodeCacheDataset(
            cache_dir,
            critic_feature_layers=self.config.critic_feature_layers,
            critic_type=self.config.critic_type,
            infer_latent_chunk_size=self.config.infer_latent_chunk_size,
            gamma=self.config.gamma,
            expected_signature=self._canonical_cache_signature(),
            lru_size=self.config.phase3_episode_cache_lru_size,
            exclude_last_n_chunks=self.config.phase3_exclude_last_n_chunks,
            episode_success_overlay=self.config.phase3_episode_success_overlay,
            window_size=self.config.window_size,
        )
        sampler_type = (
            GlobalBalancedEpisodeBatchSampler
            if self.config.phase3_global_balanced_batching
            else EpisodeLocalBatchSampler
        )
        self.sampler = sampler_type(
            self.dataset,
            batch_size=self.config.batch_size,
            episodes_per_batch=self.config.phase3_episodes_per_batch,
            rank=self.rank,
            world_size=self.world_size,
            seed=self.config.seed,
            drop_last=self.config.phase3_drop_last,
            context_length_aware=(
                self.config.phase3_context_length_aware_batching
            ),
            preserve_episode_group_order=(
                self.config.phase3_preserve_episode_group_order
            ),
        )
        if self.config.phase3_cache_residency != "on_demand":
            loader = BoundedGroupPrefetchLoader(
                self.dataset,
                self.sampler,
                group_size=self.config.phase3_cache_group_size,
                group_prefetch_depth=(
                    self.config.phase3_cache_group_prefetch_depth
                ),
                max_resident_bytes=int(
                    self.config.phase3_cache_max_resident_gib * 1024**3
                ),
                load_workers=self.config.phase3_cache_load_workers,
                batch_prefetch_depth=self.config.phase3_batch_prefetch_depth,
                batch_workers=max(1, self.config.num_workers),
                pin_memory=self.config.phase3_pin_memory,
                full=self.config.phase3_cache_residency == "full",
                warmup=self.config.phase3_cache_warmup,
            )
        else:
            loader = DataLoader(
                self.dataset,
                batch_sampler=self.sampler,
                collate_fn=phase3_canonical_collate,
                **self._dataloader_worker_kwargs(),
            )
        if (
            self.config.phase3_cache_residency == "on_demand"
            and self.config.phase3_loader_mode == "transition_range"
        ):
            loader = TransitionRangePrefetchLoader(
                self.dataset,
                self.sampler,
                transition_prefetch_max_bytes=int(
                    self.config.phase3_transition_max_inflight_gib * 1024**3
                ),
                transition_prefetch_workers=(
                    self.config.phase3_transition_prefetch_workers
                ),
                transition_prefetch_lookahead=(
                    self.config.phase3_transition_prefetch_lookahead
                ),
                transition_batch_workers=(
                    self.config.phase3_transition_batch_workers
                ),
                batch_prefetch_depth=self.config.phase3_batch_prefetch_depth,
                max_active_episodes=(
                    self.config.phase3_transition_max_active_episodes
                ),
                pin_memory=self.config.phase3_pin_memory,
            )
        elif (
            self.config.phase3_cache_residency == "on_demand"
            and self.config.phase3_episode_aware_prefetch
        ):
            loader = EpisodeAwarePrefetchLoader(
                self.dataset,
                self.sampler,
                episode_cache_max_bytes=int(
                    self.config.phase3_episode_cache_max_gib * 1024**3
                ),
                episode_load_workers=(
                    self.config.phase3_episode_prefetch_workers
                ),
                episode_lookahead_batches=(
                    self.config.phase3_episode_prefetch_lookahead_batches
                ),
                batch_prefetch_depth=self.config.phase3_batch_prefetch_depth,
                batch_workers=max(1, self.config.num_workers),
                pin_memory=self.config.phase3_pin_memory,
            )
        if self.config.phase3_cuda_prefetch:
            if (
                self.config.phase3_cache_residency == "on_demand"
                and self.config.phase3_loader_mode == "transition_range"
            ):
                self.loader = AsyncCudaPrefetcher(
                    loader,
                    self.device,
                    depth=self.config.phase3_cuda_prefetch_depth,
                )
            elif (
                self.config.phase3_cache_residency == "on_demand"
                and self.config.phase3_episode_aware_prefetch
            ):
                self.loader = AsyncCudaPrefetcher(
                    loader,
                    self.device,
                    depth=self.config.phase3_cuda_prefetch_depth,
                )
            elif (
                self.config.phase3_cache_residency != "on_demand"
                and self.config.phase3_cuda_prefetch_depth > 1
            ):
                # Preserve the legacy single-stage behavior at depth 1 while
                # allowing bounded/full residency loaders to overlap CPU batch
                # materialization, H2D transfer, and critic computation.
                self.loader = AsyncCudaPrefetcher(
                    loader,
                    self.device,
                    depth=self.config.phase3_cuda_prefetch_depth,
                )
            else:
                self.loader = CudaPrefetcher(loader, self.device)
        else:
            self.loader = loader
        _rank_log(
            self.rank,
            "canonical cache ready: "
            f"transitions={len(self.dataset)} "
            f"episodes={len(self.dataset.positions_by_episode)} "
            f"exclude_last_n_chunks={self.config.phase3_exclude_last_n_chunks} "
            f"cached_layers={list(self.cached_feature_layers)} "
            f"critic_layers={list(self.config.critic_feature_layers)} "
            f"episodes_per_batch={self.config.phase3_episodes_per_batch} "
            f"drop_last={self.config.phase3_drop_last} "
            "context_length_aware="
            f"{self.config.phase3_context_length_aware_batching} "
            "global_balanced_batching="
            f"{self.config.phase3_global_balanced_batching} "
            "preserve_episode_group_order="
            f"{self.config.phase3_preserve_episode_group_order} "
            "episode_aware_prefetch="
            f"{self.config.phase3_episode_aware_prefetch}",
        )

    def _init_online_or_episode_cache_loader(
        self,
        dataset_init_workers: int,
    ) -> None:
        _rank_log(self.rank, "loading online/cache-precompute dependencies")
        from .jepa_encoder import build_jepa_encoder
        _rank_log(self.rank, "loaded JEPA encoder dependency")
        from wan_va.dataset import MultiLatentLeRobotDataset
        _rank_log(self.rank, "loaded dataset dependency")
        from wan_va.utils import FlowMatchScheduler
        _rank_log(self.rank, "loaded scheduler dependency")
        server_prefix = self._uses_server_prefix_precompute()
        if not server_prefix:
            from wan_va.distributed.fsdp import shard_model
            _rank_log(self.rank, "loaded FSDP dependency")
            from wan_va.distributed.util import _configure_model
            _rank_log(self.rank, "loaded distributed utility dependency")

        _rank_log(self.rank, "building base latent dataset")
        base_dataset = MultiLatentLeRobotDataset(
            self.base_config,
            num_init_worker=dataset_init_workers,
        )
        missing_jepa = base_dataset.get_missing_jepa_files()
        if missing_jepa:
            raise FileNotFoundError(
                "Phase 3 requires cached actual JEPA targets; missing first "
                f"files: {missing_jepa[:5]}"
            )
        _rank_log(self.rank, "building transition dataset")
        self.dataset = ChunkTransitionDataset(
            base_dataset=base_dataset,
            infer_latent_chunk_size=self.config.infer_latent_chunk_size,
            action_per_frame=int(self.base_config.action_per_frame),
            gamma=self.config.gamma,
            reward_source=self.config.reward_source,
            include_sparse_success_reward=self.config.include_sparse_success_reward,
            jepa_reward_weight=self.config.jepa_reward_weight,
            success_reward_weight=self.config.success_reward_weight,
            require_outcomes=(
                self.config.include_sparse_success_reward
                or self._reward_terms_need_episode_metadata()
            ),
            assume_missing_success_is_success=(
                self.config.assume_missing_success_is_success
            ),
            include_previous=True,
            include_next=True,
            exclude_last_n_chunks=self.config.phase3_exclude_last_n_chunks,
        )
        self.sampler = (
            DistributedSampler(
                self.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.config.seed,
            )
            if self.world_size > 1
            else None
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.sampler is None,
            sampler=self.sampler,
            **self._dataloader_worker_kwargs(),
        )

        _rank_log(self.rank, "loading transformer checkpoint")
        if server_prefix:
            delay = float(os.getenv("PHASE3_TRANSFORMER_IMPORT_STAGGER_SECONDS", "5"))
            if delay > 0 and self.rank > 0:
                sleep_seconds = delay * int(self.rank)
                _rank_log(
                    self.rank,
                    f"staggering server-prefix transformer load by {sleep_seconds:.1f}s",
                )
                time.sleep(sleep_seconds)
            self.transformer = _load_transformer_model(
                _transformer_path(self.base_config, self.config),
                torch_dtype=self.base_config.param_dtype,
                torch_device=self.device,
                attn_mode="torch",
            )
        else:
            transformer = _load_transformer_rank0_broadcast(
                transformer_path=_transformer_path(self.base_config, self.config),
                rank=self.rank,
                device=self.device,
            )
            _rank_log(self.rank, "configuring transformer with FSDP")
            self.transformer = _configure_model(
                model=transformer,
                shard_fn=shard_model,
                param_dtype=self.base_config.param_dtype,
                device=self.device,
                eval_mode=True,
            )
        self.transformer.eval().requires_grad_(False)
        _rank_log(self.rank, "transformer ready")
        if self._uses_layer_context_features() or server_prefix:
            self.cached_feature_layers = tuple(range(len(self.transformer.blocks)))
            invalid_critic_layers = [
                layer
                for layer in self.config.critic_feature_layers
                if layer not in self.cached_feature_layers
            ]
            if invalid_critic_layers:
                raise ValueError(
                    "critic_feature_layers contain out-of-range DiT block indices: "
                    f"{invalid_critic_layers}; transformer has "
                    f"{len(self.transformer.blocks)} blocks"
                )
        else:
            self.cached_feature_layers = tuple(self.config.feature_layers)

        self.video_scheduler = FlowMatchScheduler(
            shift=self.base_config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=self.base_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )

        _rank_log(self.rank, "loading VAE")
        self.vae = _load_vae_model(
            Path(self.base_config.wan22_pretrained_model_name_or_path) / "vae",
            torch_dtype=self.base_config.param_dtype,
            torch_device=self.device,
        )
        self.vae.eval().requires_grad_(False)
        if self.load_jepa:
            _rank_log(self.rank, "loading JEPA encoder")
            self.jepa_encoder = build_jepa_encoder(
                self.config.phase3_jepa_checkpoint,
                self.device,
            )
            self.jepa_encoder.eval().requires_grad_(False)
            _rank_log(self.rank, "JEPA encoder ready")
        else:
            _rank_log(self.rank, "skipping JEPA encoder")

    def _to_device(self, batch: dict[str, torch.Tensor], key: str) -> torch.Tensor:
        tensor = batch[key].to(self.device, non_blocking=True)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype=self.base_config.param_dtype)
        return tensor

    def _sync_timing(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _time_call(
        self,
        timings: dict[str, float] | None,
        name: str,
        fn,
        *args,
        **kwargs,
    ):
        if timings is None:
            return fn(*args, **kwargs)
        self._sync_timing()
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        self._sync_timing()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - start
        return result

    def _scheduled_lr(self, completed_steps: int) -> float:
        if self.config.lr_scheduler == "constant":
            return float(self.config.learning_rate)
        denominator = max(1, int(self.config.num_steps) - 1)
        progress = min(max(int(completed_steps), 0), denominator) / float(denominator)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return float(self.config.min_lr) + (
            float(self.config.learning_rate) - float(self.config.min_lr)
        ) * cosine

    def _set_optimizer_lr(self, completed_steps: int) -> float:
        assert self.optimizer is not None
        lr = self._scheduled_lr(completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = lr * float(group.get("lr_scale", 1.0))
        return lr

    def _critic_optimizer_groups(self) -> list[dict[str, Any]]:
        assert self.bundle is not None
        inherited_parameters: list[torch.nn.Parameter] = []
        language_parameters: list[torch.nn.Parameter] = []
        for name, parameter in self.bundle.named_parameters():
            if name.startswith("target_value."):
                continue
            if is_language_critic_parameter(name):
                language_parameters.append(parameter)
            else:
                inherited_parameters.append(parameter)
        transfer_optimizer = bool(
            self.config.initialize_from
            or self.config.freeze_inherited_critic_steps > 0
            or self.config.inherited_critic_lr_scale != 1.0
            or self.config.language_critic_lr_scale != 1.0
        )
        if not transfer_optimizer:
            parameters = inherited_parameters + language_parameters
            if not parameters:
                raise RuntimeError("critic optimizer has no trainable parameters")
            return [{"params": parameters}]
        groups: list[dict[str, Any]] = []
        if inherited_parameters:
            groups.append(
                {
                    "params": inherited_parameters,
                    "name": "inherited_critic",
                    "lr_scale": self.config.inherited_critic_lr_scale,
                }
            )
        if language_parameters:
            groups.append(
                {
                    "params": language_parameters,
                    "name": "language_critic",
                    "lr_scale": self.config.language_critic_lr_scale,
                }
            )
        if not groups:
            raise RuntimeError("critic optimizer has no trainable parameter groups")
        return groups

    def _update_inherited_critic_trainability(
        self,
        *,
        force: bool = False,
    ) -> None:
        assert self.bundle is not None
        trainable = self.step >= self.config.freeze_inherited_critic_steps
        if not force and trainable == self._inherited_critic_trainable:
            return
        for name, parameter in self.bundle.named_parameters():
            if name.startswith("target_value."):
                parameter.requires_grad_(False)
            elif not is_language_critic_parameter(name):
                parameter.requires_grad_(trainable)
        self._inherited_critic_trainable = trainable
        state = "unfrozen" if trainable else "frozen"
        _rank_log(
            self.rank,
            f"inherited critic parameters {state} at step={self.step}",
        )

    def _initialize_from_parent_checkpoint(self, directory: str | Path) -> None:
        assert self.bundle is not None
        checkpoint = Path(directory)
        state_path = checkpoint / "training_state.pt"
        manifest_path = checkpoint / "manifest.json"
        if not state_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"initialize_from is not a complete checkpoint: {checkpoint}"
            )
        with manifest_path.open("rb") as handle:
            manifest_bytes = handle.read()
        manifest = json.loads(manifest_bytes)
        if manifest.get("critic_type") != "twin_query_transformer_v1":
            raise ValueError(
                "initialize_from must be a twin_query_transformer_v1 "
                f"checkpoint, found {manifest.get('critic_type')!r}"
            )
        initialize_language_critic_parameters(self.bundle)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        parent_critic = state.get("critic")
        if not isinstance(parent_critic, dict):
            raise ValueError("initialize_from training_state.pt has no critic state")
        result = self.bundle.load_state_dict(parent_critic, strict=False)
        unexpected = sorted(result.unexpected_keys)
        expected_missing = sorted(
            name
            for name in self.bundle.state_dict()
            if is_language_critic_parameter(name)
        )
        actual_missing = sorted(result.missing_keys)
        if unexpected:
            raise ValueError(
                f"initialize_from has unexpected critic keys: {unexpected}"
            )
        if actual_missing != expected_missing:
            raise ValueError(
                "initialize_from missing-key contract failed: "
                f"expected={expected_missing}, actual={actual_missing}"
            )
        if self.config.critic_type == "twin_language_query_transformer_v1":
            copy_value_language_parameters_to_target(self.bundle)
            initialization_method = (
                "identity_preserving_language_cross_attention_v1"
            )
        else:
            initialization_method = "same_architecture_critic_weight_only_v1"
        self.transfer_initialization = {
            "critic_initialization_method": initialization_method,
            "initialize_from": str(checkpoint),
            "initialize_from_step": int(state.get("step", -1)),
            "initialize_from_manifest_sha256": hashlib.sha256(
                manifest_bytes
            ).hexdigest(),
        }
        _rank_log(
            self.rank,
            "initialized critic from parent "
            f"step={self.transfer_initialization['initialize_from_step']} "
            f"missing_language_keys={len(actual_missing)}",
        )

    def _restore_transfer_metadata(self, directory: str | Path) -> None:
        manifest_path = Path(directory) / "manifest.json"
        if not manifest_path.exists():
            return
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        keys = (
            "critic_initialization_method",
            "initialize_from",
            "initialize_from_step",
            "initialize_from_manifest_sha256",
        )
        metadata = {key: manifest[key] for key in keys if key in manifest}
        self.transfer_initialization = metadata or None

    def _cache_feature_keys(self) -> tuple[str, ...]:
        if self._uses_layer_context_features():
            return PHASE3_QUERY_FEATURE_KEYS
        return PHASE3_MLP_FEATURE_KEYS

    def _uses_layer_context_features(self) -> bool:
        return self.config.critic_type in {
            "twin_query_transformer_v1",
            "twin_language_query_transformer_v1",
            "twin_flow_scalar_transformer_v1",
        }

    def _uses_server_prefix_precompute(self) -> bool:
        return self.config.phase3_precompute_mode == "server_prefix"

    @staticmethod
    def _jsonable(value: Any) -> Any:
        return json.loads(json.dumps(value))

    def _select_feature_layers(
        self,
        features: dict[str, torch.Tensor],
        cached_feature_layers: tuple[int, ...],
    ) -> dict[str, torch.Tensor]:
        layer_to_offset = {
            int(layer): offset for offset, layer in enumerate(cached_feature_layers)
        }
        if -1 in self.config.critic_feature_layers:
            layer_to_offset[-1] = len(cached_feature_layers) - 1
        missing = [
            layer
            for layer in self.config.critic_feature_layers
            if layer not in layer_to_offset
        ]
        if missing:
            raise ValueError(
                "Cached features do not contain requested critic layers: "
                f"{missing}"
            )
        offsets = torch.tensor(
            [layer_to_offset[layer] for layer in self.config.critic_feature_layers],
            device=self.device,
            dtype=torch.long,
        )
        if not self._uses_layer_context_features():
            selected = {}
            for key, tensor in features.items():
                if not key.endswith("_tokens"):
                    selected[key] = tensor
                    continue
                if tensor.ndim == 4:
                    tensor = tensor.unsqueeze(1)
                if tensor.ndim != 5:
                    raise ValueError(
                        f"{key} must have shape [B,F,S,D] or [B,L,F,S,D], "
                        f"got {tuple(tensor.shape)}"
                    )
                selected[key] = self._concat_mlp_layer_features(
                    tensor.index_select(1, offsets),
                    key=key,
                )
            return selected
        selected: dict[str, torch.Tensor] = {}
        for key, tensor in features.items():
            if key.endswith("_tokens"):
                if tensor.ndim != 4:
                    raise ValueError(
                        f"{key} must have shape [B,L,S,D], got {tuple(tensor.shape)}"
                    )
                selected[key] = tensor.index_select(1, offsets)
            elif key.endswith("_mask"):
                if tensor.ndim != 3:
                    raise ValueError(
                        f"{key} must have shape [B,L,S], got {tuple(tensor.shape)}"
                    )
                selected[key] = tensor.index_select(1, offsets)
            else:
                selected[key] = tensor
        return selected

    def _concat_mlp_layer_features(
        self,
        tokens: torch.Tensor,
        *,
        key: str,
    ) -> torch.Tensor:
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 5:
            raise ValueError(
                f"{key} must have shape [B,F,S,D] or [B,L,F,S,D], "
                f"got {tuple(tokens.shape)}"
            )
        batch, layers, frames, tokens_per_frame, dim = tokens.shape
        concatenated = (
            tokens.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(batch, frames, tokens_per_frame, layers * dim)
        )
        if concatenated.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"{key} concatenated feature_dim={concatenated.shape[-1]} "
                f"does not match config feature_dim={self.config.feature_dim}; "
                "for multi-layer MLP critics set feature_dim to "
                "base_hidden_dim * number_of_selected_layers"
            )
        return concatenated

    def _cache_signature(self) -> dict[str, Any]:
        return self._jsonable(
            {
                "base_config_name": self.config.base_config_name,
                "base_transformer_path": _transformer_path(
                    self.base_config,
                    self.config,
                ),
                "critic_type": self.config.critic_type,
                "feature_dim": self.config.feature_dim,
                "cache_feature_policy": "all_lingbot_dit_blocks",
                "cached_feature_layers": list(self.cached_feature_layers),
                "feature_aggregation": self.config.feature_aggregation,
                "feature_normalization": "raw_block_output_v1",
                "infer_latent_chunk_size": self.config.infer_latent_chunk_size,
                "phase3_precompute_mode": self.config.phase3_precompute_mode,
                "phase3_video_prediction_context": (
                    self.config.phase3_video_prediction_context
                ),
                "training_distribution": self.config.training_distribution,
                "reward_source": self.config.reward_source,
                "phase3_video_exec_step": self.config.phase3_video_exec_step,
                "phase3_video_num_inference_steps": (
                    self.config.phase3_video_num_inference_steps
                ),
                "phase3_action_num_inference_steps": (
                    self.config.phase3_action_num_inference_steps
                ),
                "phase3_jepa_checkpoint": self.config.phase3_jepa_checkpoint,
                "value_video_feature_timestep_range": list(
                    self.config.value_video_feature_timestep_range
                ),
                "value_action_feature_timestep_range": list(
                    self.config.value_action_feature_timestep_range
                ),
                "q_video_feature_timestep_range": list(
                    self.config.q_video_feature_timestep_range
                ),
                "q_action_feature_timestep_range": list(
                    self.config.q_action_feature_timestep_range
                ),
                "q_feature_action_source": "current_real_executed_action",
            }
        )

    def _cache_signature_matches(self, actual: dict[str, Any] | None) -> bool:
        expected = self._cache_signature()
        if actual == expected:
            return True
        if (
            self.config.critic_type == "twin_flow_scalar_transformer_v1"
            and isinstance(actual, dict)
        ):
            compatible = dict(expected)
            compatible["critic_type"] = "twin_query_transformer_v1"
            return actual == compatible
        return False

    def _canonical_cache_signature(
        self,
        *,
        cached_feature_layers: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        signature_layers = (
            tuple(self.cached_feature_layers)
            if cached_feature_layers is None
            else tuple(int(layer) for layer in cached_feature_layers)
        )
        return self._jsonable(
            {
                "base_config_name": self.config.base_config_name,
                "base_transformer_path": _transformer_path(
                    self.base_config,
                    self.config,
                ),
                "feature_dim": self._canonical_cache_token_dim(),
                "cache_feature_policy": "canonical_episode_branch_tokens",
                "cached_feature_layers": list(signature_layers),
                "feature_aggregation": self.config.feature_aggregation,
                "feature_normalization": self.config.feature_normalization,
                "infer_latent_chunk_size": self.config.infer_latent_chunk_size,
                "phase3_precompute_mode": self.config.phase3_precompute_mode,
                "phase3_video_prediction_context": (
                    self.config.phase3_video_prediction_context
                ),
                "training_distribution": self.config.training_distribution,
                "reward_source": self.config.reward_source,
                "phase3_video_exec_step": self.config.phase3_video_exec_step,
                "phase3_video_num_inference_steps": (
                    self.config.phase3_video_num_inference_steps
                ),
                "phase3_action_num_inference_steps": (
                    self.config.phase3_action_num_inference_steps
                ),
                "phase3_jepa_checkpoint": self.config.phase3_jepa_checkpoint,
                "value_video_feature_timestep_range": list(
                    self.config.value_video_feature_timestep_range
                ),
                "value_action_feature_timestep_range": list(
                    self.config.value_action_feature_timestep_range
                ),
                "q_video_feature_timestep_range": list(
                    self.config.q_video_feature_timestep_range
                ),
                "q_action_feature_timestep_range": list(
                    self.config.q_action_feature_timestep_range
                ),
                "q_feature_action_source": "current_real_executed_action",
            }
        )

    def _canonical_cache_token_dim(self) -> int:
        if self._uses_layer_context_features():
            return int(self.config.feature_dim)
        layer_count = max(1, len(self.config.critic_feature_layers))
        if int(self.config.feature_dim) % layer_count != 0:
            raise ValueError(
                "For multi-layer MLP canonical caches, feature_dim must be "
                "divisible by the number of critic_feature_layers"
            )
        return int(self.config.feature_dim) // layer_count

    def _canonical_cache_signature_matches(
        self,
        actual: dict[str, Any] | None,
        *,
        cached_feature_layers: tuple[int, ...] | None = None,
    ) -> bool:
        return actual == self._canonical_cache_signature(
            cached_feature_layers=cached_feature_layers
        )

    def canonical_cache_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": CANONICAL_CACHE_SCHEMA_VERSION,
            "cache_format": CANONICAL_CACHE_FORMAT,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "signature": self._canonical_cache_signature(),
            "cached_feature_layers": list(self.cached_feature_layers),
            "selected_critic_feature_layers": list(
                self.config.critic_feature_layers
            ),
            "feature_keys": list(CANONICAL_FEATURE_KEYS),
        }

    def cache_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": PHASE3_CACHE_SCHEMA_VERSION,
            "cache_format": "episode_layer_context",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "signature": self._cache_signature(),
            "cached_feature_layers": list(self.cached_feature_layers),
            "selected_critic_feature_layers": list(
                self.config.critic_feature_layers
            ),
            "feature_keys": list(self._cache_feature_keys()),
        }

    def _validate_episode_cache_payload(
        self,
        payload: dict[str, Any],
        episode_key: tuple[int, int],
        path: Path,
    ) -> None:
        if payload.get("schema_version") != PHASE3_CACHE_SCHEMA_VERSION:
            raise ValueError(f"Cache schema mismatch in {path}")
        if payload.get("cache_format") != "episode_layer_context":
            raise ValueError(
                f"Cache file {path} is not an episode_layer_context cache file"
            )
        actual_episode_key = payload.get("episode_key")
        if list(actual_episode_key or []) != [
            int(episode_key[0]),
            int(episode_key[1]),
        ]:
            raise ValueError(
                f"Cache episode_key mismatch in {path}: "
                f"{actual_episode_key} != {list(episode_key)}"
            )
        if not self._cache_signature_matches(payload.get("signature")):
            raise ValueError(
                "Phase 3 cache signature mismatch for "
                f"{path}; regenerate the cache for this config"
            )
        cached_layers = payload.get("cached_feature_layers")
        if not isinstance(cached_layers, list):
            raise ValueError(f"Cache payload has no cached_feature_layers: {path}")
        missing_layers = set(self.config.critic_feature_layers).difference(
            int(layer) for layer in cached_layers
        )
        if missing_layers:
            raise ValueError(
                f"Cache file {path} does not contain requested critic layers: "
                f"{sorted(missing_layers)}"
            )
        chunks = payload.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError(f"Cache payload has no chunk list: {path}")

    def _validate_episode_cache_chunk(
        self,
        chunk: dict[str, Any],
        transition_idx: int,
        path: Path,
    ) -> None:
        if int(chunk.get("transition_idx", -1)) != int(transition_idx):
            raise ValueError(
                f"Cache transition_idx mismatch in {path}: "
                f"{chunk.get('transition_idx')} != {transition_idx}"
            )
        features = chunk.get("features")
        if not isinstance(features, dict):
            raise ValueError(f"Cache chunk has no feature dict: {path}")
        missing = set(self._cache_feature_keys()).difference(features)
        if missing:
            raise KeyError(
                f"Cache chunk {transition_idx} in {path} is missing "
                f"features: {sorted(missing)}"
            )

    def _load_episode_cache_chunk(
        self,
        transition_idx: int,
        payload_cache: dict[Path, dict[str, Any]],
    ) -> dict[str, Any] | None:
        record = self.dataset.records[int(transition_idx)]
        path = phase3_episode_cache_file(
            self.config.phase3_precomputed_cache_dir,
            record.episode_key,
        )
        if not path.exists():
            return None
        if path not in payload_cache:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self._validate_episode_cache_payload(payload, record.episode_key, path)
            payload_cache[path] = payload
        payload = payload_cache[path]
        for chunk in payload["chunks"]:
            if int(chunk.get("transition_idx", -1)) == int(transition_idx):
                self._validate_episode_cache_chunk(chunk, transition_idx, path)
                chunk = dict(chunk)
                chunk["_cached_feature_layers"] = tuple(
                    int(layer) for layer in payload["cached_feature_layers"]
                )
                return chunk
        raise KeyError(
            f"Transition {transition_idx} is not present in episode cache {path}"
        )

    def _load_precomputed_training_inputs(
        self,
        batch: dict[str, torch.Tensor],
        timings: dict[str, float],
    ) -> dict[str, Any] | None:
        if not self.config.phase3_precomputed_cache_dir:
            return None

        def load() -> dict[str, Any] | None:
            transition_indices = [
                int(value)
                for value in batch["transition_idx"].detach().cpu().view(-1)
            ]
            chunks: list[dict[str, Any]] = []
            missing_paths: list[str] = []
            payload_cache: dict[Path, dict[str, Any]] = {}
            for transition_idx in transition_indices:
                chunk = self._load_episode_cache_chunk(
                    transition_idx,
                    payload_cache=payload_cache,
                )
                if chunk is None:
                    record = self.dataset.records[int(transition_idx)]
                    missing_paths.append(
                        str(
                            phase3_episode_cache_file(
                                self.config.phase3_precomputed_cache_dir,
                                record.episode_key,
                            )
                        )
                    )
                    continue
                chunks.append(chunk)

            if missing_paths:
                if self.config.phase3_require_precomputed_cache:
                    raise FileNotFoundError(
                        "Missing required Phase 3 precomputed cache files; "
                        f"first missing files: {missing_paths[:5]}"
                    )
                return None

            features = {
                key: torch.stack(
                    [chunk["features"][key] for chunk in chunks],
                    dim=0,
                ).to(self.device, non_blocking=True)
                for key in self._cache_feature_keys()
            }
            cached_feature_layers = tuple(chunks[0]["_cached_feature_layers"])
            if any(
                tuple(chunk["_cached_feature_layers"]) != cached_feature_layers
                for chunk in chunks
            ):
                raise ValueError(
                    "Mixed Phase 3 cache files expose different cached layers"
                )
            features = self._select_feature_layers(
                features,
                cached_feature_layers=cached_feature_layers,
            )
            distance = torch.stack(
                [chunk["jepa_distance"] for chunk in chunks],
                dim=0,
            ).to(self.device, non_blocking=True).float().view(-1)
            return {
                "features": features,
                "jepa_distance": distance,
                "reward": self._compose_reward_batch(distance, batch),
                "cache_hit": 1.0,
            }

        return self._time_call(timings, "cache_load_time", load)

    def _prepare_packed_cache_training_inputs(
        self,
        batch: dict[str, Any],
        timings: dict[str, float],
    ) -> dict[str, Any]:
        def load() -> dict[str, Any]:
            features = {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch["features"].items()
            }
            distance = batch["jepa_distance"].to(
                self.device,
                non_blocking=True,
            ).float().view(-1)
            if (
                self._reward_terms_need_episode_metadata()
                and not bool(batch.get("_packed_has_episode_metadata", False))
            ):
                self._require_reward_metadata_available(source="packed cache")
            reward = self._compose_reward_batch(distance, batch)
            return {
                "features": features,
                "jepa_distance": distance,
                "reward": reward,
                "cache_hit": 1.0,
            }

        return self._time_call(timings, "cache_load_time", load)

    def _prepare_canonical_cache_training_inputs(
        self,
        batch: dict[str, Any],
        timings: dict[str, float],
    ) -> dict[str, Any]:
        def load() -> dict[str, Any]:
            features = {
                key: value.to(self.device, non_blocking=True)
                for key, value in batch["features"].items()
            }
            distance = batch["jepa_distance"].to(
                self.device,
                non_blocking=True,
            ).float().view(-1)
            reward = self._compose_reward_batch(distance, batch)
            return {
                "features": features,
                "jepa_distance": distance,
                "reward": reward,
                "cache_hit": 1.0,
            }

        return self._time_call(timings, "cache_load_time", load)

    def build_phase3_cache_chunks(
        self,
        batch: dict[str, torch.Tensor],
        computed: dict[str, Any],
    ) -> list[dict[str, Any]]:
        feature_keys = self._cache_feature_keys()
        transition_indices = [
            int(value)
            for value in batch["transition_idx"].detach().cpu().view(-1)
        ]
        chunks: list[dict[str, Any]] = []
        for item_index, transition_idx in enumerate(transition_indices):
            record = self.dataset.records[int(transition_idx)]
            chunks.append(
                {
                    "transition_idx": transition_idx,
                    "rl_chunk_idx": int(record.rl_chunk_idx),
                    "frame_start": int(record.frame_start),
                    "valid_frames": int(record.valid_frames),
                    "done": bool(record.done),
                    "episode_success": bool(record.episode_success),
                    "episode_chunk_count": int(record.episode_chunk_count),
                    "features": {
                        key: computed["features"][key][item_index]
                        .detach()
                        .cpu()
                        .contiguous()
                        for key in feature_keys
                    },
                    "jepa_distance": computed["jepa_distance"][item_index]
                    .detach()
                    .cpu()
                    .float(),
                    "reward": computed["reward"][item_index]
                    .detach()
                    .cpu()
                    .float(),
                }
            )
        return chunks

    def build_phase3_canonical_chunks(
        self,
        batch: dict[str, torch.Tensor],
        computed: dict[str, Any],
    ) -> list[dict[str, Any]]:
        transition_indices = [
            int(value)
            for value in batch["transition_idx"].detach().cpu().view(-1)
        ]
        chunks: list[dict[str, Any]] = []
        for item_index, transition_idx in enumerate(transition_indices):
            record = self.dataset.records[int(transition_idx)]
            chunks.append(
                {
                    "transition_idx": torch.tensor(
                        transition_idx,
                        dtype=torch.long,
                    ),
                    "dataset_id": torch.tensor(
                        record.episode_key[0],
                        dtype=torch.long,
                    ),
                    "episode_index": torch.tensor(
                        record.episode_key[1],
                        dtype=torch.long,
                    ),
                    "rl_chunk_idx": torch.tensor(
                        record.rl_chunk_idx,
                        dtype=torch.long,
                    ),
                    "discount": torch.tensor(
                        record.discount,
                        dtype=torch.float32,
                    ),
                    "state_valid": torch.tensor(
                        record.previous_record_idx is not None,
                        dtype=torch.bool,
                    ),
                    "done": torch.tensor(record.done, dtype=torch.bool),
                    "episode_success": torch.tensor(
                        record.episode_success,
                        dtype=torch.bool,
                    ),
                    "episode_chunk_count": torch.tensor(
                        record.episode_chunk_count,
                        dtype=torch.long,
                    ),
                    "previous_latents_mask": batch["previous_latents_mask"][
                        item_index
                    ].detach().cpu(),
                    "latents_mask": batch["latents_mask"][
                        item_index
                    ].detach().cpu(),
                    "previous_actions_mask": batch["previous_actions_mask"][
                        item_index
                    ].detach().cpu(),
                    "actions_mask": batch["actions_mask"][
                        item_index
                    ].detach().cpu(),
                    "features": {
                        key: computed["features"][key][item_index]
                        .detach()
                        .cpu()
                        .contiguous()
                        for key in CANONICAL_FEATURE_KEYS
                    },
                    "jepa_distance": computed["jepa_distance"][item_index]
                    .detach()
                    .cpu()
                    .float(),
                    "goal_jepa_distance": computed["goal_jepa_distance"][item_index]
                    .detach()
                    .cpu()
                    .float(),
                    "cached_reward": computed["reward"][item_index]
                    .detach()
                    .cpu()
                    .float(),
                }
            )
        return chunks

    def _canonical_timestep_ranges_supported(self) -> None:
        expected = self.config.q_video_feature_timestep_range
        ranges = {
            "value_video_feature_timestep_range": (
                self.config.value_video_feature_timestep_range
            ),
            "value_action_feature_timestep_range": (
                self.config.value_action_feature_timestep_range
            ),
            "q_video_feature_timestep_range": (
                self.config.q_video_feature_timestep_range
            ),
            "q_action_feature_timestep_range": (
                self.config.q_action_feature_timestep_range
            ),
        }
        mismatched = {
            key: value for key, value in ranges.items() if value != expected
        }
        if mismatched:
            raise ValueError(
                "canonical_episode cache currently requires equal q/value "
                f"feature timestep ranges; got {mismatched}"
            )

    def _video_prediction_context(
        self,
        batch: dict[str, torch.Tensor],
        current_latents: torch.Tensor,
        current_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.phase3_video_prediction_context == "previous_chunk":
            context_latents = self._to_device(batch, "previous_latents")
            starts = torch.full(
                (current_latents.shape[0],),
                context_latents.shape[2],
                device=self.device,
                dtype=torch.long,
            )
            return (
                context_latents,
                self._to_device(batch, "previous_actions"),
                starts,
            )

        if not hasattr(self, "dataset") or not hasattr(self.dataset, "records"):
            raise RuntimeError(
                "episode_prefix video prediction context requires an online "
                "transition dataset with records"
            )

        transition_indices = [
            int(value) for value in batch["transition_idx"].detach().cpu().view(-1)
        ]
        records = [self.dataset.records[index] for index in transition_indices]
        chunk = int(self.config.infer_latent_chunk_size)
        max_total_frames = max(
            int(record.episode_frame_start) + chunk for record in records
        )
        context_latents = current_latents.new_zeros(
            current_latents.shape[0],
            current_latents.shape[1],
            max_total_frames,
            current_latents.shape[3],
            current_latents.shape[4],
        )
        context_actions = current_actions.new_zeros(
            current_actions.shape[0],
            current_actions.shape[1],
            max_total_frames,
            current_actions.shape[3],
            current_actions.shape[4],
        )
        starts = torch.empty(
            current_latents.shape[0],
            device=self.device,
            dtype=torch.long,
        )

        for row, record in enumerate(records):
            start = int(record.episode_frame_start)
            starts[row] = start
            previous_indices: list[int] = []
            previous = record.previous_record_idx
            while previous is not None:
                previous_indices.append(previous)
                previous = self.dataset.records[previous].previous_record_idx
            previous_indices.reverse()

            write_frame = 0
            for previous_index in previous_indices:
                previous_record = self.dataset.records[previous_index]
                previous_chunk = self.dataset._load_record(previous_record)
                valid = int(previous_record.valid_frames)
                latents = previous_chunk["latents"][:, :valid].to(
                    device=self.device,
                    dtype=current_latents.dtype,
                )
                actions = previous_chunk["actions"][:, :valid].to(
                    device=self.device,
                    dtype=current_actions.dtype,
                )
                context_latents[row, :, write_frame : write_frame + valid] = latents
                context_actions[row, :, write_frame : write_frame + valid] = actions
                write_frame += valid
            if write_frame != start:
                raise ValueError(
                    "Episode-prefix context frame mismatch for transition "
                    f"{int(record.rl_chunk_idx)}: wrote {write_frame}, expected {start}"
                )
        return context_latents, context_actions, starts

    @staticmethod
    def _pad_server_prefix_feature_frames(
        feature: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        """Pad a short remainder feature to the fixed cache chunk shape.

        Server-prefix inference may process one valid latent frame when a
        storage window has an odd latent length. The model and canonical
        cache still expose a fixed infer_latent_chunk_size shape; masks mark
        the added frame as invalid. Video/action feature tensors from the
        transformer are [B,L,F,S,D], so F is dimension 2 here (and dimension
        1 after selecting a batch item).
        """
        if feature.ndim not in (4, 5):
            raise ValueError(
                "Unexpected server-prefix feature rank: "
                f"shape={tuple(feature.shape)}"
            )
        frame_dim = 2 if feature.ndim == 5 else 1
        current_frames = int(feature.shape[frame_dim])
        if current_frames > target_frames:
            raise ValueError(
                "Server-prefix feature has more frames than the configured "
                f"chunk: {current_frames}>{target_frames}"
            )
        if current_frames == target_frames:
            return feature
        shape = list(feature.shape)
        shape[frame_dim] = target_frames - current_frames
        padding = feature.new_zeros(shape)
        return torch.cat((feature, padding), dim=frame_dim)

    @torch.no_grad()
    def compute_phase3_canonical_inputs(
        self,
        batch: dict[str, torch.Tensor],
        timings: dict[str, float] | None = None,
        feature_layers: tuple[int, ...] | None = None,
        defer_reward: bool = False,
    ) -> dict[str, Any]:
        self._canonical_timestep_ranges_supported()
        history_latents = self._to_device(batch, "previous_latents")
        current_latents = self._to_device(batch, "latents")
        history_actions = self._to_device(batch, "previous_actions")
        current_actions = self._to_device(batch, "actions")
        current_action_mask = self._to_device(batch, "actions_mask")
        text_emb = self._to_device(batch, "text_emb")
        video_context_latents, video_context_actions, current_frame_start = (
            self._video_prediction_context(batch, current_latents, current_actions)
        )

        predicted_latents = self._time_call(
            timings,
            "predict_video_time",
            self._predict_video_with_context,
            video_context_latents,
            current_latents,
            video_context_actions,
            current_frame_start,
            text_emb,
        )
        if self.config.training_distribution == "predicted_video_conditioned_action":
            q_current_actions = self._time_call(
                timings,
                "predict_action_time",
                self._predict_actions,
                history_latents,
                predicted_latents,
                history_actions,
                current_actions,
                current_action_mask,
                text_emb,
            )
        else:
            q_current_actions = current_actions

        selected_layers = (
            feature_layers
            if feature_layers is not None
            else (
                self.config.critic_feature_layers
                if (
                    self._uses_layer_context_features()
                    or len(self.config.critic_feature_layers) > 1
                )
                else None
            )
        )
        feature_timestep = self._sample_timestep(
            self.config.q_video_feature_timestep_range
        )
        action_timestep = self._sample_timestep(
            self.config.q_action_feature_timestep_range
        )
        pred_features = self._time_call(
            timings,
            "pred_feature_time",
            self._extract_features,
            history_latents,
            predicted_latents,
            history_actions,
            q_current_actions,
            text_emb,
            feature_timestep,
            action_timestep,
            feature_layers=selected_layers,
        )
        target_features = self._time_call(
            timings,
            "target_feature_time",
            self._extract_features,
            history_latents,
            current_latents,
            history_actions,
            current_actions,
            text_emb,
            feature_timestep,
            action_timestep,
            feature_layers=selected_layers,
        )
        distance = self._time_call(
            timings,
            "jepa_reward_time",
            self._compute_jepa_distance,
            predicted_latents,
            batch,
        )
        goal_distance = self._compute_goal_jepa_distance(batch)
        # Episode-wide goal statistics are finalized after all chunks exist.
        reward = (
            torch.zeros_like(distance)
            if defer_reward
            else self._compose_reward_batch(distance, batch)
        )
        return {
            "features": {
                "pred_video_tokens": pred_features["video_tokens"][
                    :, ..., -self.config.infer_latent_chunk_size :, :, :
                ].detach(),
                "pred_action_tokens": pred_features["action_tokens"][
                    :, ..., -self.config.infer_latent_chunk_size :, :, :
                ].detach(),
                "real_video_tokens": target_features["video_tokens"][
                    :, ..., -self.config.infer_latent_chunk_size :, :, :
                ].detach(),
                "real_action_tokens": target_features["action_tokens"][
                    :, ..., -self.config.infer_latent_chunk_size :, :, :
                ].detach(),
            },
            "jepa_distance": distance,
            "goal_jepa_distance": goal_distance,
            "reward": reward,
            "cache_hit": 0.0,
        }

    def save_phase3_canonical_episode_cache(
        self,
        episode_key: tuple[int, int],
        chunks: list[dict[str, Any]],
        output_dir: str | Path,
        overwrite: bool = False,
    ) -> bool:
        path = phase3_canonical_episode_cache_file(output_dir, episode_key)
        if path.exists() and not overwrite:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = sorted(chunks, key=lambda item: int(item["rl_chunk_idx"]))
        if chunks and "goal_jepa_distance" in chunks[0]:
            goal_max = torch.stack(
                [chunk["goal_jepa_distance"].float() for chunk in chunks],
                dim=0,
            ).max().clamp_min(1e-6)
            for chunk in chunks:
                chunk["episode_goal_jepa_distance_max"] = goal_max.clone()
        tensors: dict[str, torch.Tensor] = {}
        for key in CANONICAL_FEATURE_KEYS:
            tensors[key] = torch.stack(
                [chunk["features"][key].contiguous() for chunk in chunks],
                dim=0,
            )
        for key, dtype in (
            ("transition_idx", torch.long),
            ("discount", torch.float32),
            ("state_valid", torch.bool),
            ("jepa_distance", torch.float32),
            ("goal_jepa_distance", torch.float32),
            ("episode_goal_jepa_distance_max", torch.float32),
            ("cached_reward", torch.float32),
            ("done", torch.bool),
            ("dataset_id", torch.long),
            ("episode_index", torch.long),
            ("rl_chunk_idx", torch.long),
            ("episode_success", torch.bool),
            ("episode_chunk_count", torch.long),
            ("previous_latents_mask", torch.bool),
            ("latents_mask", torch.bool),
            ("previous_actions_mask", torch.bool),
            ("actions_mask", torch.bool),
        ):
            tensors[key] = torch.stack(
                [chunk[key].to(dtype=dtype).cpu().contiguous() for chunk in chunks],
                dim=0,
            )

        from safetensors.torch import save_file

        tmp_path = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        save_file(
            tensors,
            str(tmp_path),
            metadata={
                "schema_version": str(CANONICAL_CACHE_SCHEMA_VERSION),
                "cache_format": CANONICAL_CACHE_FORMAT,
                "episode_key": json.dumps(
                    [int(episode_key[0]), int(episode_key[1])]
                ),
            },
        )
        tmp_path.replace(path)
        return True

    def save_phase3_episode_cache(
        self,
        episode_key: tuple[int, int],
        chunks: list[dict[str, Any]],
        output_dir: str | Path,
        overwrite: bool = False,
    ) -> bool:
        path = phase3_episode_cache_file(output_dir, episode_key)
        if path.exists() and not overwrite:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        chunks = sorted(chunks, key=lambda item: int(item["rl_chunk_idx"]))
        payload = {
            "schema_version": PHASE3_CACHE_SCHEMA_VERSION,
            "cache_format": "episode_layer_context",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "episode_key": [int(episode_key[0]), int(episode_key[1])],
            "signature": self._cache_signature(),
            "cached_feature_layers": list(self.cached_feature_layers),
            "selected_critic_feature_layers": list(
                self.config.critic_feature_layers
            ),
            "feature_keys": list(self._cache_feature_keys()),
            "chunks": chunks,
        }
        tmp_path = path.with_name(
            f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
        )
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        return True

    def _packed_streams(
        self,
        history_latents: torch.Tensor,
        current_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_actions: torch.Tensor,
        text_emb: torch.Tensor,
        latent_timestep: float,
        action_timestep: float,
        condition_history_latents: torch.Tensor | None = None,
        condition_current_latents: torch.Tensor | None = None,
        condition_history_actions: torch.Tensor | None = None,
        condition_current_actions: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        latents = torch.cat([history_latents, current_latents], dim=2)
        actions = torch.cat([history_actions, current_actions], dim=2)
        if latents.shape[0] != actions.shape[0] or latents.shape[2] != actions.shape[2]:
            raise ValueError(
                "Packed Phase 3 latent/action stream mismatch: "
                f"latents batch/frames={latents.shape[0]}/{latents.shape[2]}, "
                f"actions batch/frames={actions.shape[0]}/{actions.shape[2]}"
            )
        condition_latents = torch.cat(
            [
                history_latents if condition_history_latents is None else condition_history_latents,
                current_latents if condition_current_latents is None else condition_current_latents,
            ],
            dim=2,
        )
        condition_actions = torch.cat(
            [
                history_actions if condition_history_actions is None else condition_history_actions,
                current_actions if condition_current_actions is None else condition_current_actions,
            ],
            dim=2,
        )
        return {
            "latent_dict": _zero_timestep_stream(
                latents,
                text_emb,
                tuple(self.base_config.patch_size),
                action_mode=False,
                timestep=latent_timestep,
                condition_tensor=condition_latents,
            ),
            "action_dict": _zero_timestep_stream(
                actions,
                text_emb,
                tuple(self.base_config.patch_size),
                action_mode=True,
                timestep=action_timestep,
                condition_tensor=condition_actions,
            ),
            "chunk_size": self.config.infer_latent_chunk_size,
            "window_size": self.config.window_size,
        }

    def _server_prefix_cache_name(self) -> str:
        return f"phase3_server_prefix_rank{self.rank}"

    def _server_prefix_cfg_enabled(self) -> bool:
        return float(getattr(self.base_config, "guidance_scale", 1.0)) > 1.0

    def _server_prefix_empty_text(self, text_emb: torch.Tensor) -> torch.Tensor:
        cached = getattr(self, "_server_prefix_empty_text_cache", None)
        if cached is not None:
            return cached.to(device=self.device, dtype=text_emb.dtype)
        empty_path = getattr(self.base_config, "empty_emb_path", None)
        if not empty_path:
            empty_tensor = torch.zeros_like(text_emb)
            self._server_prefix_empty_text_cache = empty_tensor
            return empty_tensor
        empty = torch.load(empty_path, map_location="cpu", weights_only=False)
        if isinstance(empty, dict):
            empty = empty.get("text_emb", next(iter(empty.values())))
        if empty.ndim == 2:
            empty = empty.unsqueeze(0)
        empty_tensor = empty.to(device=self.device, dtype=text_emb.dtype)
        self._server_prefix_empty_text_cache = empty_tensor
        return empty_tensor

    def _server_prefix_reset_cache(self, batch_size: int = 1) -> None:
        cache_name = self._server_prefix_cache_name()
        self.transformer.clear_cache(cache_name)
        if getattr(self.base_config, "env_type", None) == "robotwin_tshape":
            latent_height = ((int(self.base_config.height) // 16) * 3) // 2
            latent_width = int(self.base_config.width) // 16
        else:
            latent_height = int(self.base_config.height) // 16
            latent_width = (
                int(self.base_config.width) // 16
            ) * len(self.base_config.obs_cam_keys)
        latent_token_per_chunk = (
            int(self.config.infer_latent_chunk_size)
            * latent_height
            * latent_width
        ) // math.prod(tuple(self.base_config.patch_size))
        action_token_per_chunk = (
            int(self.config.infer_latent_chunk_size)
            * int(self.base_config.action_per_frame)
        )
        cache_batch = batch_size * (2 if self._server_prefix_cfg_enabled() else 1)
        self.transformer.create_empty_cache(
            cache_name,
            int(getattr(self.base_config, "attn_window", self.config.window_size)),
            latent_token_per_chunk,
            action_token_per_chunk,
            device=self.device,
            dtype=self.base_config.param_dtype,
            batch_size=cache_batch,
        )

    def _server_prefix_repeat_for_cfg(
        self,
        stream: dict[str, torch.Tensor],
        text_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if not self._server_prefix_cfg_enabled():
            return stream
        repeated = dict(stream)
        repeated["noisy_latents"] = stream["noisy_latents"].repeat(2, 1, 1, 1, 1)
        repeated["text_emb"] = torch.cat(
            [text_emb, self._server_prefix_empty_text(text_emb)],
            dim=0,
        )
        repeated["grid_id"] = stream["grid_id"].repeat(2, 1, 1)
        repeated["timesteps"] = stream["timesteps"].repeat(2, 1)
        return repeated

    def _server_prefix_prepare_stream(
        self,
        tensor: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        timestep: float,
        frame_start: int,
        action_mode: bool,
        condition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        from wan_va.utils.utils import get_mesh_id

        batch, _, frames, height, width = tensor.shape
        patch_size = tuple(self.base_config.patch_size)
        if action_mode:
            grid = get_mesh_id(
                frames,
                height,
                width,
                1,
                1,
                frame_start,
                action=True,
            ).to(self.device)
        else:
            grid = get_mesh_id(
                frames // patch_size[0],
                height // patch_size[1],
                width // patch_size[2],
                0,
                1,
                frame_start,
            ).to(self.device)
        stream = {
            "noisy_latents": tensor,
            "timesteps": torch.full(
                (batch, frames),
                float(timestep) * 1000.0,
                device=self.device,
                dtype=torch.float32,
            ),
            "grid_id": grid.unsqueeze(0).repeat(batch, 1, 1),
            "text_emb": text_emb,
        }
        if condition is not None:
            stream["noisy_latents"][:, :, 0:1] = condition[:, :, 0:1]
            stream["timesteps"][:, 0:1] = 0.0
        return stream

    def _server_prefix_forward_features(
        self,
        tensor: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        timestep: float,
        frame_start: int,
        action_mode: bool,
        update_cache: int,
        condition: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stream = self._server_prefix_prepare_stream(
            tensor,
            text_emb,
            timestep=timestep,
            frame_start=frame_start,
            action_mode=action_mode,
            condition=condition,
        )
        stream = self._server_prefix_repeat_for_cfg(stream, text_emb)
        prediction, features = self.transformer(
            stream,
            update_cache=update_cache,
            cache_name=self._server_prefix_cache_name(),
            action_mode=action_mode,
            return_features=True,
            critic_feature_layers=self.cached_feature_layers,
        )
        if self._server_prefix_cfg_enabled():
            prediction = prediction[:1]
            features = {
                key: (value[:1] if isinstance(value, torch.Tensor) else value)
                for key, value in features.items()
            }
        return prediction, features

    @torch.no_grad()
    def _server_prefix_predict_video_chunk(
        self,
        current_gt_latents: torch.Tensor,
        text_emb: torch.Tensor,
        *,
        frame_start: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        from wan_va.utils.utils import data_seq_to_patch

        current = torch.randn_like(current_gt_latents)
        latent_cond = current_gt_latents[:, :, 0:1] if frame_start == 0 else None
        if latent_cond is not None:
            current[:, :, 0:1] = latent_cond
        video_steps = (
            self.config.phase3_video_num_inference_steps
            or int(self.base_config.num_inference_steps)
        )
        self.video_scheduler.set_timesteps(video_steps)
        timesteps = F.pad(
            self.video_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if self.config.phase3_video_exec_step != -1:
            timesteps = timesteps[: self.config.phase3_video_exec_step]

        last_features: dict[str, torch.Tensor] | None = None
        for index, timestep in enumerate(timesteps):
            last_step = index == len(timesteps) - 1
            timestep_value = float(timestep.item()) / 1000.0
            if last_step:
                _, last_features = self._server_prefix_forward_features(
                    current,
                    text_emb,
                    timestep=timestep_value,
                    frame_start=frame_start,
                    action_mode=False,
                    update_cache=1,
                    condition=latent_cond,
                )
            else:
                stream = self._server_prefix_prepare_stream(
                    current,
                    text_emb,
                    timestep=timestep_value,
                    frame_start=frame_start,
                    action_mode=False,
                    condition=latent_cond,
                )
                stream = self._server_prefix_repeat_for_cfg(stream, text_emb)
                velocity = self.transformer(
                    stream,
                    update_cache=0,
                    cache_name=self._server_prefix_cache_name(),
                    action_mode=False,
                )
                velocity = data_seq_to_patch(
                    tuple(self.base_config.patch_size),
                    velocity,
                    current.shape[2],
                    current.shape[3],
                    current.shape[4],
                    batch_size=2 if self._server_prefix_cfg_enabled() else 1,
                )
                if self._server_prefix_cfg_enabled():
                    scale = float(getattr(self.base_config, "guidance_scale", 1.0))
                    velocity = velocity[1:] + scale * (velocity[:1] - velocity[1:])
                else:
                    velocity = velocity[:1]
                current = self.video_scheduler.step(
                    velocity,
                    timestep,
                    current,
                    return_dict=False,
                )
                if latent_cond is not None:
                    current[:, :, 0:1] = latent_cond
        assert last_features is not None
        return current, last_features

    @torch.no_grad()
    def compute_phase3_server_prefix_episode(
        self,
        transition_indices: list[int],
        timings: dict[str, float] | None = None,
        defer_reward: bool = False,
    ) -> list[dict[str, Any]]:
        if not transition_indices:
            return []
        self._server_prefix_reset_cache(batch_size=1)
        chunks: list[dict[str, Any]] = []
        for transition_idx in transition_indices:
            record = self.dataset.records[int(transition_idx)]
            batch = default_collate([self.dataset[int(transition_idx)]])
            current_latents = self._to_device(batch, "latents")
            current_actions = self._to_device(batch, "actions")
            text_emb = self._to_device(batch, "text_emb")
            valid_frames = int(record.valid_frames)
            configured_chunk = int(self.config.infer_latent_chunk_size)
            if valid_frames <= 0 or valid_frames > configured_chunk:
                raise ValueError(
                    "Invalid transition valid_frames for server-prefix: "
                    f"{valid_frames} with chunk={configured_chunk}"
                )
            if valid_frames < configured_chunk:
                # Do not put padding into the transformer KV cache. The
                # original batch retains its fixed padded shape and masks for
                # canonical serialization; only the online computation uses
                # the valid prefix.
                current_latents = current_latents.narrow(2, 0, valid_frames)
                current_actions = current_actions.narrow(2, 0, valid_frames)
            frame_start = int(record.episode_frame_start)

            predicted_latents, pred_video_features = self._time_call(
                timings,
                "predict_video_time",
                self._server_prefix_predict_video_chunk,
                current_latents,
                text_emb,
                frame_start=frame_start,
            )
            _, pred_action_features = self._time_call(
                timings,
                "pred_feature_time",
                self._server_prefix_forward_features,
                current_actions,
                text_emb,
                timestep=float(self.config.q_action_feature_timestep_range[0]),
                frame_start=frame_start,
                action_mode=True,
                update_cache=0,
                condition=None,
            )
            reward_batch = (
                batch
                if valid_frames == configured_chunk
                else {
                    **batch,
                    "jepa_target": batch["jepa_target"][:, :valid_frames],
                    "latents_mask": batch["latents_mask"][:, :valid_frames],
                    "goal_jepa_target": batch["goal_jepa_target"][
                        :, :valid_frames
                    ],
                    "goal_latents_mask": batch["goal_latents_mask"][
                        :, :valid_frames
                    ],
                }
            )
            distance = self._time_call(
                timings,
                "jepa_reward_time",
                self._compute_jepa_distance,
                predicted_latents,
                reward_batch,
            )
            goal_distance = self._compute_goal_jepa_distance(reward_batch)
            # Episode-wide goal statistics are finalized after all chunks exist.
            reward = (
                torch.zeros_like(distance)
                if defer_reward
                else self._compose_reward_batch(distance, reward_batch)
            )

            self.transformer.clear_pred_cache(self._server_prefix_cache_name())
            _, real_video_features = self._time_call(
                timings,
                "target_feature_time",
                self._server_prefix_forward_features,
                current_latents,
                text_emb,
                timestep=float(self.config.value_video_feature_timestep_range[0]),
                frame_start=frame_start,
                action_mode=False,
                update_cache=2,
                condition=None,
            )
            _, real_action_features = self._time_call(
                timings,
                "target_feature_time",
                self._server_prefix_forward_features,
                current_actions,
                text_emb,
                timestep=float(self.config.value_action_feature_timestep_range[0]),
                frame_start=frame_start,
                action_mode=True,
                update_cache=2,
                condition=None,
            )
            computed = {
                "features": {
                    "pred_video_tokens": self._pad_server_prefix_feature_frames(
                        pred_video_features["video_tokens"].detach(),
                        configured_chunk,
                    ),
                    "pred_action_tokens": self._pad_server_prefix_feature_frames(
                        pred_action_features["action_tokens"].detach(),
                        configured_chunk,
                    ),
                    "real_video_tokens": self._pad_server_prefix_feature_frames(
                        real_video_features["video_tokens"].detach(),
                        configured_chunk,
                    ),
                    "real_action_tokens": self._pad_server_prefix_feature_frames(
                        real_action_features["action_tokens"].detach(),
                        configured_chunk,
                    ),
                },
                "jepa_distance": distance,
                "goal_jepa_distance": goal_distance,
                "reward": reward,
                "cache_hit": 0.0,
            }
            chunks.extend(self.build_phase3_canonical_chunks(batch, computed))
            del (
                batch,
                current_latents,
                current_actions,
                predicted_latents,
                pred_video_features,
                pred_action_features,
                real_video_features,
                real_action_features,
                computed,
            )
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        self.transformer.clear_cache(self._server_prefix_cache_name())
        return chunks

    @torch.no_grad()
    def _predict_video_with_context(
        self,
        context_latents: torch.Tensor,
        current_gt_latents: torch.Tensor,
        context_actions: torch.Tensor,
        current_frame_start: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        current = torch.randn_like(current_gt_latents)
        video_steps = (
            self.config.phase3_video_num_inference_steps
            or int(self.base_config.num_inference_steps)
        )
        self.video_scheduler.set_timesteps(video_steps)
        timesteps = F.pad(
            self.video_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if self.config.phase3_video_exec_step != -1:
            timesteps = timesteps[: self.config.phase3_video_exec_step]

        context_latents = context_latents.to(
            device=self.device,
            dtype=current_gt_latents.dtype,
        )
        context_actions = context_actions.to(
            device=self.device,
            dtype=current_gt_latents.dtype,
        )
        current_frame_start = current_frame_start.to(
            device=self.device,
            dtype=torch.long,
        )
        first_chunk_mask = current_frame_start == 0
        first_chunk_rows = first_chunk_mask.nonzero(as_tuple=False).flatten()
        if first_chunk_rows.numel() > 0:
            current[first_chunk_rows, :, 0:1] = current_gt_latents[
                first_chunk_rows, :, 0:1
            ]
        batch, _, chunk, height, width = current.shape
        total_frames = max(
            int(context_latents.shape[2]),
            int(current_frame_start.max().item()) + chunk,
        )
        condition_latents = context_latents.new_zeros(
            batch,
            context_latents.shape[1],
            total_frames,
            height,
            width,
        )
        condition_actions = context_actions.new_zeros(
            batch,
            context_actions.shape[1],
            total_frames,
            context_actions.shape[3],
            context_actions.shape[4],
        )
        condition_latents[:, :, : context_latents.shape[2]] = context_latents
        condition_actions[:, :, : context_actions.shape[2]] = context_actions
        if first_chunk_rows.numel() > 0:
            condition_latents[first_chunk_rows, :, 0:1] = current_gt_latents[
                first_chunk_rows, :, 0:1
            ]
        noisy_actions = torch.zeros_like(condition_actions)

        for index, timestep in enumerate(timesteps):
            last_step = index == len(timesteps) - 1
            noisy_latents = torch.zeros_like(condition_latents)
            for row in range(batch):
                start = int(current_frame_start[row].item())
                noisy_latents[row : row + 1, :, start : start + chunk] = current[
                    row : row + 1
                ]
            latent_dict = _zero_timestep_stream(
                noisy_latents,
                text_emb,
                tuple(self.base_config.patch_size),
                action_mode=False,
                timestep=float(timestep.item()) / 1000.0,
                condition_tensor=condition_latents,
            )
            if first_chunk_rows.numel() > 0:
                latent_dict["noisy_latents"][first_chunk_rows, :, 0:1] = (
                    current_gt_latents[first_chunk_rows, :, 0:1]
                )
                latent_dict["timesteps"][first_chunk_rows, 0] = 0.0
            action_dict = _zero_timestep_stream(
                noisy_actions,
                text_emb,
                tuple(self.base_config.patch_size),
                action_mode=True,
                timestep=0.0,
                condition_tensor=condition_actions,
            )
            latent_pred, _ = _prediction_outputs(
                self.transformer(
                    {
                        "latent_dict": latent_dict,
                        "action_dict": action_dict,
                        "chunk_size": self.config.infer_latent_chunk_size,
                        "window_size": self.config.window_size,
                    },
                    train_mode=True,
                )
            )
            if not last_step or self.config.phase3_video_exec_step != -1:
                from wan_va.utils.utils import data_seq_to_patch

                latent_pred = data_seq_to_patch(
                    tuple(self.base_config.patch_size),
                    latent_pred,
                    total_frames,
                    height,
                    width,
                    batch_size=batch,
                )
                velocity = torch.empty_like(current)
                for row in range(batch):
                    start = int(current_frame_start[row].item())
                    velocity[row : row + 1] = latent_pred[
                        row : row + 1,
                        :,
                        start : start + chunk,
                    ]
                current = self.video_scheduler.step(
                    velocity,
                    timestep,
                    current,
                    return_dict=False,
                )
                if first_chunk_rows.numel() > 0:
                    current[first_chunk_rows, :, 0:1] = current_gt_latents[
                        first_chunk_rows, :, 0:1
                    ]
        return current

    @torch.no_grad()
    def _predict_video(
        self,
        history_latents: torch.Tensor,
        current_gt_latents: torch.Tensor,
        history_actions: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        current = torch.randn_like(current_gt_latents)
        video_steps = (
            self.config.phase3_video_num_inference_steps
            or int(self.base_config.num_inference_steps)
        )
        self.video_scheduler.set_timesteps(video_steps)
        timesteps = F.pad(
            self.video_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if self.config.phase3_video_exec_step != -1:
            timesteps = timesteps[: self.config.phase3_video_exec_step]

        history_actions = history_actions.to(
            device=history_latents.device,
            dtype=history_latents.dtype,
        )
        dummy_current_actions = torch.zeros(
            history_latents.shape[0],
            history_actions.shape[1],
            current_gt_latents.shape[2],
            history_actions.shape[3],
            history_actions.shape[4],
            device=self.device,
            dtype=history_latents.dtype,
        )
        for index, timestep in enumerate(timesteps):
            last_step = index == len(timesteps) - 1
            inputs = self._packed_streams(
                history_latents,
                current,
                history_actions,
                dummy_current_actions,
                text_emb,
                latent_timestep=float(timestep.item()) / 1000.0,
                action_timestep=0.0,
            )
            latent_pred, _ = _prediction_outputs(
                self.transformer(inputs, train_mode=True)
            )
            if not last_step or self.config.phase3_video_exec_step != -1:
                from wan_va.utils.utils import data_seq_to_patch

                latent_pred = data_seq_to_patch(
                    tuple(self.base_config.patch_size),
                    latent_pred,
                    history_latents.shape[2] + current.shape[2],
                    current.shape[3],
                    current.shape[4],
                    batch_size=current.shape[0],
                )
                velocity = latent_pred[:, :, -current.shape[2] :]
                current = self.video_scheduler.step(
                    velocity,
                    timestep,
                    current,
                    return_dict=False,
                )
        return current

    @torch.no_grad()
    def _predict_actions(
        self,
        history_latents: torch.Tensor,
        predicted_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_gt_actions: torch.Tensor,
        current_action_mask: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        current = torch.randn_like(current_gt_actions)
        current = current * current_action_mask.to(current.dtype)
        action_steps = (
            self.config.phase3_action_num_inference_steps
            or int(self.base_config.action_num_inference_steps)
        )
        self.action_scheduler.set_timesteps(action_steps)
        timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        for index, timestep in enumerate(timesteps):
            last_step = index == len(timesteps) - 1
            inputs = self._packed_streams(
                history_latents,
                predicted_latents,
                history_actions,
                current,
                text_emb,
                latent_timestep=0.0,
                action_timestep=float(timestep.item()) / 1000.0,
            )
            _, action_pred = _prediction_outputs(
                self.transformer(inputs, train_mode=True)
            )
            if not last_step:
                action_pred = rearrange(
                    action_pred,
                    "b (f n) c -> b c f n 1",
                    f=history_actions.shape[2] + current.shape[2],
                )
                velocity = action_pred[:, :, -current.shape[2] :]
                current = self.action_scheduler.step(
                    velocity,
                    timestep,
                    current,
                    return_dict=False,
                )
                current = current * current_action_mask.to(current.dtype)
        return current

    def _sample_timestep(self, value_range: tuple[float, float]) -> float:
        start, end = value_range
        if start == end:
            return float(start)
        return random.uniform(float(start), float(end))

    @torch.no_grad()
    def _extract_features(
        self,
        history_latents: torch.Tensor,
        current_latents: torch.Tensor,
        history_actions: torch.Tensor,
        current_actions: torch.Tensor,
        text_emb: torch.Tensor,
        latent_timestep: float,
        action_timestep: float,
        feature_layers: tuple[int, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = self._packed_streams(
            history_latents,
            current_latents,
            history_actions,
            current_actions,
            text_emb,
            latent_timestep=latent_timestep,
            action_timestep=action_timestep,
        )
        outputs = self.transformer(
            inputs,
            train_mode=True,
            return_features=True,
            **(
                {"critic_feature_layers": feature_layers}
                if feature_layers is not None
                else {"critic_feature_layer": self.config.feature_layer}
            ),
        )
        features = outputs[3]
        expected_feature_dim = self.config.feature_dim
        if (
            not self._uses_layer_context_features()
            and feature_layers is not None
            and len(feature_layers) > 1
        ):
            if self.config.feature_dim % len(feature_layers) != 0:
                raise ValueError(
                    "For multi-layer MLP critics, feature_dim must be divisible "
                    "by the number of selected feature layers"
                )
            expected_feature_dim = self.config.feature_dim // len(feature_layers)
        if features["video_tokens"].shape[-1] != expected_feature_dim:
            raise ValueError("Transformer video feature_dim does not match config")
        if features["action_tokens"].shape[-1] != expected_feature_dim:
            raise ValueError("Transformer action feature_dim does not match config")
        return {
            "video_tokens": features["video_tokens"].detach(),
            "action_tokens": features["action_tokens"].detach(),
        }

    def _flatten_video_context(
        self,
        tokens: torch.Tensor,
        frame_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim not in {4, 5}:
            raise ValueError(
                "video tokens must have shape [B,F,S,D] or [B,L,F,S,D], "
                f"got {tuple(tokens.shape)}"
            )
        if frame_mask.ndim != 2:
            raise ValueError(
                f"video frame mask must have shape [B,F], got {tuple(frame_mask.shape)}"
            )
        token_mask = frame_mask.to(device=tokens.device, dtype=torch.bool)
        if tokens.ndim == 4:
            token_mask = token_mask.unsqueeze(-1).expand(-1, -1, tokens.shape[2])
            return tokens.flatten(1, 2), token_mask.flatten(1, 2)
        token_mask = (
            token_mask[:, None, :, None]
            .expand(-1, tokens.shape[1], -1, tokens.shape[3])
        )
        return tokens.flatten(2, 3), token_mask.flatten(2, 3)

    def _flatten_action_context(
        self,
        tokens: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim not in {4, 5}:
            raise ValueError(
                "action tokens must have shape [B,F,N,D] or [B,L,F,N,D], "
                f"got {tuple(tokens.shape)}"
            )
        if action_mask.ndim != 5:
            raise ValueError(
                "action mask must have shape [B,C,F,N,1], got "
                f"{tuple(action_mask.shape)}"
            )
        token_mask = action_mask.to(device=tokens.device, dtype=torch.bool)
        token_mask = token_mask.squeeze(-1).any(dim=1)
        if token_mask.shape != tokens.shape[:3]:
            if not (
                tokens.ndim == 5
                and token_mask.shape
                == (tokens.shape[0], tokens.shape[2], tokens.shape[3])
            ):
                raise ValueError(
                    "action mask shape does not match action tokens: "
                    f"{tuple(token_mask.shape)} vs {tuple(tokens.shape)}"
                )
        if tokens.ndim == 4:
            return tokens.flatten(1, 2), token_mask.flatten(1, 2)
        token_mask = (
            token_mask[:, None, :, :]
            .expand(-1, tokens.shape[1], -1, -1)
        )
        return tokens.flatten(2, 3), token_mask.flatten(2, 3)

    @staticmethod
    def _cat_context(
        *parts: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_dim = 2 if parts[0][0].ndim == 4 else 1
        mask_dim = 2 if parts[0][1].ndim == 3 else 1
        return (
            torch.cat([tokens for tokens, _ in parts], dim=token_dim),
            torch.cat([mask for _, mask in parts], dim=mask_dim),
        )

    def _build_transformer_critic_contexts(
        self,
        batch: dict[str, torch.Tensor],
        predicted_latents: torch.Tensor,
        q_current_actions: torch.Tensor,
        timings: dict[str, float] | None = None,
        feature_layers: tuple[int, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        # Legacy online/debug path only. Normal Phase 3 training loads
        # canonical episode caches assembled by compute_phase3_canonical_inputs().
        if feature_layers is None:
            feature_layers = self.config.critic_feature_layers
        history_latents = self._to_device(batch, "previous_latents")
        current_latents = self._to_device(batch, "latents")
        history_actions = self._to_device(batch, "previous_actions")
        current_actions = self._to_device(batch, "actions")
        text_emb = self._to_device(batch, "text_emb")

        value_features = self._time_call(
            timings,
            "value_feature_time",
            self._extract_features,
            history_latents,
            predicted_latents,
            history_actions,
            q_current_actions,
            text_emb,
            self._sample_timestep(
                self.config.value_video_feature_timestep_range
            ),
            self._sample_timestep(
                self.config.value_action_feature_timestep_range
            ),
            feature_layers=feature_layers,
        )
        q_features = self._time_call(
            timings,
            "q_feature_time",
            self._extract_features,
            history_latents,
            predicted_latents,
            history_actions,
            q_current_actions,
            text_emb,
            self._sample_timestep(
                self.config.q_video_feature_timestep_range
            ),
            self._sample_timestep(
                self.config.q_action_feature_timestep_range
            ),
            feature_layers=feature_layers,
        )
        target_features = self._time_call(
            timings,
            "target_value_feature_time",
            self._extract_features,
            history_latents,
            current_latents,
            history_actions,
            current_actions,
            text_emb,
            self._sample_timestep(
                self.config.value_video_feature_timestep_range
            ),
            self._sample_timestep(
                self.config.value_action_feature_timestep_range
            ),
            feature_layers=feature_layers,
        )

        chunk = self.config.infer_latent_chunk_size
        previous_latent_mask = batch["previous_latents_mask"].to(
            self.device,
            non_blocking=True,
        )
        current_latent_mask = batch["latents_mask"].to(
            self.device,
            non_blocking=True,
        )
        current_action_mask = batch["actions_mask"].to(
            self.device,
            non_blocking=True,
        )

        value_context = self._flatten_video_context(
            value_features["video_tokens"][:, :, :chunk],
            previous_latent_mask,
        )
        q_context = self._cat_context(
            self._flatten_video_context(
                q_features["video_tokens"][:, :, :chunk],
                previous_latent_mask,
            ),
            self._flatten_action_context(
                q_features["action_tokens"][:, :, -chunk:],
                current_action_mask,
            ),
        )
        target_value_context = self._flatten_video_context(
            target_features["video_tokens"],
            torch.cat([previous_latent_mask, current_latent_mask], dim=1),
        )
        return {
            "value_context_tokens": value_context[0],
            "value_context_mask": value_context[1],
            "q_context_tokens": q_context[0],
            "q_context_mask": q_context[1],
            "target_value_context_tokens": target_value_context[0],
            "target_value_context_mask": target_value_context[1],
        }

    def _build_legacy_mlp_features(
        self,
        batch: dict[str, torch.Tensor],
        predicted_latents: torch.Tensor,
        q_current_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature_layers = (
            self.config.critic_feature_layers
            if len(self.config.critic_feature_layers) > 1
            else None
        )
        features = self._extract_features(
            self._to_device(batch, "previous_latents"),
            predicted_latents,
            self._to_device(batch, "previous_actions"),
            q_current_actions,
            self._to_device(batch, "text_emb"),
            latent_timestep=self._sample_timestep(
                self.config.value_video_feature_timestep_range
            ),
            action_timestep=self._sample_timestep(
                self.config.q_action_feature_timestep_range
            ),
            feature_layers=feature_layers,
        )
        target_features = self._extract_features(
            self._to_device(batch, "previous_latents"),
            self._to_device(batch, "latents"),
            self._to_device(batch, "previous_actions"),
            self._to_device(batch, "actions"),
            self._to_device(batch, "text_emb"),
            latent_timestep=self._sample_timestep(
                self.config.value_video_feature_timestep_range
            ),
            action_timestep=0.0,
            feature_layers=feature_layers,
        )
        chunk = self.config.infer_latent_chunk_size
        if feature_layers is not None:
            return {
                "previous_video_tokens": self._concat_mlp_layer_features(
                    features["video_tokens"][:, :, :chunk],
                    key="previous_video_tokens",
                ),
                "action_tokens": self._concat_mlp_layer_features(
                    features["action_tokens"][:, :, -chunk:],
                    key="action_tokens",
                ),
                "current_real_video_tokens": self._concat_mlp_layer_features(
                    target_features["video_tokens"][:, :, -chunk:],
                    key="current_real_video_tokens",
                ),
            }
        return {
            "previous_video_tokens": features["video_tokens"][:, :chunk],
            "action_tokens": features["action_tokens"][:, -chunk:],
            "current_real_video_tokens": target_features["video_tokens"][:, -chunk:],
        }

    def _split_robotwin_latents(
        self,
        latents: torch.Tensor,
    ) -> list[tuple[str, torch.Tensor]]:
        if self.base_config.env_type == "robotwin_tshape":
            high_h = int(self.base_config.height) // 16
            high_w = int(self.base_config.width) // 16
            wrist_h = high_h // 2
            wrist_w = high_w // 2
            wrist = latents[:, :, :, :wrist_h, :]
            high = latents[:, :, :, wrist_h : wrist_h + high_h, :high_w]
            left = wrist[:, :, :, :, :wrist_w]
            right = wrist[:, :, :, :, wrist_w : wrist_w + wrist_w]
            return [
                ("observation.images.cam_high", high),
                ("observation.images.cam_left_wrist", left),
                ("observation.images.cam_right_wrist", right),
            ]
        cameras = list(self.base_config.obs_cam_keys)
        width = latents.shape[-1] // len(cameras)
        return [
            (camera, latents[:, :, :, :, i * width : (i + 1) * width])
            for i, camera in enumerate(cameras)
        ]

    def _decode_latent_camera(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        decoded = self.vae.decode(
            latents / latents_std + latents_mean,
            return_dict=False,
        )[0]
        return (decoded.float() * 0.5 + 0.5).clamp(0.0, 1.0)

    @staticmethod
    def _align_jepa_to_latents(
        features: torch.Tensor,
        latent_frames: int,
    ) -> torch.Tensor:
        aligned = torch.empty(
            features.shape[0],
            latent_frames,
            features.shape[2],
            features.shape[3],
            features.shape[4],
            dtype=torch.float32,
            device=features.device,
        )
        feats = features.float()
        aligned[:, 0] = feats[:, 0]
        if latent_frames == 1:
            return aligned
        odd = feats[:, 1::2]
        even = feats[:, 2::2]
        pairs = min(odd.shape[1], even.shape[1], latent_frames - 1)
        if pairs > 0:
            aligned[:, 1 : 1 + pairs] = 0.5 * (
                odd[:, :pairs] + even[:, :pairs]
            )
        if pairs < latent_frames - 1:
            last = aligned[:, pairs] if pairs > 0 else aligned[:, 0]
            aligned[:, 1 + pairs :] = last[:, None].expand(
                -1, latent_frames - 1 - pairs, -1, -1, -1
            )
        return aligned

    @torch.no_grad()
    def _predicted_jepa_target(self, latents: torch.Tensor) -> torch.Tensor:
        aligned: list[torch.Tensor] = []
        for _, camera_latents in self._split_robotwin_latents(latents):
            video = self._decode_latent_camera(camera_latents)
            video = torch.cat([video[:, :, 0:1], video], dim=2)
            mean = torch.tensor(
                [0.485, 0.456, 0.406],
                device=video.device,
                dtype=video.dtype,
            ).view(1, 3, 1, 1, 1)
            std = torch.tensor(
                [0.229, 0.224, 0.225],
                device=video.device,
                dtype=video.dtype,
            ).view(1, 3, 1, 1, 1)
            feats = self.jepa_encoder((video - mean) / std)
            batch, _, frames, height, width = video.shape
            h_patches = height // 16
            w_patches = width // 16
            t_jepa = frames // 2
            feats = feats.view(
                batch,
                t_jepa,
                h_patches,
                w_patches,
                feats.shape[-1],
            )
            aligned.append(self._align_jepa_to_latents(feats, latents.shape[2]))

        if self.base_config.env_type == "robotwin_tshape":
            wrist = torch.cat(aligned[1:], dim=3)
            combined = torch.cat([wrist, aligned[0]], dim=2)
        else:
            combined = torch.cat(aligned, dim=3)
        batch, frames, height, width, dim = combined.shape
        pooled = F.avg_pool2d(
            combined.permute(0, 1, 4, 2, 3).flatten(0, 1),
            kernel_size=2,
            stride=2,
        )
        return pooled.view(batch, frames, dim, height // 2, width // 2).permute(
            0, 1, 3, 4, 2
        )

    @torch.no_grad()
    def _phase3_sample(
        self,
        batch: dict[str, torch.Tensor],
        current_prefix: str,
        history_prefix: str,
        timings: dict[str, float] | None = None,
        feature_layers: tuple[int, ...] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        history_latents = self._to_device(batch, f"{history_prefix}latents")
        current_latents = self._to_device(batch, f"{current_prefix}latents")
        history_actions = self._to_device(batch, f"{history_prefix}actions")
        current_actions = self._to_device(batch, f"{current_prefix}actions")
        current_action_mask = self._to_device(
            batch,
            f"{current_prefix}actions_mask",
        )
        text_emb = self._to_device(batch, f"{current_prefix}text_emb")
        video_context_latents, video_context_actions, current_frame_start = (
            self._video_prediction_context(batch, current_latents, current_actions)
        )

        predicted_latents = self._time_call(
            timings,
            "predict_video_time",
            self._predict_video_with_context,
            video_context_latents,
            current_latents,
            video_context_actions,
            current_frame_start,
            text_emb,
        )
        if self.config.training_distribution == "predicted_video_conditioned_action":
            q_current_actions = self._time_call(
                timings,
                "predict_action_time",
                self._predict_actions,
                history_latents,
                predicted_latents,
                history_actions,
                current_actions,
                current_action_mask,
                text_emb,
            )
        else:
            q_current_actions = current_actions

        if self._uses_layer_context_features():
            features = self._build_transformer_critic_contexts(
                batch,
                predicted_latents,
                q_current_actions,
                timings=timings,
                feature_layers=feature_layers,
            )
        else:
            features = self._build_legacy_mlp_features(
                batch,
                predicted_latents,
                q_current_actions,
            )
        return predicted_latents, q_current_actions, features

    def _compute_jepa_distance(
        self,
        predicted_latents: torch.Tensor,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        predicted_jepa = self._predicted_jepa_target(predicted_latents)
        actual_jepa = self._to_device(batch, "jepa_target")
        latent_mask = batch["latents_mask"].to(self.device, non_blocking=True)
        jepa_available = batch.get("jepa_available")
        if jepa_available is not None:
            jepa_available = jepa_available.to(self.device, non_blocking=True)
        distance = _masked_mean_distance(
            predicted_jepa,
            actual_jepa,
            latent_mask,
            jepa_available,
        )
        return distance

    def _compute_goal_jepa_distance(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if "goal_jepa_target" not in batch:
            self._require_goal_distance_available(source="transition dataset")
        current_jepa = self._to_device(batch, "jepa_target")
        goal_jepa = self._to_device(batch, "goal_jepa_target")
        current_mask = batch["latents_mask"].to(self.device, non_blocking=True).bool()
        goal_mask = batch["goal_latents_mask"].to(self.device, non_blocking=True).bool()
        latent_mask = current_mask & goal_mask
        current_available = batch.get("jepa_available")
        goal_available = batch.get("goal_jepa_available")
        if current_available is not None:
            current_available = current_available.to(self.device, non_blocking=True).bool()
        if goal_available is not None:
            goal_available = goal_available.to(self.device, non_blocking=True).bool()
        if current_available is not None and goal_available is not None:
            available = current_available & goal_available
        else:
            available = current_available if current_available is not None else goal_available
        return _masked_mean_distance(
            current_jepa,
            goal_jepa,
            latent_mask,
            available,
        )

    @torch.no_grad()
    def compute_phase3_training_inputs(
        self,
        batch: dict[str, torch.Tensor],
        timings: dict[str, float] | None = None,
        feature_layers: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        raise RuntimeError(
            "Uncached Phase 3 critic training is disabled for clarity. "
            "Use phase3_cache_format='canonical_episode' with "
            "phase3_precomputed_cache_dir, use "
            "phase3_cache_format='federated_canonical_episode' with "
            "phase3_federated_cache_catalog, or run "
            "wan_va.rl.precompute_phase3_cache first. Canonical cache precompute "
            "uses compute_phase3_canonical_inputs() and is unaffected."
        )

    def _accumulate_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        timings: dict[str, float] = {}
        loader_metrics = {
            str(key): float(value)
            for key, value in batch.get("_phase3_loader_metrics", {}).items()
        }
        self._sync_timing()
        microbatch_start = time.perf_counter()
        if batch.get("_phase3_packed_cache", False):
            computed = self._prepare_packed_cache_training_inputs(batch, timings)
        elif batch.get("_phase3_canonical_cache", False):
            computed = self._prepare_canonical_cache_training_inputs(batch, timings)
        else:
            computed = self._load_precomputed_training_inputs(batch, timings)
            if computed is None:
                computed = self.compute_phase3_training_inputs(batch, timings)

        assert self.bundle is not None
        assert self.bundle.value is not None

        features = computed["features"]
        reward = computed["reward"]
        distance = computed["jepa_distance"]
        discount = self._chunk_discount_batch(batch)

        self._sync_timing()
        critic_start = time.perf_counter()
        next_value = None
        if self.config.rl_objective == "mc_flow":
            target = self._mc_return_targets(batch)
            flow_sample = sample_mc_flow_target(
                target,
                timestep_range=self.config.scalar_flow_timestep_range,
            )
            q_predictions = self.bundle.q(
                flow_sample.x_t,
                flow_sample.timestep,
                features["q_context_tokens"],
                features["q_context_mask"],
            )
            value = self.bundle.value(
                flow_sample.x_t,
                flow_sample.timestep,
                features["value_context_tokens"],
                features["value_context_mask"],
            )
            losses = mc_flow_losses(
                q_predictions=q_predictions,
                value_prediction=value,
                target_velocity=flow_sample.target_velocity,
                target=target,
                timestep=flow_sample.timestep,
                value_weight=self.config.value_loss_weight,
                value_mask=batch["state_valid"].to(self.device),
                loss_type=self.config.scalar_flow_loss,
                huber_delta=self.config.scalar_flow_huber_delta,
            )
            q_stack = torch.stack(
                [prediction.detach().float() for prediction in q_predictions],
                dim=0,
            )
            q1 = q_predictions[0]
            q2 = q_predictions[min(1, len(q_predictions) - 1)]
            extra_metrics = {
                "mc_return_target_mean": float(target.detach().mean().cpu()),
                "mc_return_target_std": float(
                    target.detach().float().std(unbiased=False).cpu()
                ),
                "scalar_flow_timestep_mean": float(
                    flow_sample.timestep.detach().mean().cpu()
                ),
                "q_velocity_mean": float(q_stack.mean().cpu()),
                "value_velocity_mean": float(
                    value.detach().float().mean().cpu()
                ),
            }
        elif self.config.rl_objective == "mc_return":
            q1, q2 = self.bundle.q(
                features["q_context_tokens"],
                features["q_context_mask"],
            )
            value = self.bundle.value(
                features["value_context_tokens"],
                features["value_context_mask"],
            )
            target = self._mc_return_targets(batch)
            losses = mc_return_losses(
                q1=q1,
                q2=q2,
                value=value,
                returns=target,
                value_weight=self.config.value_loss_weight,
                value_mask=batch["state_valid"].to(self.device),
                loss_type=self.config.q_loss_type,
                huber_delta=self.config.q_huber_delta,
            )
            extra_metrics = {
                "mc_return_target_mean": float(target.detach().mean().cpu()),
                "mc_return_target_std": float(
                    target.detach().float().std(unbiased=False).cpu()
                ),
            }
        elif self.config.critic_type in {
            "twin_query_transformer_v1",
            "twin_language_query_transformer_v1",
        }:
            assert self.bundle.target_value is not None
            if self.config.critic_type == "twin_language_query_transformer_v1":
                text_args = (
                    features["text_tokens"],
                    features["text_mask"],
                )
            else:
                text_args = ()
            q1, q2 = self.bundle.q(
                features["q_context_tokens"],
                features["q_context_mask"],
                *text_args,
            )
            value = self.bundle.value(
                features["value_context_tokens"],
                features["value_context_mask"],
                *text_args,
            )
            next_value = self.bundle.target_value(
                features["target_value_context_tokens"],
                features["target_value_context_mask"],
                *text_args,
            )
            losses = iql_losses(
                q1=q1,
                q2=q2,
                value=value,
                next_target_value=next_value,
                reward=reward,
                discount=discount,
                expectile=self.config.expectile,
                value_weight=self.config.value_loss_weight,
                value_mask=batch["state_valid"].to(self.device),
                q_loss_type=self.config.q_loss_type,
                q_huber_delta=self.config.q_huber_delta,
            )
            extra_metrics = {}
        else:
            assert self.bundle.target_value is not None
            latent_mask = batch["latents_mask"].to(
                self.device,
                non_blocking=True,
            )
            action_mask = batch["actions_mask"].to(self.device, non_blocking=True)
            q1, q2 = self.bundle.q(features["action_tokens"], action_mask)
            previous_latent_mask = batch["previous_latents_mask"].to(
                self.device,
                non_blocking=True,
            )
            value = self.bundle.value(
                features["previous_video_tokens"],
                previous_latent_mask,
            )
            next_value = self.bundle.target_value(
                features["current_real_video_tokens"],
                latent_mask,
            )
            losses = iql_losses(
                q1=q1,
                q2=q2,
                value=value,
                next_target_value=next_value,
                reward=reward,
                discount=discount,
                expectile=self.config.expectile,
                value_weight=self.config.value_loss_weight,
                value_mask=batch["state_valid"].to(self.device),
                q_loss_type=self.config.q_loss_type,
                q_huber_delta=self.config.q_huber_delta,
            )
            extra_metrics = {}

        (losses.total / self.config.gradient_accumulation_steps).backward()
        self._sync_timing()
        timings["critic_backward_time"] = (
            timings.get("critic_backward_time", 0.0)
            + time.perf_counter()
            - critic_start
        )
        timings["microbatch_time"] = time.perf_counter() - microbatch_start
        reward_float = reward.detach().float()
        state_valid = batch["state_valid"].to(self.device, non_blocking=True).float()
        done = batch["done"].to(self.device, non_blocking=True).float()
        episode_success = batch.get("episode_success")
        episode_chunk_count = batch.get("episode_chunk_count")
        data_metrics = {
            "local_microbatch_size": float(state_valid.shape[0]),
            "state_valid_frac": float(state_valid.mean().cpu()),
            "done_frac": float(done.mean().cpu()),
            "rl_chunk_idx_mean": float(
                batch["rl_chunk_idx"]
                .to(self.device, non_blocking=True)
                .float()
                .mean()
                .cpu()
            ),
        }
        dataset_id = batch.get("dataset_id")
        if dataset_id is not None:
            dataset_id = dataset_id.to(
                self.device,
                non_blocking=True,
            )
            data_metrics["clean_dataset_frac"] = float(
                (dataset_id == 0).float().mean().cpu()
            )
            data_metrics["augmented_dataset_frac"] = float(
                (dataset_id == 1).float().mean().cpu()
            )
            data_metrics["other_dataset_frac"] = float(
                (dataset_id > 1).float().mean().cpu()
            )
        for metric_name, mask_key in (
            ("value_context", "value_context_mask"),
            ("q_context", "q_context_mask"),
            ("target_value_context", "target_value_context_mask"),
        ):
            mask = features.get(mask_key)
            if mask is None:
                continue
            mask_float = mask.detach().float()
            token_count = mask_float.reshape(mask_float.shape[0], -1).sum(dim=1)
            data_metrics[f"{metric_name}_token_count_mean"] = float(
                token_count.mean().cpu()
            )
            data_metrics[f"{metric_name}_token_count_max"] = float(
                token_count.max().cpu()
            )
            data_metrics[f"{metric_name}_valid_token_frac"] = float(
                mask_float.mean().cpu()
            )
        if episode_success is not None:
            data_metrics["episode_success_frac"] = float(
                episode_success.to(self.device, non_blocking=True)
                .float()
                .mean()
                .cpu()
            )
        if episode_chunk_count is not None:
            data_metrics["episode_chunk_count_mean"] = float(
                episode_chunk_count.to(self.device, non_blocking=True)
                .float()
                .mean()
                .cpu()
            )
        q_min = torch.minimum(q1, q2).detach().float()
        advantage = q_min - value.detach().float()
        return {
            "total_loss": float(losses.total.detach().cpu()),
            "q_loss": float(losses.q.detach().cpu()),
            "value_loss": float(losses.value.detach().cpu()),
            "q_target_mean": float(losses.target.detach().float().mean().cpu()),
            "q_target_std": float(
                losses.target.detach().float().std(unbiased=False).cpu()
            ),
            "v_target_mean": float(
                q_min.mean().cpu()
            ),
            "v_target_std": float(
                q_min.std(unbiased=False).cpu()
            ),
            "advantage_mean": float(advantage.mean().cpu()),
            "advantage_std": float(advantage.std(unbiased=False).cpu()),
            "positive_advantage_frac": float(
                (advantage > 0).float().mean().cpu()
            ),
            "discount_mean": float(discount.detach().float().mean().cpu()),
            "reward_mean": float(reward_float.mean().cpu()),
            "reward_std": float(reward_float.std(unbiased=False).cpu()),
            "reward_min": float(reward_float.min().cpu()),
            "reward_max": float(reward_float.max().cpu()),
            "jepa_distance": float(distance.detach().float().mean().cpu()),
            "q1_mean": float(q1.detach().float().mean().cpu()),
            "q2_mean": float(q2.detach().float().mean().cpu()),
            "q_disagreement": float(
                (q1 - q2).detach().abs().float().mean().cpu()
            ),
            "value_mean": float(value.detach().float().mean().cpu()),
            **(
                {
                    "next_value_mean": float(
                        next_value.detach().float().mean().cpu()
                    )
                }
                if next_value is not None
                else {}
            ),
            "cache_hit": float(computed.get("cache_hit", 0.0)),
            **self._reward_term_batch_metrics(distance, batch),
            **data_metrics,
            **extra_metrics,
            **timings,
            **loader_metrics,
        }

    def _gradient_buckets(self) -> list[list[torch.Tensor]]:
        bucket_limit = max(
            1,
            int(self.config.gradient_allreduce_bucket_mb * 1024 * 1024),
        )
        buckets: list[list[torch.Tensor]] = []
        current: list[torch.Tensor] = []
        current_key: tuple[torch.device, torch.dtype] | None = None
        current_bytes = 0
        for parameter in self.bundle.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            key = (grad.device, grad.dtype)
            grad_bytes = grad.numel() * grad.element_size()
            if current and (
                key != current_key
                or current_bytes + grad_bytes > bucket_limit
            ):
                buckets.append(current)
                current = []
                current_bytes = 0
            current.append(grad)
            current_key = key
            current_bytes += grad_bytes
        if current:
            buckets.append(current)
        return buckets

    def _all_reduce_gradients(self) -> tuple[int, int]:
        buckets = self._gradient_buckets()
        total_bytes = 0
        for gradients in buckets:
            flat = _flatten_dense_tensors(gradients)
            total_bytes += flat.numel() * flat.element_size()
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            for gradient, reduced in zip(
                gradients,
                _unflatten_dense_tensors(flat, gradients),
            ):
                gradient.copy_(reduced)
        return len(buckets), total_bytes

    def _submodule_gradient_norms(self) -> dict[str, float]:
        squared: dict[str, torch.Tensor] = {}
        for name, parameter in self.bundle.named_parameters():
            if parameter.grad is None:
                continue
            if name.startswith("q.q1."):
                group = "q1"
            elif name.startswith("q.q2."):
                group = "q2"
            elif name.startswith("value."):
                group = "value"
            else:
                continue
            value = parameter.grad.detach().float().square().sum()
            squared[group] = squared.get(group, torch.zeros_like(value)) + value
        return {
            f"optimizer/{group}_grad_norm": float(value.sqrt().cpu())
            for group, value in squared.items()
        }

    def _optimizer_step(
        self,
        timings: dict[str, float] | None = None,
        *,
        collect_detailed_metrics: bool = False,
    ) -> float:
        assert self.bundle is not None
        assert self.optimizer is not None
        total_start = time.perf_counter()
        lr = self._set_optimizer_lr(self.step)
        if timings is not None:
            timings["learning_rate"] = lr
        if self.world_size > 1:
            self._sync_timing()
            all_reduce_start = time.perf_counter()
            bucket_count, gradient_bytes = self._all_reduce_gradients()
            self._sync_timing()
            if timings is not None:
                timings["optimizer/all_reduce_time"] = (
                    time.perf_counter() - all_reduce_start
                )
                timings["optimizer/all_reduce_bucket_count"] = float(
                    bucket_count
                )
                timings["optimizer/all_reduce_bytes"] = float(gradient_bytes)
        if collect_detailed_metrics and timings is not None:
            timings.update(self._submodule_gradient_norms())
        self._sync_timing()
        clip_start = time.perf_counter()
        max_grad_norm = (
            float("inf")
            if self.config.max_grad_norm is None
            else float(self.config.max_grad_norm)
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in self.bundle.parameters()
                if parameter.requires_grad
            ],
            max_norm=max_grad_norm,
        )
        self._sync_timing()
        if timings is not None:
            timings["optimizer/grad_clip_time"] = time.perf_counter() - clip_start
        self._sync_timing()
        step_start = time.perf_counter()
        self.optimizer.step()
        self._sync_timing()
        if timings is not None:
            timings["optimizer/adam_step_time"] = time.perf_counter() - step_start
        self._sync_timing()
        target_start = time.perf_counter()
        self.bundle.update_target(self.config.target_ema_rate)
        self._sync_timing()
        if timings is not None:
            timings["optimizer/target_update_time"] = (
                time.perf_counter() - target_start
            )
        zero_start = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        if timings is not None:
            timings["optimizer/zero_grad_time"] = time.perf_counter() - zero_start
            timings["optimizer_time"] = time.perf_counter() - total_start
        return float(torch.as_tensor(grad_norm).float().cpu())

    def _distributed_step_metrics(
        self,
        metrics: dict[str, float],
    ) -> None:
        local = torch.tensor(
            [
                metrics["effective_samples_per_optimizer_step"],
                metrics.get("data_wait_time", 0.0),
                metrics.get("microbatch_time", 0.0),
                metrics.get("optimizer_time", 0.0),
            ],
            device=self.device,
            dtype=torch.float64,
        )
        if self.world_size > 1:
            gathered = [torch.zeros_like(local) for _ in range(self.world_size)]
            dist.all_gather(gathered, local)
            values = torch.stack(gathered)
        else:
            values = local.unsqueeze(0)
        metrics["estimated_global_batch_size"] = float(values[:, 0].sum().cpu())
        metrics["rank_batch_size_min"] = float(values[:, 0].min().cpu())
        metrics["rank_batch_size_max"] = float(values[:, 0].max().cpu())
        for index, name in enumerate(
            ("data_wait", "microbatch", "optimizer"),
            start=1,
        ):
            minimum = values[:, index].min()
            maximum = values[:, index].max()
            metrics[f"rank_{name}_time_min"] = float(minimum.cpu())
            metrics[f"rank_{name}_time_max"] = float(maximum.cpu())
            metrics[f"rank_{name}_time_skew"] = float(
                (maximum - minimum).cpu()
            )

    def _gpu_memory_metrics(self) -> dict[str, float]:
        if self.device.type != "cuda":
            return {}
        gib = float(1024**3)
        metrics = {
            "gpu_memory_allocated_gib": (
                torch.cuda.memory_allocated(self.device) / gib
            ),
            "gpu_memory_reserved_gib": (
                torch.cuda.memory_reserved(self.device) / gib
            ),
            "gpu_memory_peak_allocated_gib": (
                torch.cuda.max_memory_allocated(self.device) / gib
            ),
            "gpu_memory_peak_reserved_gib": (
                torch.cuda.max_memory_reserved(self.device) / gib
            ),
        }
        torch.cuda.reset_peak_memory_stats(self.device)
        return metrics

    @staticmethod
    def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
        if not metrics:
            raise ValueError("metrics must be non-empty")
        keys = set().union(*(item.keys() for item in metrics))
        return {
            key: sum(item.get(key, 0.0) for item in metrics) / len(metrics)
            for key in sorted(keys)
        }

    @torch.no_grad()
    def _parameter_norm(self) -> float:
        squared_norm = torch.zeros((), device=self.device, dtype=torch.float32)
        for parameter in self.bundle.parameters():
            if parameter.requires_grad:
                squared_norm += parameter.detach().float().square().sum()
        return float(squared_norm.sqrt().cpu())

    def _save(self) -> None:
        if self.rank != 0:
            return
        save_critic_checkpoint(
            directory=self.checkpoint_dir / f"checkpoint_{self.step:08d}",
            bundle=self.bundle,
            optimizer=self.optimizer,
            config=self.config,
            step=self.step,
            manifest={
                "base_config_name": self.config.base_config_name,
                "base_transformer_path": _transformer_path(
                    self.base_config,
                    self.config,
                ),
                "training_distribution": self.config.training_distribution,
                "phase3_precompute_mode": self.config.phase3_precompute_mode,
                "phase3_video_prediction_context": (
                    self.config.phase3_video_prediction_context
                ),
                "reward_source": self.config.reward_source,
                "rl_objective": self.config.rl_objective,
                "gradient_accumulation_steps": (
                    self.config.gradient_accumulation_steps
                ),
                "critic_feature_layers": list(self.config.critic_feature_layers),
                "critic_num_query_tokens": self.config.critic_num_query_tokens,
                "critic_transformer_hidden_dim": (
                    self.config.critic_transformer_hidden_dim
                ),
                "critic_transformer_num_layers": (
                    self.config.critic_transformer_layer_count
                ),
                "critic_transformer_num_heads": (
                    self.config.critic_transformer_num_heads
                ),
                "critic_transformer_ffn_dim": (
                    self.config.critic_transformer_ffn_dim
                ),
                "critic_transformer_dropout": (
                    self.config.critic_transformer_dropout
                ),
                "critic_text_embed_dim": self.config.critic_text_embed_dim,
                "num_q_heads": self.config.num_q_heads,
                "q_ensemble_reduce": self.config.q_ensemble_reduce,
                "q_loss_type": self.config.q_loss_type,
                "q_huber_delta": self.config.q_huber_delta,
                "lr_scheduler": self.config.lr_scheduler,
                "min_lr": self.config.min_lr,
                "learning_rate": self.config.learning_rate,
                "freeze_inherited_critic_steps": (
                    self.config.freeze_inherited_critic_steps
                ),
                "inherited_critic_lr_scale": (
                    self.config.inherited_critic_lr_scale
                ),
                "language_critic_lr_scale": (
                    self.config.language_critic_lr_scale
                ),
                **(self.transfer_initialization or {}),
                "final_reward_normalization": (
                    self.config.final_reward_normalization
                ),
                "final_reward_stats": self.final_reward_stats,
                "mc_return_normalization": self.config.mc_return_normalization,
                "mc_return_mean": self.mc_return_mean,
                "mc_return_std": self.mc_return_std,
                "mc_return_clip": self.config.mc_return_clip,
                "scalar_flow_loss": self.config.scalar_flow_loss,
                "scalar_flow_huber_delta": self.config.scalar_flow_huber_delta,
                "scalar_flow_timestep_range": (
                    self.config.scalar_flow_timestep_range
                ),
                "scalar_flow_eval_timestep": (
                    self.config.scalar_flow_eval_timestep
                ),
                "scalar_flow_eval_noise": self.config.scalar_flow_eval_noise,
                "q_feature_action_source": "current_real_executed_action",
                "value_video_feature_timestep_range": (
                    self.config.value_video_feature_timestep_range
                ),
                "value_action_feature_timestep_range": (
                    self.config.value_action_feature_timestep_range
                ),
                "q_video_feature_timestep_range": (
                    self.config.q_video_feature_timestep_range
                ),
                "q_action_feature_timestep_range": (
                    self.config.q_action_feature_timestep_range
                ),
                "phase3_jepa_checkpoint": self.config.phase3_jepa_checkpoint,
                **(
                    {
                        "phase3_federated_cache_catalog_sha256": (
                            self.dataset.catalog_sha256
                        )
                    }
                    if isinstance(
                        self.dataset,
                        Phase3FederatedCanonicalEpisodeCacheDataset,
                    )
                    else {}
                ),
                "value_state_alignment": (
                    "previous_real_video_action_for_v_previous_current_real_for_target_v"
                ),
                "q_context": (
                    "previous_real_video_action_plus_current_real_action"
                ),
            },
        )

    def train(self) -> None:
        assert self.bundle is not None
        assert self.optimizer is not None
        self.bundle.train()
        epoch = 0
        resume_batches_to_skip = 0
        exact_resume_cursor = False
        if self.config.resume_from and self.sampler is not None:
            batches_per_epoch = len(self.sampler)
            if batches_per_epoch <= 0:
                raise RuntimeError("cannot resume an empty deterministic sampler")
            completed_batches = (
                self.step * self.config.gradient_accumulation_steps
            )
            epoch, resume_batches_to_skip = divmod(
                completed_batches,
                batches_per_epoch,
            )
            exact_resume_cursor = self.config.phase3_exact_resume_cursor
            if exact_resume_cursor and not callable(
                getattr(self.sampler, "set_start_batch", None)
            ):
                raise RuntimeError(
                    "phase3_exact_resume_cursor requires a sampler with "
                    "set_start_batch() support"
                )
            _rank_log(
                self.rank,
                "resuming deterministic sampler "
                f"step={self.step} epoch={epoch} "
                f"batch_offset={resume_batches_to_skip} "
                f"batches_per_epoch={batches_per_epoch} "
                f"remaining_batches={batches_per_epoch - resume_batches_to_skip} "
                f"exact_cursor={exact_resume_cursor}",
            )
        pending_metrics: list[dict[str, float]] = []
        self.optimizer.zero_grad(set_to_none=True)
        while self.step < self.config.num_steps:
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)
                if exact_resume_cursor:
                    self.sampler.set_start_batch(resume_batches_to_skip)
            data_wait_start = time.perf_counter()
            step_wall_start = data_wait_start
            for batch_index, batch in enumerate(self.loader):
                if (
                    not exact_resume_cursor
                    and batch_index < resume_batches_to_skip
                ):
                    data_wait_start = time.perf_counter()
                    continue
                if not pending_metrics:
                    step_wall_start = data_wait_start
                data_wait_time = time.perf_counter() - data_wait_start
                actual_batch_size = int(batch["state_valid"].shape[0])
                if (
                    self.config.phase3_drop_last
                    and actual_batch_size != self.config.batch_size
                ):
                    raise RuntimeError(
                        "fixed-size Phase 3 batch contract violated: "
                        f"expected={self.config.batch_size} "
                        f"actual={actual_batch_size}"
                    )
                if self.step == 0 and not pending_metrics:
                    _rank_log(self.rank, "starting first optimization batch")
                batch_metrics = self._accumulate_batch(batch)
                batch_metrics["data_wait_time"] = data_wait_time
                pending_metrics.append(batch_metrics)
                if (
                    len(pending_metrics)
                    < self.config.gradient_accumulation_steps
                ):
                    data_wait_start = time.perf_counter()
                    continue
                next_step = self.step + 1
                wandb_interval = (
                    self.config.wandb_log_interval
                    or self.config.log_interval
                )
                should_log_json = next_step % self.config.log_interval == 0
                should_log_wandb = (
                    self.config.enable_wandb
                    and next_step % wandb_interval == 0
                )
                collect_detailed_metrics = should_log_json or should_log_wandb
                optimizer_timings: dict[str, float] = {}
                grad_norm = self._optimizer_step(
                    optimizer_timings,
                    collect_detailed_metrics=collect_detailed_metrics,
                )
                metrics = self._mean_metrics(pending_metrics)
                effective_samples = sum(
                    item["local_microbatch_size"] for item in pending_metrics
                )
                metrics.update(optimizer_timings)
                metrics["grad_norm"] = grad_norm
                metrics["effective_samples_per_optimizer_step"] = float(
                    effective_samples
                )
                pending_metrics.clear()
                self.step = next_step
                if collect_detailed_metrics:
                    self._distributed_step_metrics(metrics)
                    metrics.update(self._gpu_memory_metrics())
                if self.step % self.config.save_interval == 0:
                    checkpoint_start = time.perf_counter()
                    self._save()
                    if self.rank == 0:
                        metrics["checkpoint_time"] = (
                            time.perf_counter() - checkpoint_start
                        )
                metrics["step_wall_time"] = time.perf_counter() - step_wall_start
                if collect_detailed_metrics:
                    global_batch = metrics.get(
                        "estimated_global_batch_size",
                        effective_samples * self.world_size,
                    )
                    metrics["samples_per_second"] = (
                        global_batch / max(metrics["step_wall_time"], 1e-12)
                    )
                if self.rank == 0:
                    param_norm = (
                        self._parameter_norm()
                        if should_log_json or should_log_wandb
                        else None
                    )
                    if self.step % self.config.log_interval == 0:
                        entry = {
                            "step": self.step,
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            **metrics,
                            "param_norm": param_norm,
                        }
                        with self.log_path.open("a") as handle:
                            handle.write(json.dumps(entry) + "\n")
                        _rank_log(
                            self.rank,
                            "step "
                            f"{self.step} total_loss={metrics['total_loss']:.4f} "
                            f"q_loss={metrics['q_loss']:.4f} "
                            f"value_loss={metrics['value_loss']:.4f} "
                            f"reward={metrics['reward_mean']:.3f} "
                            f"lr={metrics.get('learning_rate', self.config.learning_rate):.3e} "
                            f"microbatch_time={metrics.get('microbatch_time', 0.0):.2f}s "
                            f"optimizer_time={metrics.get('optimizer_time', 0.0):.2f}s "
                            f"cache_hit={metrics.get('cache_hit', 0.0):.2f}",
                        )
                    if should_log_wandb:
                        self.wandb_logger.log(
                            critic_wandb_payload(
                                metrics,
                                config=self.config,
                                param_norm=param_norm,
                                step_time=metrics.get("step_wall_time"),
                            ),
                            step=self.step,
                        )
                if self.step >= self.config.num_steps:
                    break
                data_wait_start = time.perf_counter()
            epoch += 1
            resume_batches_to_skip = 0
        if self.step % self.config.save_interval != 0:
            self._save()
        if self.rank == 0:
            self.wandb_logger.finish()


def run(config_path: str) -> None:
    config = CriticTrainingConfig.from_json(config_path)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rank, local_rank, world_size, device = _init_runtime()
    trainer = Phase3CriticTrainer(config, rank, local_rank, world_size, device)
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Phase 3 generated-video-conditioned IQL critics"
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    init_logger()
    try:
        main()
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
