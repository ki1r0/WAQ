from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any


def _parse_layer_spec(
    value: Any,
    *,
    key: str,
    allow_final_norm: bool,
) -> tuple[int, ...]:
    if isinstance(value, dict):
        allowed = {"start", "stop", "step"}
        unknown = sorted(set(value).difference(allowed))
        if unknown:
            raise ValueError(f"{key} has unknown range keys: {unknown}")
        if "start" not in value or "stop" not in value:
            raise ValueError(f"{key} range form requires start and stop")
        start = int(value["start"])
        stop = int(value["stop"])
        step = int(value.get("step", 1))
        if step == 0:
            raise ValueError(f"{key}.step must be non-zero")
        layers = tuple(range(start, stop, step))
    elif isinstance(value, (list, tuple)):
        layers = tuple(value)
    else:
        raise TypeError(
            f"{key} must be a list/tuple of layer indices or a "
            "{start, stop, step} range dict"
        )

    if not layers:
        raise ValueError(f"{key} must not be empty")
    if any(not isinstance(layer, int) or isinstance(layer, bool) for layer in layers):
        raise TypeError(f"{key} must contain only integers")
    minimum = -1 if allow_final_norm else 0
    if any(layer < minimum for layer in layers):
        if allow_final_norm:
            raise ValueError(f"{key} layers must be -1 or non-negative")
        raise ValueError(f"{key} layers must be non-negative DiT block indices")
    if not allow_final_norm and any(layer == -1 for layer in layers):
        raise ValueError(f"{key} does not support -1 final-normalized features")
    if len(set(layers)) != len(layers):
        raise ValueError(f"{key} must not contain duplicate layers")
    return layers


_LEGACY_ALGORITHM_TO_OBJECTIVE = {
    "mc": "mc_return",
    "iql": "iql",
    "mc_return": "mc_return",
    "mc_flow": "mc_flow",
}
_OBJECTIVE_TO_LEGACY_ALGORITHM = {
    "mc_return": "mc_return",
    "iql": "iql",
    "mc_flow": "mc_flow",
}
_SCALAR_QV_CRITICS = {
    "twin_mlp_v1",
    "twin_query_transformer_v1",
    "twin_language_query_transformer_v1",
}
_FLOW_SCALAR_CRITICS = {"twin_flow_scalar_transformer_v1"}
_REWARD_TERM_NAMES = {
    "negative_predicted_actual_jepa_distance",
    "terminal_success",
    "terminal_outcome_cost",
    "successful_episode_progress",
    "goal_jepa_progress",
    "goal_jepa_potential_shaping",
}
_REWARD_NORMALIZATION_MODES = {
    False,
    True,
    "z_score",
    "scale_neg1_pos1",
}

_DATASET_CONFIG_FIELDS = {
    "cache_format": "phase3_cache_format",
    "precomputed_cache_dir": "phase3_precomputed_cache_dir",
    "federated_cache_catalog": "phase3_federated_cache_catalog",
    "federated_cache_artifacts_manifest": "phase3_federated_cache_artifacts_manifest",
    "language_sidecar_catalog": "phase3_language_sidecar_catalog",
    "episode_success_overlay": "phase3_episode_success_overlay",
    "require_precomputed_cache": "phase3_require_precomputed_cache",
    "use_packed_cache": "phase3_use_packed_cache",
    "packed_cache_dir": "phase3_packed_cache_dir",
    "packed_cache_shard_size": "phase3_packed_cache_shard_size",
    "cache_batching": "phase3_cache_batching",
    "episodes_per_batch": "phase3_episodes_per_batch",
    "drop_last": "phase3_drop_last",
    "context_length_aware_batching": "phase3_context_length_aware_batching",
    "global_balanced_batching": "phase3_global_balanced_batching",
    "exact_resume_cursor": "phase3_exact_resume_cursor",
    "federated_source_balanced_sampling": "phase3_federated_source_balanced_sampling",
    "preserve_episode_group_order": "phase3_preserve_episode_group_order",
    "episode_cache_lru_size": "phase3_episode_cache_lru_size",
    "cuda_prefetch": "phase3_cuda_prefetch",
    "pin_memory": "phase3_pin_memory",
    "persistent_workers": "phase3_persistent_workers",
    "prefetch_factor": "phase3_prefetch_factor",
    "episode_aware_prefetch": "phase3_episode_aware_prefetch",
    "episode_prefetch_workers": "phase3_episode_prefetch_workers",
    "episode_prefetch_lookahead_batches": "phase3_episode_prefetch_lookahead_batches",
    "batch_prefetch_depth": "phase3_batch_prefetch_depth",
    "episode_cache_max_gib": "phase3_episode_cache_max_gib",
    "cuda_prefetch_depth": "phase3_cuda_prefetch_depth",
    "loader_mode": "phase3_loader_mode",
    "transition_prefetch_lookahead": "phase3_transition_prefetch_lookahead",
    "transition_prefetch_workers": "phase3_transition_prefetch_workers",
    "transition_batch_workers": "phase3_transition_batch_workers",
    "transition_max_active_episodes": "phase3_transition_max_active_episodes",
    "transition_max_inflight_gib": "phase3_transition_max_inflight_gib",
    "precompute_mode": "phase3_precompute_mode",
    "video_prediction_context": "phase3_video_prediction_context",
    "exclude_last_n_chunks": "phase3_exclude_last_n_chunks",
    "cache_residency": "phase3_cache_residency",
    "cache_group_size": "phase3_cache_group_size",
    "cache_group_prefetch_depth": "phase3_cache_group_prefetch_depth",
    "cache_max_resident_gib": "phase3_cache_max_resident_gib",
    "cache_load_workers": "phase3_cache_load_workers",
    "cache_warmup": "phase3_cache_warmup",
}
_DATASET_PATH_FIELDS = {
    "precomputed_cache_dir", "federated_cache_catalog",
    "federated_cache_artifacts_manifest",
    "language_sidecar_catalog", "episode_success_overlay", "packed_cache_dir",
}


def _resolve_dataset_config(training_path: Path, values: dict[str, Any]) -> dict[str, Any]:
    reference = values.get("dataset_config")
    if reference is None:
        return values
    if not isinstance(reference, str) or not reference.strip():
        raise TypeError("dataset_config must be a non-empty JSON path")
    dataset_path = Path(reference)
    if not dataset_path.is_absolute():
        dataset_path = (training_path.parent / dataset_path).resolve()
    try:
        dataset_values = json.loads(dataset_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing dataset_config: {dataset_path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"dataset_config must be valid JSON: {dataset_path}") from error
    if not isinstance(dataset_values, dict):
        raise TypeError("dataset_config must contain a JSON object")
    unknown = sorted(set(dataset_values).difference(_DATASET_CONFIG_FIELDS))
    if unknown:
        raise ValueError(f"Unknown dataset config keys: {unknown}")
    expanded = dict(values)
    for dataset_key, flat_key in _DATASET_CONFIG_FIELDS.items():
        if dataset_key not in dataset_values:
            continue
        value = dataset_values[dataset_key]
        if dataset_key in _DATASET_PATH_FIELDS and value is not None:
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"dataset_config.{dataset_key} must be a path or null")
            path_value = Path(value)
            if not path_value.is_absolute():
                path_value = (dataset_path.parent / path_value).resolve()
            value = str(path_value)
        if flat_key in expanded and expanded[flat_key] != value:
            raise ValueError(f"Conflicting dataset setting: {flat_key!r} is declared both in the training config and dataset_config")
        expanded[flat_key] = value
    expanded["dataset_config"] = str(dataset_path)
    return expanded


def _normalize_rl_objective(value: str, *, key: str) -> str:
    if value not in _LEGACY_ALGORITHM_TO_OBJECTIVE:
        allowed = sorted(set(_LEGACY_ALGORITHM_TO_OBJECTIVE))
        raise ValueError(f"{key} must be one of {allowed}")
    return _LEGACY_ALGORITHM_TO_OBJECTIVE[value]


def _is_checkpoint_config(path: Path) -> bool:
    return path.name == "config.json" and (
        (path.parent / "training_state.pt").exists()
        or (path.parent / "manifest.json").exists()
    )


def _backup_and_write_migrated_config(
    path: Path,
    original: dict[str, Any],
    migrated: dict[str, Any],
) -> None:
    backup = path.with_name("config.json.legacy_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(migrated, indent=2, sort_keys=True) + "\n")


def _normalize_reward_terms(
    value: Any,
    *,
    reward_source: str,
    include_sparse_success_reward: bool,
    jepa_reward_weight: float,
    success_reward_weight: float,
) -> tuple[dict[str, Any], ...]:
    if value is None:
        terms: list[dict[str, Any]] = []
        if reward_source == "negative_predicted_actual_jepa_distance":
            terms.append(
                {
                    "name": "negative_predicted_actual_jepa_distance",
                    "weight": float(jepa_reward_weight),
                    "normalize": False,
                }
            )
            if include_sparse_success_reward:
                terms.append(
                    {
                        "name": "terminal_success",
                        "weight": float(success_reward_weight),
                        "normalize": False,
                    }
                )
        elif reward_source == "sparse_success":
            terms.append(
                {
                    "name": "terminal_success",
                    "weight": float(success_reward_weight),
                    "normalize": False,
                }
            )
        elif reward_source == "jepa_delta_distance":
            # Legacy dataset-side reward is still handled by ChunkTransitionDataset.
            terms = []
        else:
            raise ValueError(f"Unsupported reward_source {reward_source!r}")
        return tuple(terms)

    if not isinstance(value, list):
        raise TypeError("reward_terms must be a list of reward term objects")
    terms = []
    for index, term in enumerate(value):
        if not isinstance(term, dict):
            raise TypeError(f"reward_terms[{index}] must be an object")
        allowed = {"name", "weight", "normalize"}
        unknown = sorted(set(term).difference(allowed))
        if unknown:
            raise ValueError(
                f"reward_terms[{index}] has unknown keys: {unknown}"
            )
        name = term.get("name")
        if name not in _REWARD_TERM_NAMES:
            raise ValueError(
                f"reward_terms[{index}].name must be one of "
                f"{sorted(_REWARD_TERM_NAMES)}"
            )
        normalize = term.get("normalize", False)
        if normalize not in _REWARD_NORMALIZATION_MODES:
            raise ValueError(
                f"reward_terms[{index}].normalize must be false, true, "
                "'z_score', or 'scale_neg1_pos1'"
            )
        if name in {"terminal_success", "terminal_outcome_cost"} and normalize:
            raise ValueError(f"{name} reward must not be normalized")
        terms.append(
            {
                "name": str(name),
                "weight": float(term.get("weight", 1.0)),
                "normalize": normalize,
            }
        )
    return tuple(terms)


@dataclass(frozen=True)
class CriticTrainingConfig:
    base_config_name: str
    output_dir: str
    infer_latent_chunk_size: int
    algorithm: str = "mc"
    rl_objective: str | None = None
    critic_type: str = "twin_mlp_v1"
    transformer_path: str | None = None
    feature_dim: int = 3072
    feature_layers: tuple[int, ...] = (-1,)
    critic_feature_layers: Any = None
    feature_aggregation: str = "single"
    hidden_dim: int = 512
    num_layers: int = 2
    critic_num_query_tokens: int = 1
    critic_transformer_hidden_dim: int = 512
    critic_transformer_num_layers: int | None = None
    critic_transformer_num_heads: int = 8
    critic_transformer_ffn_dim: int | None = None
    critic_transformer_dropout: float = 0.0
    critic_text_embed_dim: int = 4096
    num_q_heads: int = 2
    q_ensemble_reduce: str = "min"
    mc_return_normalization: str = "dataset_mc_return"
    mc_return_clip: tuple[float, float] | None = (-10.0, 10.0)
    scalar_flow_loss: str = "huber"
    scalar_flow_huber_delta: float = 1.0
    scalar_flow_timestep_range: tuple[float, float] = (0.0, 1.0)
    scalar_flow_eval_timestep: float = 0.5
    scalar_flow_eval_noise: str = "zero"
    gamma: float = 0.99
    reward_source: str = "sparse_success"
    include_sparse_success_reward: bool = False
    jepa_reward_weight: float = 1.0
    success_reward_weight: float = 1.0
    reward_terms: Any = None
    final_reward_normalization: str | bool | None = "none"
    goal_jepa_progress_lambda: float = 0.83
    goal_jepa_progress_beta: float = 0.52
    goal_jepa_potential_quantile_low: float | None = None
    goal_jepa_potential_quantile_high: float | None = None
    training_distribution: str = "clean_dataset"
    phase3_video_exec_step: int = -1
    phase3_q_feature_timestep: float = 0.0
    phase3_v_feature_timestep: float = 0.0
    value_video_feature_timestep_range: tuple[float, float] = (0.0, 0.0)
    value_action_feature_timestep_range: tuple[float, float] = (0.0, 0.0)
    q_video_feature_timestep_range: tuple[float, float] = (0.0, 0.0)
    q_action_feature_timestep_range: tuple[float, float] = (0.0, 0.0)
    phase3_video_num_inference_steps: int | None = None
    phase3_action_num_inference_steps: int | None = None
    phase3_jepa_checkpoint: str = (
        "/luhongchao/shared/weights/vjepa2.1/vjepa2_1_vitG_384.pt"
    )
    phase3_jepa_checkpoint_key: str = "target_encoder"
    phase3_jepa_config: str | None = None
    phase3_recompute_jepa_targets: bool = False
    phase3_precomputed_cache_dir: str | None = None
    phase3_federated_cache_catalog: str | None = None
    phase3_federated_cache_artifacts_manifest: str | None = None
    phase3_language_sidecar_catalog: str | None = None
    phase3_episode_success_overlay: str | None = None
    phase3_cache_format: str = "expanded_transition"
    phase3_require_precomputed_cache: bool = False
    phase3_use_packed_cache: bool = False
    phase3_packed_cache_dir: str | None = None
    phase3_packed_cache_shard_size: int = 64
    phase3_cache_batching: str = "random"
    phase3_episodes_per_batch: int = 2
    phase3_drop_last: bool = False
    phase3_context_length_aware_batching: bool = False
    phase3_global_balanced_batching: bool = False
    phase3_exact_resume_cursor: bool = False
    phase3_federated_source_balanced_sampling: bool = False
    phase3_preserve_episode_group_order: bool = False
    phase3_episode_cache_lru_size: int = 4
    phase3_cuda_prefetch: bool = False
    phase3_pin_memory: bool = True
    phase3_persistent_workers: bool = True
    phase3_prefetch_factor: int | None = 2
    phase3_episode_aware_prefetch: bool = False
    phase3_episode_prefetch_workers: int = 2
    phase3_episode_prefetch_lookahead_batches: int = 24
    phase3_batch_prefetch_depth: int = 4
    phase3_episode_cache_max_gib: float = 8.0
    phase3_cuda_prefetch_depth: int = 2
    # "episode" is the legacy complete-episode loader.  The range mode is
    # opt-in so old training and evaluation configs retain their behavior.
    phase3_loader_mode: str = "episode"
    phase3_transition_prefetch_lookahead: int = 8
    phase3_transition_prefetch_workers: int = 1
    phase3_transition_batch_workers: int = 2
    phase3_transition_max_active_episodes: int = 2
    phase3_transition_max_inflight_gib: float = 16.0
    # Opt-in cache residency policy.  ``on_demand`` preserves all legacy
    # loaders.  ``bounded_group_prefetching`` keeps the active episode group
    # and byte-bounded future groups resident.  ``full`` admits all episodes
    # before training starts, subject to the same safety budget.
    phase3_cache_residency: str = "on_demand"
    phase3_cache_group_size: int = 1
    phase3_cache_group_prefetch_depth: int = 1
    phase3_cache_max_resident_gib: float = 8.0
    phase3_cache_load_workers: int = 2
    phase3_cache_warmup: bool = False
    phase3_precompute_mode: str = "packed_train"
    phase3_video_prediction_context: str = "episode_prefix"
    phase3_exclude_last_n_chunks: int = 0
    assume_missing_success_is_success: bool = False
    expectile: float = 0.7
    value_loss_weight: float = 1.0
    q_loss_type: str = "huber"
    q_huber_delta: float = 1.0
    target_ema_rate: float = 0.005
    max_grad_norm: float | None = 10.0
    gradient_allreduce_bucket_mb: float = 25.0
    lr_scheduler: str = "constant"
    min_lr: float = 0.0
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_workers: int = 4
    num_steps: int = 10000
    log_interval: int = 10
    save_interval: int = 1000
    enable_wandb: bool = False
    wandb_mode: str = "offline"
    wandb_dir: str | None = "/luhongchao/wy/wandb"
    wandb_project: str = "lingbot-va-rl"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_log_interval: int | None = None
    wandb_hist_interval: int = 0
    wandb_fail_open: bool = True
    wandb_init_timeout: int = 30
    window_size: int = 64
    seed: int = 42
    initialize_from: str | None = None
    freeze_inherited_critic_steps: int = 0
    inherited_critic_lr_scale: float = 1.0
    language_critic_lr_scale: float = 1.0
    resume_from: str | None = None
    dataset_config: str | None = None

    def __post_init__(self) -> None:
        feature_layers = _parse_layer_spec(
            self.feature_layers,
            key="feature_layers",
            allow_final_norm=True,
        )
        object.__setattr__(self, "feature_layers", feature_layers)

        if self.critic_feature_layers is None:
            critic_feature_layers = (
                (0,)
                if self.critic_type
                in {
                    "twin_query_transformer_v1",
                    "twin_language_query_transformer_v1",
                    "twin_flow_scalar_transformer_v1",
                }
                and feature_layers == (-1,)
                else feature_layers
            )
        else:
            critic_feature_layers = _parse_layer_spec(
                self.critic_feature_layers,
                key="critic_feature_layers",
                allow_final_norm=False,
            )
        object.__setattr__(
            self,
            "critic_feature_layers",
            tuple(critic_feature_layers),
        )

        layer_context_critic_types = {
            "twin_query_transformer_v1",
            "twin_language_query_transformer_v1",
            "twin_flow_scalar_transformer_v1",
        }
        if self.critic_type in layer_context_critic_types:
            if any(layer < 0 for layer in self.critic_feature_layers):
                raise ValueError(
                    "critic_feature_layers must be non-negative DiT block indices"
                )
        elif self.critic_type != "twin_mlp_v1" and len(self.feature_layers) != 1:
            raise ValueError(
                "non-query, non-MLP critics require exactly one feature layer"
            )
        if self.feature_aggregation != "single":
            raise ValueError("feature_aggregation must be 'single'")
        if self.critic_num_query_tokens <= 0:
            raise ValueError("critic_num_query_tokens must be positive")
        if self.critic_transformer_hidden_dim <= 0:
            raise ValueError("critic_transformer_hidden_dim must be positive")
        if self.critic_text_embed_dim <= 0:
            raise ValueError("critic_text_embed_dim must be positive")
        if (
            self.critic_type == "twin_language_query_transformer_v1"
            and not self.phase3_language_sidecar_catalog
        ):
            raise ValueError(
                "language-conditioned critic requires "
                "phase3_language_sidecar_catalog"
            )
        if self.initialize_from and self.resume_from:
            raise ValueError(
                "initialize_from and resume_from cannot both be set"
            )
        if self.initialize_from and self.critic_type not in {
            "twin_query_transformer_v1",
            "twin_language_query_transformer_v1",
        }:
            raise ValueError(
                "initialize_from requires a query-transformer critic: "
                "critic_type must be 'twin_query_transformer_v1' or "
                "'twin_language_query_transformer_v1'"
            )
        if self.freeze_inherited_critic_steps < 0:
            raise ValueError(
                "freeze_inherited_critic_steps must be non-negative"
            )
        if self.inherited_critic_lr_scale <= 0:
            raise ValueError("inherited_critic_lr_scale must be positive")
        if self.language_critic_lr_scale <= 0:
            raise ValueError("language_critic_lr_scale must be positive")
        if (
            self.critic_transformer_num_layers is not None
            and self.critic_transformer_num_layers <= 0
        ):
            raise ValueError("critic_transformer_num_layers must be positive")
        if self.critic_transformer_num_heads <= 0:
            raise ValueError("critic_transformer_num_heads must be positive")
        if (
            self.critic_transformer_hidden_dim
            % self.critic_transformer_num_heads
            != 0
        ):
            raise ValueError(
                "critic_transformer_hidden_dim must be divisible by "
                "critic_transformer_num_heads"
            )
        if (
            self.critic_transformer_ffn_dim is not None
            and self.critic_transformer_ffn_dim <= 0
        ):
            raise ValueError("critic_transformer_ffn_dim must be positive")
        if not 0.0 <= self.critic_transformer_dropout < 1.0:
            raise ValueError("critic_transformer_dropout must be in [0, 1)")
        if self.num_q_heads <= 0:
            raise ValueError("num_q_heads must be positive")
        if self.q_ensemble_reduce not in {"min", "mean"}:
            raise ValueError("q_ensemble_reduce must be 'min' or 'mean'")
        if self.mc_return_normalization != "dataset_mc_return":
            raise ValueError(
                "mc_return_normalization must be 'dataset_mc_return'"
            )
        if self.mc_return_clip is not None:
            if (
                not isinstance(self.mc_return_clip, (list, tuple))
                or len(self.mc_return_clip) != 2
            ):
                raise TypeError("mc_return_clip must be null or two floats")
            clip_min, clip_max = (
                float(self.mc_return_clip[0]),
                float(self.mc_return_clip[1]),
            )
            if clip_min >= clip_max:
                raise ValueError("mc_return_clip must satisfy min < max")
            object.__setattr__(self, "mc_return_clip", (clip_min, clip_max))
        if self.scalar_flow_loss not in {"huber", "mse"}:
            raise ValueError("scalar_flow_loss must be 'huber' or 'mse'")
        if self.scalar_flow_huber_delta <= 0.0:
            raise ValueError("scalar_flow_huber_delta must be positive")
        if (
            not isinstance(self.scalar_flow_timestep_range, (list, tuple))
            or len(self.scalar_flow_timestep_range) != 2
        ):
            raise TypeError("scalar_flow_timestep_range must contain two floats")
        flow_start, flow_end = (
            float(self.scalar_flow_timestep_range[0]),
            float(self.scalar_flow_timestep_range[1]),
        )
        if not (0.0 <= flow_start <= flow_end <= 1.0):
            raise ValueError(
                "scalar_flow_timestep_range must satisfy 0 <= start <= end <= 1"
            )
        object.__setattr__(
            self,
            "scalar_flow_timestep_range",
            (flow_start, flow_end),
        )
        if not 0.0 <= self.scalar_flow_eval_timestep <= 1.0:
            raise ValueError("scalar_flow_eval_timestep must be in [0, 1]")
        if self.scalar_flow_eval_noise not in {"zero", "normal"}:
            raise ValueError("scalar_flow_eval_noise must be 'zero' or 'normal'")

        legacy_objective = _normalize_rl_objective(
            self.algorithm,
            key="algorithm",
        )
        rl_objective = (
            legacy_objective
            if self.rl_objective is None
            else _normalize_rl_objective(self.rl_objective, key="rl_objective")
        )
        if self.rl_objective is not None and rl_objective != legacy_objective:
            raise ValueError(
                "algorithm and rl_objective disagree after normalization: "
                f"{self.algorithm!r} -> {legacy_objective!r}, "
                f"rl_objective={self.rl_objective!r} -> {rl_objective!r}"
            )
        object.__setattr__(self, "rl_objective", rl_objective)
        object.__setattr__(
            self,
            "algorithm",
            _OBJECTIVE_TO_LEGACY_ALGORITHM[rl_objective],
        )

        if rl_objective == "mc_flow" and self.critic_type not in _FLOW_SCALAR_CRITICS:
            raise ValueError(
                "rl_objective='mc_flow' requires a flow-scalar critic_type"
            )
        if rl_objective in {"iql", "mc_return"} and self.critic_type not in _SCALAR_QV_CRITICS:
            raise ValueError(
                "rl_objective='iql' and 'mc_return' require scalar Q/V "
                f"critic_type, got {self.critic_type!r}"
            )
        if self.training_distribution not in {
            "clean_dataset",
            "predicted_video_conditioned_action",
            "predicted_video_conditioned_real_action",
        }:
            raise ValueError(
                "training_distribution must be 'clean_dataset', "
                "'predicted_video_conditioned_action', or "
                "'predicted_video_conditioned_real_action'"
            )
        if self.reward_source not in {
            "sparse_success",
            "jepa_delta_distance",
            "negative_predicted_actual_jepa_distance",
        }:
            raise ValueError(
                "reward_source must be 'sparse_success', "
                "'jepa_delta_distance', or "
                "'negative_predicted_actual_jepa_distance'"
            )
        if (
            self.training_distribution
            in {
                "predicted_video_conditioned_action",
                "predicted_video_conditioned_real_action",
            }
            and self.reward_source
            != "negative_predicted_actual_jepa_distance"
        ):
            raise ValueError(
                "Phase 3 predicted-video training requires "
                "reward_source='negative_predicted_actual_jepa_distance'"
            )
        if self.reward_source == "negative_predicted_actual_jepa_distance":
            if (
                self.training_distribution
                not in {
                    "predicted_video_conditioned_action",
                    "predicted_video_conditioned_real_action",
                }
            ):
                raise ValueError(
                    "negative_predicted_actual_jepa_distance requires "
                    "predicted-video-conditioned training_distribution"
                )
            if self.rl_objective not in {"iql", "mc_flow", "mc_return"}:
                raise ValueError(
                    "Phase 3 currently requires rl_objective='iql', "
                    "'mc_flow', or 'mc_return'"
                )
        reward_terms = _normalize_reward_terms(
            self.reward_terms,
            reward_source=self.reward_source,
            include_sparse_success_reward=self.include_sparse_success_reward,
            jepa_reward_weight=self.jepa_reward_weight,
            success_reward_weight=self.success_reward_weight,
        )
        object.__setattr__(self, "reward_terms", reward_terms)
        final_reward_normalization = self.final_reward_normalization
        if final_reward_normalization in {False, None}:
            final_reward_normalization = "none"
        elif final_reward_normalization is True:
            final_reward_normalization = "scale_neg1_pos1"
        if final_reward_normalization not in {"none", "scale_neg1_pos1"}:
            raise ValueError(
                "final_reward_normalization must be 'none', "
                "'scale_neg1_pos1', false, true, or null"
            )
        object.__setattr__(
            self,
            "final_reward_normalization",
            str(final_reward_normalization),
        )
        if not 0.0 <= self.phase3_q_feature_timestep <= 1.0:
            raise ValueError("phase3_q_feature_timestep must be in [0, 1]")
        if not 0.0 <= self.phase3_v_feature_timestep <= 1.0:
            raise ValueError("phase3_v_feature_timestep must be in [0, 1]")
        for key in (
            "value_video_feature_timestep_range",
            "value_action_feature_timestep_range",
            "q_video_feature_timestep_range",
            "q_action_feature_timestep_range",
        ):
            value = getattr(self, key)
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise TypeError(f"{key} must contain two floats")
            start, end = float(value[0]), float(value[1])
            if not (0.0 <= start <= end <= 1.0):
                raise ValueError(f"{key} must satisfy 0 <= start <= end <= 1")
            object.__setattr__(self, key, (start, end))
        for key, value in (
            ("phase3_video_num_inference_steps", self.phase3_video_num_inference_steps),
            ("phase3_action_num_inference_steps", self.phase3_action_num_inference_steps),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{key} must be positive when set")
        if self.infer_latent_chunk_size <= 0:
            raise ValueError("infer_latent_chunk_size must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")
        if self.save_interval <= 0:
            raise ValueError("save_interval must be positive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be 'online', 'offline', or 'disabled'")
        if not isinstance(self.wandb_tags, (list, tuple)):
            raise TypeError("wandb_tags must be a list or tuple of strings")
        object.__setattr__(
            self,
            "wandb_tags",
            tuple(str(tag) for tag in self.wandb_tags),
        )
        if self.wandb_log_interval is not None and self.wandb_log_interval <= 0:
            raise ValueError("wandb_log_interval must be positive when set")
        if self.wandb_hist_interval < 0:
            raise ValueError("wandb_hist_interval must be non-negative")
        if self.wandb_init_timeout <= 0:
            raise ValueError("wandb_init_timeout must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 < self.expectile < 1.0:
            raise ValueError("expectile must be in (0, 1)")
        if self.q_loss_type not in {"huber", "mse"}:
            raise ValueError("q_loss_type must be 'huber' or 'mse'")
        if self.q_huber_delta <= 0.0:
            raise ValueError("q_huber_delta must be positive")
        if self.max_grad_norm is not None:
            max_grad_norm = float(self.max_grad_norm)
            object.__setattr__(
                self,
                "max_grad_norm",
                None if max_grad_norm <= 0.0 else max_grad_norm,
            )
        if self.gradient_allreduce_bucket_mb <= 0.0:
            raise ValueError("gradient_allreduce_bucket_mb must be positive")
        if self.lr_scheduler not in {"constant", "cosine"}:
            raise ValueError("lr_scheduler must be 'constant' or 'cosine'")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.min_lr < 0.0:
            raise ValueError("min_lr must be non-negative")
        if self.min_lr > self.learning_rate:
            raise ValueError("min_lr must be <= learning_rate")
        has_canonical_cache = bool(self.phase3_precomputed_cache_dir)
        has_federated_cache = bool(self.phase3_federated_cache_catalog)
        if (
            self.phase3_require_precomputed_cache
            and self.dataset_config is not None
            and self.phase3_cache_format
            in {"canonical_episode", "federated_canonical_episode"}
        ):
            if not self.phase3_episode_success_overlay:
                raise ValueError(
                    "precomputed episode-cache training requires "
                    "phase3_episode_success_overlay; use an explicit identity "
                    "overlay to preserve raw labels"
                )
            overlay_path = Path(self.phase3_episode_success_overlay)
            if not overlay_path.is_file():
                raise FileNotFoundError(
                    f"Missing phase3_episode_success_overlay: {overlay_path}"
                )
            overlay = json.loads(overlay_path.read_text())
            if (
                not isinstance(overlay, dict)
                or overlay.get("schema_version") != 1
                or overlay.get("format") != "phase3_episode_success_overlay_v1"
                or overlay.get("mode") not in {"identity", "replace"}
            ):
                raise ValueError("Invalid phase3_episode_success_overlay format")
        if has_canonical_cache and has_federated_cache:
            raise ValueError(
                "Configure either phase3_precomputed_cache_dir or "
                "phase3_federated_cache_catalog, not both"
            )
        if self.phase3_federated_cache_artifacts_manifest and not has_federated_cache:
            raise ValueError(
                "phase3_federated_cache_artifacts_manifest requires "
                "phase3_federated_cache_catalog"
            )
        if self.phase3_use_packed_cache:
            if not self.phase3_packed_cache_dir:
                raise ValueError(
                    "phase3_use_packed_cache requires phase3_packed_cache_dir"
                )
            if has_federated_cache:
                raise ValueError(
                    "Packed and federated canonical cache routes cannot be "
                    "enabled together"
                )
        if self.phase3_cache_format not in {
            "expanded_transition",
            "canonical_episode",
            "federated_canonical_episode",
        }:
            raise ValueError(
                "phase3_cache_format must be 'expanded_transition', "
                "'canonical_episode', or 'federated_canonical_episode'"
            )
        if self.phase3_cache_format == "federated_canonical_episode":
            if not has_federated_cache:
                raise ValueError(
                    "federated_canonical_episode requires "
                    "phase3_federated_cache_catalog"
                )
            if not self.phase3_require_precomputed_cache:
                raise ValueError(
                    "federated_canonical_episode requires "
                    "phase3_require_precomputed_cache=true"
                )
        elif has_federated_cache:
            raise ValueError(
                "phase3_federated_cache_catalog requires "
                "phase3_cache_format='federated_canonical_episode'"
            )
        if (
            self.phase3_require_precomputed_cache
            and not (
                has_canonical_cache
                or has_federated_cache
                or self.phase3_use_packed_cache
            )
        ):
            raise ValueError(
                "phase3_require_precomputed_cache requires a canonical, "
                "federated canonical, or packed cache route"
            )
        if self.phase3_packed_cache_shard_size <= 0:
            raise ValueError("phase3_packed_cache_shard_size must be positive")
        if self.phase3_cache_batching not in {"random", "shard_local"}:
            raise ValueError(
                "phase3_cache_batching must be 'random' or 'shard_local'"
            )
        if self.phase3_episodes_per_batch <= 0:
            raise ValueError("phase3_episodes_per_batch must be positive")
        if self.phase3_global_balanced_batching:
            if self.phase3_cache_format not in {
                "canonical_episode",
                "federated_canonical_episode",
            }:
                raise ValueError(
                    "phase3_global_balanced_batching requires a canonical "
                    "episode cache"
                )
            if not self.phase3_drop_last:
                raise ValueError(
                    "phase3_global_balanced_batching requires "
                    "phase3_drop_last=true"
                )
            if (
                self.phase3_preserve_episode_group_order
                and self.phase3_federated_source_balanced_sampling
            ):
                raise ValueError(
                    "phase3_preserve_episode_group_order=true is "
                    "incompatible with federated source-balanced sampling"
                )
        if (
            self.phase3_federated_source_balanced_sampling
            and not self.phase3_global_balanced_batching
        ):
            raise ValueError(
                "phase3_federated_source_balanced_sampling requires "
                "phase3_global_balanced_batching=true"
            )
        if (
            self.phase3_exact_resume_cursor
            and not self.phase3_global_balanced_batching
        ):
            raise ValueError(
                "phase3_exact_resume_cursor requires "
                "phase3_global_balanced_batching=true"
            )
        if self.phase3_episode_cache_lru_size <= 0:
            raise ValueError("phase3_episode_cache_lru_size must be positive")
        if (
            self.phase3_prefetch_factor is not None
            and self.phase3_prefetch_factor <= 0
        ):
            raise ValueError("phase3_prefetch_factor must be positive when set")
        if self.phase3_episode_prefetch_workers <= 0:
            raise ValueError("phase3_episode_prefetch_workers must be positive")
        if self.phase3_episode_prefetch_lookahead_batches <= 0:
            raise ValueError(
                "phase3_episode_prefetch_lookahead_batches must be positive"
            )
        if self.phase3_batch_prefetch_depth <= 0:
            raise ValueError("phase3_batch_prefetch_depth must be positive")
        if self.phase3_episode_cache_max_gib <= 0:
            raise ValueError("phase3_episode_cache_max_gib must be positive")
        if self.phase3_cuda_prefetch_depth <= 0:
            raise ValueError("phase3_cuda_prefetch_depth must be positive")
        if self.phase3_loader_mode not in {"episode", "transition_range"}:
            raise ValueError(
                "phase3_loader_mode must be 'episode' or 'transition_range'"
            )
        if self.phase3_transition_prefetch_lookahead <= 0:
            raise ValueError(
                "phase3_transition_prefetch_lookahead must be positive"
            )
        if self.phase3_transition_prefetch_workers <= 0:
            raise ValueError(
                "phase3_transition_prefetch_workers must be positive"
            )
        if self.phase3_transition_batch_workers <= 0:
            raise ValueError("phase3_transition_batch_workers must be positive")
        if self.phase3_transition_max_active_episodes <= 0:
            raise ValueError(
                "phase3_transition_max_active_episodes must be positive"
            )
        if self.phase3_transition_max_inflight_gib <= 0:
            raise ValueError(
                "phase3_transition_max_inflight_gib must be positive"
            )
        if self.phase3_cache_residency not in {
            "on_demand",
            "bounded_group_prefetching",
            "full",
        }:
            raise ValueError(
                "phase3_cache_residency must be 'on_demand', "
                "'bounded_group_prefetching', or 'full'"
            )
        if self.phase3_cache_group_size <= 0:
            raise ValueError("phase3_cache_group_size must be positive")
        if self.phase3_cache_group_prefetch_depth <= 0:
            raise ValueError(
                "phase3_cache_group_prefetch_depth must be positive"
            )
        if self.phase3_cache_max_resident_gib <= 0:
            raise ValueError("phase3_cache_max_resident_gib must be positive")
        if self.phase3_cache_load_workers <= 0:
            raise ValueError("phase3_cache_load_workers must be positive")
        if (
            self.phase3_cache_residency == "bounded_group_prefetching"
            and not self.phase3_preserve_episode_group_order
        ):
            raise ValueError(
                "bounded_group_prefetching requires "
                "phase3_preserve_episode_group_order=true"
            )
        if self.phase3_precompute_mode not in {"packed_train", "server_prefix"}:
            raise ValueError(
                "phase3_precompute_mode must be 'packed_train' or "
                "'server_prefix'"
            )
        if self.phase3_video_prediction_context not in {
            "previous_chunk",
            "episode_prefix",
            "server_prefix",
        }:
            raise ValueError(
                "phase3_video_prediction_context must be 'previous_chunk', "
                "'episode_prefix', or 'server_prefix'"
            )
        if self.phase3_exclude_last_n_chunks < 0:
            raise ValueError("phase3_exclude_last_n_chunks must be non-negative")
        if self.goal_jepa_progress_lambda < 0:
            raise ValueError("goal_jepa_progress_lambda must be non-negative")
        if self.goal_jepa_progress_beta < 0:
            raise ValueError("goal_jepa_progress_beta must be non-negative")
        if self.rl_objective in {"mc_flow", "mc_return"} and (
            self.reward_source == "negative_predicted_actual_jepa_distance"
            and not (
                self.phase3_precomputed_cache_dir
                or self.phase3_federated_cache_catalog
                or self.phase3_use_packed_cache
            )
        ):
            raise ValueError(
                "MC-return phase-3 JEPA reward training requires "
                "a canonical, federated canonical, or packed cache so "
                "discounted MC returns can be built from cached rewards"
            )

    @property
    def feature_layer(self) -> int:
        if len(self.feature_layers) != 1:
            raise ValueError(
                "feature_layer is only defined when exactly one legacy "
                "feature layer is configured"
            )
        return self.feature_layers[0]

    @property
    def critic_transformer_layer_count(self) -> int:
        return len(self.critic_feature_layers)

    @property
    def feature_normalization(self) -> str:
        if self.critic_type in {
            "twin_query_transformer_v1",
            "twin_language_query_transformer_v1",
            "twin_flow_scalar_transformer_v1",
        }:
            return "raw_block_output_v1"
        if len(self.feature_layers) == 1 and self.feature_layers[0] == -1:
            return "final_adaptive_norm_v1"
        return "raw_block_output_v1"

    @classmethod
    def from_json(cls, path: str | Path) -> "CriticTrainingConfig":
        path = Path(path)
        with path.open() as handle:
            values: dict[str, Any] = json.load(handle)
        if not isinstance(values, dict):
            raise TypeError("critic config must contain a JSON object")
        values = _resolve_dataset_config(path, values)
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values).difference(known))
        if unknown:
            raise ValueError(f"Unknown critic config keys: {unknown}")

        migrated = dict(values)
        if "rl_objective" not in migrated:
            migrated["rl_objective"] = _normalize_rl_objective(
                str(migrated.get("algorithm", "mc")),
                key="algorithm",
            )
        if "algorithm" not in migrated:
            migrated["algorithm"] = _OBJECTIVE_TO_LEGACY_ALGORITHM[
                _normalize_rl_objective(
                    str(migrated["rl_objective"]),
                    key="rl_objective",
                )
            ]
        if _is_checkpoint_config(path) and migrated != values:
            _backup_and_write_migrated_config(path, values, migrated)
        values = migrated

        for key in (
            "value_video_feature_timestep_range",
            "value_action_feature_timestep_range",
            "q_video_feature_timestep_range",
            "q_action_feature_timestep_range",
            "mc_return_clip",
            "scalar_flow_timestep_range",
        ):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
