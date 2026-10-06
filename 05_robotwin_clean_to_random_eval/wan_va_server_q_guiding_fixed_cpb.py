# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

try:
    from torch.nn.attention.flex_attention import (
        flex_attention as raw_flex_attention,
    )
except ModuleNotFoundError:
    raw_flex_attention = None

from wan_va.rl.guidance import (
    build_action_guidance_mask,
    build_clean_feature_input,
    build_query_q_context,
    denoising_time_scaling,
    load_q_guidance_artifact,
)
from wan_va.rl.progress_budget import (
    beta_envelope,
    blended_task_budget,
    calibrated_progress,
    distance_confidence,
    update_ema,
)
try:
    from progress_budget import direct_progress_task_budget
except ImportError:
    from wan_va.rl.progress_budget import direct_progress_task_budget
from wan_va.wan_va_server import (
    VA_CONFIGS,
    VA_Server,
    data_seq_to_patch,
    init_distributed,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class QGuidedVA_Server(VA_Server):
    """VA server variant that applies Q-gradient guidance to action sampling."""

    def __init__(
        self,
        job_config,
        q_checkpoint: str,
        q_guidance_scale: float,
        q_guidance_beta: float = 2.0,
        q_guidance_weight_scheduler: str = "auto",
        q_guidance_start_step: int = 0,
        q_guidance_end_step: int = -1,
        q_guidance_interval: int = 1,
        q_objective: str = "min",
        q_grad_clip: float = 0.0,
        q_grad_normalize: bool = False,
        q_history_chunks: int = 10,
        q_guidance_activation_checkpointing: bool = False,
        robotwin_eval: bool = False,
        q_log_dir: str | None = None,
        q_log_denoising_steps: bool = False,
        q_control_path: str | None = None,
        q_progress_guidance: bool = False,
        q_progress_goal_embedding: str | None = None,
        q_progress_jepa_checkpoint: str = "/luhongchao/shared/weights/vjepa2.1/vjepa2_1_vitG_384.pt",
        q_progress_jepa_config: str | None = None,
        q_progress_jepa_repo: str | None = None,
        q_progress_jepa_checkpoint_key: str = "target_encoder",
        q_progress_camera_key: str | None = None,
        q_progress_clip_frames: int = 1,
        q_progress_beta_max: float | None = None,
        q_progress_beta_min: float = 0.0,
        q_progress_kappa: float = 1.0,
        q_progress_rho: float = 1.0,
        q_progress_eps: float = 1e-6,
        q_progress_scheduler: str = "progress_decay",
        q_progress_exp_beta: float = 10.0,
        q_cpb_success_reference: float | None = None,
        q_cpb_confidence: float = 0.0,
        q_cpb_fixed_budget: float | None = None,
        q_cpb_budget_min: float | None = None,
        q_cpb_budget_max: float | None = None,
        q_cpb_ema_alpha: float = 0.25,
        q_cpb_sigma_peak: float = 0.25,
        q_cpb_sigma_concentration: float = 4.0,
        q_cpb_distance_low: float | None = None,
        q_cpb_distance_high: float | None = None,
        q_cpb_ood_margin: float = 0.0,
        q_cpb_direct_progress_budget: bool = False,
    ) -> None:
        super().__init__(job_config, robotwin_eval=robotwin_eval)
        # Guidance differentiates Q with respect to actions, never Wan/Q weights.
        self.transformer.eval().requires_grad_(False)
        self.q_checkpoint = str(q_checkpoint)
        self.q_guidance_scale = float(q_guidance_scale)
        self.q_guidance_beta = float(q_guidance_beta)
        if q_guidance_weight_scheduler not in {
            "auto",
            "fixed",
            "denoising_time",
            "progress_decay",
            "exp_progress",
            "cpb",
            "fixed_cpb",
        }:
            raise ValueError(
                "q_guidance_weight_scheduler must be one of "
                "'auto', 'fixed', 'denoising_time', 'progress_decay', "
                f"'exp_progress', 'cpb', or 'fixed_cpb', got {q_guidance_weight_scheduler!r}"
            )
        self.q_guidance_weight_scheduler = q_guidance_weight_scheduler
        self.q_guidance_start_step = int(q_guidance_start_step)
        self.q_guidance_end_step = int(q_guidance_end_step)
        if q_guidance_interval <= 0:
            raise ValueError("q_guidance_interval must be positive")
        self.q_guidance_interval = int(q_guidance_interval)
        self.q_objective = q_objective
        self.q_grad_clip = float(q_grad_clip)
        self.q_grad_normalize = bool(q_grad_normalize)
        if q_history_chunks <= 0:
            raise ValueError("q_history_chunks must be positive")
        self.q_history_chunks = int(q_history_chunks)
        self.q_guidance_activation_checkpointing = bool(
            q_guidance_activation_checkpointing
        )
        self.q_log_dir = Path(q_log_dir) if q_log_dir else None
        self.q_log_denoising_steps = bool(q_log_denoising_steps)
        self.q_summary_log_path: Path | None = None
        self.q_step_log_path: Path | None = None
        self.q_control_path = Path(q_control_path) if q_control_path else None
        self._q_control_mtime_ns: int | None = None
        self.q_progress_guidance = bool(q_progress_guidance)
        self.q_progress_goal_embedding_path = (
            Path(q_progress_goal_embedding) if q_progress_goal_embedding else None
        )
        self.q_progress_jepa_checkpoint = q_progress_jepa_checkpoint
        self.q_progress_jepa_config = q_progress_jepa_config
        self.q_progress_jepa_repo = q_progress_jepa_repo
        self.q_progress_jepa_checkpoint_key = q_progress_jepa_checkpoint_key
        self.q_progress_camera_key = q_progress_camera_key
        if q_progress_clip_frames <= 0:
            raise ValueError("q_progress_clip_frames must be positive")
        self.q_progress_clip_frames = int(q_progress_clip_frames)
        self.q_progress_beta_max = (
            float(q_progress_beta_max)
            if q_progress_beta_max is not None
            else float(q_guidance_scale)
        )
        self.q_progress_beta_min = float(q_progress_beta_min)
        self.q_progress_kappa = float(q_progress_kappa)
        self.q_progress_rho = float(q_progress_rho)
        self.q_progress_eps = float(q_progress_eps)
        if q_progress_scheduler not in {"progress_decay", "exp_progress", "cpb"}:
            raise ValueError(
                "q_progress_scheduler must be 'progress_decay', "
                f"'exp_progress', or 'cpb', got {q_progress_scheduler!r}"
            )
        if q_progress_exp_beta <= 0:
            raise ValueError("q_progress_exp_beta must be positive")
        self.q_progress_scheduler = q_progress_scheduler
        self.q_progress_exp_beta = float(q_progress_exp_beta)
        self.q_cpb_success_reference = (
            None if q_cpb_success_reference is None else float(q_cpb_success_reference)
        )
        self.q_cpb_confidence = float(q_cpb_confidence)
        self.q_cpb_fixed_budget = (
            None if q_cpb_fixed_budget is None else float(q_cpb_fixed_budget)
        )
        self.q_cpb_budget_min = (
            None if q_cpb_budget_min is None else float(q_cpb_budget_min)
        )
        self.q_cpb_budget_max = (
            None if q_cpb_budget_max is None else float(q_cpb_budget_max)
        )
        self.q_cpb_ema_alpha = float(q_cpb_ema_alpha)
        self.q_cpb_sigma_peak = float(q_cpb_sigma_peak)
        self.q_cpb_sigma_concentration = float(q_cpb_sigma_concentration)
        self.q_cpb_distance_low = (
            None if q_cpb_distance_low is None else float(q_cpb_distance_low)
        )
        self.q_cpb_distance_high = (
            None if q_cpb_distance_high is None else float(q_cpb_distance_high)
        )
        self.q_cpb_ood_margin = float(q_cpb_ood_margin)
        self.q_cpb_direct_progress_budget = bool(q_cpb_direct_progress_budget)
        cpb_enabled = q_guidance_weight_scheduler in {"cpb", "fixed_cpb"} or (
            q_guidance_weight_scheduler == "auto" and q_progress_scheduler == "cpb"
        )
        if cpb_enabled:
            required = {
                "q_cpb_fixed_budget": self.q_cpb_fixed_budget,
            }
            if q_guidance_weight_scheduler != "fixed_cpb":
                required.update(
                    {
                        "q_cpb_success_reference": self.q_cpb_success_reference,
                        "q_cpb_budget_min": self.q_cpb_budget_min,
                        "q_cpb_budget_max": self.q_cpb_budget_max,
                    }
                )
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"CPB requires explicit values for {missing}")
            if self.q_cpb_fixed_budget is not None and self.q_cpb_fixed_budget < 0.0:
                raise ValueError("q_cpb_fixed_budget must be non-negative")
            if not 0.0 <= self.q_cpb_confidence <= 1.0:
                raise ValueError("q_cpb_confidence must be in [0, 1]")
            if not 0.0 < self.q_cpb_ema_alpha <= 1.0:
                raise ValueError("q_cpb_ema_alpha must be in (0, 1]")
            if (self.q_cpb_distance_low is None) != (self.q_cpb_distance_high is None):
                raise ValueError("CPB distance low/high bounds must be provided together")
        self._progress_goal_embedding: torch.Tensor | None = None
        self._progress_init_embedding: torch.Tensor | None = None
        self._progress_current_embedding: torch.Tensor | None = None
        self._progress_jepa_encoder: torch.nn.Module | None = None
        self._progress_jepa_is_multiseq = False
        self._progress_value = torch.tensor(0.0, device=self.device)
        self._cpb_progress_value = torch.tensor(0.0, device=self.device)
        self._cpb_confidence_value = torch.tensor(0.0, device=self.device)
        self._progress_d_init: torch.Tensor | None = None
        self._progress_d_curr: torch.Tensor | None = None
        self._progress_d_ema: torch.Tensor | None = None
        self.q_artifact = load_q_guidance_artifact(q_checkpoint, self.device)
        self.q_artifact.adapter.eval().requires_grad_(False)
        self._q_previous_latents: torch.Tensor | None = None
        self._q_previous_actions: torch.Tensor | None = None
        self._q_previous_latent_mask: torch.Tensor | None = None
        self._q_previous_action_mask: torch.Tensor | None = None
        self._last_predicted_actions: torch.Tensor | None = None
        self._last_predicted_latents: torch.Tensor | None = None
        self._init_progress_guidance()
        self._validate_q_artifact()
        self._log_loaded_q_artifact()
        self._init_q_control_file()

    def _reset(self, prompt=None):
        super()._reset(prompt=prompt)
        self._q_previous_latents = None
        self._q_previous_actions = None
        self._q_previous_latent_mask = None
        self._q_previous_action_mask = None
        self._last_predicted_actions = None
        self._last_predicted_latents = None
        self._progress_init_embedding = None
        self._progress_current_embedding = None
        self._progress_value = torch.tensor(0.0, device=self.device)
        self._cpb_progress_value = torch.tensor(0.0, device=self.device)
        self._cpb_confidence_value = torch.tensor(0.0, device=self.device)
        self._progress_d_init = None
        self._progress_d_curr = None
        self._progress_d_ema = None
        self._init_state_normalization()
        self._init_q_logs()

    def _init_q_logs(self) -> None:
        log_dir = self.q_log_dir or Path(self.exp_save_root) / "q_guidance"
        log_dir.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(self.job_config, "local_rank", 0))
        self.q_summary_log_path = log_dir / f"q_guidance_summary_rank{rank}.jsonl"
        self.q_step_log_path = log_dir / f"q_guidance_steps_rank{rank}.jsonl"

    def _init_progress_guidance(self) -> None:
        if not self.q_progress_guidance:
            return
        if self.q_progress_goal_embedding_path is None:
            raise ValueError(
                "--q_progress_goal_embedding is required when "
                "--q_progress_guidance is enabled"
            )
        data = torch.load(
            self.q_progress_goal_embedding_path,
            map_location="cpu",
            weights_only=False,
        )
        if isinstance(data, dict):
            if "embedding" in data:
                data = data["embedding"]
            elif "goal_embedding" in data:
                data = data["goal_embedding"]
            else:
                tensor_values = [
                    value for value in data.values() if isinstance(value, torch.Tensor)
                ]
                if len(tensor_values) != 1:
                    raise ValueError(
                        "Progress goal embedding file must contain an "
                        "'embedding' tensor or a single tensor value"
                    )
                data = tensor_values[0]
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        self._progress_goal_embedding = data.float().flatten().to(self.device)
        if self._progress_goal_embedding.numel() == 0:
            raise ValueError("Progress goal embedding is empty")
        logger.info(
            f"Loaded progress goal embedding {self.q_progress_goal_embedding_path} "
            f"with {self._progress_goal_embedding.numel()} values"
        )

    def _progress_encoder(self) -> torch.nn.Module:
        if self._progress_jepa_encoder is None:
            if self.q_progress_jepa_config:
                import yaml

                if not self.q_progress_jepa_repo:
                    raise ValueError(
                        "--q_progress_jepa_repo is required with "
                        "--q_progress_jepa_config"
                    )
                repo = str(Path(self.q_progress_jepa_repo).resolve())
                if repo not in sys.path:
                    sys.path.insert(0, repo)
                import app.vjepa_2_1.models.vision_transformer as video_vit
                from app.vjepa_2_1.wrappers import MultiSeqWrapper

                with Path(self.q_progress_jepa_config).open("r", encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle)
                data_cfg = cfg["data"]
                model_cfg = cfg["model"]
                meta_cfg = cfg.get("meta", {})
                encoder = video_vit.__dict__[model_cfg["model_name"]](
                    img_size=data_cfg.get("crop_size", 256),
                    patch_size=data_cfg["patch_size"],
                    num_frames=max(data_cfg["dataset_fpcs"]),
                    tubelet_size=data_cfg["tubelet_size"],
                    uniform_power=model_cfg.get("uniform_power", False),
                    use_sdpa=meta_cfg.get("use_sdpa", False),
                    use_rope=model_cfg.get("use_rope", False),
                    use_silu=model_cfg.get("use_silu", False),
                    wide_silu=model_cfg.get("wide_silu", True),
                    use_activation_checkpointing=False,
                    is_causal=model_cfg.get("is_causal", False),
                    init_type=model_cfg.get("init_type", "default"),
                    img_temporal_dim_size=model_cfg.get("img_temporal_dim_size"),
                    n_registers=model_cfg.get("n_registers", 0),
                    has_cls_first=model_cfg.get("has_cls_first", False),
                    interpolate_rope=model_cfg.get("interpolate_rope", False),
                    modality_embedding=model_cfg.get("modality_embedding", False),
                )
                self._progress_jepa_encoder = MultiSeqWrapper(encoder).to(self.device)
                self._progress_jepa_is_multiseq = True
                checkpoint = torch.load(
                    self.q_progress_jepa_checkpoint,
                    map_location="cpu",
                    weights_only=False,
                )
                state = checkpoint.get(self.q_progress_jepa_checkpoint_key)
                if not isinstance(state, dict):
                    raise KeyError(
                        f"{self.q_progress_jepa_checkpoint} has no "
                        f"{self.q_progress_jepa_checkpoint_key!r} state dict"
                    )
                state = {
                    key.removeprefix("module."): value
                    for key, value in state.items()
                }
                missing, unexpected = self._progress_jepa_encoder.load_state_dict(
                    state,
                    strict=False,
                )
                if missing or unexpected:
                    raise RuntimeError(
                        "Phase-3 progress encoder state mismatch: "
                        f"missing={missing[:8]} unexpected={unexpected[:8]}"
                    )
                self._progress_jepa_encoder = self._progress_jepa_encoder.to(
                    device=self.device,
                    dtype=torch.float32,
                )
                logger.info(
                    "Loaded exact phase-3 progress encoder "
                    f"checkpoint={self.q_progress_jepa_checkpoint} "
                    f"key={self.q_progress_jepa_checkpoint_key} "
                    f"config={self.q_progress_jepa_config}"
                )
            else:
                from preprocessing.extract_jepa_features import build_jepa_encoder

                self._progress_jepa_encoder = build_jepa_encoder(
                    self.q_progress_jepa_checkpoint,
                    self.device,
                )
            self._progress_jepa_encoder.eval().requires_grad_(False)
        return self._progress_jepa_encoder

    def _progress_clip_tensor(self, images: list[np.ndarray]) -> torch.Tensor:
        if not images:
            raise ValueError("Progress clip must contain at least one frame")
        images = images[-self.q_progress_clip_frames :]
        while len(images) < self.q_progress_clip_frames:
            images.insert(0, images[0])
        frames = []
        for image in images:
            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[-1] < 3:
                raise ValueError(f"Expected HWC RGB image, got shape {frame.shape}")
            frame = frame[..., :3]
            tensor = torch.from_numpy(np.array(frame, copy=True, order="C")).float()
            if frame.dtype == np.uint8 or (
                tensor.numel() and float(tensor.max()) > 1.5
            ):
                tensor = tensor / 255.0
            frames.append(tensor.clamp(0.0, 1.0))
        tensor = torch.stack(frames, dim=0).permute(3, 0, 1, 2).unsqueeze(0)
        height = max(16, (tensor.shape[-2] // 16) * 16)
        width = max(16, (tensor.shape[-1] // 16) * 16)
        if (height, width) != tuple(tensor.shape[-2:]):
            tensor = F.interpolate(
                tensor.flatten(0, 1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).view(1, 3, 1, height, width)
        if tensor.shape[2] == 1:
            tensor = torch.cat([tensor, tensor], dim=2)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
        return (tensor - mean) / std

    @torch.inference_mode()
    def _progress_embedding_from_observations(
        self,
        observations: list[dict[str, object]],
    ) -> torch.Tensor | None:
        encoder = self._progress_encoder()
        camera_vectors: list[torch.Tensor] = []
        camera_keys = (
            [self.q_progress_camera_key]
            if self.q_progress_camera_key
            else self.job_config.obs_cam_keys
        )
        for camera_key in camera_keys:
            images = [
                observation[camera_key]
                for observation in observations
                if camera_key in observation
            ]
            if not images:
                continue
            tensor = self._progress_clip_tensor(images).to(
                device=self.device,
                dtype=torch.float32,
            )
            features = (
                encoder([tensor], training_mode=False)[0]
                if self._progress_jepa_is_multiseq
                else encoder(tensor)
            )
            camera_vectors.append(features.float().mean(dim=(0, 1)))
        if not camera_vectors:
            return None
        return torch.stack(camera_vectors, dim=0).mean(dim=0).flatten()

    def _progress_distance(
        self,
        embedding: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        embedding = embedding.float().flatten()
        goal = goal.float().flatten()
        if embedding.numel() != goal.numel():
            raise ValueError(
                "Progress embedding dimension mismatch: "
                f"{embedding.numel()} != {goal.numel()}"
            )
        embedding = F.normalize(embedding, dim=0, eps=self.q_progress_eps)
        goal = F.normalize(goal, dim=0, eps=self.q_progress_eps)
        return torch.linalg.vector_norm(embedding - goal)

    def _update_progress_from_obs(
        self,
        request: dict[str, object],
        *,
        initialize: bool = False,
    ) -> None:
        if not self.q_progress_guidance or self._progress_goal_embedding is None:
            return
        raw_obs = request.get("obs")
        if raw_obs is None:
            return
        observations = raw_obs if isinstance(raw_obs, list) else [raw_obs]
        observations = [obs for obs in observations if isinstance(obs, dict)]
        if not observations:
            return
        selected = [observations[0]] if initialize else observations
        embedding = self._progress_embedding_from_observations(selected)
        if embedding is None:
            return
        self._progress_current_embedding = embedding
        if initialize or self._progress_init_embedding is None:
            self._progress_init_embedding = embedding
        if self._progress_init_embedding is None:
            return
        goal = self._progress_goal_embedding
        self._progress_d_init = self._progress_distance(
            self._progress_init_embedding,
            goal,
        )
        self._progress_d_curr = self._progress_distance(
            self._progress_current_embedding,
            goal,
        )
        self._progress_d_ema = update_ema(
            self._progress_d_ema,
            self._progress_d_curr,
            self.q_cpb_ema_alpha,
        )
        self._progress_value = (
            1.0
            - self._progress_d_curr
            / self._progress_d_init.clamp_min(self.q_progress_eps)
        ).clamp(0.0, 1.0)
        if self.q_cpb_success_reference is not None:
            self._cpb_progress_value = calibrated_progress(
                self._progress_d_init,
                self._progress_d_ema,
                self.q_cpb_success_reference,
                self.q_progress_eps,
            )

    def _guidance_step_selected(self, step_index: int) -> bool:
        if step_index < self.q_guidance_start_step:
            return False
        if self.q_guidance_end_step >= 0 and step_index > self.q_guidance_end_step:
            return False
        return not (
            (step_index - self.q_guidance_start_step) % self.q_guidance_interval
        )

    def _cpb_step_alpha(
        self,
        step_index: int,
        sigma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        timesteps = self.action_scheduler.timesteps
        selected = [
            index
            for index in range(max(0, len(timesteps) - 1))
            if self._guidance_step_selected(index)
        ]
        if step_index not in selected:
            return torch.zeros_like(sigma), torch.zeros_like(sigma)
        sigmas = torch.stack(
            [
                self._scheduler_sigma(timesteps[index], dtype=torch.float32)
                for index in selected
            ]
        ).to(device=sigma.device)
        envelope = beta_envelope(
            sigmas,
            peak=self.q_cpb_sigma_peak,
            concentration=self.q_cpb_sigma_concentration,
            eps=self.q_progress_eps,
        )
        position = selected.index(step_index)
        return envelope[position].to(sigma), envelope.max().to(sigma)

    def _progress_guidance_weight(
        self,
        sigma: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        scheduler = self.q_guidance_weight_scheduler
        if scheduler == "auto":
            scheduler = (
                self.q_progress_scheduler
                if self.q_progress_guidance
                else "denoising_time"
            )

        if scheduler == "fixed":
            weight = torch.full_like(
                sigma,
                float(self.q_guidance_scale),
                dtype=torch.float32,
            )
            return weight, {
                "flow_time": (1.0 - sigma).clamp(0.0, 1.0),
                "guidance_weight_scheduler": scheduler,
                "progress_guidance_enabled": False,
                "guidance_scale": float(self.q_guidance_scale),
            }

        if scheduler == "denoising_time":
            time = sigma.clamp(0.0, 1.0)
            step_scale = denoising_time_scaling(
                time,
                beta=self.q_guidance_beta,
            )
            return self.q_guidance_scale * step_scale, {
                "flow_time": 1.0 - time,
                "guidance_weight_scheduler": scheduler,
                "progress_guidance_enabled": False,
                "guidance_scale": float(self.q_guidance_scale),
                "legacy_step_scale": step_scale,
            }

        if scheduler == "fixed_cpb":
            if self.q_cpb_fixed_budget is None:
                raise RuntimeError("fixed_cpb requires --q_cpb_fixed_budget")
            denoising_alpha, envelope_peak = self._cpb_step_alpha(step_index, sigma)
            weight = torch.as_tensor(
                float(self.q_cpb_fixed_budget),
                device=sigma.device,
                dtype=torch.float32,
            ) * denoising_alpha
            return weight, {
                "flow_time": (1.0 - sigma).clamp(0.0, 1.0),
                "guidance_weight_scheduler": scheduler,
                "progress_guidance_enabled": False,
                "progress_scheduler": None,
                "cpb_task_budget": float(self.q_cpb_fixed_budget),
                "cpb_distance_budget": None,
                "cpb_fixed_budget": float(self.q_cpb_fixed_budget),
                "cpb_confidence": 0.0,
                "cpb_ood_confidence": None,
                "cpb_denoising_alpha": denoising_alpha,
                "cpb_envelope_peak_weight": envelope_peak,
                "cpb_sigma_peak": self.q_cpb_sigma_peak,
                "cpb_sigma_concentration": self.q_cpb_sigma_concentration,
                "progress_weight": weight,
                "fixed_budget_mode": True,
            }

        if scheduler in {"progress_decay", "exp_progress", "cpb"} and not self.q_progress_guidance:
            raise RuntimeError(
                f"q_guidance_weight_scheduler={scheduler!r} requires "
                "--q_progress_guidance"
            )

        flow_time = (1.0 - sigma).clamp(0.0, 1.0)
        progress = self._progress_value.to(device=sigma.device, dtype=torch.float32)
        distance = (
            self._progress_d_curr.to(device=sigma.device, dtype=torch.float32)
            if self._progress_d_curr is not None
            else torch.zeros((), device=sigma.device, dtype=torch.float32)
        )
        if scheduler == "cpb":
            smoothed_distance = (
                self._progress_d_ema.to(device=sigma.device, dtype=torch.float32)
                if self._progress_d_ema is not None
                else distance
            )
            if self.q_cpb_direct_progress_budget:
                distance_budget = direct_progress_task_budget(
                    self._cpb_progress_value.to(sigma),
                    minimum_budget=float(self.q_cpb_budget_min),
                    maximum_budget=float(self.q_cpb_budget_max),
                    kappa=self.q_progress_kappa,
                )
                task_budget = distance_budget
                ood_confidence = None
                effective_confidence = None
                self._cpb_confidence_value = torch.ones_like(task_budget)
            else:
                ood_confidence = torch.ones_like(smoothed_distance)
                if self.q_cpb_distance_low is not None and self.q_cpb_distance_high is not None:
                    ood_confidence = distance_confidence(
                        smoothed_distance,
                        self.q_cpb_distance_low,
                        self.q_cpb_distance_high,
                        self.q_cpb_ood_margin,
                        self.q_progress_eps,
                    )
                effective_confidence = ood_confidence * self.q_cpb_confidence
                self._cpb_confidence_value = effective_confidence
                task_budget, distance_budget = blended_task_budget(
                    self._cpb_progress_value.to(sigma),
                    effective_confidence.to(sigma),
                    fixed_budget=float(self.q_cpb_fixed_budget),
                    minimum_budget=float(self.q_cpb_budget_min),
                    maximum_budget=float(self.q_cpb_budget_max),
                    kappa=self.q_progress_kappa,
                )
            denoising_alpha, envelope_peak = self._cpb_step_alpha(step_index, sigma)
            weight = task_budget * denoising_alpha
            return weight, {
                "flow_time": flow_time,
                "progress": self._cpb_progress_value,
                "progress_d_init": self._progress_d_init,
                "progress_d_curr": self._progress_d_curr,
                "progress_d_ema": self._progress_d_ema,
                "guidance_weight_scheduler": scheduler,
                "progress_guidance_enabled": True,
                "progress_scheduler": scheduler,
                "cpb_task_budget": task_budget,
                "cpb_distance_budget": distance_budget,
                "cpb_fixed_budget": self.q_cpb_fixed_budget,
                "cpb_confidence": effective_confidence,
                "cpb_ood_confidence": ood_confidence,
                "cpb_direct_progress_budget": self.q_cpb_direct_progress_budget,
                "cpb_denoising_alpha": denoising_alpha,
                "cpb_envelope_peak_weight": envelope_peak,
                "cpb_sigma_peak": self.q_cpb_sigma_peak,
                "cpb_sigma_concentration": self.q_cpb_sigma_concentration,
                "progress_weight": weight,
            }
        if scheduler == "exp_progress":
            beta_task = 1.0 / (
                self.q_progress_exp_beta + torch.exp(-distance)
            ).clamp_min(self.q_progress_eps)
            beta_flow = torch.ones_like(flow_time)
            weight = beta_task
        else:
            beta_task = self.q_progress_beta_min + (
                self.q_progress_beta_max - self.q_progress_beta_min
            ) * (1.0 - progress).clamp(0.0, 1.0).pow(self.q_progress_kappa)
            beta_flow = flow_time.pow(self.q_progress_rho)
            weight = beta_task * beta_flow
        return weight, {
            "flow_time": flow_time,
            "progress": progress,
            "progress_d_init": self._progress_d_init,
            "progress_d_curr": self._progress_d_curr,
            "guidance_weight_scheduler": scheduler,
            "progress_guidance_enabled": True,
            "guidance_scale": float(self.q_guidance_scale),
            "progress_scheduler": scheduler,
            "progress_beta_task": beta_task,
            "progress_beta_flow": beta_flow,
            "progress_exp_beta": self.q_progress_exp_beta,
            "progress_weight": weight,
        }

    def _apply_q_control(
        self,
        control: dict[str, object],
        *,
        source: str,
    ) -> bool:
        changed = False
        if "q_guidance_scale" in control:
            new_scale = float(control["q_guidance_scale"])
            if new_scale != self.q_guidance_scale:
                logger.info(
                    f"Update q_guidance_scale from {self.q_guidance_scale} "
                    f"to {new_scale} via {source}"
                )
                self.q_guidance_scale = new_scale
                changed = True
        if "q_guidance_weight_scheduler" in control:
            scheduler = str(control["q_guidance_weight_scheduler"])
            if scheduler not in {
                "auto",
                "fixed",
                "denoising_time",
                "progress_decay",
                "exp_progress",
                "cpb",
                "fixed_cpb",
            }:
                raise ValueError(
                    "q_guidance_weight_scheduler must be one of "
                    "'auto', 'fixed', 'denoising_time', 'progress_decay', "
                    f"'exp_progress', 'cpb', or 'fixed_cpb', got {scheduler!r}"
                )
            if scheduler != self.q_guidance_weight_scheduler:
                logger.info(
                    "Update q_guidance_weight_scheduler from "
                    f"{self.q_guidance_weight_scheduler} to {scheduler} via {source}"
                )
                self.q_guidance_weight_scheduler = scheduler
                changed = True
        if "q_guidance_interval" in control:
            interval = int(control["q_guidance_interval"])
            if interval <= 0:
                raise ValueError("q_guidance_interval must be positive")
            if interval != self.q_guidance_interval:
                logger.info(
                    f"Update q_guidance_interval from {self.q_guidance_interval} "
                    f"to {interval} via {source}"
                )
                self.q_guidance_interval = interval
                changed = True
        if changed:
            self._write_q_record(
                self.q_summary_log_path,
                {
                    "event": "q_guidance_control",
                    "source": source,
                    "q_guidance_scale": self.q_guidance_scale,
                    "q_guidance_weight_scheduler": self.q_guidance_weight_scheduler,
                    "q_guidance_interval": self.q_guidance_interval,
                },
            )
        return changed

    def _poll_q_control_file(self) -> None:
        if self.q_control_path is None or not self.q_control_path.exists():
            return
        stat = self.q_control_path.stat()
        if self._q_control_mtime_ns == stat.st_mtime_ns:
            return
        with self.q_control_path.open("r", encoding="utf-8") as handle:
            control = json.load(handle)
        if not isinstance(control, dict):
            raise ValueError(f"Q control file must contain a JSON object: {self.q_control_path}")
        self._q_control_mtime_ns = stat.st_mtime_ns
        self._apply_q_control(control, source=str(self.q_control_path))

    def _init_q_control_file(self) -> None:
        if self.q_control_path is None:
            return
        self.q_control_path.parent.mkdir(parents=True, exist_ok=True)
        if self.q_control_path.exists():
            self._poll_q_control_file()
            return
        control = {
            "q_guidance_scale": self.q_guidance_scale,
            "q_guidance_weight_scheduler": self.q_guidance_weight_scheduler,
            "q_guidance_interval": self.q_guidance_interval,
        }
        self.q_control_path.write_text(
            json.dumps(control, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._q_control_mtime_ns = self.q_control_path.stat().st_mtime_ns
        logger.info(f"Wrote initial Q guidance control file {self.q_control_path}")

    def _q_control_state(self) -> dict[str, object]:
        return {
            "q_guidance_scale": self.q_guidance_scale,
            "q_guidance_weight_scheduler": self.q_guidance_weight_scheduler,
            "q_guidance_interval": self.q_guidance_interval,
            "q_guidance_start_step": self.q_guidance_start_step,
            "q_guidance_end_step": self.q_guidance_end_step,
            "q_checkpoint": self.q_checkpoint,
        }

    def _write_q_record(
        self,
        path: Path | None,
        record: dict[str, object],
    ) -> None:
        if path is None:
            return
        record = {
            "exp_name": getattr(self, "exp_name", None),
            "exp_save_root": getattr(self, "exp_save_root", None),
            **record,
        }
        serializable = {
            key: self._json_value(value)
            for key, value in record.items()
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serializable, sort_keys=True) + "\n")

    @staticmethod
    def _json_value(value: object) -> object:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                return float(value.item())
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        return value

    def _validate_q_artifact(self) -> None:
        config = self.q_artifact.config
        manifest = self.q_artifact.manifest
        if config.infer_latent_chunk_size != self.job_config.frame_chunk_size:
            raise ValueError(
                "Q checkpoint infer_latent_chunk_size must match inference "
                f"frame_chunk_size: {config.infer_latent_chunk_size} != "
                f"{self.job_config.frame_chunk_size}"
            )
        action_per_frame = manifest.get("action_per_frame")
        if (
            action_per_frame is not None
            and int(action_per_frame) != int(self.job_config.action_per_frame)
        ):
            raise ValueError(
                "Q checkpoint action_per_frame must match inference "
                f"action_per_frame: {action_per_frame} != "
                f"{self.job_config.action_per_frame}"
            )
        if self.q_artifact.feature_mode == "mlp_action_tokens":
            invalid = [
                layer
                for layer in config.critic_feature_layers
                if layer >= len(self.transformer.blocks)
            ]
            if invalid:
                raise ValueError(
                    "Q checkpoint critic_feature_layers are out of range for "
                    f"{len(self.transformer.blocks)} transformer blocks: {invalid}"
                )
        elif self.q_artifact.feature_mode == "query_context":
            invalid = [
                layer
                for layer in config.critic_feature_layers
                if layer >= len(self.transformer.blocks)
            ]
            if invalid:
                raise ValueError(
                    "Q checkpoint critic_feature_layers are out of range for "
                    f"{len(self.transformer.blocks)} transformer blocks: {invalid}"
                )
        else:
            raise ValueError(f"Unknown guidance feature mode {self.q_artifact.feature_mode}")

    def _log_loaded_q_artifact(self) -> None:
        param_count = sum(
            parameter.numel()
            for parameter in self.q_artifact.adapter.parameters()
        )
        logger.info(
            "Loaded Q guidance checkpoint "
            f"{self.q_checkpoint} "
            f"critic_type={self.q_artifact.config.critic_type} "
            f"feature_mode={self.q_artifact.feature_mode} "
            f"params={param_count}"
        )

    def _init_state_normalization(self) -> None:
        used_ids = np.array(self.job_config.used_action_channel_ids)
        inverse_ids = np.array(self.job_config.inverse_used_action_channel_ids)
        q01 = np.array(self.job_config.norm_stat["q01"], dtype=np.float32)
        q99 = np.array(self.job_config.norm_stat["q99"], dtype=np.float32)
        q01_used = np.pad(q01[used_ids], (0, 1))
        q99_used = np.pad(q99[used_ids], (0, 1))
        self._q_state_q01 = q01_used[inverse_ids]
        self._q_state_q99 = q99_used[inverse_ids]
        self._q_state_valid = inverse_ids < len(used_ids)

    def _should_apply_q_guidance(self, step_index: int) -> bool:
        if self.q_guidance_scale == 0.0:
            return False
        return self._guidance_step_selected(step_index)

    def _scheduler_sigma(
        self,
        timestep: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep_cpu = timestep.detach().cpu()
        timestep_id = torch.argmin(
            (self.action_scheduler.timesteps - timestep_cpu).abs()
        )
        sigma = self.action_scheduler.sigmas[timestep_id]
        return sigma.to(device=self.device, dtype=dtype)

    def _feature_timestep(self, value_range: tuple[float, float]) -> float:
        start, end = value_range
        return 0.5 * (float(start) + float(end))

    def _flex_attention_classes(self) -> set[type]:
        classes = set()
        for block in self.transformer.blocks:
            attn_op = getattr(block.attn1, "attn_op", None)
            attn_cls = type(attn_op)
            if hasattr(attn_cls, "flex_attn"):
                classes.add(attn_cls)
        return classes

    @contextmanager
    def _raw_flex_attention_context(self):
        if raw_flex_attention is None:
            yield
            return
        saved = {
            attn_cls: attn_cls.__dict__.get("flex_attn")
            for attn_cls in self._flex_attention_classes()
        }
        for attn_cls in saved:
            attn_cls.flex_attn = staticmethod(raw_flex_attention)
        try:
            yield
        finally:
            for attn_cls, flex_attn in saved.items():
                if flex_attn is None:
                    delattr(attn_cls, "flex_attn")
                else:
                    attn_cls.flex_attn = flex_attn

    def _clear_flex_attention_masks(self) -> None:
        for block in self.transformer.blocks:
            attn_op = getattr(block.attn1, "attn_op", None)
            attn_cls = type(attn_op)
            if hasattr(attn_cls, "attention_mask"):
                attn_cls.attention_mask = None
            if hasattr(attn_cls, "cross_attention_mask"):
                attn_cls.cross_attention_mask = None

    @contextmanager
    def _q_feature_extraction_context(self):
        missing = object()
        saved_caches = []
        for block in self.transformer.blocks:
            caches = getattr(block.attn1, "attn_caches", None)
            if caches is None:
                continue
            saved = caches.get(self.cache_name, missing)
            saved_caches.append((caches, saved))
            caches[self.cache_name] = None
        try:
            yield
        finally:
            for caches, saved in saved_caches:
                if saved is missing:
                    caches.pop(self.cache_name, None)
                else:
                    caches[self.cache_name] = saved
            self._clear_flex_attention_masks()

    def _valid_action_mask(
        self,
        actions: torch.Tensor,
        *,
        clamp_first_frame: bool,
    ) -> torch.Tensor:
        return build_action_guidance_mask(
            actions,
            valid_action_channels=self.action_mask,
            clamp_first_frame=clamp_first_frame,
        )

    def _full_action_mask(self, actions: torch.Tensor) -> torch.Tensor:
        return build_action_guidance_mask(
            actions,
            valid_action_channels=self.action_mask,
            clamp_first_frame=False,
        )

    def _store_previous_context(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        new_latents = latents.detach().to(
            device=self.device,
            dtype=self.dtype,
        )
        new_actions = actions.detach().to(
            device=self.device,
            dtype=self.dtype,
        )
        new_latent_mask = torch.ones(
            new_latents.shape[0],
            new_latents.shape[2],
            device=self.device,
            dtype=torch.bool,
        )
        latent_frames = new_latents.shape[2]
        action_frames = new_actions.shape[2]
        if latent_frames != action_frames:
            target_frames = max(latent_frames, action_frames)
            if latent_frames < target_frames:
                pad_frames = target_frames - latent_frames
                pad_shape = list(new_latents.shape)
                pad_shape[2] = pad_frames
                new_latents = torch.cat(
                    [torch.zeros(pad_shape, device=new_latents.device, dtype=new_latents.dtype), new_latents],
                    dim=2,
                ).contiguous()
                new_latent_mask = torch.cat(
                    [
                        torch.zeros(
                            new_latent_mask.shape[0],
                            pad_frames,
                            device=new_latent_mask.device,
                            dtype=torch.bool,
                        ),
                        new_latent_mask,
                    ],
                    dim=1,
                ).contiguous()
            if action_frames < target_frames:
                pad_frames = target_frames - action_frames
                pad_shape = list(new_actions.shape)
                pad_shape[2] = pad_frames
                new_actions = torch.cat(
                    [torch.zeros(pad_shape, device=new_actions.device, dtype=new_actions.dtype), new_actions],
                    dim=2,
                ).contiguous()
        new_action_mask = self._full_action_mask(new_actions)
        if action_frames < new_actions.shape[2]:
            new_action_mask[:, :, : new_actions.shape[2] - action_frames] = False
        if (
            self._q_previous_latents is not None
            and self._q_previous_actions is not None
            and self._q_previous_latent_mask is not None
            and self._q_previous_action_mask is not None
        ):
            self._q_previous_latents = torch.cat(
                [self._q_previous_latents.to(new_latents), new_latents],
                dim=2,
            )
            self._q_previous_actions = torch.cat(
                [self._q_previous_actions.to(new_actions), new_actions],
                dim=2,
            )
            self._q_previous_latent_mask = torch.cat(
                [
                    self._q_previous_latent_mask.to(new_latent_mask.device),
                    new_latent_mask,
                ],
                dim=1,
            )
            self._q_previous_action_mask = torch.cat(
                [
                    self._q_previous_action_mask.to(new_action_mask.device),
                    new_action_mask,
                ],
                dim=2,
            )
            self._trim_previous_context()
            return

        self._q_previous_latents = new_latents
        self._q_previous_actions = new_actions
        self._q_previous_latent_mask = new_latent_mask
        self._q_previous_action_mask = new_action_mask
        self._trim_previous_context()

    def _trim_previous_context(self) -> None:
        if (
            self._q_previous_latents is None
            or self._q_previous_actions is None
            or self._q_previous_latent_mask is None
            or self._q_previous_action_mask is None
        ):
            return
        chunk = self.q_artifact.config.infer_latent_chunk_size
        max_frames = self.q_history_chunks * chunk
        if self._q_previous_latents.shape[2] <= max_frames:
            return
        self._q_previous_latents = (
            self._q_previous_latents[:, :, -max_frames:].contiguous()
        )
        self._q_previous_actions = (
            self._q_previous_actions[:, :, -max_frames:].contiguous()
        )
        self._q_previous_latent_mask = (
            self._q_previous_latent_mask[:, -max_frames:].contiguous()
        )
        self._q_previous_action_mask = (
            self._q_previous_action_mask[:, :, -max_frames:].contiguous()
        )

    def _last_chunk_with_mask(
        self,
        tensor: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frame_dim = 2
        frame_count = tensor.shape[frame_dim]
        if frame_count >= chunk_size:
            start = frame_count - chunk_size
            chunk = tensor.narrow(frame_dim, start, chunk_size).clone()
            mask = torch.ones(
                tensor.shape[0],
                chunk_size,
                device=tensor.device,
                dtype=torch.bool,
            )
            return chunk, mask

        output_shape = list(tensor.shape)
        output_shape[frame_dim] = chunk_size
        output = torch.zeros(
            output_shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        frame_slice = [slice(None)] * tensor.ndim
        frame_slice[frame_dim] = slice(chunk_size - frame_count, chunk_size)
        output[tuple(frame_slice)] = tensor
        mask = torch.zeros(
            tensor.shape[0],
            chunk_size,
            device=tensor.device,
            dtype=torch.bool,
        )
        mask[:, chunk_size - frame_count :] = True
        return output, mask

    def _previous_context_or_zeros(
        self,
        current_latents: torch.Tensor,
        current_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk = self.q_artifact.config.infer_latent_chunk_size
        if (
            self._q_previous_latents is not None
            and self._q_previous_actions is not None
            and self._q_previous_latent_mask is not None
            and self._q_previous_action_mask is not None
        ):
            previous_latents = self._q_previous_latents.to(current_latents)
            previous_actions = self._q_previous_actions.to(current_actions)
            previous_latent_mask = self._q_previous_latent_mask.to(
                current_latents.device
            )
            previous_action_mask = self._q_previous_action_mask.to(
                current_actions.device
            )
            history_frames = previous_latents.shape[2]
            if history_frames < chunk:
                pad_frames = chunk - history_frames
                latent_pad_shape = list(previous_latents.shape)
                latent_pad_shape[2] = pad_frames
                action_pad_shape = list(previous_actions.shape)
                action_pad_shape[2] = pad_frames
                previous_latents = torch.cat(
                    [
                        torch.zeros(
                            latent_pad_shape,
                            device=previous_latents.device,
                            dtype=previous_latents.dtype,
                        ),
                        previous_latents,
                    ],
                    dim=2,
                ).contiguous()
                previous_actions = torch.cat(
                    [
                        torch.zeros(
                            action_pad_shape,
                            device=previous_actions.device,
                            dtype=previous_actions.dtype,
                        ),
                        previous_actions,
                    ],
                    dim=2,
                ).contiguous()
                previous_latent_mask = torch.cat(
                    [
                        torch.zeros(
                            previous_latent_mask.shape[0],
                            pad_frames,
                            device=previous_latent_mask.device,
                            dtype=torch.bool,
                        ),
                        previous_latent_mask,
                    ],
                    dim=1,
                ).contiguous()
                previous_action_mask = torch.cat(
                    [
                        torch.zeros(
                            previous_action_mask.shape[0],
                            previous_action_mask.shape[1],
                            pad_frames,
                            previous_action_mask.shape[3],
                            previous_action_mask.shape[4],
                            device=previous_action_mask.device,
                            dtype=torch.bool,
                        ),
                        previous_action_mask,
                    ],
                    dim=2,
                ).contiguous()
            return (
                previous_latents,
                previous_actions,
                previous_latent_mask,
                previous_action_mask,
            )
        previous_latents = torch.zeros_like(current_latents)
        previous_actions = torch.zeros_like(current_actions)
        previous_latent_mask = torch.zeros(
            current_latents.shape[0],
            chunk,
            device=current_latents.device,
            dtype=torch.bool,
        )
        previous_action_mask = torch.zeros(
            current_actions.shape[0],
            current_actions.shape[1],
            chunk,
            current_actions.shape[3],
            1,
            device=current_actions.device,
            dtype=torch.bool,
        )
        return (
            previous_latents,
            previous_actions,
            previous_latent_mask,
            previous_action_mask,
        )

    def _previous_chunk_or_zeros(
        self,
        current_latents: torch.Tensor,
        current_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_latents, previous_actions, previous_latent_mask, _ = (
            self._previous_context_or_zeros(current_latents, current_actions)
        )
        chunk = self.q_artifact.config.infer_latent_chunk_size
        previous_latents, latent_mask = self._last_chunk_with_mask(
            previous_latents,
            chunk,
        )
        previous_actions, _ = self._last_chunk_with_mask(
            previous_actions,
            chunk,
        )
        if previous_latent_mask.shape[1] >= chunk:
            latent_mask = previous_latent_mask[:, -chunk:].to(
                current_latents.device
            )
        return previous_latents, previous_actions, latent_mask

    def _extract_q_features(
        self,
        latents: torch.Tensor,
        clean_action: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        config = self.q_artifact.config
        text_emb = self.prompt_embeds.to(self.dtype).clone()
        if self.q_artifact.feature_mode == "mlp_action_tokens":
            feature_input = build_clean_feature_input(
                latents=latents.detach().to(dtype=self.dtype),
                actions=clean_action,
                text_emb=text_emb,
                patch_size=tuple(self.job_config.patch_size),
                chunk_size=config.infer_latent_chunk_size,
                window_size=config.window_size,
                latent_timestep=self._feature_timestep(
                    config.q_video_feature_timestep_range
                ),
                action_timestep=self._feature_timestep(
                    config.q_action_feature_timestep_range
                ),
            )
            outputs = self.transformer(
                feature_input,
                train_mode=True,
                return_features=True,
                activation_checkpointing=self.q_guidance_activation_checkpointing,
                **self._mlp_feature_layer_kwargs(config),
            )
            features = outputs[3]
            if len(config.critic_feature_layers) > 1:
                features = dict(features)
                features["action_tokens"] = self._concat_mlp_layer_features(
                    features["action_tokens"],
                    key="action_tokens",
                )
                features["video_tokens"] = self._concat_mlp_layer_features(
                    features["video_tokens"],
                    key="video_tokens",
                )
            return features

        previous_latents, previous_actions, previous_latent_mask, previous_action_mask = (
            self._previous_context_or_zeros(latents, clean_action)
        )
        feature_input = build_clean_feature_input(
            latents=torch.cat([previous_latents, latents.detach().to(self.dtype)], dim=2),
            actions=torch.cat([previous_actions, clean_action], dim=2),
            text_emb=text_emb,
            patch_size=tuple(self.job_config.patch_size),
            chunk_size=config.infer_latent_chunk_size,
            window_size=config.window_size,
            latent_timestep=self._feature_timestep(
                config.q_video_feature_timestep_range
            ),
            action_timestep=self._feature_timestep(
                config.q_action_feature_timestep_range
            ),
        )
        outputs = self.transformer(
            feature_input,
            train_mode=True,
            return_features=True,
            critic_feature_layers=config.critic_feature_layers,
            stop_after_critic_feature_layer=True,
            activation_checkpointing=self.q_guidance_activation_checkpointing,
        )
        return build_query_q_context(
            outputs[3],
            previous_latent_mask=previous_latent_mask,
            previous_action_mask=previous_action_mask,
            current_action_mask=action_mask,
            chunk_size=config.infer_latent_chunk_size,
        )

    def _mlp_feature_layer_kwargs(self, config) -> dict[str, object]:
        if len(config.critic_feature_layers) > 1:
            return {"critic_feature_layers": config.critic_feature_layers}
        return {"critic_feature_layer": config.critic_feature_layers[0]}

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
        expected = int(self.q_artifact.config.feature_dim)
        if concatenated.shape[-1] != expected:
            raise ValueError(
                f"{key} concatenated feature_dim={concatenated.shape[-1]} "
                f"does not match Q config feature_dim={expected}"
            )
        return concatenated

    def _select_q_objective(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
    ) -> torch.Tensor:
        if self.q_objective == "min":
            return torch.minimum(q1, q2)
        if self.q_objective == "mean":
            return 0.5 * (q1 + q2)
        if self.q_objective == "q1":
            return q1
        if self.q_objective == "q2":
            return q2
        raise ValueError(f"Unknown q_objective: {self.q_objective}")

    def _evaluate_mlp_value(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, bool]:
        bundle = getattr(self.q_artifact.adapter, "bundle", None)
        if bundle is None or getattr(bundle, "value", None) is None:
            return None, False
        if self.q_artifact.feature_mode != "mlp_action_tokens":
            return None, False

        config = self.q_artifact.config
        previous_latents, previous_actions, previous_latent_mask = (
            self._previous_chunk_or_zeros(latents, actions)
        )
        state_valid = bool(previous_latent_mask.any().item())
        feature_input = build_clean_feature_input(
            latents=previous_latents,
            actions=previous_actions,
            text_emb=self.prompt_embeds.to(self.dtype).clone(),
            patch_size=tuple(self.job_config.patch_size),
            chunk_size=config.infer_latent_chunk_size,
            window_size=config.window_size,
            latent_timestep=self._feature_timestep(
                config.value_video_feature_timestep_range
            ),
            action_timestep=self._feature_timestep(
                config.value_action_feature_timestep_range
            ),
        )
        outputs = self.transformer(
            feature_input,
            train_mode=True,
            return_features=True,
            **self._mlp_feature_layer_kwargs(config),
        )
        video_tokens = outputs[3]["video_tokens"]
        if len(config.critic_feature_layers) > 1:
            video_tokens = self._concat_mlp_layer_features(
                video_tokens,
                key="video_tokens",
            )
        value = bundle.value(video_tokens, previous_latent_mask)
        return value.detach().float(), state_valid

    def _log_chunk_qv(
        self,
        *,
        frame_st_id: int,
        latents: torch.Tensor,
        actions: torch.Tensor,
        action_cond: torch.Tensor | None,
    ) -> None:
        if self.q_guidance_scale == 0.0 and not self.q_log_denoising_steps:
            return
        if self.q_summary_log_path is None:
            return
        action_mask = self._valid_action_mask(
            actions,
            clamp_first_frame=False,
        )
        try:
            with (
                torch.no_grad(),
                self._q_feature_extraction_context(),
                self._raw_flex_attention_context(),
            ):
                features = self._extract_q_features(
                    latents=latents,
                    clean_action=actions.to(dtype=self.dtype),
                    action_mask=action_mask,
                )
                q1, q2 = self.q_artifact.adapter.q_values(features, action_mask)
                q_objective = self._select_q_objective(q1, q2)
                value, state_valid = self._evaluate_mlp_value(latents, actions)
        except Exception as error:
            self._write_q_record(
                self.q_summary_log_path,
                {
                    "event": "chunk_qv_error",
                    "frame_st_id": int(frame_st_id),
                    "error": repr(error),
                },
            )
            return

        record: dict[str, object] = {
            "event": "chunk_qv",
            "frame_st_id": int(frame_st_id),
            "critic_type": self.q_artifact.config.critic_type,
            "feature_mode": self.q_artifact.feature_mode,
            "q_checkpoint": self.q_checkpoint,
            "q_objective_mode": self.q_objective,
            "q_history_chunks": self.q_history_chunks,
            "q_guidance_scale": self.q_guidance_scale,
            "q_guidance_weight_scheduler": self.q_guidance_weight_scheduler,
            "guidance_coordinate_scaling": "none",
            "progress_guidance_enabled": bool(self.q_progress_guidance),
            "guidance_formula": (
                "scheduler_weight * normalized_dQ_da_clean"
                if self.q_grad_normalize
                else "scheduler_weight * dQ_da_clean"
            ),
            "effective_guidance_formula": (
                "scheduler_weight * normalized_dQ_da_clean"
                if self.q_grad_normalize
                else "scheduler_weight * dQ_da_clean"
            ),
            "q1": q1.detach().float().cpu().tolist(),
            "q2": q2.detach().float().cpu().tolist(),
            "q_min": torch.minimum(q1, q2).detach().float().cpu().tolist(),
            "q_objective": q_objective.detach().float().cpu().tolist(),
            "state_valid": state_valid,
            "action_cond_clamped": action_cond is not None,
        }
        if value is not None:
            record["value"] = value.cpu().tolist()
        if self.q_progress_guidance:
            record["progress"] = (
                self._cpb_progress_value
                if self.q_guidance_weight_scheduler == "cpb"
                or (
                    self.q_guidance_weight_scheduler == "auto"
                    and self.q_progress_scheduler == "cpb"
                )
                else self._progress_value
            )
            record["progress_d_init"] = self._progress_d_init
            record["progress_d_curr"] = self._progress_d_curr
            record["progress_d_ema"] = self._progress_d_ema
            record["cpb_confidence"] = self._cpb_confidence_value
        self._write_q_record(self.q_summary_log_path, record)

    def _q_guided_velocity(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        action_velocity: torch.Tensor,
        timestep: torch.Tensor,
        action_cond: torch.Tensor | None,
        step_index: int,
    ) -> torch.Tensor:
        if not self._should_apply_q_guidance(step_index):
            return action_velocity

        sigma = self._scheduler_sigma(timestep, dtype=torch.float32)
        clean_action = (
            actions.float() - sigma.view(1, 1, 1, 1, 1) * action_velocity.float()
        ).detach()
        if action_cond is not None:
            clean_action = clean_action.clone()
            clean_action[:, :, 0:1] = action_cond.float()
        clean_action = clean_action.to(dtype=self.dtype).requires_grad_(True)

        action_mask = self._valid_action_mask(
            clean_action,
            clamp_first_frame=action_cond is not None,
        )
        with (
            torch.enable_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            features = self._extract_q_features(
                latents=latents,
                clean_action=clean_action,
                action_mask=action_mask,
            )
            q1, q2 = self.q_artifact.adapter.q_values(features, action_mask)
            q_values_for_objective = self._select_q_objective(q1, q2)
            q_objective = q_values_for_objective.sum()
            grad = torch.autograd.grad(
                q_objective,
                clean_action,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]

        grad = grad.float() * action_mask.to(grad.dtype)
        if self.q_grad_normalize:
            reduce_dims = tuple(range(1, grad.ndim))
            grad_norm = grad.square().sum(dim=reduce_dims, keepdim=True).sqrt()
            grad = grad / grad_norm.clamp_min(self.q_progress_eps)
        if self.q_grad_clip > 0:
            grad = grad.clamp(-self.q_grad_clip, self.q_grad_clip)

        guidance_weight, weight_record = self._progress_guidance_weight(
            sigma,
            step_index,
        )
        guidance_weight = guidance_weight.view(1, 1, 1, 1, 1)
        guidance = guidance_weight * grad
        guided_velocity = action_velocity.float() - guidance
        if self.q_log_denoising_steps:
            active = action_mask.to(grad.dtype).sum().clamp_min(1.0)
            self._write_q_record(
                self.q_step_log_path,
                {
                    "event": "denoising_step",
                    "frame_st_id": int(getattr(self, "_current_frame_st_id", -1)),
                    "action_step": int(step_index),
                    "sigma": sigma,
                    **weight_record,
                    "guidance_weight": guidance_weight.squeeze(),
                    "guidance_coordinate_scaling": "none",
                    "guidance_coordinate_scale": 1.0,
                    "effective_guidance_weight": guidance_weight.squeeze(),
                    "effective_guidance_formula": (
                        "scheduler_weight * normalized_dQ_da_clean"
                        if self.q_grad_normalize
                        else "scheduler_weight * dQ_da_clean"
                    ),
                    "guidance_gradient_normalized": self.q_grad_normalize,
                    "guidance_gradient_clip": self.q_grad_clip,
                    "q1": q1.detach().float().cpu().tolist(),
                    "q2": q2.detach().float().cpu().tolist(),
                    "q_objective": (
                        q_values_for_objective.detach().float().cpu().tolist()
                    ),
                    "grad_l2": grad.square().sum().sqrt(),
                    "grad_rms": (grad.square().sum() / active).sqrt(),
                    "guidance_rms": (
                        guidance.square().sum()
                        / action_mask.to(guidance.dtype).sum().clamp_min(1.0)
                    ).sqrt(),
                },
            )
        return guided_velocity.to(dtype=action_velocity.dtype)

    def _initial_action_cond(self, initial_state) -> torch.Tensor:
        if self.robotwin_eval:
            return torch.zeros(
                [
                    1,
                    self.job_config.action_dim,
                    1,
                    self.action_per_frame,
                    1,
                ],
                device=self.device,
                dtype=self.dtype,
            )
        if initial_state is None:
            raise ValueError(
                "_infer requires initial_state for q-guided frame_st_id == 0. "
                "Pass the current robot state in used-channel space."
            )

        state = np.asarray(initial_state, dtype=np.float32)
        if state.ndim >= 3:
            return self.preprocess_action(state).to(self.device, self.dtype)[:, :, 0:1]
        if state.ndim == 1:
            state = state[None, :]
        expected_width = len(self.job_config.used_action_channel_ids)
        if state.shape[-1] != expected_width:
            raise ValueError(
                "Initial state must use the configured action representation: "
                f"expected {expected_width} channels, got {state.shape[-1]}"
            )
        state_rep = np.tile(state[:1], (self.action_per_frame, 1))
        inv_ids = np.array(self.job_config.inverse_used_action_channel_ids)
        required_width = int(inv_ids.max()) + 1 if inv_ids.size else state_rep.shape[1]
        pad_width = max(0, required_width - state_rep.shape[1])
        state_padded = np.pad(state_rep, ((0, 0), (0, pad_width)))
        state_aligned = state_padded[:, inv_ids]
        denom = np.maximum(self._q_state_q99 - self._q_state_q01, 1e-2)
        state_norm = np.where(
            self._q_state_valid[None, :],
            (state_aligned - self._q_state_q01[None, :])
            / (denom[None, :] + 1e-6)
            * 2.0
            - 1.0,
            0.0,
        )
        return torch.from_numpy(
            state_norm.T[None, :, None, :, None].astype(np.float32)
        ).to(self.device, self.dtype)

    def _infer(self, obs, frame_st_id=0, initial_state=None):
        self._current_frame_st_id = int(frame_st_id)
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            self._update_progress_from_obs(obs, initialize=True)
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
            device=self.device,
            dtype=self.dtype,
        )
        actions = torch.randn(
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
            device=self.device,
            dtype=self.dtype,
        )

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = F.pad(
            self.scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )
        if video_step != -1:
            timesteps = timesteps[:video_step]
        action_timesteps = F.pad(
            self.action_scheduler.timesteps,
            (0, 1),
            mode="constant",
            value=0,
        )

        with torch.no_grad():
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = (
                    init_latent[:, :, 0:1].to(self.dtype)
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id,
                )
                self._clear_flex_attention_masks()
                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                )
                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size,
                        video_noise_pred,
                        frame_chunk_size,
                        self.latent_height,
                        self.latent_width,
                        batch_size=2 if self.use_cfg else 1,
                    )
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = (
                            video_noise_pred[1:]
                            + self.job_config.guidance_scale
                            * (video_noise_pred[:1] - video_noise_pred[1:])
                        )
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(
                        video_noise_pred,
                        t,
                        latents,
                        return_dict=False,
                    )
                latents[:, :, 0:1] = (
                    latent_cond if frame_st_id == 0 else latents[:, :, 0:1]
                )

        action_cond = (
            self._initial_action_cond(initial_state) if frame_st_id == 0 else None
        )
        for i, t in enumerate(tqdm(action_timesteps)):
            last_step = i == len(action_timesteps) - 1
            input_dict = self._prepare_latent_input(
                None,
                actions,
                t,
                t,
                None,
                action_cond,
                frame_st_id=frame_st_id,
            )
            with torch.no_grad():
                self._clear_flex_attention_masks()
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                )

            if not last_step:
                action_noise_pred = rearrange(
                    action_noise_pred,
                    "b (f n) c -> b c f n 1",
                    f=frame_chunk_size,
                )
                if self.job_config.action_guidance_scale > 1:
                    action_noise_pred = (
                        action_noise_pred[1:]
                        + self.job_config.action_guidance_scale
                        * (action_noise_pred[:1] - action_noise_pred[1:])
                    )
                else:
                    action_noise_pred = action_noise_pred[:1]
                action_noise_pred = self._q_guided_velocity(
                    actions,
                    latents,
                    action_noise_pred,
                    t,
                    action_cond,
                    i,
                )
                with torch.no_grad():
                    actions = self.action_scheduler.step(
                        action_noise_pred,
                        t,
                        actions,
                        return_dict=False,
                    )
            actions[:, :, 0:1] = (
                action_cond if frame_st_id == 0 else actions[:, :, 0:1]
            )

        actions[:, ~self.action_mask] *= 0
        self._last_predicted_actions = actions.clone()
        self._last_predicted_latents = latents.clone()
        self._log_chunk_qv(
            frame_st_id=frame_st_id,
            latents=latents,
            actions=actions,
            action_cond=action_cond,
        )

        save_async(
            latents,
            os.path.join(self.exp_save_root, f"latents_{frame_st_id}.pt"),
        )
        save_async(
            actions,
            os.path.join(self.exp_save_root, f"actions_{frame_st_id}.pt"),
        )

        actions_np = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        return actions_np, latents

    def _decode_predicted_video(self, latents: torch.Tensor) -> np.ndarray:
        if not hasattr(self, "video_processor"):
            from diffusers.video_processor import VideoProcessor

            self.video_processor = VideoProcessor(vae_scale_factor=1)
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        with torch.no_grad():
            return self.decode_one_video(latents.detach(), "np")[0]

    def _prediction_latent_metrics(
        self,
        actual_latents: torch.Tensor,
    ) -> dict[str, object] | None:
        if self._last_predicted_latents is None or actual_latents is None:
            return None
        predicted = self._last_predicted_latents.detach().to(
            device=actual_latents.device,
            dtype=torch.float32,
        )
        actual = actual_latents.detach().to(dtype=torch.float32)
        frames = min(predicted.shape[2], actual.shape[2])
        if frames <= 0:
            return None
        predicted = predicted[:, :, -frames:]
        actual = actual[:, :, -frames:]
        diff = predicted - actual
        mse = diff.square().mean()
        mae = diff.abs().mean()
        return {
            "latent_frames": int(frames),
            "latent_mse": float(mse.detach().cpu()),
            "latent_mae": float(mae.detach().cpu()),
            "latent_reward": float((-mse).detach().cpu()),
        }

    def _compute_kv_cache(self, obs):
        self.transformer.clear_pred_cache(self.cache_name)
        obs_frames = obs.get("obs")
        try:
            frame_count = len(obs_frames) if obs_frames is not None else 0
        except TypeError:
            frame_count = 1
        if frame_count == 0:
            logger.warning(
                "Skipping Q-guided KV cache update because no observation frames were provided."
            )
            return None
        save_async(
            obs_frames,
            os.path.join(self.exp_save_root, f"obs_data_{self.frame_st_id}.pt"),
        )
        try:
            latent_model_input = self._encode_obs(obs)
        except RuntimeError as exc:
            message = str(exc)
            if (
                "Calculated padded input size" not in message
                and "must match the size of tensor b" not in message
                and "must match the size of tensor a" not in message
            ):
                raise
            self.streaming_vae.clear_cache()
            if self.streaming_vae_half is not None:
                self.streaming_vae_half.clear_cache()
            logger.warning(
                "Skipping Q-guided KV cache update because WAN VAE observation "
                "encoding failed on a short or partial temporal chunk: %s",
                message,
            )
            return None
        prediction_metrics = self._prediction_latent_metrics(latent_model_input)
        self._update_progress_from_obs(obs, initialize=False)
        if self.robotwin_eval and self.frame_st_id == 0 and self.init_latent is not None:
            latent_model_input = (
                torch.cat([self.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else self.init_latent
            )
        elif latent_model_input is None:
            raise ValueError("KV-cache update requires at least one observation frame.")
        elif (
            self.frame_st_id == 0
            and self.init_latent is not None
        ):
            latent_model_input = latent_model_input.clone()
            latent_model_input[:, :, 0:1] = self.init_latent[:, :, 0:1].to(
                latent_model_input
            )
        state = obs.get("state", None)
        if state is not None:
            action_model_input = self.preprocess_action(state).to(latent_model_input)
        elif self._last_predicted_actions is not None:
            action_model_input = self._last_predicted_actions.to(latent_model_input)
        else:
            raise RuntimeError(
                "_compute_kv_cache called without obs['state'] before any "
                "q-guided predicted actions are available."
            )
        if action_model_input.shape[2] != latent_model_input.shape[2]:
            raise ValueError(
                "KV-cache temporal mismatch: encoded observation latents and "
                "action/state frames must match, got "
                f"{latent_model_input.shape[2]} and {action_model_input.shape[2]}."
            )
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(
            latent_model_input,
            action_model_input,
            frame_st_id=self.frame_st_id,
        )
        with torch.no_grad():
            self.transformer(
                self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=False,
            )
            self.transformer(
                self._repeat_input_for_cfg(input_dict["action_res_lst"]),
                update_cache=2,
                cache_name=self.cache_name,
                action_mode=True,
            )
        self._store_previous_context(latent_model_input, action_model_input)
        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]
        return prediction_metrics

    def infer(self, obs):
        reset = obs.get("reset", False)
        prompt = obs.get("prompt", None)
        compute_kv_cache = obs.get("compute_kv_cache", False)
        self._poll_q_control_file()
        request_control = {}
        if isinstance(obs.get("q_guidance_control"), dict):
            request_control.update(obs["q_guidance_control"])
        for key in (
            "q_guidance_scale",
            "q_guidance_weight_scheduler",
            "q_guidance_interval",
        ):
            if key in obs:
                request_control[key] = obs[key]
        if request_control:
            self._apply_q_control(request_control, source="request")
        if obs.get("q_control_only", False):
            return {"q_guidance_control": self._q_control_state()}

        if reset:
            logger.info("******************* Reset Q-guided server ******************")
            with torch.no_grad():
                self._reset(prompt=prompt)
            return dict()
        if compute_kv_cache:
            logger.info("################# Compute Q-guided KV Cache #################")
            with torch.no_grad():
                prediction_metrics = self._compute_kv_cache(obs)
            if prediction_metrics is None:
                return dict()
            return dict(prediction_metrics=prediction_metrics)

        logger.info("################# Infer One Q-guided Chunk #################")
        action, latents = self._infer(
            obs,
            frame_st_id=self.frame_st_id,
            initial_state=obs.get("state", None),
        )
        output = dict(action=action)
        if obs.get("return_predicted_video", False):
            output["video"] = self._decode_predicted_video(latents)
        return output


def run(args):
    config = VA_CONFIGS[args.config_name]
    if args.eval_model_path is not None:
        config.wan22_finetuned_model_name_or_path = args.eval_model_path
        eval_root = Path(args.eval_model_path)
        if all(
            (eval_root / name).exists()
            for name in ("vae", "tokenizer", "text_encoder")
        ):
            config.wan22_pretrained_model_name_or_path = args.eval_model_path
    config.attn_mode = args.attn_mode

    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        init_distributed(world_size, local_rank, rank)
    else:
        torch.cuda.set_device(local_rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = QGuidedVA_Server(
        config,
        q_checkpoint=args.q_checkpoint,
        q_guidance_scale=args.q_guidance_scale,
        q_guidance_beta=args.q_guidance_beta,
        q_guidance_weight_scheduler=args.q_guidance_weight_scheduler,
        q_guidance_start_step=args.q_guidance_start_step,
        q_guidance_end_step=args.q_guidance_end_step,
        q_guidance_interval=args.q_guidance_interval,
        q_objective=args.q_objective,
        q_grad_clip=args.q_grad_clip,
        q_grad_normalize=args.q_grad_normalize,
        q_history_chunks=args.q_history_chunks,
        q_guidance_activation_checkpointing=args.q_guidance_activation_checkpointing,
        robotwin_eval=args.robotwin,
        q_log_dir=args.q_log_dir,
        q_log_denoising_steps=args.q_log_denoising_steps,
        q_control_path=args.q_control_path,
        q_progress_guidance=args.q_progress_guidance,
        q_progress_goal_embedding=args.q_progress_goal_embedding,
        q_progress_jepa_checkpoint=args.q_progress_jepa_checkpoint,
        q_progress_jepa_config=args.q_progress_jepa_config,
        q_progress_jepa_repo=args.q_progress_jepa_repo,
        q_progress_jepa_checkpoint_key=args.q_progress_jepa_checkpoint_key,
        q_progress_camera_key=args.q_progress_camera_key,
        q_progress_clip_frames=args.q_progress_clip_frames,
        q_progress_beta_max=args.q_progress_beta_max,
        q_progress_beta_min=args.q_progress_beta_min,
        q_progress_kappa=args.q_progress_kappa,
        q_progress_rho=args.q_progress_rho,
        q_progress_eps=args.q_progress_eps,
        q_progress_scheduler=args.q_progress_scheduler,
        q_progress_exp_beta=args.q_progress_exp_beta,
        q_cpb_success_reference=args.q_cpb_success_reference,
        q_cpb_confidence=args.q_cpb_confidence,
        q_cpb_fixed_budget=args.q_cpb_fixed_budget,
        q_cpb_budget_min=args.q_cpb_budget_min,
        q_cpb_budget_max=args.q_cpb_budget_max,
        q_cpb_ema_alpha=args.q_cpb_ema_alpha,
        q_cpb_sigma_peak=args.q_cpb_sigma_peak,
        q_cpb_sigma_concentration=args.q_cpb_sigma_concentration,
        q_cpb_distance_low=args.q_cpb_distance_low,
        q_cpb_distance_high=args.q_cpb_distance_high,
        q_cpb_ood_margin=args.q_cpb_ood_margin,
        q_cpb_direct_progress_budget=args.q_cpb_direct_progress_budget,
    )
    if config.infer_mode == "i2va":
        logger.info(
            "******************************USE Q-guided I2AV mode******************************"
        )
        model.generate()
    elif config.infer_mode == "server":
        logger.info(
            "***********************USE Q-guided Server mode***********************"
        )
        metadata = {
            "action_per_frame": config.action_per_frame,
            "frame_chunk_size": config.frame_chunk_size,
            "q_guidance": "q_gradient_v1",
            "q_guidance_control": {
                "q_guidance_scale": args.q_guidance_scale,
                "q_guidance_weight_scheduler": args.q_guidance_weight_scheduler,
                "q_guidance_activation_checkpointing": args.q_guidance_activation_checkpointing,
                "q_control_path": args.q_control_path,
            },
        }
        run_async_server_mode(model, local_rank, config.host, port, metadata=metadata)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", type=str, default="robotwin")
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--eval_model_path", type=str, default=None)
    parser.add_argument("--robotwin", action="store_true")
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="torch",
        choices=("flex", "torch", "flashattn"),
        help="Stored on the config for compatibility; the original server currently loads torch attention.",
    )
    parser.add_argument("--q_checkpoint", type=str, required=True)
    parser.add_argument("--q_guidance_scale", type=float, default=0.0)
    parser.add_argument("--q_guidance_beta", type=float, default=2.0)
    parser.add_argument(
        "--q_guidance_weight_scheduler",
        type=str,
        default="auto",
        choices=(
            "auto",
            "fixed",
            "denoising_time",
            "progress_decay",
            "exp_progress",
            "cpb",
            "fixed_cpb",
        ),
        help=(
            "How to turn q_guidance_scale into per-step guidance weight. "
            "'auto' preserves legacy behavior; 'fixed' uses q_guidance_scale "
            "directly at every guided denoising step."
        ),
    )
    parser.add_argument("--q_guidance_start_step", type=int, default=0)
    parser.add_argument("--q_guidance_end_step", type=int, default=-1)
    parser.add_argument("--q_guidance_interval", type=int, default=1)
    parser.add_argument(
        "--q_objective",
        type=str,
        default="min",
        choices=("min", "mean", "q1", "q2"),
    )
    parser.add_argument("--q_grad_clip", type=float, default=0.0)
    parser.add_argument("--q_grad_normalize", action="store_true")
    parser.add_argument(
        "--q_history_chunks",
        type=int,
        default=10,
        help="Maximum number of previous inference chunks kept for Q guidance context.",
    )
    parser.add_argument(
        "--q_guidance_activation_checkpointing",
        action="store_true",
        help="Recompute Q-feature transformer blocks during dQ/daction to reduce activation memory.",
    )
    parser.add_argument("--q_log_dir", type=str, default=None)
    parser.add_argument("--q_log_denoising_steps", action="store_true")
    parser.add_argument(
        "--q_control_path",
        type=str,
        default=None,
        help=(
            "Optional JSON control file polled on each request. Supported keys: "
            "q_guidance_scale, q_guidance_weight_scheduler, q_guidance_interval."
        ),
    )
    parser.add_argument("--q_progress_guidance", action="store_true")
    parser.add_argument("--q_progress_goal_embedding", type=str, default=None)
    parser.add_argument(
        "--q_progress_jepa_checkpoint",
        type=str,
        default="/luhongchao/shared/weights/vjepa2.1/vjepa2_1_vitG_384.pt",
    )
    parser.add_argument("--q_progress_jepa_config", type=str, default=None)
    parser.add_argument("--q_progress_jepa_repo", type=str, default=None)
    parser.add_argument(
        "--q_progress_jepa_checkpoint_key",
        type=str,
        default="target_encoder",
    )
    parser.add_argument("--q_progress_camera_key", type=str, default=None)
    parser.add_argument("--q_progress_clip_frames", type=int, default=1)
    parser.add_argument("--q_progress_beta_max", type=float, default=None)
    parser.add_argument("--q_progress_beta_min", type=float, default=0.0)
    parser.add_argument("--q_progress_kappa", type=float, default=1.0)
    parser.add_argument("--q_progress_rho", type=float, default=1.0)
    parser.add_argument("--q_progress_eps", type=float, default=1e-6)
    parser.add_argument(
        "--q_progress_scheduler",
        type=str,
        default="progress_decay",
        choices=("progress_decay", "exp_progress", "cpb"),
    )
    parser.add_argument("--q_progress_exp_beta", type=float, default=10.0)
    parser.add_argument("--q_cpb_success_reference", type=float, default=None)
    parser.add_argument("--q_cpb_confidence", type=float, default=0.0)
    parser.add_argument("--q_cpb_fixed_budget", type=float, default=None)
    parser.add_argument("--q_cpb_budget_min", type=float, default=None)
    parser.add_argument("--q_cpb_budget_max", type=float, default=None)
    parser.add_argument("--q_cpb_ema_alpha", type=float, default=0.25)
    parser.add_argument("--q_cpb_sigma_peak", type=float, default=0.25)
    parser.add_argument("--q_cpb_sigma_concentration", type=float, default=4.0)
    parser.add_argument("--q_cpb_distance_low", type=float, default=None)
    parser.add_argument("--q_cpb_distance_high", type=float, default=None)
    parser.add_argument("--q_cpb_ood_margin", type=float, default=0.0)
    parser.add_argument("--q_cpb_direct_progress_budget", action="store_true")
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
