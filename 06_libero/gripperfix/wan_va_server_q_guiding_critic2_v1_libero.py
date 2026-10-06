# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

try:
    from wan_va.replayable_noise import (
        load_replay_episode_keys,
        replayable_randn,
        tensor_sha256,
    )
except ImportError:
    from replayable_noise import (
        load_replay_episode_keys,
        replayable_randn,
        tensor_sha256,
    )

try:
    from wan_va.rl.q_observation_noise import (
        parse_nonnegative_int_csv,
        parse_positive_float_csv,
        run_observation_noise_probe,
    )
except ModuleNotFoundError:
    from q_observation_noise import (
        parse_nonnegative_int_csv,
        parse_positive_float_csv,
        run_observation_noise_probe,
    )

try:
    from wan_va.rl.q_action_gradient_distribution import (
        run_action_gradient_distribution_probe,
    )
except ModuleNotFoundError:
    from q_action_gradient_distribution import (
        run_action_gradient_distribution_probe,
    )

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
try:
    from wan_va.rl.adaptive_cpb import (
        AdaptiveCPBTrustRegionConfig,
        AdaptiveCPBTrustRegionController,
        QRelativeCollapseTracker,
        adaptive_cpb_qdelta_safety_gate,
        adaptive_cpb_qdelta_smooth_budget,
        adaptive_cpb_budget,
        build_fixed_anchor_probe_budgets,
        centered_adaptive_cpb_budget,
        clip_gradient_l2_norm,
        cpb_implied_clean_actions,
        first_order_local_q_surrogates,
        fixed_floor_centered_adaptive_cpb_budget,
        local_uncertainty_ascent_direction,
        q_vgm_residual_velocity_target,
        select_local_uncertainty_cpb,
        select_fixed_anchor_probe,
        select_multibudget_probe,
        twin_grad_cosine_confidence,
    )
except ImportError as error:
    missing_module = isinstance(error, ModuleNotFoundError)
    missing_gate_helper = any(
        name in str(error)
        for name in (
            "AdaptiveCPBTrustRegionConfig",
            "AdaptiveCPBTrustRegionController",
            "QRelativeCollapseTracker",
            "adaptive_cpb_qdelta_safety_gate",
            "adaptive_cpb_qdelta_smooth_budget",
            "build_fixed_anchor_probe_budgets",
            "clip_gradient_l2_norm",
            "cpb_implied_clean_actions",
            "first_order_local_q_surrogates",
            "fixed_floor_centered_adaptive_cpb_budget",
            "local_uncertainty_ascent_direction",
            "q_vgm_residual_velocity_target",
            "select_local_uncertainty_cpb",
            "select_fixed_anchor_probe",
            "select_multibudget_probe",
        )
    )
    if (missing_module and error.name != "wan_va.rl.adaptive_cpb") or (
        not missing_module and not missing_gate_helper
    ):
        raise
    from adaptive_cpb import (
        AdaptiveCPBTrustRegionConfig,
        AdaptiveCPBTrustRegionController,
        QRelativeCollapseTracker,
        adaptive_cpb_qdelta_safety_gate,
        adaptive_cpb_qdelta_smooth_budget,
        adaptive_cpb_budget,
        build_fixed_anchor_probe_budgets,
        centered_adaptive_cpb_budget,
        clip_gradient_l2_norm,
        cpb_implied_clean_actions,
        first_order_local_q_surrogates,
        fixed_floor_centered_adaptive_cpb_budget,
        local_uncertainty_ascent_direction,
        q_vgm_residual_velocity_target,
        select_local_uncertainty_cpb,
        select_fixed_anchor_probe,
        select_multibudget_probe,
        twin_grad_cosine_confidence,
    )
from wan_va.rl.progress_budget import (
    beta_envelope,
    blended_task_budget,
    calibrated_progress,
    distance_confidence,
    update_ema,
)
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
        q_replay_noise_seed: int | None = None,
        q_replay_manifest_path: str | None = None,
        q_log_replay_noise_hashes: bool = False,
        q_log_local_guidance_delta: bool = False,
        q_log_second_order_cosine: bool = False,
        q_second_order_finite_difference_epsilon: float = 0.25,
        q_second_order_observation_steps: str = "25,35,40",
        q_log_observation_noise_sensitivity: bool = False,
        q_observation_noise_levels: str = "0.001,0.003,0.01,0.03",
        q_observation_noise_draws: int = 4,
        q_observation_noise_steps: str = "25,35,40",
        q_observation_noise_request_indices: str = "2,5,10",
        q_observation_noise_seed: int = 2026082501,
        q_log_action_gradient_distribution: bool = False,
        q_action_gradient_noise_scale: float = 0.03,
        q_action_gradient_draws: int = 4,
        q_action_gradient_steps: str = "35",
        q_action_gradient_request_indices: str = "1,3,5,7,9",
        q_action_gradient_seed: int = 2026082601,
        q_control_path: str | None = None,
        q_progress_guidance: bool = False,
        q_log_jepa_progress: bool = False,
        q_progress_goal_embedding: str | None = None,
        q_progress_jepa_checkpoint: str = "/luhongchao/shared/weights/vjepa2.1/vjepa2_1_vitG_384.pt",
        q_progress_jepa_config: str | None = None,
        q_progress_jepa_repo: str | None = None,
        q_progress_jepa_checkpoint_key: str = "target_encoder",
        q_progress_camera_key: str | None = None,
        q_progress_clip_frames: int = 1,
        q_progress_jepa_offload: bool = False,
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
        q_adaptive_cpb_budget_min: float = 0.02,
        q_adaptive_cpb_budget_max: float = 0.10,
        q_adaptive_cpb_cosine_tau: float = 0.3,
        q_adaptive_cpb_cosine_temperature: float = 5.0,
        q_adaptive_cpb_cosine_pool: str | None = None,
        q_adaptive_cpb_cosine_pool_offset: int = 0,
        q_adaptive_cpb_gate_steps: str = "25,30,35,40",
        q_adaptive_cpb_q_relative_low: float = 0.30,
        q_adaptive_cpb_q_relative_high: float = 0.55,
        q_adaptive_cpb_grad_cosine_low: float = -0.15,
        q_adaptive_cpb_grad_cosine_high: float = 0.15,
        q_adaptive_cpb_delta_relative_margin: float = 0.002,
        q_adaptive_cpb_effect_consistency_floor: float = 0.75,
        q_adaptive_cpb_smooth_probe_budget: float = 0.08,
        q_adaptive_cpb_delta_gain_center: float = 14.0,
        q_adaptive_cpb_delta_gain_scale: float = 6.0,
        q_adaptive_cpb_delta_score_softness: float = 0.20,
        q_adaptive_cpb_delta_logit_strength: float = 0.60,
        q_adaptive_cpb_request_center: float = 1.75,
        q_adaptive_cpb_request_softness: float = 0.35,
        q_adaptive_cpb_step_center: float = 32.5,
        q_adaptive_cpb_step_softness: float = 2.5,
        q_adaptive_cpb_q_penalty_strength: float = 0.60,
        q_adaptive_cpb_probe_subdivisions: int = 8,
        q_adaptive_cpb_probe_score_tolerance: float = 1e-5,
        q_adaptive_cpb_probe_noise_multiplier: float = 1.0,
        q_adaptive_cpb_probe_fixed_budget: float = 0.08,
        q_adaptive_cpb_trust_warmup_requests: int = 2,
        q_adaptive_cpb_trust_target_ratio: float = 0.0,
        q_adaptive_cpb_trust_feedback_gain: float = 0.5,
        q_adaptive_cpb_trust_log_deadband: float = 0.15,
        q_adaptive_cpb_trust_max_log_step: float = 0.22314355131420976,
        q_adaptive_cpb_trust_reliability_floor: float = 0.45,
        q_adaptive_cpb_trust_hysteresis_requests: int = 2,
        q_adaptive_cpb_collapse_reference_step: int = 35,
        q_adaptive_cpb_collapse_threshold: float = 0.35,
        q_adaptive_cpb_collapse_patience: int = 2,
        q_adaptive_cpb_recovery_threshold: float = 0.60,
        q_uncertainty_cpb_steps: str = "25,30,35,40",
        q_uncertainty_cpb_samples: int = 4,
        q_uncertainty_cpb_noise_scale: float = 0.03,
        q_uncertainty_cpb_uncertainty_max: float | None = None,
        q_uncertainty_cpb_uncertainty_gradient_ratio: float = 0.25,
        q_uncertainty_cpb_trust_penalty: float = 0.1,
        q_uncertainty_cpb_barrier_strength: float = 0.01,
        q_uncertainty_cpb_gradient_cosine_min: float = 0.0,
        q_uncertainty_cpb_trust_radius: float = 0.10,
        q_uncertainty_cpb_action_bound: float = 1.0,
        q_uncertainty_cpb_grad_norm_max: float = 1.0,
        q_uncertainty_cpb_ascent_steps: int = 1,
        q_uncertainty_cpb_backtracks: int = 3,
        q_uncertainty_cpb_backtrack_factor: float = 0.5,
        q_uncertainty_cpb_min_score_improvement: float = 0.0,
        q_uncertainty_cpb_score_tolerance: float = 1e-5,
        q_uncertainty_cpb_smooth_abs_eps: float = 1e-4,
        q_uncertainty_cpb_seed: int = 2026082701,
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
            "adaptive_cpb_gradcos",
            "adaptive_cpb_gradcos_centered",
            "adaptive_cpb_gradcos_centered_fixed_floor",
            "adaptive_cpb_gradcos_qdelta_gated",
            "adaptive_cpb_gradcos_qdelta_smooth",
            "adaptive_cpb_gradcos_multibudget_probe",
            "adaptive_cpb_gradcos_fixed_anchor_probe",
            "adaptive_cpb_gradcos_fixed_anchor_increase_only",
            "adaptive_cpb_gradcos_shuffled",
            "adaptive_cpb_trust_region",
            "local_uncertainty_cpb",
        }:
            raise ValueError(
                "q_guidance_weight_scheduler must be one of "
                "'auto', 'fixed', 'denoising_time', 'progress_decay', "
                "'exp_progress', 'cpb', 'fixed_cpb', "
                "'adaptive_cpb_gradcos', 'adaptive_cpb_gradcos_centered', "
                "'adaptive_cpb_gradcos_centered_fixed_floor', "
                "'adaptive_cpb_gradcos_qdelta_gated', "
                "'adaptive_cpb_gradcos_qdelta_smooth', "
                "'adaptive_cpb_gradcos_multibudget_probe', "
                "'adaptive_cpb_gradcos_fixed_anchor_probe', "
                "'adaptive_cpb_gradcos_fixed_anchor_increase_only', or "
                "'adaptive_cpb_gradcos_shuffled', or "
                "'adaptive_cpb_trust_region', or 'local_uncertainty_cpb', "
                f"got {q_guidance_weight_scheduler!r}"
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
        self.q_replay_noise_seed = (
            None if q_replay_noise_seed is None else int(q_replay_noise_seed)
        )
        if self.q_replay_noise_seed is not None and self.q_replay_noise_seed < 0:
            raise ValueError("q_replay_noise_seed must be non-negative")
        self.q_log_replay_noise_hashes = bool(q_log_replay_noise_hashes)
        if self.q_log_replay_noise_hashes and self.q_replay_noise_seed is None:
            raise ValueError(
                "q_log_replay_noise_hashes requires q_replay_noise_seed"
            )
        self._q_replay_episode_keys: tuple[int, ...] | None = None
        if q_replay_manifest_path is not None:
            self._q_replay_episode_keys = load_replay_episode_keys(
                q_replay_manifest_path
            )
        if self._q_replay_episode_keys is not None and self.q_replay_noise_seed is None:
            raise ValueError("q_replay_manifest_path requires q_replay_noise_seed")
        self.q_log_local_guidance_delta = bool(q_log_local_guidance_delta)
        self.q_log_second_order_cosine = bool(q_log_second_order_cosine)
        self.q_second_order_finite_difference_epsilon = float(
            q_second_order_finite_difference_epsilon
        )
        if self.q_second_order_finite_difference_epsilon <= 0.0:
            raise ValueError(
                "q_second_order_finite_difference_epsilon must be positive"
            )
        try:
            self.q_second_order_observation_steps = {
                int(value.strip())
                for value in q_second_order_observation_steps.split(",")
                if value.strip()
            }
        except ValueError as error:
            raise ValueError(
                "q_second_order_observation_steps must be comma-separated integers"
            ) from error
        if not self.q_second_order_observation_steps:
            raise ValueError("q_second_order_observation_steps must not be empty")
        self.q_log_observation_noise_sensitivity = bool(
            q_log_observation_noise_sensitivity
        )
        self.q_observation_noise_levels = parse_positive_float_csv(
            q_observation_noise_levels,
            name="q_observation_noise_levels",
        )
        if q_observation_noise_draws <= 0:
            raise ValueError("q_observation_noise_draws must be positive")
        self.q_observation_noise_draws = int(q_observation_noise_draws)
        self.q_observation_noise_steps = parse_nonnegative_int_csv(
            q_observation_noise_steps,
            name="q_observation_noise_steps",
        )
        self.q_observation_noise_request_indices = parse_nonnegative_int_csv(
            q_observation_noise_request_indices,
            name="q_observation_noise_request_indices",
        )
        if q_observation_noise_seed < 0:
            raise ValueError("q_observation_noise_seed must be non-negative")
        self.q_observation_noise_seed = int(q_observation_noise_seed)
        self.q_log_action_gradient_distribution = bool(
            q_log_action_gradient_distribution
        )
        self.q_action_gradient_noise_scale = float(q_action_gradient_noise_scale)
        if self.q_action_gradient_noise_scale <= 0.0:
            raise ValueError("q_action_gradient_noise_scale must be positive")
        if q_action_gradient_draws <= 0:
            raise ValueError("q_action_gradient_draws must be positive")
        self.q_action_gradient_draws = int(q_action_gradient_draws)
        self.q_action_gradient_steps = parse_nonnegative_int_csv(
            q_action_gradient_steps,
            name="q_action_gradient_steps",
        )
        self.q_action_gradient_request_indices = parse_nonnegative_int_csv(
            q_action_gradient_request_indices,
            name="q_action_gradient_request_indices",
        )
        if q_action_gradient_seed < 0:
            raise ValueError("q_action_gradient_seed must be non-negative")
        self.q_action_gradient_seed = int(q_action_gradient_seed)
        self.q_log_denoising_steps = bool(
            q_log_denoising_steps
            or self.q_log_local_guidance_delta
            or self.q_log_second_order_cosine
            or self.q_log_observation_noise_sensitivity
            or self.q_log_action_gradient_distribution
            or q_guidance_weight_scheduler
            in {
                "adaptive_cpb_gradcos_qdelta_gated",
                "adaptive_cpb_gradcos_qdelta_smooth",
                "adaptive_cpb_gradcos_multibudget_probe",
                "adaptive_cpb_gradcos_fixed_anchor_probe",
                "adaptive_cpb_gradcos_fixed_anchor_increase_only",
                "adaptive_cpb_trust_region",
                "local_uncertainty_cpb",
            }
        )
        self.q_summary_log_path: Path | None = None
        self.q_step_log_path: Path | None = None
        self.q_observation_noise_log_path: Path | None = None
        self.q_action_gradient_log_path: Path | None = None
        self.q_control_path = Path(q_control_path) if q_control_path else None
        self._q_control_mtime_ns: int | None = None
        self.q_progress_guidance = bool(q_progress_guidance)
        self.q_log_jepa_progress = bool(q_log_jepa_progress)
        self.q_progress_goal_embedding_path = (
            Path(q_progress_goal_embedding) if q_progress_goal_embedding else None
        )
        self.q_progress_jepa_checkpoint = q_progress_jepa_checkpoint
        self.q_progress_jepa_config = q_progress_jepa_config
        self.q_progress_jepa_repo = q_progress_jepa_repo
        self.q_progress_jepa_checkpoint_key = q_progress_jepa_checkpoint_key
        self.q_progress_camera_key = q_progress_camera_key
        self.q_progress_jepa_offload = bool(q_progress_jepa_offload)
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
        self.q_adaptive_cpb_budget_min = float(q_adaptive_cpb_budget_min)
        self.q_adaptive_cpb_budget_max = float(q_adaptive_cpb_budget_max)
        self.q_adaptive_cpb_cosine_tau = float(q_adaptive_cpb_cosine_tau)
        self.q_adaptive_cpb_cosine_temperature = float(
            q_adaptive_cpb_cosine_temperature
        )
        if not 0.0 <= self.q_adaptive_cpb_budget_min <= self.q_adaptive_cpb_budget_max:
            raise ValueError(
                "adaptive CPB budgets must satisfy 0 <= minimum <= maximum"
            )
        if self.q_adaptive_cpb_cosine_temperature <= 0.0:
            raise ValueError("adaptive CPB cosine temperature must be positive")
        try:
            self.q_adaptive_cpb_gate_steps = {
                int(value.strip())
                for value in q_adaptive_cpb_gate_steps.split(",")
                if value.strip()
            }
        except ValueError as error:
            raise ValueError(
                "q_adaptive_cpb_gate_steps must be comma-separated integers"
            ) from error
        if not self.q_adaptive_cpb_gate_steps:
            raise ValueError("q_adaptive_cpb_gate_steps must not be empty")
        self.q_adaptive_cpb_q_relative_low = float(q_adaptive_cpb_q_relative_low)
        self.q_adaptive_cpb_q_relative_high = float(q_adaptive_cpb_q_relative_high)
        self.q_adaptive_cpb_grad_cosine_low = float(
            q_adaptive_cpb_grad_cosine_low
        )
        self.q_adaptive_cpb_grad_cosine_high = float(
            q_adaptive_cpb_grad_cosine_high
        )
        self.q_adaptive_cpb_delta_relative_margin = float(
            q_adaptive_cpb_delta_relative_margin
        )
        self.q_adaptive_cpb_effect_consistency_floor = float(
            q_adaptive_cpb_effect_consistency_floor
        )
        self.q_adaptive_cpb_smooth_probe_budget = float(
            q_adaptive_cpb_smooth_probe_budget
        )
        if self.q_adaptive_cpb_smooth_probe_budget <= 0.0:
            raise ValueError("q_adaptive_cpb_smooth_probe_budget must be positive")
        self.q_adaptive_cpb_delta_gain_center = float(
            q_adaptive_cpb_delta_gain_center
        )
        self.q_adaptive_cpb_delta_gain_scale = float(q_adaptive_cpb_delta_gain_scale)
        self.q_adaptive_cpb_delta_score_softness = float(
            q_adaptive_cpb_delta_score_softness
        )
        self.q_adaptive_cpb_delta_logit_strength = float(
            q_adaptive_cpb_delta_logit_strength
        )
        self.q_adaptive_cpb_request_center = float(q_adaptive_cpb_request_center)
        self.q_adaptive_cpb_request_softness = float(
            q_adaptive_cpb_request_softness
        )
        self.q_adaptive_cpb_step_center = float(q_adaptive_cpb_step_center)
        self.q_adaptive_cpb_step_softness = float(q_adaptive_cpb_step_softness)
        self.q_adaptive_cpb_q_penalty_strength = float(
            q_adaptive_cpb_q_penalty_strength
        )
        self.q_adaptive_cpb_probe_subdivisions = int(
            q_adaptive_cpb_probe_subdivisions
        )
        self.q_adaptive_cpb_probe_score_tolerance = float(
            q_adaptive_cpb_probe_score_tolerance
        )
        self.q_adaptive_cpb_probe_noise_multiplier = float(
            q_adaptive_cpb_probe_noise_multiplier
        )
        self.q_adaptive_cpb_probe_fixed_budget = float(
            q_adaptive_cpb_probe_fixed_budget
        )
        if self.q_adaptive_cpb_probe_subdivisions <= 0:
            raise ValueError("q_adaptive_cpb_probe_subdivisions must be positive")
        if self.q_adaptive_cpb_probe_score_tolerance < 0.0:
            raise ValueError("q_adaptive_cpb_probe_score_tolerance must be non-negative")
        if self.q_adaptive_cpb_probe_noise_multiplier < 0.0:
            raise ValueError("q_adaptive_cpb_probe_noise_multiplier must be non-negative")
        if q_guidance_weight_scheduler in {
            "adaptive_cpb_gradcos_fixed_anchor_probe",
            "adaptive_cpb_gradcos_fixed_anchor_increase_only",
        }:
            build_fixed_anchor_probe_budgets(
                minimum_budget=self.q_adaptive_cpb_budget_min,
                maximum_budget=self.q_adaptive_cpb_budget_max,
                fixed_budget=self.q_adaptive_cpb_probe_fixed_budget,
                subdivisions=self.q_adaptive_cpb_probe_subdivisions,
            )
        self.q_adaptive_cpb_collapse_reference_step = int(
            q_adaptive_cpb_collapse_reference_step
        )
        if (
            self.q_adaptive_cpb_collapse_reference_step
            not in self.q_adaptive_cpb_gate_steps
        ):
            raise ValueError(
                "q_adaptive_cpb_collapse_reference_step must be included in "
                "q_adaptive_cpb_gate_steps"
            )
        self._q_relative_collapse_tracker = QRelativeCollapseTracker(
            reference_step=self.q_adaptive_cpb_collapse_reference_step,
            collapse_threshold=float(q_adaptive_cpb_collapse_threshold),
            collapse_patience=int(q_adaptive_cpb_collapse_patience),
            recovery_threshold=float(q_adaptive_cpb_recovery_threshold),
            eps=self.q_progress_eps,
        )
        # Validate all stateless gate bounds before loading the critic.
        adaptive_cpb_qdelta_safety_gate(
            q_relative=torch.ones(1),
            grad_cosine=torch.ones(1),
            delta_q1=torch.ones(1),
            delta_q2=torch.ones(1),
            q1_before=torch.ones(1),
            q2_before=torch.ones(1),
            q_relative_low=self.q_adaptive_cpb_q_relative_low,
            q_relative_high=self.q_adaptive_cpb_q_relative_high,
            grad_cosine_low=self.q_adaptive_cpb_grad_cosine_low,
            grad_cosine_high=self.q_adaptive_cpb_grad_cosine_high,
            delta_relative_margin=self.q_adaptive_cpb_delta_relative_margin,
            effect_consistency_floor=(
                self.q_adaptive_cpb_effect_consistency_floor
            ),
            eps=self.q_progress_eps,
        )
        adaptive_cpb_qdelta_smooth_budget(
            candidate_budget=torch.full((1,), 0.5 * (
                self.q_adaptive_cpb_budget_min + self.q_adaptive_cpb_budget_max
            )),
            delta_q1=torch.ones(1),
            delta_q2=torch.ones(1),
            q1_before=torch.ones(1),
            q2_before=torch.ones(1),
            action_delta_rms=torch.ones(1),
            q_relative=torch.ones(1),
            request_index=2,
            denoising_step=35,
            minimum_budget=self.q_adaptive_cpb_budget_min,
            maximum_budget=self.q_adaptive_cpb_budget_max,
            delta_gain_center=self.q_adaptive_cpb_delta_gain_center,
            delta_gain_scale=self.q_adaptive_cpb_delta_gain_scale,
            delta_score_softness=self.q_adaptive_cpb_delta_score_softness,
            delta_logit_strength=self.q_adaptive_cpb_delta_logit_strength,
            request_center=self.q_adaptive_cpb_request_center,
            request_softness=self.q_adaptive_cpb_request_softness,
            step_center=self.q_adaptive_cpb_step_center,
            step_softness=self.q_adaptive_cpb_step_softness,
            q_relative_low=self.q_adaptive_cpb_q_relative_low,
            q_relative_high=self.q_adaptive_cpb_q_relative_high,
            q_penalty_strength=self.q_adaptive_cpb_q_penalty_strength,
            eps=self.q_progress_eps,
        )
        select_multibudget_probe(
            q1_candidates=torch.tensor([[0.0, 1.0]]),
            q2_candidates=torch.tensor([[0.0, 1.0]]),
            candidate_lambdas=torch.tensor([0.0, 1.0]),
            q1_zero_repeat=torch.zeros(1),
            q2_zero_repeat=torch.zeros(1),
            score_tolerance=self.q_adaptive_cpb_probe_score_tolerance,
            noise_multiplier=self.q_adaptive_cpb_probe_noise_multiplier,
        )
        select_fixed_anchor_probe(
            q1_candidates=torch.tensor([[0.0, 1.0, 0.0]]),
            q2_candidates=torch.tensor([[0.0, 1.0, 0.0]]),
            candidate_budgets=torch.tensor([0.04, 0.08, 0.12]),
            fixed_index=1,
            q1_fixed_repeat=torch.ones(1),
            q2_fixed_repeat=torch.ones(1),
            score_tolerance=self.q_adaptive_cpb_probe_score_tolerance,
            noise_multiplier=self.q_adaptive_cpb_probe_noise_multiplier,
        )
        self.q_uncertainty_cpb_steps = parse_nonnegative_int_csv(
            q_uncertainty_cpb_steps,
            name="q_uncertainty_cpb_steps",
        )
        self.q_uncertainty_cpb_samples = int(q_uncertainty_cpb_samples)
        if (
            self.q_uncertainty_cpb_samples <= 0
            or self.q_uncertainty_cpb_samples % 2 != 0
        ):
            raise ValueError("q_uncertainty_cpb_samples must be a positive even integer")
        self.q_uncertainty_cpb_noise_scale = float(q_uncertainty_cpb_noise_scale)
        self.q_uncertainty_cpb_uncertainty_max = (
            None
            if q_uncertainty_cpb_uncertainty_max is None
            else float(q_uncertainty_cpb_uncertainty_max)
        )
        self.q_uncertainty_cpb_uncertainty_gradient_ratio = float(
            q_uncertainty_cpb_uncertainty_gradient_ratio
        )
        self.q_uncertainty_cpb_trust_penalty = float(q_uncertainty_cpb_trust_penalty)
        self.q_uncertainty_cpb_barrier_strength = float(
            q_uncertainty_cpb_barrier_strength
        )
        self.q_uncertainty_cpb_gradient_cosine_min = float(
            q_uncertainty_cpb_gradient_cosine_min
        )
        self.q_uncertainty_cpb_trust_radius = float(q_uncertainty_cpb_trust_radius)
        self.q_uncertainty_cpb_action_bound = float(q_uncertainty_cpb_action_bound)
        self.q_uncertainty_cpb_grad_norm_max = float(
            q_uncertainty_cpb_grad_norm_max
        )
        self.q_uncertainty_cpb_ascent_steps = int(
            q_uncertainty_cpb_ascent_steps
        )
        self.q_uncertainty_cpb_backtracks = int(q_uncertainty_cpb_backtracks)
        self.q_uncertainty_cpb_backtrack_factor = float(
            q_uncertainty_cpb_backtrack_factor
        )
        self.q_uncertainty_cpb_min_score_improvement = float(
            q_uncertainty_cpb_min_score_improvement
        )
        self.q_uncertainty_cpb_score_tolerance = float(
            q_uncertainty_cpb_score_tolerance
        )
        self.q_uncertainty_cpb_smooth_abs_eps = float(
            q_uncertainty_cpb_smooth_abs_eps
        )
        self.q_uncertainty_cpb_seed = int(q_uncertainty_cpb_seed)
        if self.q_uncertainty_cpb_noise_scale <= 0.0:
            raise ValueError("q_uncertainty_cpb_noise_scale must be positive")
        if self.q_guidance_weight_scheduler == "local_uncertainty_cpb":
            if self.q_objective != "mean":
                raise ValueError(
                    "local_uncertainty_cpb requires q_objective='mean' to avoid "
                    "double-counting twin disagreement"
                )
            if (
                self.q_uncertainty_cpb_uncertainty_max is None
                or self.q_uncertainty_cpb_uncertainty_max <= 0.0
            ):
                raise ValueError(
                    "local_uncertainty_cpb requires a positive, calibrated "
                    "q_uncertainty_cpb_uncertainty_max"
                )
        if not 0.0 <= self.q_uncertainty_cpb_uncertainty_gradient_ratio <= 1.0:
            raise ValueError(
                "q_uncertainty_cpb_uncertainty_gradient_ratio must lie in [0, 1]"
            )
        if self.q_uncertainty_cpb_trust_penalty < 0.0:
            raise ValueError("q_uncertainty_cpb_trust_penalty must be non-negative")
        if self.q_uncertainty_cpb_barrier_strength < 0.0:
            raise ValueError("q_uncertainty_cpb_barrier_strength must be non-negative")
        if not -1.0 <= self.q_uncertainty_cpb_gradient_cosine_min <= 1.0:
            raise ValueError(
                "q_uncertainty_cpb_gradient_cosine_min must lie in [-1, 1]"
            )
        if self.q_uncertainty_cpb_trust_radius <= 0.0:
            raise ValueError("q_uncertainty_cpb_trust_radius must be positive")
        if self.q_uncertainty_cpb_action_bound <= 0.0:
            raise ValueError("q_uncertainty_cpb_action_bound must be positive")
        if self.q_uncertainty_cpb_grad_norm_max <= 0.0:
            raise ValueError("q_uncertainty_cpb_grad_norm_max must be positive")
        if self.q_uncertainty_cpb_ascent_steps != 1:
            raise ValueError(
                "this Q-VGM inference path performs one ascent update per selected "
                "denoising step, so proposal parameter J must be 1"
            )
        if self.q_uncertainty_cpb_backtracks < 0:
            raise ValueError("q_uncertainty_cpb_backtracks must be non-negative")
        if not 0.0 < self.q_uncertainty_cpb_backtrack_factor < 1.0:
            raise ValueError("q_uncertainty_cpb_backtrack_factor must lie in (0, 1)")
        if self.q_uncertainty_cpb_seed < 0:
            raise ValueError("q_uncertainty_cpb_seed must be non-negative")
        if self.q_uncertainty_cpb_smooth_abs_eps <= 0.0:
            raise ValueError("q_uncertainty_cpb_smooth_abs_eps must be positive")
        self._adaptive_cpb_cosine_pool: torch.Tensor | None = None
        self._adaptive_cpb_cosine_pool_index = int(q_adaptive_cpb_cosine_pool_offset)
        if q_adaptive_cpb_cosine_pool:
            pool_payload = json.loads(Path(q_adaptive_cpb_cosine_pool).read_text())
            if isinstance(pool_payload, dict):
                pool_payload = pool_payload.get("cosines")
            if not isinstance(pool_payload, list) or not pool_payload:
                raise ValueError(
                    "adaptive CPB cosine pool must be a non-empty JSON list or "
                    "an object with a non-empty 'cosines' list"
                )
            pool = torch.tensor(pool_payload, dtype=torch.float32)
            if not bool(torch.isfinite(pool).all()):
                raise ValueError("adaptive CPB cosine pool must contain finite values")
            self._adaptive_cpb_cosine_pool = pool.clamp(-1.0, 1.0)
            self._adaptive_cpb_cosine_pool_index %= len(pool)
        if (
            q_guidance_weight_scheduler == "adaptive_cpb_gradcos_shuffled"
            and self._adaptive_cpb_cosine_pool is None
        ):
            raise ValueError(
                "adaptive_cpb_gradcos_shuffled requires "
                "--q_adaptive_cpb_cosine_pool"
            )
        cpb_enabled = q_guidance_weight_scheduler == "cpb" or (
            q_guidance_weight_scheduler == "auto" and q_progress_scheduler == "cpb"
        )
        if q_guidance_weight_scheduler in {
            "fixed_cpb",
            "adaptive_cpb_gradcos_centered_fixed_floor",
            "adaptive_cpb_trust_region",
            "local_uncertainty_cpb",
        }:
            if self.q_cpb_fixed_budget is None:
                raise ValueError(
                    f"{q_guidance_weight_scheduler} requires --q_cpb_fixed_budget"
                )
        if (
            q_guidance_weight_scheduler
            in {
                "adaptive_cpb_gradcos_centered_fixed_floor",
                "adaptive_cpb_trust_region",
            }
            and not self.q_adaptive_cpb_budget_min
            <= float(self.q_cpb_fixed_budget)
            <= self.q_adaptive_cpb_budget_max
        ):
            raise ValueError(
                "fixed anchor must lie within the adaptive budget bounds"
            )
        self._adaptive_cpb_trust_region_controller = None
        self._adaptive_cpb_trust_region_previous_grad: torch.Tensor | None = None
        self._adaptive_cpb_trust_region_request_record: dict[str, object] = {}
        if q_guidance_weight_scheduler == "adaptive_cpb_trust_region":
            target_ratio = float(q_adaptive_cpb_trust_target_ratio)
            self._adaptive_cpb_trust_region_controller = (
                AdaptiveCPBTrustRegionController(
                    AdaptiveCPBTrustRegionConfig(
                        minimum_budget=self.q_adaptive_cpb_budget_min,
                        maximum_budget=self.q_adaptive_cpb_budget_max,
                        anchor_budget=float(self.q_cpb_fixed_budget),
                        warmup_requests=int(
                            q_adaptive_cpb_trust_warmup_requests
                        ),
                        target_intervention_ratio=(
                            None if target_ratio == 0.0 else target_ratio
                        ),
                        feedback_gain=float(
                            q_adaptive_cpb_trust_feedback_gain
                        ),
                        log_deadband=float(
                            q_adaptive_cpb_trust_log_deadband
                        ),
                        max_log_step=float(
                            q_adaptive_cpb_trust_max_log_step
                        ),
                        reliability_floor=float(
                            q_adaptive_cpb_trust_reliability_floor
                        ),
                        hysteresis_requests=int(
                            q_adaptive_cpb_trust_hysteresis_requests
                        ),
                        eps=self.q_progress_eps,
                    )
                )
            )
        if cpb_enabled:
            required = {
                "q_cpb_success_reference": self.q_cpb_success_reference,
                "q_cpb_fixed_budget": self.q_cpb_fixed_budget,
                "q_cpb_budget_min": self.q_cpb_budget_min,
                "q_cpb_budget_max": self.q_cpb_budget_max,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise ValueError(f"CPB requires explicit values for {missing}")
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
        self._progress_raw_value = torch.tensor(0.0, device=self.device)
        self._cpb_progress_value = torch.tensor(0.0, device=self.device)
        self._cpb_confidence_value = torch.tensor(0.0, device=self.device)
        self._progress_d_init: torch.Tensor | None = None
        self._progress_d_curr: torch.Tensor | None = None
        self._progress_d_ema: torch.Tensor | None = None
        self._progress_prev_value: torch.Tensor | None = None
        self._progress_prev_raw_value: torch.Tensor | None = None
        self._progress_d_prev: torch.Tensor | None = None
        self._progress_d_delta: torch.Tensor | None = None
        self._progress_delta: torch.Tensor | None = None
        self._progress_delta_raw: torch.Tensor | None = None
        self.q_artifact = load_q_guidance_artifact(q_checkpoint, self.device)
        self.q_artifact.adapter.eval().requires_grad_(False)
        self._q_previous_latents: torch.Tensor | None = None
        self._q_previous_actions: torch.Tensor | None = None
        self._q_previous_latent_mask: torch.Tensor | None = None
        self._q_previous_action_mask: torch.Tensor | None = None
        self._last_predicted_actions: torch.Tensor | None = None
        self._last_predicted_latents: torch.Tensor | None = None
        self._q_action_request_index = -1
        self._q_replay_local_episode_index = -1
        self._q_replay_episode_index = -1
        self._init_progress_guidance()
        self._validate_q_artifact()
        self._log_loaded_q_artifact()
        self._init_q_control_file()

    def _reset(self, prompt=None):
        super()._reset(prompt=prompt)
        self._q_replay_local_episode_index += 1
        if self._q_replay_episode_keys is None:
            self._q_replay_episode_index = self._q_replay_local_episode_index
        else:
            if self._q_replay_local_episode_index >= len(self._q_replay_episode_keys):
                raise RuntimeError(
                    "Server received more episode resets than replay manifest rows: "
                    f"reset={self._q_replay_local_episode_index + 1} "
                    f"rows={len(self._q_replay_episode_keys)}"
                )
            self._q_replay_episode_index = self._q_replay_episode_keys[
                self._q_replay_local_episode_index
            ]
        self._q_previous_latents = None
        self._q_previous_actions = None
        self._q_previous_latent_mask = None
        self._q_previous_action_mask = None
        self._last_predicted_actions = None
        self._last_predicted_latents = None
        self._q_action_request_index = -1
        if self._adaptive_cpb_trust_region_controller is not None:
            self._adaptive_cpb_trust_region_controller.reset()
        self._adaptive_cpb_trust_region_previous_grad = None
        self._adaptive_cpb_trust_region_request_record = {}
        self._progress_init_embedding = None
        self._progress_current_embedding = None
        self._progress_value = torch.tensor(0.0, device=self.device)
        self._progress_raw_value = torch.tensor(0.0, device=self.device)
        self._cpb_progress_value = torch.tensor(0.0, device=self.device)
        self._cpb_confidence_value = torch.tensor(0.0, device=self.device)
        self._progress_d_init = None
        self._progress_d_curr = None
        self._progress_d_ema = None
        self._progress_prev_value = None
        self._progress_prev_raw_value = None
        self._progress_d_prev = None
        self._progress_d_delta = None
        self._progress_delta = None
        self._progress_delta_raw = None
        self._q_relative_collapse_tracker.reset()
        self._init_state_normalization()
        self._init_q_logs()

    def _init_q_logs(self) -> None:
        log_dir = self.q_log_dir or Path(self.exp_save_root) / "q_guidance"
        log_dir.mkdir(parents=True, exist_ok=True)
        rank = int(getattr(self.job_config, "local_rank", 0))
        self.q_summary_log_path = log_dir / f"q_guidance_summary_rank{rank}.jsonl"
        self.q_step_log_path = log_dir / f"q_guidance_steps_rank{rank}.jsonl"
        self.q_observation_noise_log_path = (
            log_dir / f"q_observation_noise_rank{rank}.jsonl"
        )
        self.q_action_gradient_log_path = (
            log_dir / f"q_action_gradient_distribution_rank{rank}.jsonl"
        )

    def _init_progress_guidance(self) -> None:
        if not self._progress_observation_enabled:
            return
        if self.q_progress_goal_embedding_path is None:
            raise ValueError(
                "--q_progress_goal_embedding is required when "
                "--q_progress_guidance or --q_log_jepa_progress is enabled"
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

    @property
    def _progress_observation_enabled(self) -> bool:
        return self.q_progress_guidance or self.q_log_jepa_progress

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
        elif self.q_progress_jepa_offload:
            self._progress_jepa_encoder = self._progress_jepa_encoder.to(
                device=self.device,
                dtype=torch.float32,
            )
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
        try:
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
        finally:
            if self.q_progress_jepa_offload:
                encoder.to("cpu")
                torch.cuda.empty_cache()
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
        if (
            not self._progress_observation_enabled
            or self._progress_goal_embedding is None
        ):
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
        previous_progress = self._progress_value.detach().clone()
        previous_progress_raw = self._progress_raw_value.detach().clone()
        previous_distance = (
            self._progress_d_curr.detach().clone()
            if self._progress_d_curr is not None
            else None
        )
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
        self._progress_raw_value = (
            1.0
            - self._progress_d_curr
            / self._progress_d_init.clamp_min(self.q_progress_eps)
        )
        self._progress_value = self._progress_raw_value.clamp(0.0, 1.0)
        if previous_distance is None:
            self._progress_prev_value = None
            self._progress_prev_raw_value = None
            self._progress_d_prev = None
            self._progress_d_delta = None
            self._progress_delta = None
            self._progress_delta_raw = None
        else:
            self._progress_prev_value = previous_progress
            self._progress_prev_raw_value = previous_progress_raw
            self._progress_d_prev = previous_distance
            self._progress_d_delta = self._progress_d_curr - previous_distance
            self._progress_delta = self._progress_value - previous_progress
            self._progress_delta_raw = (
                self._progress_raw_value - previous_progress_raw
            )
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
        adaptive_confidence: torch.Tensor | None = None,
        adaptive_grad_cosine: torch.Tensor | None = None,
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
                "guidance_scale": float(self.q_guidance_scale),
                "legacy_step_scale": step_scale,
            }

        if scheduler in {"fixed_cpb", "local_uncertainty_cpb"}:
            if self.q_cpb_fixed_budget is None:
                raise RuntimeError(
                    f"{scheduler} requires --q_cpb_fixed_budget"
                )
            denoising_alpha, envelope_peak = self._cpb_step_alpha(step_index, sigma)
            task_budget = torch.as_tensor(
                float(self.q_cpb_fixed_budget),
                device=sigma.device,
                dtype=torch.float32,
            )
            weight = task_budget * denoising_alpha
            return weight, {
                "flow_time": (1.0 - sigma).clamp(0.0, 1.0),
                "guidance_weight_scheduler": scheduler,
                "guidance_scale": float(self.q_guidance_scale),
                "cpb_task_budget": task_budget,
                "cpb_fixed_budget": float(self.q_cpb_fixed_budget),
                "cpb_denoising_alpha": denoising_alpha,
                "cpb_envelope_peak_weight": envelope_peak,
                "cpb_sigma_peak": self.q_cpb_sigma_peak,
                "cpb_sigma_concentration": self.q_cpb_sigma_concentration,
                "progress_weight": weight,
                "fixed_budget_mode": scheduler == "fixed_cpb",
                "local_uncertainty_mode": scheduler == "local_uncertainty_cpb",
            }

        if scheduler == "adaptive_cpb_trust_region":
            controller = self._adaptive_cpb_trust_region_controller
            if controller is None:
                raise RuntimeError("adaptive CPB trust-region controller is unavailable")
            denoising_alpha, envelope_peak = self._cpb_step_alpha(step_index, sigma)
            task_budget = torch.as_tensor(
                controller.current_budget,
                device=sigma.device,
                dtype=torch.float32,
            )
            weight = task_budget * denoising_alpha
            return weight, {
                "flow_time": (1.0 - sigma).clamp(0.0, 1.0),
                "guidance_weight_scheduler": scheduler,
                "guidance_scale": float(self.q_guidance_scale),
                "cpb_task_budget": task_budget,
                "cpb_denoising_alpha": denoising_alpha,
                "cpb_envelope_peak_weight": envelope_peak,
                "cpb_sigma_peak": self.q_cpb_sigma_peak,
                "cpb_sigma_concentration": self.q_cpb_sigma_concentration,
                "progress_weight": weight,
                **self._adaptive_cpb_trust_region_request_record,
            }

        adaptive_schedulers = {
            "adaptive_cpb_gradcos",
            "adaptive_cpb_gradcos_centered",
            "adaptive_cpb_gradcos_centered_fixed_floor",
            "adaptive_cpb_gradcos_qdelta_gated",
            "adaptive_cpb_gradcos_qdelta_smooth",
            "adaptive_cpb_gradcos_multibudget_probe",
            "adaptive_cpb_gradcos_fixed_anchor_probe",
            "adaptive_cpb_gradcos_fixed_anchor_increase_only",
            "adaptive_cpb_gradcos_shuffled",
        }
        if scheduler in adaptive_schedulers:
            if adaptive_confidence is None or adaptive_grad_cosine is None:
                raise RuntimeError(
                    f"{scheduler} requires twin-Q gradient confidence"
                )
            confidence = adaptive_confidence.to(device=sigma.device, dtype=torch.float32)
            grad_cosine = adaptive_grad_cosine.to(device=sigma.device, dtype=torch.float32)
            budget_cosine = grad_cosine
            confidence_source = "twin_grad_cosine"
            pool_index = None
            if scheduler == "adaptive_cpb_gradcos_shuffled":
                if self._adaptive_cpb_cosine_pool is None:
                    raise RuntimeError("shuffled adaptive CPB cosine pool is unavailable")
                pool_index = self._adaptive_cpb_cosine_pool_index
                budget_cosine = self._adaptive_cpb_cosine_pool[pool_index].to(sigma)
                self._adaptive_cpb_cosine_pool_index = (
                    pool_index + 1
                ) % len(self._adaptive_cpb_cosine_pool)
                confidence_source = "shuffled_empirical_cosine_pool"

            if scheduler == "adaptive_cpb_gradcos":
                task_budget = adaptive_cpb_budget(
                    confidence,
                    minimum_budget=self.q_adaptive_cpb_budget_min,
                    maximum_budget=self.q_adaptive_cpb_budget_max,
                ).to(sigma)
            else:
                unfloored_task_budget = centered_adaptive_cpb_budget(
                    budget_cosine,
                    minimum_budget=self.q_adaptive_cpb_budget_min,
                    maximum_budget=self.q_adaptive_cpb_budget_max,
                    tau=self.q_adaptive_cpb_cosine_tau,
                    temperature=self.q_adaptive_cpb_cosine_temperature,
                ).to(sigma)
                if scheduler == "adaptive_cpb_gradcos_centered_fixed_floor":
                    task_budget = fixed_floor_centered_adaptive_cpb_budget(
                        budget_cosine,
                        minimum_budget=self.q_adaptive_cpb_budget_min,
                        maximum_budget=self.q_adaptive_cpb_budget_max,
                        fixed_budget=float(self.q_cpb_fixed_budget),
                        tau=self.q_adaptive_cpb_cosine_tau,
                        temperature=self.q_adaptive_cpb_cosine_temperature,
                    ).to(sigma)
                else:
                    task_budget = unfloored_task_budget
            denoising_alpha, envelope_peak = self._cpb_step_alpha(step_index, sigma)
            weight = task_budget * denoising_alpha
            return weight, {
                "flow_time": (1.0 - sigma).clamp(0.0, 1.0),
                "guidance_weight_scheduler": scheduler,
                "guidance_scale": float(self.q_guidance_scale),
                "adaptive_cpb_confidence_source": confidence_source,
                "adaptive_cpb_grad_cosine": grad_cosine,
                "adaptive_cpb_budget_cosine": budget_cosine,
                "adaptive_cpb_cosine_pool_index": pool_index,
                "adaptive_cpb_confidence": confidence,
                "adaptive_cpb_cosine_tau": self.q_adaptive_cpb_cosine_tau,
                "adaptive_cpb_cosine_temperature": self.q_adaptive_cpb_cosine_temperature,
                "adaptive_cpb_budget_min": self.q_adaptive_cpb_budget_min,
                "adaptive_cpb_budget_max": self.q_adaptive_cpb_budget_max,
                "adaptive_cpb_unfloored_task_budget": (
                    unfloored_task_budget
                    if scheduler != "adaptive_cpb_gradcos"
                    else None
                ),
                "adaptive_cpb_fixed_floor": (
                    float(self.q_cpb_fixed_budget)
                    if scheduler == "adaptive_cpb_gradcos_centered_fixed_floor"
                    else None
                ),
                "adaptive_cpb_floor_applied": (
                    task_budget > unfloored_task_budget
                    if scheduler == "adaptive_cpb_gradcos_centered_fixed_floor"
                    else None
                ),
                "cpb_task_budget": task_budget,
                "cpb_denoising_alpha": denoising_alpha,
                "cpb_envelope_peak_weight": envelope_peak,
                "cpb_sigma_peak": self.q_cpb_sigma_peak,
                "cpb_sigma_concentration": self.q_cpb_sigma_concentration,
                "progress_weight": weight,
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
                "progress_scheduler": scheduler,
                "cpb_task_budget": task_budget,
                "cpb_distance_budget": distance_budget,
                "cpb_fixed_budget": self.q_cpb_fixed_budget,
                "cpb_confidence": effective_confidence,
                "cpb_ood_confidence": ood_confidence,
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
                "adaptive_cpb_gradcos",
                "adaptive_cpb_gradcos_centered",
                "adaptive_cpb_gradcos_centered_fixed_floor",
                "adaptive_cpb_gradcos_qdelta_gated",
                "adaptive_cpb_gradcos_qdelta_smooth",
                "adaptive_cpb_gradcos_multibudget_probe",
                "adaptive_cpb_gradcos_fixed_anchor_probe",
                "adaptive_cpb_gradcos_fixed_anchor_increase_only",
                "adaptive_cpb_gradcos_shuffled",
                "adaptive_cpb_trust_region",
            }:
                raise ValueError(
                    "q_guidance_weight_scheduler must be one of "
                    "'auto', 'fixed', 'denoising_time', 'progress_decay', "
                    "'exp_progress', 'cpb', 'fixed_cpb', "
                    "'adaptive_cpb_gradcos', 'adaptive_cpb_gradcos_centered', "
                    "'adaptive_cpb_gradcos_centered_fixed_floor', "
                    "'adaptive_cpb_gradcos_qdelta_gated', "
                    "'adaptive_cpb_gradcos_qdelta_smooth', "
                    "'adaptive_cpb_gradcos_multibudget_probe', "
                    "'adaptive_cpb_gradcos_fixed_anchor_probe', "
                    "'adaptive_cpb_gradcos_fixed_anchor_increase_only', or "
                    "'adaptive_cpb_gradcos_shuffled', or "
                    "'adaptive_cpb_trust_region', "
                    f"got {scheduler!r}"
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
            target_batch = current_actions.shape[0]

            def match_batch(tensor: torch.Tensor) -> torch.Tensor:
                if tensor.shape[0] == target_batch:
                    return tensor
                if tensor.shape[0] != 1:
                    raise ValueError(
                        "Q history batch cannot be expanded from "
                        f"{tensor.shape[0]} to {target_batch}"
                    )
                return tensor.expand(target_batch, *tensor.shape[1:]).contiguous()

            previous_latents = match_batch(previous_latents)
            previous_actions = match_batch(previous_actions)
            previous_latent_mask = match_batch(previous_latent_mask)
            previous_action_mask = match_batch(previous_action_mask)
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
        target_batch = clean_action.shape[0]
        if text_emb.shape[0] != target_batch:
            if text_emb.shape[0] != 1:
                raise ValueError(
                    "prompt embedding batch cannot be expanded from "
                    f"{text_emb.shape[0]} to {target_batch}"
                )
            text_emb = text_emb.expand(
                target_batch,
                *text_emb.shape[1:],
            ).contiguous()
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
        layerwise_features = outputs[3]
        features = build_query_q_context(
            layerwise_features,
            previous_latent_mask=previous_latent_mask,
            previous_action_mask=previous_action_mask,
            current_action_mask=action_mask,
            chunk_size=config.infer_latent_chunk_size,
        )
        action_tokens = layerwise_features["action_tokens"]
        if action_tokens.ndim != 5:
            raise ValueError(
                "observation-noise probing requires layerwise action tokens "
                f"[B,L,F,T,D], got {tuple(action_tokens.shape)}"
            )
        current_action_token_count = (
            config.infer_latent_chunk_size * action_tokens.shape[3]
        )
        observation_token_count = (
            features["q_context_tokens"].shape[2] - current_action_token_count
        )
        if observation_token_count <= 0:
            raise ValueError("Q context has no history-video observation tokens")
        features["_q_observation_token_count"] = observation_token_count
        if config.critic_type == "twin_language_query_transformer_v1":
            if text_emb.ndim != 3 or text_emb.shape[-1] != 4096:
                raise ValueError(
                    "language Q guidance requires text embeddings with shape "
                    f"[B,L,4096], got {tuple(text_emb.shape)}"
                )
            text_mask = text_emb.float().abs().sum(dim=-1).gt(0)
            if not bool(text_mask.any(dim=1).all()):
                raise ValueError(
                    "language Q guidance received an empty instruction embedding"
                )
            features["text_tokens"] = text_emb
            features["text_mask"] = text_mask
        return features

    def _observation_noise_probe_selected(self, step_index: int) -> bool:
        return bool(
            self.q_log_observation_noise_sensitivity
            and step_index in self.q_observation_noise_steps
            and self._q_action_request_index
            in self.q_observation_noise_request_indices
        )

    def _run_observation_noise_probe(
        self,
        *,
        features: dict[str, torch.Tensor | int],
        action_mask: torch.Tensor,
        step_index: int,
    ) -> None:
        if not self._observation_noise_probe_selected(step_index):
            return
        request_index = int(self._q_action_request_index)
        episode_index = max(int(self._q_replay_episode_index), 0)
        probe_seed = (
            self.q_observation_noise_seed
            + episode_index * 10_000_019
            + request_index * 100_003
            + int(step_index) * 1_009
        )
        try:
            observation_token_count = int(features["_q_observation_token_count"])
            probe_features = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in features.items()
                if not key.startswith("_q_")
            }
            context_tokens = probe_features["q_context_tokens"]
            context_mask = probe_features["q_context_mask"]

            def q_values_fn(
                perturbed_context: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                candidate_features = dict(probe_features)
                candidate_features["q_context_tokens"] = perturbed_context
                return self.q_artifact.adapter.q_values(
                    candidate_features,
                    action_mask,
                )

            record = run_observation_noise_probe(
                context_tokens=context_tokens,
                context_mask=context_mask,
                observation_token_count=observation_token_count,
                q_values_fn=q_values_fn,
                objective_mode=self.q_objective,
                noise_levels=self.q_observation_noise_levels,
                draws=self.q_observation_noise_draws,
                seed=probe_seed,
                eps=self.q_progress_eps,
            )
            record.update(
                {
                    "event": "q_observation_noise_probe",
                    "episode_index": episode_index,
                    "action_request_index": request_index,
                    "frame_st_id": int(
                        getattr(self, "_current_frame_st_id", -1)
                    ),
                    "action_step": int(step_index),
                    "probe_observation_only": True,
                    "probe_changes_guidance": False,
                }
            )
        except Exception as error:
            record = {
                "event": "q_observation_noise_probe",
                "episode_index": episode_index,
                "action_request_index": request_index,
                "frame_st_id": int(getattr(self, "_current_frame_st_id", -1)),
                "action_step": int(step_index),
                "probe_observation_only": True,
                "probe_changes_guidance": False,
                "probe_error": repr(error),
            }
        self._write_q_record(self.q_observation_noise_log_path, record)

    def _action_gradient_distribution_probe_selected(
        self,
        step_index: int,
    ) -> bool:
        return bool(
            self.q_log_action_gradient_distribution
            and step_index in self.q_action_gradient_steps
            and self._q_action_request_index
            in self.q_action_gradient_request_indices
        )

    def _run_action_gradient_distribution_probe(
        self,
        *,
        latents: torch.Tensor,
        clean_action: torch.Tensor,
        action_mask: torch.Tensor,
        step_index: int,
    ) -> None:
        if not self._action_gradient_distribution_probe_selected(step_index):
            return
        request_index = int(self._q_action_request_index)
        episode_index = max(int(self._q_replay_episode_index), 0)
        probe_seed = (
            self.q_action_gradient_seed
            + episode_index * 10_000_019
            + request_index * 100_003
            + int(step_index) * 1_009
        )
        try:
            probe_latents = latents.detach()
            probe_action = clean_action.detach()

            def q_values_fn(
                candidate_action: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                candidate_features = self._extract_q_features(
                    latents=probe_latents,
                    clean_action=candidate_action,
                    action_mask=action_mask,
                )
                return self.q_artifact.adapter.q_values(
                    candidate_features,
                    action_mask,
                )

            record = run_action_gradient_distribution_probe(
                clean_action=probe_action,
                action_mask=action_mask,
                q_values_fn=q_values_fn,
                noise_scale=self.q_action_gradient_noise_scale,
                draws=self.q_action_gradient_draws,
                seed=probe_seed,
                eps=self.q_progress_eps,
            )
            record.update(
                {
                    "event": "q_action_gradient_distribution_probe",
                    "episode_index": episode_index,
                    "action_request_index": request_index,
                    "frame_st_id": int(getattr(self, "_current_frame_st_id", -1)),
                    "action_step": int(step_index),
                    "q_guidance_scale": self.q_guidance_scale,
                    "observation_only": True,
                }
            )
        except Exception as error:
            record = {
                "event": "q_action_gradient_distribution_probe",
                "episode_index": episode_index,
                "action_request_index": request_index,
                "frame_st_id": int(getattr(self, "_current_frame_st_id", -1)),
                "action_step": int(step_index),
                "q_guidance_scale": self.q_guidance_scale,
                "observation_only": True,
                "probe_changes_guidance": False,
                "probe_error": repr(error),
            }
        self._write_q_record(self.q_action_gradient_log_path, record)

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

    def _select_q_objective_gradient(
        self,
        q1: torch.Tensor,
        q2: torch.Tensor,
        grad_q1: torch.Tensor,
        grad_q2: torch.Tensor,
    ) -> torch.Tensor:
        """Reuse twin gradients for the configured scalar-per-sample objective."""

        if self.q_objective == "q1":
            return grad_q1
        if self.q_objective == "q2":
            return grad_q2
        if self.q_objective == "mean":
            return 0.5 * (grad_q1 + grad_q2)
        if self.q_objective != "min":
            raise ValueError(f"Unknown q_objective: {self.q_objective}")
        if q1.numel() != q1.shape[0] or q2.numel() != q2.shape[0]:
            raise ValueError(
                "gradient reuse requires one Q value per batch element, got "
                f"q1={tuple(q1.shape)} q2={tuple(q2.shape)}"
            )

        lower_q1 = q1 < q2
        equal = q1 == q2
        q1_weight = lower_q1.to(grad_q1.dtype) + 0.5 * equal.to(grad_q1.dtype)
        q2_weight = (q2 < q1).to(grad_q2.dtype) + 0.5 * equal.to(grad_q2.dtype)
        while q1_weight.ndim < grad_q1.ndim:
            q1_weight = q1_weight.unsqueeze(-1)
            q2_weight = q2_weight.unsqueeze(-1)
        return q1_weight * grad_q1 + q2_weight * grad_q2

    def _prepare_q_guidance_gradient(
        self,
        grad: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        grad = grad.detach().float() * action_mask.to(dtype=torch.float32)
        if self.q_grad_normalize:
            reduce_dims = tuple(range(1, grad.ndim))
            grad_norm = grad.square().sum(dim=reduce_dims, keepdim=True).sqrt()
            grad = grad / grad_norm.clamp_min(self.q_progress_eps)
        if self.q_grad_clip > 0:
            grad = grad.clamp(-self.q_grad_clip, self.q_grad_clip)
        return grad

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
            "q1": q1.detach().float().cpu().tolist(),
            "q2": q2.detach().float().cpu().tolist(),
            "q_min": torch.minimum(q1, q2).detach().float().cpu().tolist(),
            "q_objective": q_objective.detach().float().cpu().tolist(),
            "state_valid": state_valid,
            "action_cond_clamped": action_cond is not None,
        }
        if value is not None:
            record["value"] = value.cpu().tolist()
        if self._progress_observation_enabled:
            record["progress"] = (
                self._cpb_progress_value
                if self.q_guidance_weight_scheduler == "cpb"
                or (
                    self.q_guidance_weight_scheduler == "auto"
                    and self.q_progress_scheduler == "cpb"
                )
                else self._progress_value
            )
            record["progress_raw"] = self._progress_raw_value
            record["progress_d_init"] = self._progress_d_init
            record["progress_d_curr"] = self._progress_d_curr
            record["progress_d_ema"] = self._progress_d_ema
            record["progress_prev"] = self._progress_prev_value
            record["progress_prev_raw"] = self._progress_prev_raw_value
            record["progress_d_prev"] = self._progress_d_prev
            record["delta_distance"] = self._progress_d_delta
            record["delta_progress"] = self._progress_delta
            record["delta_progress_raw"] = self._progress_delta_raw
            record["jepa_progress_observation_only"] = (
                self.q_log_jepa_progress and not self.q_progress_guidance
            )
            record["cpb_confidence"] = self._cpb_confidence_value
        self._write_q_record(self.q_summary_log_path, record)

    def _local_q_delta_probe(
        self,
        *,
        latents: torch.Tensor,
        clean_action: torch.Tensor,
        action_cond: torch.Tensor | None,
        action_mask: torch.Tensor,
        sigma: torch.Tensor,
        candidate_guidance: torch.Tensor,
        q1_before: torch.Tensor,
        q2_before: torch.Tensor,
        q_objective_before: torch.Tensor,
    ) -> dict[str, object]:
        with (
            torch.no_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            sigma_view = sigma.view(1, 1, 1, 1, 1)
            local_action_before = clean_action.detach().float()
            # The velocity correction implies x0_after = x0_before + sigma*g.
            local_action_after = (
                local_action_before
                + sigma_view * candidate_guidance.detach().float()
            ).to(dtype=self.dtype)
            if action_cond is not None:
                local_action_after = local_action_after.clone()
                local_action_after[:, :, 0:1] = action_cond.to(dtype=self.dtype)
            local_after_features = self._extract_q_features(
                latents=latents,
                clean_action=local_action_after,
                action_mask=action_mask,
            )
            local_q1_after, local_q2_after = self.q_artifact.adapter.q_values(
                local_after_features,
                action_mask,
            )
            local_q_objective_after = self._select_q_objective(
                local_q1_after,
                local_q2_after,
            ).float()
            local_q1_after = local_q1_after.float()
            local_q2_after = local_q2_after.float()
            local_q_min_before = torch.minimum(q1_before, q2_before)
            local_q_min_after = torch.minimum(local_q1_after, local_q2_after)
            local_action_delta = (
                local_action_after.float() - local_action_before
            ) * action_mask.to(dtype=torch.float32)
            local_reduce_dims = tuple(range(1, local_action_delta.ndim))
            local_active = action_mask.to(dtype=torch.float32).sum(
                dim=local_reduce_dims
            ).clamp_min(1.0)
            local_delta_q1 = local_q1_after - q1_before
            local_delta_q2 = local_q2_after - q2_before
            local_delta_objective = (
                local_q_objective_after - q_objective_before
            )
            return {
                "local_q_delta_action_definition": (
                    "same_step_implied_clean_action_after_candidate_velocity_correction"
                ),
                "local_q1_before": q1_before,
                "local_q2_before": q2_before,
                "local_q_min_before": local_q_min_before,
                "local_q_objective_before": q_objective_before,
                "local_q1_after": local_q1_after,
                "local_q2_after": local_q2_after,
                "local_q_min_after": local_q_min_after,
                "local_q_objective_after": local_q_objective_after,
                "local_delta_q1": local_delta_q1,
                "local_delta_q2": local_delta_q2,
                "local_delta_q_min": local_q_min_after - local_q_min_before,
                "local_delta_q_objective": local_delta_objective,
                "local_both_q_improved": (
                    (local_delta_q1 > 0.0) & (local_delta_q2 > 0.0)
                ),
                "local_q_sign_disagreement": (
                    (local_delta_q1 > 0.0) != (local_delta_q2 > 0.0)
                ),
                "local_action_delta_l2": local_action_delta.square()
                .sum(dim=local_reduce_dims)
                .sqrt(),
                "local_action_delta_rms": (
                    local_action_delta.square().sum(dim=local_reduce_dims)
                    / local_active
                ).sqrt(),
            }

    def _local_uncertainty_cpb_probe(
        self,
        *,
        latents: torch.Tensor,
        clean_action: torch.Tensor,
        action_cond: torch.Tensor | None,
        action_mask: torch.Tensor,
        sigma: torch.Tensor,
        vanilla_grad: torch.Tensor,
        task_budget: torch.Tensor,
        denoising_alpha: torch.Tensor,
        step_index: int,
    ) -> dict[str, object]:
        if clean_action.shape[0] != 1:
            raise ValueError("local uncertainty CPB currently requires batch size one")
        if task_budget.numel() != 1 or denoising_alpha.numel() != 1:
            raise ValueError("local uncertainty CPB budget and alpha must be scalar")

        factors = torch.tensor(
            [0.0]
            + [
                self.q_uncertainty_cpb_backtrack_factor**index
                for index in range(self.q_uncertainty_cpb_backtracks + 1)
            ],
            device=clean_action.device,
            dtype=torch.float32,
        )
        candidate_count = factors.numel()
        samples = self.q_uncertainty_cpb_samples
        budget = task_budget.float().reshape(())
        alpha = denoising_alpha.float().reshape(())
        candidate_weights = factors * budget * alpha
        mask = action_mask.detach().bool()
        mask_float = mask.float()
        reduce_dims = tuple(range(1, clean_action.ndim))
        active = mask_float.sum(dim=reduce_dims).clamp_min(1.0)
        action_scale = torch.ones_like(clean_action, dtype=torch.float32)

        seed = (
            self.q_uncertainty_cpb_seed
            + max(0, self._q_action_request_index) * 1000003
            + int(step_index) * 9176
        )
        generator = torch.Generator(device=clean_action.device)
        generator.manual_seed(seed)
        directions: list[torch.Tensor] = []
        for _ in range(samples // 2):
            direction = torch.randn(
                clean_action.shape,
                generator=generator,
                device=clean_action.device,
                dtype=torch.float32,
            )
            if direction.ndim == 5 and direction.shape[2] >= 3:
                direction = F.avg_pool3d(
                    direction,
                    kernel_size=(3, 1, 1),
                    stride=1,
                    padding=(1, 0, 0),
                )
            direction = direction * mask_float
            direction_rms = (
                direction.square().sum(dim=reduce_dims) / active
            ).sqrt().clamp_min(self.q_progress_eps)
            direction = direction / direction_rms.view(
                clean_action.shape[0],
                *([1] * (clean_action.ndim - 1)),
            )
            directions.extend((-direction, direction))
        local_directions = torch.cat(directions, dim=0)
        local_deltas = (
            self.q_uncertainty_cpb_noise_scale
            * action_scale
            * local_directions
        )

        cuda_device = clean_action.device if clean_action.is_cuda else None
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            memory_before = torch.cuda.memory_allocated(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        else:
            memory_before = 0
        started_at = time.perf_counter()

        with (
            torch.enable_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            center_action = clean_action.detach().requires_grad_(True)
            center_features = self._extract_q_features(
                latents=latents.detach(),
                clean_action=center_action.to(dtype=self.dtype),
                action_mask=mask,
            )
            center_q1, center_q2 = self.q_artifact.adapter.q_values(
                center_features,
                mask,
            )
            center_q1 = center_q1.float().reshape(1, -1)
            center_q2 = center_q2.float().reshape(1, -1)
            if center_q1.shape[-1] != 1 or center_q2.shape[-1] != 1:
                raise ValueError(
                    "local uncertainty CPB expected one scalar from each critic "
                    f"at the center, got {tuple(center_q1.shape)} and "
                    f"{tuple(center_q2.shape)}"
                )
            grad_q1_center = torch.autograd.grad(
                center_q1[0, 0],
                center_action,
                retain_graph=True,
                create_graph=False,
                only_inputs=True,
            )[0].detach()
            grad_q2_center = torch.autograd.grad(
                center_q2[0, 0],
                center_action,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0].detach()

        del center_action, center_features, center_q1, center_q2
        if cuda_device is not None:
            torch.cuda.empty_cache()

        with (
            torch.enable_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            objective_center = clean_action.detach().requires_grad_(True)
            objective_q1_values: list[torch.Tensor] = []
            objective_q2_values: list[torch.Tensor] = []
            objective_grad_q1_samples: list[torch.Tensor] = []
            objective_grad_q2_samples: list[torch.Tensor] = []
            for sample_index in range(samples):
                evaluation_action = (
                    objective_center.float()
                    + local_deltas[sample_index : sample_index + 1].to(
                        objective_center
                    )
                ).clamp(
                    -self.q_uncertainty_cpb_action_bound,
                    self.q_uncertainty_cpb_action_bound,
                )
                if action_cond is not None:
                    evaluation_action = evaluation_action.clone()
                    evaluation_action[:, :, 0:1] = action_cond.to(
                        dtype=self.dtype
                    )
                evaluation_features = self._extract_q_features(
                    latents=latents.detach(),
                    clean_action=evaluation_action.to(dtype=self.dtype),
                    action_mask=mask,
                )
                sample_q1, sample_q2 = self.q_artifact.adapter.q_values(
                    evaluation_features,
                    mask,
                )
                sample_q1 = sample_q1.float().reshape(1, -1)
                sample_q2 = sample_q2.float().reshape(1, -1)
                if sample_q1.shape[-1] != 1 or sample_q2.shape[-1] != 1:
                    raise ValueError(
                        "local uncertainty CPB expected one scalar from each "
                        "critic per sample, got "
                        f"{tuple(sample_q1.shape)} and {tuple(sample_q2.shape)}"
                    )
                sample_grad_q1 = torch.autograd.grad(
                    sample_q1[0, 0],
                    objective_center,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                )[0].detach()
                sample_grad_q2 = torch.autograd.grad(
                    sample_q2[0, 0],
                    objective_center,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True,
                )[0].detach()
                objective_q1_values.append(sample_q1[0, 0].detach())
                objective_q2_values.append(sample_q2[0, 0].detach())
                objective_grad_q1_samples.append(sample_grad_q1)
                objective_grad_q2_samples.append(sample_grad_q2)
                del (
                    evaluation_action,
                    evaluation_features,
                    sample_q1,
                    sample_q2,
                    sample_grad_q1,
                    sample_grad_q2,
                )
                if cuda_device is not None:
                    torch.cuda.empty_cache()

            objective_q1, objective_q2 = first_order_local_q_surrogates(
                center_action=objective_center,
                q1_values=torch.stack(objective_q1_values),
                q2_values=torch.stack(objective_q2_values),
                grad_q1_samples=torch.stack(objective_grad_q1_samples),
                grad_q2_samples=torch.stack(objective_grad_q2_samples),
            )
            objective_gradient, objective_record = (
                local_uncertainty_ascent_direction(
                    center_action=objective_center,
                    frozen_base_action=clean_action.detach(),
                    q1_local=objective_q1,
                    q2_local=objective_q2,
                    grad_q1_center=grad_q1_center,
                    grad_q2_center=grad_q2_center,
                    action_mask=mask,
                    action_scale=action_scale,
                    uncertainty_max=self.q_uncertainty_cpb_uncertainty_max,
                    uncertainty_gradient_ratio=(
                        self.q_uncertainty_cpb_uncertainty_gradient_ratio
                    ),
                    trust_penalty=self.q_uncertainty_cpb_trust_penalty,
                    barrier_strength=self.q_uncertainty_cpb_barrier_strength,
                    gradient_cosine_min=(
                        self.q_uncertainty_cpb_gradient_cosine_min
                    ),
                    smooth_abs_eps=self.q_uncertainty_cpb_smooth_abs_eps,
                    eps=self.q_progress_eps,
                )
            )

        del (
            objective_center,
            objective_q1_values,
            objective_q2_values,
            objective_grad_q1_samples,
            objective_grad_q2_samples,
            objective_q1,
            objective_q2,
            grad_q1_center,
            grad_q2_center,
        )
        if cuda_device is not None:
            torch.cuda.empty_cache()

        prepared_gradient, raw_objective_gradient_l2, gradient_clip_scale = (
            clip_gradient_l2_norm(
                objective_gradient,
                action_mask,
                max_norm=self.q_uncertainty_cpb_grad_norm_max,
                eps=self.q_progress_eps,
            )
        )
        clipped_objective_gradient_l2 = prepared_gradient.square().sum(
            dim=reduce_dims
        ).sqrt()
        agreement_gate = objective_record["gradient_gate"].to(prepared_gradient)
        proposal_direction = prepared_gradient * agreement_gate
        direction_cosine_to_vanilla = F.cosine_similarity(
            proposal_direction.float().reshape(1, -1),
            vanilla_grad.detach().float().reshape(1, -1),
            dim=-1,
            eps=self.q_progress_eps,
        ).squeeze(0)

        with (
            torch.no_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            sigma_scalar = sigma.float().reshape(())
            if sigma_scalar <= self.q_progress_eps:
                raise ValueError("local uncertainty CPB requires positive sigma")
            candidate_guidance = candidate_weights.view(
                candidate_count, 1, 1, 1, 1
            ) * proposal_direction.float().expand(
                candidate_count, *proposal_direction.shape[1:]
            )
            frozen_base = clean_action.detach().float().expand(
                candidate_count, *clean_action.shape[1:]
            )
            nominal_centers = frozen_base + sigma_scalar * candidate_guidance
            bounded_centers = nominal_centers.clamp(
                -self.q_uncertainty_cpb_action_bound,
                self.q_uncertainty_cpb_action_bound,
            )
            candidate_centers = torch.where(
                (factors > 0.0).view(candidate_count, 1, 1, 1, 1),
                bounded_centers,
                frozen_base,
            ).to(dtype=self.dtype)
            if action_cond is not None:
                candidate_centers = candidate_centers.clone()
                candidate_centers[:, :, 0:1] = action_cond.to(
                    dtype=self.dtype
                ).expand(candidate_count, *action_cond.shape[1:])

            center_mask = mask.expand(
                candidate_count, *mask.shape[1:]
            ).contiguous()
            center_delta = (
                candidate_centers.float() - frozen_base
            ) * center_mask.float()
            nominal_step = (
                nominal_centers.float() - frozen_base
            ) * center_mask.float()
            nonzero_ascent_step = nominal_step.reshape(
                candidate_count, -1
            ).abs().amax(dim=-1) > self.q_progress_eps
            center_active = center_mask.float().sum(dim=reduce_dims).clamp_min(1.0)
            center_delta_rms = (
                center_delta.square().sum(dim=reduce_dims) / center_active
            ).sqrt()
            trust_distances = (
                (center_delta / action_scale.expand_as(center_delta)).square().sum(
                    dim=reduce_dims
                )
                / center_active
            )
            bounds_feasible = (
                candidate_centers.float().abs()
                <= self.q_uncertainty_cpb_action_bound + self.q_progress_eps
            ).logical_or(~center_mask).all(dim=reduce_dims)
            bounds_feasible = bounds_feasible & (
                (factors <= 0.0) | nonzero_ascent_step
            )
            bounds_clipped = (
                (bounded_centers - nominal_centers).abs() * center_mask.float()
            ).reshape(candidate_count, -1).amax(dim=-1) > self.q_progress_eps

            local_actions = (
                candidate_centers.float().unsqueeze(1)
                + local_deltas.to(candidate_centers).unsqueeze(0)
            ).clamp(
                -self.q_uncertainty_cpb_action_bound,
                self.q_uncertainty_cpb_action_bound,
            )
            if action_cond is not None:
                local_actions = local_actions.clone()
                local_actions[:, :, :, 0:1] = action_cond.to(
                    dtype=self.dtype
                ).reshape(1, 1, *action_cond.shape[1:])
            try:
                candidate_microbatch_size = max(
                    1,
                    int(os.environ.get("Q_UNCERTAINTY_CPB_MICROBATCH_SIZE", "1")),
                )
            except ValueError as error:
                raise ValueError(
                    "Q_UNCERTAINTY_CPB_MICROBATCH_SIZE must be an integer"
                ) from error
            local_q1_rows: list[torch.Tensor] = []
            local_q2_rows: list[torch.Tensor] = []
            for candidate_index in range(candidate_count):
                candidate_q1_samples: list[torch.Tensor] = []
                candidate_q2_samples: list[torch.Tensor] = []
                for sample_start in range(0, samples, candidate_microbatch_size):
                    sample_end = min(
                        samples,
                        sample_start + candidate_microbatch_size,
                    )
                    evaluation_action = local_actions[
                        candidate_index, sample_start:sample_end
                    ].to(dtype=self.dtype)
                    batch_size = evaluation_action.shape[0]
                    evaluation_latents = latents.detach()
                    evaluation_mask = mask
                    if evaluation_latents.shape[0] == 1 and batch_size != 1:
                        evaluation_latents = evaluation_latents.expand(
                            batch_size,
                            *evaluation_latents.shape[1:],
                        ).contiguous()
                    if evaluation_mask.shape[0] == 1 and batch_size != 1:
                        evaluation_mask = evaluation_mask.expand(
                            batch_size,
                            *evaluation_mask.shape[1:],
                        ).contiguous()
                    evaluation_features = self._extract_q_features(
                        latents=evaluation_latents,
                        clean_action=evaluation_action,
                        action_mask=evaluation_mask,
                    )
                    candidate_q1, candidate_q2 = (
                        self.q_artifact.adapter.q_values(
                            evaluation_features,
                            evaluation_mask,
                        )
                    )
                    candidate_q1 = candidate_q1.float().reshape(batch_size, -1)
                    candidate_q2 = candidate_q2.float().reshape(batch_size, -1)
                    if (
                        candidate_q1.shape[-1] != 1
                        or candidate_q2.shape[-1] != 1
                    ):
                        raise ValueError(
                            "local uncertainty CPB expected one scalar from each "
                            "critic per sample, got "
                            f"{tuple(candidate_q1.shape)} and "
                            f"{tuple(candidate_q2.shape)}"
                        )
                    candidate_q1_samples.extend(candidate_q1[:, 0].unbind(0))
                    candidate_q2_samples.extend(candidate_q2[:, 0].unbind(0))
                    del (
                        evaluation_action,
                        evaluation_latents,
                        evaluation_mask,
                        evaluation_features,
                        candidate_q1,
                        candidate_q2,
                    )
                    if cuda_device is not None:
                        torch.cuda.empty_cache()
                local_q1_rows.append(torch.stack(candidate_q1_samples))
                local_q2_rows.append(torch.stack(candidate_q2_samples))
            local_q1 = torch.stack(local_q1_rows, dim=0)
            local_q2 = torch.stack(local_q2_rows, dim=0)

        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
            memory_after = torch.cuda.memory_allocated(cuda_device)
        else:
            peak_allocated = 0
            memory_after = 0
        latency_ms = 1000.0 * (time.perf_counter() - started_at)

        selected_factor, selection = select_local_uncertainty_cpb(
            q1_local=local_q1,
            q2_local=local_q2,
            candidate_factors=factors,
            trust_distances=trust_distances,
            action_bounds_feasible=bounds_feasible,
            effective_lambda=objective_record["effective_lambda"],
            uncertainty_max=self.q_uncertainty_cpb_uncertainty_max,
            trust_penalty=self.q_uncertainty_cpb_trust_penalty,
            barrier_strength=self.q_uncertainty_cpb_barrier_strength,
            trust_radius=self.q_uncertainty_cpb_trust_radius,
            gradient_cosine=objective_record["gradient_cosine"],
            gradient_cosine_min=self.q_uncertainty_cpb_gradient_cosine_min,
            min_score_improvement=self.q_uncertainty_cpb_min_score_improvement,
            score_tolerance=self.q_uncertainty_cpb_score_tolerance,
            smooth_abs_eps=self.q_uncertainty_cpb_smooth_abs_eps,
            eps=self.q_progress_eps,
        )
        selected_factor = selected_factor.reshape(())
        selected_index = selection["selected_index"].reshape(()).long()
        selected_center = candidate_centers.index_select(0, selected_index.view(1))
        selected_center, selected_guidance = q_vgm_residual_velocity_target(
            selected_center,
            clean_action,
            sigma_scalar,
            eps=self.q_progress_eps,
        )
        selected_action_delta = (
            selected_center.float() - clean_action.detach().float()
        ) * mask_float
        selected_guidance = selected_guidance * mask_float
        selected_task_budget = budget * selected_factor
        selected_guidance_weight = selected_task_budget * alpha
        if not bool(objective_record["base_uncertainty_feasible"].item()):
            rejection_reason = "base_uncertainty_exceeds_calibrated_max"
        elif not bool(selection["gradient_feasible"].item()):
            rejection_reason = "gradient_agreement_below_threshold"
        elif bool(selection["failed_closed"].item()):
            rejection_reason = "no_feasible_robust_score_improvement"
        else:
            rejection_reason = "accepted"

        return {
            "_local_uncertainty_selected_guidance": selected_guidance,
            "_local_uncertainty_selected_direction": proposal_direction.detach(),
            "local_uncertainty_sampled": True,
            "local_uncertainty_schema_version": 2,
            "local_uncertainty_action_definition": (
                "stop_gradient_best_feasible_bounded_clean_action_target"
            ),
            "local_uncertainty_frozen_base_definition": (
                "A_base=x_minus_sigma_times_v_base_in_scheduler_convention"
            ),
            "local_uncertainty_residual_velocity_definition": (
                "h_Q=(A_Q-A_base)/sigma_with_sigma_equal_one_minus_flow_time"
            ),
            "local_uncertainty_stop_gradient_target": True,
            "local_uncertainty_noise_definition": (
                "antithetic_three_tap_temporally_smoothed_normalized_action_scale"
            ),
            "local_uncertainty_objective_definition": (
                "local_mean_q-lambda_paired_disagreement-rho_frozen_base_"
                "distance+mu_log_uncertainty_barrier"
            ),
            "local_uncertainty_lambda_definition": (
                "eta_times_grad_q_norm_over_grad_uncertainty_norm"
            ),
            "local_uncertainty_threshold_definition": (
                "externally_calibrated_absolute_paired_disagreement_quantile"
            ),
            "local_uncertainty_seed": seed,
            "local_uncertainty_samples": samples,
            "local_uncertainty_noise_scale": self.q_uncertainty_cpb_noise_scale,
            "local_uncertainty_uncertainty_max": (
                self.q_uncertainty_cpb_uncertainty_max
            ),
            "local_uncertainty_uncertainty_gradient_ratio": (
                self.q_uncertainty_cpb_uncertainty_gradient_ratio
            ),
            "local_uncertainty_trust_penalty": (
                self.q_uncertainty_cpb_trust_penalty
            ),
            "local_uncertainty_barrier_strength": (
                self.q_uncertainty_cpb_barrier_strength
            ),
            "local_uncertainty_gradient_cosine_min": (
                self.q_uncertainty_cpb_gradient_cosine_min
            ),
            "local_uncertainty_trust_radius": self.q_uncertainty_cpb_trust_radius,
            "local_uncertainty_action_bound": self.q_uncertainty_cpb_action_bound,
            "local_uncertainty_ascent_steps": self.q_uncertainty_cpb_ascent_steps,
            "local_uncertainty_grad_clip_definition": "masked_l2_norm_clip",
            "local_uncertainty_grad_norm_max": (
                self.q_uncertainty_cpb_grad_norm_max
            ),
            "local_uncertainty_grad_clip_scale": gradient_clip_scale.squeeze(),
            "local_uncertainty_raw_objective_gradient_l2": (
                raw_objective_gradient_l2.squeeze()
            ),
            "local_uncertainty_clipped_objective_gradient_l2": (
                clipped_objective_gradient_l2.squeeze()
            ),
            "local_uncertainty_candidate_count": candidate_count,
            "local_uncertainty_center_evaluation_batch_size": 1,
            "local_uncertainty_objective_evaluation_batch_size": 1,
            "local_uncertainty_objective_evaluation_count": samples,
            "local_uncertainty_candidate_evaluation_count": (
                candidate_count * samples
            ),
            "local_uncertainty_evaluation_batch_size": candidate_microbatch_size,
            "local_uncertainty_candidate_factors": factors,
            "local_uncertainty_candidate_task_budgets": factors * budget,
            "local_uncertainty_candidate_guidance_weights": candidate_weights,
            "local_uncertainty_q1": local_q1,
            "local_uncertainty_q2": local_q2,
            "local_uncertainty_center_delta_rms": center_delta_rms,
            "local_uncertainty_nonzero_ascent_step": nonzero_ascent_step,
            "local_uncertainty_bounds_clipped": bounds_clipped,
            "local_uncertainty_direction_cosine_to_vanilla_q": (
                direction_cosine_to_vanilla
            ),
            "local_uncertainty_selected_factor": selected_factor,
            "local_uncertainty_selected_index": selected_index,
            "local_uncertainty_selected_task_budget": selected_task_budget,
            "local_uncertainty_selected_guidance_weight": (
                selected_guidance_weight
            ),
            "local_uncertainty_selected_action_delta_rms": (
                selected_action_delta.square().sum().div(active).sqrt()
            ),
            "local_uncertainty_rejection_reason": rejection_reason,
            "local_uncertainty_latency_ms": latency_ms,
            "local_uncertainty_memory_before_bytes": memory_before,
            "local_uncertainty_memory_after_bytes": memory_after,
            "local_uncertainty_peak_allocated_bytes": peak_allocated,
            "local_uncertainty_peak_extra_allocated_bytes": max(
                0, peak_allocated - memory_before
            ),
            **{
                f"local_uncertainty_objective_{name}": value
                for name, value in objective_record.items()
            },
            **{
                f"local_uncertainty_{name}": value
                for name, value in selection.items()
            },
        }

    def _multibudget_cpb_probe(
        self,
        *,
        latents: torch.Tensor,
        clean_action: torch.Tensor,
        action_cond: torch.Tensor | None,
        action_mask: torch.Tensor,
        sigma: torch.Tensor,
        grad: torch.Tensor,
        envelope_task_budget: torch.Tensor,
        denoising_alpha: torch.Tensor,
        q1_before: torch.Tensor,
        q2_before: torch.Tensor,
    ) -> dict[str, object]:
        if clean_action.shape[0] != 1:
            raise ValueError("multi-budget CPB probe currently requires batch size one")
        if envelope_task_budget.numel() != 1 or denoising_alpha.numel() != 1:
            raise ValueError("multi-budget CPB envelope must be scalar")

        subdivisions = self.q_adaptive_cpb_probe_subdivisions
        lambdas = torch.linspace(
            0.0,
            1.0,
            subdivisions + 1,
            device=clean_action.device,
            dtype=torch.float32,
        )
        # The last evaluation repeats lambda=0 to estimate forward noise.
        evaluation_lambdas = torch.cat([lambdas, lambdas.new_zeros(1)])
        evaluation_count = evaluation_lambdas.numel()
        envelope_budget = envelope_task_budget.float().reshape(())
        alpha = denoising_alpha.float().reshape(())
        evaluation_weights = evaluation_lambdas * envelope_budget * alpha

        cuda_device = clean_action.device if clean_action.is_cuda else None
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            memory_before = torch.cuda.memory_allocated(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        else:
            memory_before = 0
        started_at = time.perf_counter()

        with (
            torch.no_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            sigma_view = sigma.float().reshape(1, 1, 1, 1, 1)
            direction = grad.detach().float().expand(
                evaluation_count,
                *grad.shape[1:],
            )
            candidate_guidance = evaluation_weights.view(
                evaluation_count,
                1,
                1,
                1,
                1,
            ) * direction
            candidate_actions = (
                clean_action.detach().float().expand(
                    evaluation_count,
                    *clean_action.shape[1:],
                )
                + sigma_view * candidate_guidance
            ).to(dtype=self.dtype)
            if action_cond is not None:
                candidate_actions = candidate_actions.clone()
                candidate_actions[:, :, 0:1] = action_cond.to(
                    dtype=self.dtype
                ).expand(evaluation_count, *action_cond.shape[1:])
            candidate_latents = latents.detach().expand(
                evaluation_count,
                *latents.shape[1:],
            ).contiguous()
            candidate_mask = action_mask.expand(
                evaluation_count,
                *action_mask.shape[1:],
            ).contiguous()
            candidate_features = self._extract_q_features(
                latents=candidate_latents,
                clean_action=candidate_actions,
                action_mask=candidate_mask,
            )
            probe_q1, probe_q2 = self.q_artifact.adapter.q_values(
                candidate_features,
                candidate_mask,
            )
            probe_q1 = probe_q1.float().reshape(evaluation_count, -1)
            probe_q2 = probe_q2.float().reshape(evaluation_count, -1)
            if probe_q1.shape[1] != 1 or probe_q2.shape[1] != 1:
                raise ValueError(
                    "multi-budget probe expected one scalar from each critic "
                    f"per candidate, got {tuple(probe_q1.shape)} and "
                    f"{tuple(probe_q2.shape)}"
                )
            probe_q1 = probe_q1[:, 0]
            probe_q2 = probe_q2[:, 0]

        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
            memory_after = torch.cuda.memory_allocated(cuda_device)
        else:
            peak_allocated = 0
            memory_after = 0
        latency_ms = 1000.0 * (time.perf_counter() - started_at)

        q1_curve = probe_q1[:-1].unsqueeze(0)
        q2_curve = probe_q2[:-1].unsqueeze(0)
        selected_lambda, selection = select_multibudget_probe(
            q1_candidates=q1_curve,
            q2_candidates=q2_curve,
            candidate_lambdas=lambdas,
            q1_zero_repeat=probe_q1[-1:].reshape(1),
            q2_zero_repeat=probe_q2[-1:].reshape(1),
            score_tolerance=self.q_adaptive_cpb_probe_score_tolerance,
            noise_multiplier=self.q_adaptive_cpb_probe_noise_multiplier,
        )
        selected_lambda = selected_lambda.reshape(())
        selected_task_budget = envelope_budget * selected_lambda
        selected_guidance_weight = selected_task_budget * alpha
        selected_index = selection["selected_index"].reshape(())
        global_index = selection["global_index"].reshape(())
        candidate_budgets = lambdas * envelope_budget

        return {
            "multibudget_probe_sampled": True,
            "multibudget_probe_selection_policy": (
                "global_peak_min_budget_tie_break"
            ),
            "multibudget_probe_action_definition": (
                "same_step_implied_clean_action_from_fixed_q_gradient_direction"
            ),
            "multibudget_probe_single_batched_forward": True,
            "multibudget_probe_subdivisions": subdivisions,
            "multibudget_probe_candidate_count": lambdas.numel(),
            "multibudget_probe_evaluation_batch_size": evaluation_count,
            "multibudget_probe_lambdas": lambdas,
            "multibudget_probe_task_budgets": candidate_budgets,
            "multibudget_probe_q1_curve": q1_curve.squeeze(0),
            "multibudget_probe_q2_curve": q2_curve.squeeze(0),
            "multibudget_probe_delta_q1_curve": selection[
                "delta_q1_curve"
            ].squeeze(0),
            "multibudget_probe_delta_q2_curve": selection[
                "delta_q2_curve"
            ].squeeze(0),
            "multibudget_probe_score_curve": selection[
                "robust_score_curve"
            ].squeeze(0),
            "multibudget_probe_q1_zero_repeat": probe_q1[-1],
            "multibudget_probe_q2_zero_repeat": probe_q2[-1],
            "multibudget_probe_zero_noise": selection[
                "zero_repeat_noise"
            ].reshape(()),
            "multibudget_probe_noise_threshold": selection[
                "noise_threshold"
            ].reshape(()),
            "multibudget_probe_selected_index": selected_index,
            "multibudget_probe_selected_lambda": selected_lambda,
            "multibudget_probe_selected_score": selection[
                "selected_score"
            ].reshape(()),
            "multibudget_probe_selected_task_budget": selected_task_budget,
            "multibudget_probe_selected_guidance_weight": (
                selected_guidance_weight
            ),
            "multibudget_probe_global_index": global_index,
            "multibudget_probe_global_lambda": lambdas[global_index],
            "multibudget_probe_global_score": selection[
                "global_score"
            ].reshape(()),
            "multibudget_probe_selected_zero": selection[
                "selected_zero"
            ].reshape(()),
            "multibudget_probe_noise_fallback": selection[
                "noise_fallback"
            ].reshape(()),
            "multibudget_probe_boundary_optimum": selection[
                "boundary_optimum"
            ].reshape(()),
            "multibudget_probe_first_peak_selected": selection[
                "first_peak_selected"
            ].reshape(()),
            "multibudget_probe_global_peak_selected": selection[
                "global_peak_selected"
            ].reshape(()),
            "multibudget_probe_far_global_peak_rejected": selection[
                "far_global_peak_rejected"
            ].reshape(()),
            "multibudget_probe_q1_gradient_baseline": q1_before,
            "multibudget_probe_q2_gradient_baseline": q2_before,
            "multibudget_probe_latency_ms": latency_ms,
            "multibudget_probe_memory_before_bytes": memory_before,
            "multibudget_probe_memory_after_bytes": memory_after,
            "multibudget_probe_peak_allocated_bytes": peak_allocated,
            "multibudget_probe_peak_extra_allocated_bytes": max(
                0,
                peak_allocated - memory_before,
            ),
        }

    def _fixed_anchor_cpb_probe(
        self,
        *,
        latents: torch.Tensor,
        clean_action: torch.Tensor,
        action_cond: torch.Tensor | None,
        action_mask: torch.Tensor,
        sigma: torch.Tensor,
        grad: torch.Tensor,
        centered_envelope_budget: torch.Tensor,
        denoising_alpha: torch.Tensor,
        q1_before: torch.Tensor,
        q2_before: torch.Tensor,
        increase_only: bool = False,
    ) -> dict[str, object]:
        if clean_action.shape[0] != 1:
            raise ValueError("fixed-anchor CPB probe currently requires batch size one")
        if centered_envelope_budget.numel() != 1 or denoising_alpha.numel() != 1:
            raise ValueError("fixed-anchor CPB inputs must be scalar")

        candidate_budgets, fixed_index = build_fixed_anchor_probe_budgets(
            minimum_budget=self.q_adaptive_cpb_budget_min,
            maximum_budget=self.q_adaptive_cpb_budget_max,
            fixed_budget=self.q_adaptive_cpb_probe_fixed_budget,
            subdivisions=self.q_adaptive_cpb_probe_subdivisions,
            device=clean_action.device,
        )
        fixed_budget = candidate_budgets[fixed_index]
        # The final item exactly repeats the fixed anchor to estimate forward noise.
        evaluation_budgets = torch.cat(
            [candidate_budgets, fixed_budget.reshape(1)]
        )
        evaluation_count = evaluation_budgets.numel()
        alpha = denoising_alpha.float().reshape(())

        cuda_device = clean_action.device if clean_action.is_cuda else None
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            memory_before = torch.cuda.memory_allocated(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        else:
            memory_before = 0
        started_at = time.perf_counter()

        with (
            torch.no_grad(),
            self._q_feature_extraction_context(),
            self._raw_flex_attention_context(),
        ):
            candidate_actions = cpb_implied_clean_actions(
                clean_action=clean_action,
                sigma=sigma,
                denoising_alpha=denoising_alpha,
                task_budgets=evaluation_budgets,
                q_gradient_direction=grad,
            ).to(dtype=self.dtype)
            if action_cond is not None:
                candidate_actions = candidate_actions.clone()
                candidate_actions[:, :, 0:1] = action_cond.to(
                    dtype=self.dtype
                ).expand(evaluation_count, *action_cond.shape[1:])
            candidate_latents = latents.detach().expand(
                evaluation_count,
                *latents.shape[1:],
            ).contiguous()
            candidate_mask = action_mask.expand(
                evaluation_count,
                *action_mask.shape[1:],
            ).contiguous()
            candidate_features = self._extract_q_features(
                latents=candidate_latents,
                clean_action=candidate_actions,
                action_mask=candidate_mask,
            )
            probe_q1, probe_q2 = self.q_artifact.adapter.q_values(
                candidate_features,
                candidate_mask,
            )
            probe_q1 = probe_q1.float().reshape(evaluation_count, -1)
            probe_q2 = probe_q2.float().reshape(evaluation_count, -1)
            if probe_q1.shape[1] != 1 or probe_q2.shape[1] != 1:
                raise ValueError(
                    "fixed-anchor probe expected one scalar from each critic "
                    f"per candidate, got {tuple(probe_q1.shape)} and "
                    f"{tuple(probe_q2.shape)}"
                )
            probe_q1 = probe_q1[:, 0]
            probe_q2 = probe_q2[:, 0]

        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            peak_allocated = torch.cuda.max_memory_allocated(cuda_device)
            memory_after = torch.cuda.memory_allocated(cuda_device)
        else:
            peak_allocated = 0
            memory_after = 0
        latency_ms = 1000.0 * (time.perf_counter() - started_at)

        q1_curve = probe_q1[:-1].unsqueeze(0)
        q2_curve = probe_q2[:-1].unsqueeze(0)
        selected_budget, selection = select_fixed_anchor_probe(
            q1_candidates=q1_curve,
            q2_candidates=q2_curve,
            candidate_budgets=candidate_budgets,
            fixed_index=fixed_index,
            q1_fixed_repeat=probe_q1[-1:].reshape(1),
            q2_fixed_repeat=probe_q2[-1:].reshape(1),
            score_tolerance=self.q_adaptive_cpb_probe_score_tolerance,
            noise_multiplier=self.q_adaptive_cpb_probe_noise_multiplier,
            allow_decrease=not increase_only,
        )
        selected_budget = selected_budget.reshape(())
        selected_guidance_weight = selected_budget * alpha
        selected_index = selection["selected_index"].reshape(())
        selected_direction = int(
            selection["selected_direction"].reshape(()).item()
        )
        direction_name = {
            -1: "decrease",
            0: "fixed",
            1: "increase",
        }[selected_direction]
        selection_policy = (
            "fixed_default_increase_only_global_peak_nearest_anchor_tie_break"
            if increase_only
            else "fixed_default_global_peak_nearest_anchor_tie_break"
        )

        return {
            "fixed_anchor_probe_sampled": True,
            "fixed_anchor_probe_selection_policy": selection_policy,
            "fixed_anchor_probe_increase_only": increase_only,
            "fixed_anchor_probe_decrease_allowed": selection[
                "decrease_allowed"
            ].reshape(()),
            "fixed_anchor_probe_action_definition": (
                "same_step_implied_clean_action_from_fixed_q_gradient_direction"
            ),
            "fixed_anchor_probe_single_batched_forward": True,
            "fixed_anchor_probe_direction_reused": True,
            "fixed_anchor_probe_subdivisions": (
                self.q_adaptive_cpb_probe_subdivisions
            ),
            "fixed_anchor_probe_candidate_count": candidate_budgets.numel(),
            "fixed_anchor_probe_evaluation_batch_size": evaluation_count,
            "fixed_anchor_probe_candidate_budgets": candidate_budgets,
            "fixed_anchor_probe_candidate_lambdas": selection[
                "candidate_lambdas"
            ],
            "fixed_anchor_probe_fixed_index": fixed_index,
            "fixed_anchor_probe_fixed_budget": fixed_budget,
            "fixed_anchor_probe_centered_envelope_budget": (
                centered_envelope_budget.float().reshape(())
            ),
            "fixed_anchor_probe_q1_curve": q1_curve.squeeze(0),
            "fixed_anchor_probe_q2_curve": q2_curve.squeeze(0),
            "fixed_anchor_probe_delta_q1_curve": selection[
                "delta_q1_curve"
            ].squeeze(0),
            "fixed_anchor_probe_delta_q2_curve": selection[
                "delta_q2_curve"
            ].squeeze(0),
            "fixed_anchor_probe_score_curve": selection[
                "robust_score_curve"
            ].squeeze(0),
            "fixed_anchor_probe_q1_fixed_repeat": probe_q1[-1],
            "fixed_anchor_probe_q2_fixed_repeat": probe_q2[-1],
            "fixed_anchor_probe_fixed_noise_q1": selection[
                "fixed_repeat_noise_q1"
            ].reshape(()),
            "fixed_anchor_probe_fixed_noise_q2": selection[
                "fixed_repeat_noise_q2"
            ].reshape(()),
            "fixed_anchor_probe_fixed_noise": selection[
                "fixed_repeat_noise"
            ].reshape(()),
            "fixed_anchor_probe_noise_threshold": selection[
                "noise_threshold"
            ].reshape(()),
            "fixed_anchor_probe_selected_index": selected_index,
            "fixed_anchor_probe_selected_lambda": selection[
                "selected_lambda"
            ].reshape(()),
            "fixed_anchor_probe_selected_score": selection[
                "selected_score"
            ].reshape(()),
            "fixed_anchor_probe_selected_task_budget": selected_budget,
            "fixed_anchor_probe_selected_guidance_weight": (
                selected_guidance_weight
            ),
            "fixed_anchor_probe_selected_deviation": selection[
                "selected_deviation"
            ].reshape(()),
            "fixed_anchor_probe_selected_direction": direction_name,
            "fixed_anchor_probe_selected_direction_code": selected_direction,
            "fixed_anchor_probe_selected_fixed": selection[
                "selected_fixed"
            ].reshape(()),
            "fixed_anchor_probe_fixed_fallback": selection[
                "fixed_fallback"
            ].reshape(()),
            "fixed_anchor_probe_noise_fallback": selection[
                "noise_fallback"
            ].reshape(()),
            "fixed_anchor_probe_boundary_optimum": selection[
                "boundary_optimum"
            ].reshape(()),
            "fixed_anchor_probe_q1_gradient_baseline": q1_before,
            "fixed_anchor_probe_q2_gradient_baseline": q2_before,
            "fixed_anchor_probe_latency_ms": latency_ms,
            "fixed_anchor_probe_memory_before_bytes": memory_before,
            "fixed_anchor_probe_memory_after_bytes": memory_after,
            "fixed_anchor_probe_peak_allocated_bytes": peak_allocated,
            "fixed_anchor_probe_peak_extra_allocated_bytes": max(
                0,
                peak_allocated - memory_before,
            ),
        }

    def _observe_adaptive_cpb_trust_region_step(
        self,
        *,
        action_velocity: torch.Tensor,
        guidance: torch.Tensor,
        grad: torch.Tensor,
        action_mask: torch.Tensor,
        adaptive_grad_cosine: torch.Tensor,
        grad_q1_l2: torch.Tensor,
        grad_q2_l2: torch.Tensor,
    ) -> dict[str, object]:
        controller = self._adaptive_cpb_trust_region_controller
        if controller is None:
            return {}

        mask = action_mask.to(device=grad.device, dtype=torch.float32)
        active = mask.sum().clamp_min(1.0)
        base_flow = action_velocity.detach().float() * mask
        guidance_flow = guidance.detach().float() * mask
        base_flow_rms = (base_flow.square().sum() / active).sqrt()
        guidance_rms = (guidance_flow.square().sum() / active).sqrt()

        direction_stability = None
        previous_grad = self._adaptive_cpb_trust_region_previous_grad
        current_grad = grad.detach().float() * mask
        if previous_grad is not None and previous_grad.shape == current_grad.shape:
            direction_stability = F.cosine_similarity(
                previous_grad.flatten(start_dim=1),
                current_grad.flatten(start_dim=1),
                dim=-1,
                eps=self.q_progress_eps,
            ).mean()
        self._adaptive_cpb_trust_region_previous_grad = current_grad.clone()

        observed = controller.observe(
            guidance_rms=float(guidance_rms.item()),
            base_flow_rms=float(base_flow_rms.item()),
            twin_grad_cosine=float(adaptive_grad_cosine.float().mean().item()),
            grad_q1_l2=float(grad_q1_l2.float().mean().item()),
            grad_q2_l2=float(grad_q2_l2.float().mean().item()),
            direction_stability_cosine=(
                None
                if direction_stability is None
                else float(direction_stability.item())
            ),
        )
        return {
            "adaptive_cpb_trust_region_step_observed": observed,
            "adaptive_cpb_trust_region_base_flow_rms": base_flow_rms,
            "adaptive_cpb_trust_region_guidance_rms": guidance_rms,
            "adaptive_cpb_trust_region_step_intervention_ratio": (
                guidance_rms / base_flow_rms.clamp_min(self.q_progress_eps)
            ),
            "adaptive_cpb_trust_region_direction_stability": direction_stability,
        }

    def _q_guided_velocity(
        self,
        actions: torch.Tensor,
        latents: torch.Tensor,
        action_velocity: torch.Tensor,
        timestep: torch.Tensor,
        action_cond: torch.Tensor | None,
        step_index: int,
    ) -> torch.Tensor:
        apply_q_guidance = self._should_apply_q_guidance(step_index)
        observation_probe_selected = self._observation_noise_probe_selected(
            step_index
        )
        action_gradient_probe_selected = (
            self._action_gradient_distribution_probe_selected(step_index)
        )
        if not (
            apply_q_guidance
            or observation_probe_selected
            or action_gradient_probe_selected
        ):
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
            q1_log = q1.detach().float().cpu().tolist()
            q2_log = q2.detach().float().cpu().tolist()
            q1_before = q1.detach().float()
            q2_before = q2.detach().float()
            q_objective_before = q_values_for_objective.detach().float()
            q_objective_log = (
                q_values_for_objective.detach().float().cpu().tolist()
            )
            self._run_observation_noise_probe(
                features=features,
                action_mask=action_mask,
                step_index=step_index,
            )
            self._run_action_gradient_distribution_probe(
                latents=latents,
                clean_action=clean_action,
                action_mask=action_mask,
                step_index=step_index,
            )
            if not apply_q_guidance:
                return action_velocity
            adaptive_confidence = None
            adaptive_grad_cosine = None
            grad_q1_l2 = None
            grad_q2_l2 = None
            adaptive_schedulers = {
                "adaptive_cpb_gradcos",
                "adaptive_cpb_gradcos_centered",
                "adaptive_cpb_gradcos_centered_fixed_floor",
                "adaptive_cpb_gradcos_qdelta_gated",
                "adaptive_cpb_gradcos_qdelta_smooth",
                "adaptive_cpb_gradcos_multibudget_probe",
                "adaptive_cpb_gradcos_fixed_anchor_probe",
                "adaptive_cpb_gradcos_fixed_anchor_increase_only",
                "adaptive_cpb_gradcos_shuffled",
                "adaptive_cpb_trust_region",
            }
            needs_twin_gradients = (
                self.q_guidance_weight_scheduler in adaptive_schedulers
                or self.q_log_second_order_cosine
            )
            second_order_record: dict[str, object] = {}
            if needs_twin_gradients:
                grad_q1 = torch.autograd.grad(
                    q1.sum(),
                    clean_action,
                    retain_graph=True,
                    create_graph=False,
                    only_inputs=True,
                )[0]
                grad_q2 = torch.autograd.grad(
                    q2.sum(),
                    clean_action,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True,
                )[0]
                if self.q_guidance_weight_scheduler in adaptive_schedulers:
                    adaptive_confidence, adaptive_grad_cosine = (
                        twin_grad_cosine_confidence(
                            grad_q1.detach(),
                            grad_q2.detach(),
                            action_mask,
                            tau=self.q_adaptive_cpb_cosine_tau,
                            temperature=self.q_adaptive_cpb_cosine_temperature,
                            eps=self.q_progress_eps,
                        )
                    )
                raw_reduce_dims = tuple(range(1, grad_q1.ndim))
                grad_q1_l2 = grad_q1.float().square().sum(
                    dim=raw_reduce_dims
                ).sqrt().detach()
                grad_q2_l2 = grad_q2.float().square().sum(
                    dim=raw_reduce_dims
                ).sqrt().detach()
                grad = self._select_q_objective_gradient(
                    q1,
                    q2,
                    grad_q1,
                    grad_q2,
                )
                grad = self._prepare_q_guidance_gradient(grad, action_mask)
                if self.q_log_second_order_cosine:
                    sampled = step_index in self.q_second_order_observation_steps
                    second_order_record = {
                        "second_order_observation_only": True,
                        "second_order_sampled": sampled,
                        "second_order_backend": "central_difference_first_order",
                        "second_order_activation_checkpointing": False,
                        "second_order_direction": (
                            "masked_normalized_selected_q_gradient"
                        ),
                        "second_order_finite_difference_epsilon": (
                            self.q_second_order_finite_difference_epsilon
                        ),
                    }
                    del (
                        features,
                        q1,
                        q2,
                        q_values_for_objective,
                        q_objective,
                        grad_q1,
                        grad_q2,
                    )
                    if sampled:
                        try:
                            epsilon = (
                                self.q_second_order_finite_difference_epsilon
                            )
                            direction = grad.to(dtype=torch.float32)
                            center = clean_action.detach().float()
                            plus_action = (
                                center + epsilon * direction
                            ).to(dtype=self.dtype)
                            minus_action = (
                                center - epsilon * direction
                            ).to(dtype=self.dtype)
                            mask = action_mask.to(dtype=torch.float32)
                            realized_half_delta = (
                                0.5
                                * (plus_action.float() - minus_action.float())
                                * mask
                            )

                            def first_order_gradients(
                                action: torch.Tensor,
                            ) -> tuple[torch.Tensor, torch.Tensor]:
                                action = action.detach().requires_grad_(True)
                                perturbed_features = self._extract_q_features(
                                    latents=latents,
                                    clean_action=action,
                                    action_mask=action_mask,
                                )
                                perturbed_q1, perturbed_q2 = (
                                    self.q_artifact.adapter.q_values(
                                        perturbed_features,
                                        action_mask,
                                    )
                                )
                                perturbed_grad_q1 = torch.autograd.grad(
                                    perturbed_q1.sum(),
                                    action,
                                    retain_graph=True,
                                    create_graph=False,
                                    only_inputs=True,
                                )[0]
                                perturbed_grad_q2 = torch.autograd.grad(
                                    perturbed_q2.sum(),
                                    action,
                                    retain_graph=False,
                                    create_graph=False,
                                    only_inputs=True,
                                )[0]
                                return (
                                    perturbed_grad_q1.detach(),
                                    perturbed_grad_q2.detach(),
                                )

                            plus_grad_q1, plus_grad_q2 = first_order_gradients(
                                plus_action
                            )
                            minus_grad_q1, minus_grad_q2 = first_order_gradients(
                                minus_action
                            )
                            hvp_q1 = (
                                (plus_grad_q1.float() - minus_grad_q1.float())
                                * mask
                                / (2.0 * epsilon)
                            )
                            hvp_q2 = (
                                (plus_grad_q2.float() - minus_grad_q2.float())
                                * mask
                                / (2.0 * epsilon)
                            )
                            second_confidence, second_cosine = (
                                twin_grad_cosine_confidence(
                                    hvp_q1,
                                    hvp_q2,
                                    action_mask,
                                    tau=self.q_adaptive_cpb_cosine_tau,
                                    temperature=(
                                        self.q_adaptive_cpb_cosine_temperature
                                    ),
                                    eps=self.q_progress_eps,
                                )
                            )
                            reduce_dims = tuple(range(1, hvp_q1.ndim))
                            realized_l2 = realized_half_delta.square().sum(
                                dim=reduce_dims
                            ).sqrt()
                            direction_l2 = direction.square().sum(
                                dim=reduce_dims
                            ).sqrt()
                            realized_direction_cosine = (
                                (realized_half_delta * direction).sum(
                                    dim=reduce_dims
                                )
                                / (
                                    realized_l2
                                    * direction_l2
                                ).clamp_min(self.q_progress_eps)
                            )
                            active = mask.sum(dim=reduce_dims).clamp_min(1.0)
                            changed_fraction = (
                                ((plus_action != minus_action).float() * mask).sum(
                                    dim=reduce_dims
                                )
                                / active
                            )
                            second_order_record.update(
                                {
                                    "second_order_grad_cosine": second_cosine,
                                    "second_order_confidence": second_confidence,
                                    "second_order_hvp_q1_l2": hvp_q1.square()
                                    .sum(dim=reduce_dims)
                                    .sqrt(),
                                    "second_order_hvp_q2_l2": hvp_q2.square()
                                    .sum(dim=reduce_dims)
                                    .sqrt(),
                                    "second_order_direction_l2": direction_l2,
                                    "second_order_realized_half_delta_l2": (
                                        realized_l2
                                    ),
                                    "second_order_realized_direction_cosine": (
                                        realized_direction_cosine
                                    ),
                                    "second_order_changed_fraction": (
                                        changed_fraction
                                    ),
                                    "second_order_directional_curvature_q1": (
                                        hvp_q1 * direction
                                    ).sum(dim=reduce_dims),
                                    "second_order_directional_curvature_q2": (
                                        hvp_q2 * direction
                                    ).sum(dim=reduce_dims),
                                }
                            )
                        except Exception as error:
                            second_order_record["second_order_error"] = repr(error)
            else:
                grad = torch.autograd.grad(
                    q_objective,
                    clean_action,
                    retain_graph=False,
                    create_graph=False,
                    only_inputs=True,
                )[0]
                grad = self._prepare_q_guidance_gradient(grad, action_mask)

        if (
            self.q_guidance_weight_scheduler == "local_uncertainty_cpb"
            and not self.q_log_second_order_cosine
        ):
            del features, q1, q2, q_values_for_objective, q_objective
            if clean_action.is_cuda:
                torch.cuda.empty_cache()

        guidance_weight, weight_record = self._progress_guidance_weight(
            sigma,
            step_index,
            adaptive_confidence=adaptive_confidence,
            adaptive_grad_cosine=adaptive_grad_cosine,
        )
        candidate_guidance_weight = guidance_weight
        candidate_guidance = candidate_guidance_weight.view(1, 1, 1, 1, 1) * grad
        uncertainty_scheduler = (
            self.q_guidance_weight_scheduler == "local_uncertainty_cpb"
        )
        uncertainty_sampled = (
            uncertainty_scheduler
            and step_index in self.q_uncertainty_cpb_steps
        )
        gated_scheduler = (
            self.q_guidance_weight_scheduler
            == "adaptive_cpb_gradcos_qdelta_gated"
        )
        smooth_scheduler = (
            self.q_guidance_weight_scheduler
            == "adaptive_cpb_gradcos_qdelta_smooth"
        )
        multibudget_scheduler = (
            self.q_guidance_weight_scheduler
            == "adaptive_cpb_gradcos_multibudget_probe"
        )
        fixed_anchor_scheduler = self.q_guidance_weight_scheduler in {
            "adaptive_cpb_gradcos_fixed_anchor_probe",
            "adaptive_cpb_gradcos_fixed_anchor_increase_only",
        }
        fixed_anchor_increase_only = (
            self.q_guidance_weight_scheduler
            == "adaptive_cpb_gradcos_fixed_anchor_increase_only"
        )
        gate_sampled = (
            (gated_scheduler or smooth_scheduler)
            and step_index in self.q_adaptive_cpb_gate_steps
        )
        probe_guidance = candidate_guidance
        probe_guidance_weight = candidate_guidance_weight
        if smooth_scheduler and gate_sampled:
            probe_guidance_weight = (
                torch.full_like(
                    candidate_guidance_weight,
                    self.q_adaptive_cpb_smooth_probe_budget,
                    dtype=torch.float32,
                )
                * weight_record["cpb_denoising_alpha"].to(
                    candidate_guidance_weight
                )
            )
            probe_guidance = probe_guidance_weight.view(1, 1, 1, 1, 1) * grad
        local_q_delta_record: dict[str, object] = {}
        if self.q_log_local_guidance_delta or gate_sampled:
            try:
                local_q_delta_record = self._local_q_delta_probe(
                    latents=latents,
                    clean_action=clean_action,
                    action_cond=action_cond,
                    action_mask=action_mask,
                    sigma=sigma,
                    candidate_guidance=probe_guidance,
                    q1_before=q1_before,
                    q2_before=q2_before,
                    q_objective_before=q_objective_before,
                )
                local_q_delta_record.update(
                    {
                        "local_q_delta_observation_only": not gate_sampled,
                        "local_q_delta_gate_probe": gate_sampled,
                        "local_q_delta_probe_task_budget": (
                            self.q_adaptive_cpb_smooth_probe_budget
                            if smooth_scheduler and gate_sampled
                            else None
                        ),
                        "local_q_delta_probe_guidance_weight": (
                            probe_guidance_weight
                        ),
                    }
                )
            except Exception as error:
                local_q_delta_record = {
                    "local_q_delta_observation_only": not gate_sampled,
                    "local_q_delta_gate_probe": gate_sampled,
                    "local_q_delta_error": repr(error),
                }

        safety_gate = torch.ones_like(candidate_guidance_weight, dtype=torch.float32)
        guidance_override: torch.Tensor | None = None
        gate_record: dict[str, object] = {}
        if uncertainty_scheduler:
            candidate_task_budget = weight_record["cpb_task_budget"]
            final_task_budget = candidate_task_budget
            uncertainty_record: dict[str, object] = {
                "local_uncertainty_sampled": False,
            }
            if uncertainty_sampled:
                try:
                    uncertainty_record = self._local_uncertainty_cpb_probe(
                        latents=latents,
                        clean_action=clean_action,
                        action_cond=action_cond,
                        action_mask=action_mask,
                        sigma=sigma,
                        vanilla_grad=grad,
                        task_budget=candidate_task_budget,
                        denoising_alpha=weight_record["cpb_denoising_alpha"],
                        step_index=step_index,
                    )
                    final_task_budget = uncertainty_record[
                        "local_uncertainty_selected_task_budget"
                    ]
                    guidance_weight = uncertainty_record[
                        "local_uncertainty_selected_guidance_weight"
                    ]
                    guidance_override = uncertainty_record.pop(
                        "_local_uncertainty_selected_guidance"
                    )
                    grad = uncertainty_record.pop(
                        "_local_uncertainty_selected_direction"
                    )
                except Exception as error:
                    if clean_action.is_cuda:
                        torch.cuda.empty_cache()
                    final_task_budget = torch.zeros_like(
                        candidate_task_budget,
                        dtype=torch.float32,
                    )
                    guidance_weight = torch.zeros_like(
                        candidate_guidance_weight,
                        dtype=torch.float32,
                    )
                    uncertainty_record = {
                        "local_uncertainty_sampled": True,
                        "local_uncertainty_failed_closed": True,
                        "local_uncertainty_error": repr(error),
                        "local_uncertainty_selected_factor": 0.0,
                        "local_uncertainty_selected_task_budget": (
                            final_task_budget
                        ),
                        "local_uncertainty_selected_guidance_weight": (
                            guidance_weight
                        ),
                    }
            weight_record.update(
                {
                    "cpb_candidate_task_budget": candidate_task_budget,
                    "cpb_task_budget": final_task_budget,
                    "candidate_guidance_weight": candidate_guidance_weight,
                    "progress_weight": guidance_weight,
                    **uncertainty_record,
                }
            )
        elif gated_scheduler:
            hard_veto = self._q_relative_collapse_tracker.collapsed
            if gate_sampled and "local_q_delta_error" not in local_q_delta_record:
                local_q_min_before = local_q_delta_record["local_q_min_before"]
                tracker_record = self._q_relative_collapse_tracker.observe(
                    step_index,
                    float(local_q_min_before.float().mean().item()),
                )
                hard_veto = bool(tracker_record["q_collapsed"])
                q_relative = torch.full_like(
                    local_q_min_before,
                    float(tracker_record["q_relative"]),
                    dtype=torch.float32,
                )
                safety_gate, gate_components = adaptive_cpb_qdelta_safety_gate(
                    q_relative=q_relative,
                    grad_cosine=adaptive_grad_cosine,
                    delta_q1=local_q_delta_record["local_delta_q1"],
                    delta_q2=local_q_delta_record["local_delta_q2"],
                    q1_before=q1_before,
                    q2_before=q2_before,
                    q_relative_low=self.q_adaptive_cpb_q_relative_low,
                    q_relative_high=self.q_adaptive_cpb_q_relative_high,
                    grad_cosine_low=self.q_adaptive_cpb_grad_cosine_low,
                    grad_cosine_high=self.q_adaptive_cpb_grad_cosine_high,
                    delta_relative_margin=(
                        self.q_adaptive_cpb_delta_relative_margin
                    ),
                    effect_consistency_floor=(
                        self.q_adaptive_cpb_effect_consistency_floor
                    ),
                    hard_veto=hard_veto,
                    eps=self.q_progress_eps,
                )
                gate_record = {
                    "qdelta_gate_sampled": True,
                    "qdelta_gate_q_relative": q_relative,
                    "qdelta_gate_q_peak": tracker_record["q_peak"],
                    "qdelta_gate_reference_trend": tracker_record[
                        "q_reference_trend"
                    ],
                    "qdelta_gate_low_streak": tracker_record[
                        "q_collapse_low_streak"
                    ],
                    "qdelta_gate_collapsed_before": tracker_record[
                        "q_collapsed_before"
                    ],
                    "qdelta_gate_collapsed": hard_veto,
                    **{
                        f"qdelta_gate_{name}": value
                        for name, value in gate_components.items()
                    },
                }
            elif gate_sampled:
                # A missing gate probe is not evidence that the direction is safe.
                safety_gate = torch.zeros_like(
                    candidate_guidance_weight,
                    dtype=torch.float32,
                )
                hard_veto = True
                gate_record = {
                    "qdelta_gate_sampled": True,
                    "qdelta_gate_probe_failed_closed": True,
                    "qdelta_gate_collapsed": (
                        self._q_relative_collapse_tracker.collapsed
                    ),
                }
            elif hard_veto:
                safety_gate = torch.zeros_like(
                    candidate_guidance_weight,
                    dtype=torch.float32,
                )
                gate_record = {
                    "qdelta_gate_sampled": False,
                    "qdelta_gate_collapsed": True,
                }
            else:
                gate_record = {
                    "qdelta_gate_sampled": False,
                    "qdelta_gate_collapsed": False,
                }

            candidate_task_budget = weight_record["cpb_task_budget"]
            guidance_weight = candidate_guidance_weight * safety_gate.to(
                candidate_guidance_weight
            )
            weight_record.update(
                {
                    "cpb_candidate_task_budget": candidate_task_budget,
                    "cpb_task_budget": candidate_task_budget
                    * safety_gate.to(candidate_task_budget),
                    "candidate_guidance_weight": candidate_guidance_weight,
                    "qdelta_safety_gate": safety_gate,
                    "qdelta_gate_hard_veto": hard_veto,
                    "progress_weight": guidance_weight,
                    **gate_record,
                }
            )
        elif smooth_scheduler:
            candidate_task_budget = weight_record["cpb_task_budget"]
            final_task_budget = candidate_task_budget
            smooth_record: dict[str, object] = {
                "qdelta_smooth_sampled": False,
            }
            if gate_sampled and "local_q_delta_error" not in local_q_delta_record:
                local_q_min_before = local_q_delta_record["local_q_min_before"]
                tracker_record = self._q_relative_collapse_tracker.observe(
                    step_index,
                    float(local_q_min_before.float().mean().item()),
                )
                q_relative = torch.full_like(
                    local_q_min_before,
                    float(tracker_record["q_relative"]),
                    dtype=torch.float32,
                )
                final_task_budget, smooth_components = (
                    adaptive_cpb_qdelta_smooth_budget(
                        candidate_budget=candidate_task_budget,
                        delta_q1=local_q_delta_record["local_delta_q1"],
                        delta_q2=local_q_delta_record["local_delta_q2"],
                        q1_before=q1_before,
                        q2_before=q2_before,
                        action_delta_rms=local_q_delta_record[
                            "local_action_delta_rms"
                        ],
                        q_relative=q_relative,
                        request_index=self._q_action_request_index,
                        denoising_step=step_index,
                        minimum_budget=self.q_adaptive_cpb_budget_min,
                        maximum_budget=self.q_adaptive_cpb_budget_max,
                        delta_gain_center=self.q_adaptive_cpb_delta_gain_center,
                        delta_gain_scale=self.q_adaptive_cpb_delta_gain_scale,
                        delta_score_softness=(
                            self.q_adaptive_cpb_delta_score_softness
                        ),
                        delta_logit_strength=(
                            self.q_adaptive_cpb_delta_logit_strength
                        ),
                        request_center=self.q_adaptive_cpb_request_center,
                        request_softness=self.q_adaptive_cpb_request_softness,
                        step_center=self.q_adaptive_cpb_step_center,
                        step_softness=self.q_adaptive_cpb_step_softness,
                        q_relative_low=self.q_adaptive_cpb_q_relative_low,
                        q_relative_high=self.q_adaptive_cpb_q_relative_high,
                        q_penalty_strength=(
                            self.q_adaptive_cpb_q_penalty_strength
                        ),
                        eps=self.q_progress_eps,
                    )
                )
                smooth_record = {
                    "qdelta_smooth_sampled": True,
                    "qdelta_smooth_q_relative": q_relative,
                    "qdelta_smooth_q_peak": tracker_record["q_peak"],
                    "qdelta_smooth_reference_trend": tracker_record[
                        "q_reference_trend"
                    ],
                    **{
                        f"qdelta_smooth_{name}": value
                        for name, value in smooth_components.items()
                    },
                }
            elif gate_sampled:
                smooth_record = {
                    "qdelta_smooth_sampled": True,
                    "qdelta_smooth_probe_failed_open": True,
                }

            guidance_weight = final_task_budget * weight_record[
                "cpb_denoising_alpha"
            ].to(final_task_budget)
            weight_record.update(
                {
                    "cpb_candidate_task_budget": candidate_task_budget,
                    "cpb_task_budget": final_task_budget,
                    "candidate_guidance_weight": candidate_guidance_weight,
                    "qdelta_smooth_probe_guidance_weight": (
                        probe_guidance_weight
                    ),
                    "qdelta_smooth_modulation_factor": (
                        final_task_budget
                        / candidate_task_budget.clamp_min(self.q_progress_eps)
                    ),
                    "progress_weight": guidance_weight,
                    **smooth_record,
                }
            )
        elif multibudget_scheduler:
            candidate_task_budget = weight_record["cpb_task_budget"]
            try:
                multibudget_record = self._multibudget_cpb_probe(
                    latents=latents,
                    clean_action=clean_action,
                    action_cond=action_cond,
                    action_mask=action_mask,
                    sigma=sigma,
                    grad=grad,
                    envelope_task_budget=candidate_task_budget,
                    denoising_alpha=weight_record["cpb_denoising_alpha"],
                    q1_before=q1_before,
                    q2_before=q2_before,
                )
                final_task_budget = multibudget_record[
                    "multibudget_probe_selected_task_budget"
                ]
                guidance_weight = multibudget_record[
                    "multibudget_probe_selected_guidance_weight"
                ]
            except Exception as error:
                if clean_action.is_cuda:
                    torch.cuda.empty_cache()
                final_task_budget = torch.zeros_like(
                    candidate_task_budget,
                    dtype=torch.float32,
                )
                guidance_weight = torch.zeros_like(
                    candidate_guidance_weight,
                    dtype=torch.float32,
                )
                multibudget_record = {
                    "multibudget_probe_sampled": True,
                    "multibudget_probe_selection_policy": (
                        "global_peak_min_budget_tie_break"
                    ),
                    "multibudget_probe_failed_closed": True,
                    "multibudget_probe_error": repr(error),
                    "multibudget_probe_selected_lambda": 0.0,
                    "multibudget_probe_selected_task_budget": (
                        final_task_budget
                    ),
                    "multibudget_probe_selected_guidance_weight": (
                        guidance_weight
                    ),
                }
            weight_record.update(
                {
                    "cpb_candidate_task_budget": candidate_task_budget,
                    "cpb_task_budget": final_task_budget,
                    "candidate_guidance_weight": candidate_guidance_weight,
                    "progress_weight": guidance_weight,
                    **multibudget_record,
                }
            )
        elif fixed_anchor_scheduler:
            centered_envelope_budget = weight_record["cpb_task_budget"]
            fixed_budget = torch.full_like(
                centered_envelope_budget,
                self.q_adaptive_cpb_probe_fixed_budget,
                dtype=torch.float32,
            )
            denoising_alpha = weight_record["cpb_denoising_alpha"].to(
                fixed_budget
            )
            try:
                fixed_anchor_record = self._fixed_anchor_cpb_probe(
                    latents=latents,
                    clean_action=clean_action,
                    action_cond=action_cond,
                    action_mask=action_mask,
                    sigma=sigma,
                    grad=grad,
                    centered_envelope_budget=centered_envelope_budget,
                    denoising_alpha=denoising_alpha,
                    q1_before=q1_before,
                    q2_before=q2_before,
                    increase_only=fixed_anchor_increase_only,
                )
                final_task_budget = fixed_anchor_record[
                    "fixed_anchor_probe_selected_task_budget"
                ]
                guidance_weight = fixed_anchor_record[
                    "fixed_anchor_probe_selected_guidance_weight"
                ]
            except Exception as error:
                if clean_action.is_cuda:
                    torch.cuda.empty_cache()
                final_task_budget = fixed_budget
                guidance_weight = fixed_budget * denoising_alpha
                selection_policy = (
                    "fixed_default_increase_only_global_peak_nearest_anchor_tie_break"
                    if fixed_anchor_increase_only
                    else "fixed_default_global_peak_nearest_anchor_tie_break"
                )
                fixed_anchor_record = {
                    "fixed_anchor_probe_sampled": True,
                    "fixed_anchor_probe_selection_policy": selection_policy,
                    "fixed_anchor_probe_increase_only": fixed_anchor_increase_only,
                    "fixed_anchor_probe_decrease_allowed": (
                        not fixed_anchor_increase_only
                    ),
                    "fixed_anchor_probe_failed_to_fixed": True,
                    "fixed_anchor_probe_error": repr(error),
                    "fixed_anchor_probe_fixed_budget": fixed_budget,
                    "fixed_anchor_probe_selected_lambda": 1.0,
                    "fixed_anchor_probe_selected_task_budget": (
                        final_task_budget
                    ),
                    "fixed_anchor_probe_selected_guidance_weight": (
                        guidance_weight
                    ),
                    "fixed_anchor_probe_selected_deviation": 0.0,
                    "fixed_anchor_probe_selected_direction": "fixed",
                    "fixed_anchor_probe_selected_direction_code": 0,
                    "fixed_anchor_probe_selected_fixed": True,
                    "fixed_anchor_probe_fixed_fallback": True,
                }
            weight_record.update(
                {
                    "cpb_candidate_task_budget": centered_envelope_budget,
                    "cpb_task_budget": final_task_budget,
                    "candidate_guidance_weight": candidate_guidance_weight,
                    "progress_weight": guidance_weight,
                    **fixed_anchor_record,
                }
            )
        else:
            guidance_weight = candidate_guidance_weight

        guidance_weight = guidance_weight.view(1, 1, 1, 1, 1)
        guidance = (
            guidance_override
            if guidance_override is not None
            else guidance_weight * grad
        )
        trust_region_step_record: dict[str, object] = {}
        if self.q_guidance_weight_scheduler == "adaptive_cpb_trust_region":
            trust_region_step_record = (
                self._observe_adaptive_cpb_trust_region_step(
                    action_velocity=action_velocity,
                    guidance=guidance,
                    grad=grad,
                    action_mask=action_mask,
                    adaptive_grad_cosine=adaptive_grad_cosine,
                    grad_q1_l2=grad_q1_l2,
                    grad_q2_l2=grad_q2_l2,
                )
            )
            weight_record.update(trust_region_step_record)
        guided_velocity = action_velocity.float() - guidance
        if self.q_log_denoising_steps:
            active = action_mask.to(grad.dtype).sum().clamp_min(1.0)
            self._write_q_record(
                self.q_step_log_path,
                {
                    "event": "denoising_step",
                    "frame_st_id": int(getattr(self, "_current_frame_st_id", -1)),
                    "action_request_index": int(self._q_action_request_index),
                    "action_step": int(step_index),
                    "sigma": sigma,
                    **weight_record,
                    "guidance_weight": guidance_weight.squeeze(),
                    "q1": q1_log,
                    "q2": q2_log,
                    "q_objective": q_objective_log,
                    "grad_q1_l2": grad_q1_l2,
                    "grad_q2_l2": grad_q2_l2,
                    "grad_l2": grad.square().sum().sqrt(),
                    "grad_rms": (grad.square().sum() / active).sqrt(),
                    "guidance_rms": (
                        guidance.square().sum()
                        / action_mask.to(guidance.dtype).sum().clamp_min(1.0)
                    ).sqrt(),
                    **trust_region_step_record,
                    **local_q_delta_record,
                    **second_order_record,
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

        latent_shape = (
            1,
            48,
            frame_chunk_size,
            self.latent_height,
            self.latent_width,
        )
        action_shape = (
            1,
            self.job_config.action_dim,
            frame_chunk_size,
            self.action_per_frame,
            1,
        )
        if self.q_replay_noise_seed is None:
            latents = torch.randn(
                latent_shape,
                device=self.device,
                dtype=self.dtype,
            )
            actions = torch.randn(
                action_shape,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            episode_index = max(self._q_replay_episode_index, 0)
            request_index = max(self._q_action_request_index, 0)
            latents, video_noise_seed = replayable_randn(
                latent_shape,
                device=self.device,
                dtype=self.dtype,
                base_seed=self.q_replay_noise_seed,
                episode_index=episode_index,
                request_index=request_index,
                stream="video",
            )
            actions, action_noise_seed = replayable_randn(
                action_shape,
                device=self.device,
                dtype=self.dtype,
                base_seed=self.q_replay_noise_seed,
                episode_index=episode_index,
                request_index=request_index,
                stream="action",
            )
            if self.q_log_replay_noise_hashes:
                self._write_q_record(
                    self.q_summary_log_path,
                    {
                        "event": "replay_noise",
                        "base_seed": self.q_replay_noise_seed,
                        "episode_index": episode_index,
                        "local_episode_index": self._q_replay_local_episode_index,
                        "request_index": request_index,
                        "frame_st_id": int(frame_st_id),
                        "video_seed": video_noise_seed,
                        "action_seed": action_noise_seed,
                        "video_sha256": tensor_sha256(latents),
                        "action_sha256": tensor_sha256(actions),
                    },
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
        if self.robotwin_eval and self.frame_st_id == 0 and self.init_latent is None:
            raise RuntimeError(
                "--robotwin requires an initial policy inference before the first KV refresh."
            )
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
        self._q_action_request_index += 1
        if self.q_guidance_weight_scheduler == "adaptive_cpb_trust_region":
            controller = self._adaptive_cpb_trust_region_controller
            if controller is None:
                raise RuntimeError("adaptive CPB trust-region controller is unavailable")
            _, self._adaptive_cpb_trust_region_request_record = (
                controller.start_request(self._q_action_request_index)
            )
            self._adaptive_cpb_trust_region_previous_grad = None
            self._write_q_record(
                self.q_summary_log_path,
                {
                    "event": "adaptive_cpb_trust_region_request",
                    **self._adaptive_cpb_trust_region_request_record,
                },
            )
        action, latents = self._infer(
            obs,
            frame_st_id=self.frame_st_id,
            initial_state=obs.get("state", None),
        )
        if self.q_log_replay_noise_hashes:
            self._write_q_record(
                self.q_summary_log_path,
                {
                    "event": "replay_output",
                    "base_seed": self.q_replay_noise_seed,
                    "episode_index": max(self._q_replay_episode_index, 0),
                    "local_episode_index": self._q_replay_local_episode_index,
                    "request_index": max(self._q_action_request_index, 0),
                    "frame_st_id": int(self.frame_st_id),
                    "action_sha256": tensor_sha256(torch.as_tensor(action)),
                },
            )
        output = dict(action=action)
        if obs.get("return_predicted_video", False):
            output["video"] = self._decode_predicted_video(latents)
        return output


def run(args):
    if args.robotwin and args.libero:
        raise ValueError("--robotwin and --libero are mutually exclusive")
    if args.q_deterministic_replay:
        if args.q_replay_noise_seed is None:
            raise ValueError("--q_deterministic_replay requires --q_replay_noise_seed")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(args.q_replay_noise_seed)
        torch.cuda.manual_seed_all(args.q_replay_noise_seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False

    config = deepcopy(VA_CONFIGS[args.config_name])
    if args.frame_chunk_size_override is not None:
        if not args.libero:
            raise ValueError(
                "--frame_chunk_size_override is restricted to --libero"
            )
        if args.frame_chunk_size_override <= 0:
            raise ValueError("--frame_chunk_size_override must be positive")
        config.frame_chunk_size = int(args.frame_chunk_size_override)
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
    original_protocol = bool(args.robotwin or args.libero)
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
        robotwin_eval=original_protocol,
        q_log_dir=args.q_log_dir,
        q_log_denoising_steps=args.q_log_denoising_steps,
        q_replay_noise_seed=args.q_replay_noise_seed,
        q_replay_manifest_path=args.q_replay_manifest_path,
        q_log_replay_noise_hashes=args.q_log_replay_noise_hashes,
        q_log_local_guidance_delta=args.q_log_local_guidance_delta,
        q_log_second_order_cosine=args.q_log_second_order_cosine,
        q_second_order_finite_difference_epsilon=(
            args.q_second_order_finite_difference_epsilon
        ),
        q_second_order_observation_steps=args.q_second_order_observation_steps,
        q_log_observation_noise_sensitivity=(
            args.q_log_observation_noise_sensitivity
        ),
        q_observation_noise_levels=args.q_observation_noise_levels,
        q_observation_noise_draws=args.q_observation_noise_draws,
        q_observation_noise_steps=args.q_observation_noise_steps,
        q_observation_noise_request_indices=(
            args.q_observation_noise_request_indices
        ),
        q_observation_noise_seed=args.q_observation_noise_seed,
        q_log_action_gradient_distribution=(
            args.q_log_action_gradient_distribution
        ),
        q_action_gradient_noise_scale=args.q_action_gradient_noise_scale,
        q_action_gradient_draws=args.q_action_gradient_draws,
        q_action_gradient_steps=args.q_action_gradient_steps,
        q_action_gradient_request_indices=(
            args.q_action_gradient_request_indices
        ),
        q_action_gradient_seed=args.q_action_gradient_seed,
        q_control_path=args.q_control_path,
        q_progress_guidance=args.q_progress_guidance,
        q_log_jepa_progress=args.q_log_jepa_progress,
        q_progress_goal_embedding=args.q_progress_goal_embedding,
        q_progress_jepa_checkpoint=args.q_progress_jepa_checkpoint,
        q_progress_jepa_config=args.q_progress_jepa_config,
        q_progress_jepa_repo=args.q_progress_jepa_repo,
        q_progress_jepa_checkpoint_key=args.q_progress_jepa_checkpoint_key,
        q_progress_camera_key=args.q_progress_camera_key,
        q_progress_clip_frames=args.q_progress_clip_frames,
        q_progress_jepa_offload=args.q_progress_jepa_offload,
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
        q_adaptive_cpb_budget_min=args.q_adaptive_cpb_budget_min,
        q_adaptive_cpb_budget_max=args.q_adaptive_cpb_budget_max,
        q_adaptive_cpb_cosine_tau=args.q_adaptive_cpb_cosine_tau,
        q_adaptive_cpb_cosine_temperature=args.q_adaptive_cpb_cosine_temperature,
        q_adaptive_cpb_cosine_pool=args.q_adaptive_cpb_cosine_pool,
        q_adaptive_cpb_cosine_pool_offset=args.q_adaptive_cpb_cosine_pool_offset,
        q_adaptive_cpb_gate_steps=args.q_adaptive_cpb_gate_steps,
        q_adaptive_cpb_q_relative_low=args.q_adaptive_cpb_q_relative_low,
        q_adaptive_cpb_q_relative_high=args.q_adaptive_cpb_q_relative_high,
        q_adaptive_cpb_grad_cosine_low=args.q_adaptive_cpb_grad_cosine_low,
        q_adaptive_cpb_grad_cosine_high=args.q_adaptive_cpb_grad_cosine_high,
        q_adaptive_cpb_delta_relative_margin=(
            args.q_adaptive_cpb_delta_relative_margin
        ),
        q_adaptive_cpb_effect_consistency_floor=(
            args.q_adaptive_cpb_effect_consistency_floor
        ),
        q_adaptive_cpb_smooth_probe_budget=(
            args.q_adaptive_cpb_smooth_probe_budget
        ),
        q_adaptive_cpb_delta_gain_center=args.q_adaptive_cpb_delta_gain_center,
        q_adaptive_cpb_delta_gain_scale=args.q_adaptive_cpb_delta_gain_scale,
        q_adaptive_cpb_delta_score_softness=(
            args.q_adaptive_cpb_delta_score_softness
        ),
        q_adaptive_cpb_delta_logit_strength=(
            args.q_adaptive_cpb_delta_logit_strength
        ),
        q_adaptive_cpb_request_center=args.q_adaptive_cpb_request_center,
        q_adaptive_cpb_request_softness=args.q_adaptive_cpb_request_softness,
        q_adaptive_cpb_step_center=args.q_adaptive_cpb_step_center,
        q_adaptive_cpb_step_softness=args.q_adaptive_cpb_step_softness,
        q_adaptive_cpb_q_penalty_strength=(
            args.q_adaptive_cpb_q_penalty_strength
        ),
        q_adaptive_cpb_probe_subdivisions=(
            args.q_adaptive_cpb_probe_subdivisions
        ),
        q_adaptive_cpb_probe_score_tolerance=(
            args.q_adaptive_cpb_probe_score_tolerance
        ),
        q_adaptive_cpb_probe_noise_multiplier=(
            args.q_adaptive_cpb_probe_noise_multiplier
        ),
        q_adaptive_cpb_probe_fixed_budget=(
            args.q_adaptive_cpb_probe_fixed_budget
        ),
        q_adaptive_cpb_trust_warmup_requests=(
            args.q_adaptive_cpb_trust_warmup_requests
        ),
        q_adaptive_cpb_trust_target_ratio=(
            args.q_adaptive_cpb_trust_target_ratio
        ),
        q_adaptive_cpb_trust_feedback_gain=(
            args.q_adaptive_cpb_trust_feedback_gain
        ),
        q_adaptive_cpb_trust_log_deadband=(
            args.q_adaptive_cpb_trust_log_deadband
        ),
        q_adaptive_cpb_trust_max_log_step=(
            args.q_adaptive_cpb_trust_max_log_step
        ),
        q_adaptive_cpb_trust_reliability_floor=(
            args.q_adaptive_cpb_trust_reliability_floor
        ),
        q_adaptive_cpb_trust_hysteresis_requests=(
            args.q_adaptive_cpb_trust_hysteresis_requests
        ),
        q_adaptive_cpb_collapse_reference_step=(
            args.q_adaptive_cpb_collapse_reference_step
        ),
        q_adaptive_cpb_collapse_threshold=(
            args.q_adaptive_cpb_collapse_threshold
        ),
        q_adaptive_cpb_collapse_patience=(
            args.q_adaptive_cpb_collapse_patience
        ),
        q_adaptive_cpb_recovery_threshold=(
            args.q_adaptive_cpb_recovery_threshold
        ),
        q_uncertainty_cpb_steps=args.q_uncertainty_cpb_steps,
        q_uncertainty_cpb_samples=args.q_uncertainty_cpb_samples,
        q_uncertainty_cpb_noise_scale=args.q_uncertainty_cpb_noise_scale,
        q_uncertainty_cpb_uncertainty_max=(
            args.q_uncertainty_cpb_uncertainty_max
        ),
        q_uncertainty_cpb_uncertainty_gradient_ratio=(
            args.q_uncertainty_cpb_uncertainty_gradient_ratio
        ),
        q_uncertainty_cpb_trust_penalty=args.q_uncertainty_cpb_trust_penalty,
        q_uncertainty_cpb_barrier_strength=(
            args.q_uncertainty_cpb_barrier_strength
        ),
        q_uncertainty_cpb_gradient_cosine_min=(
            args.q_uncertainty_cpb_gradient_cosine_min
        ),
        q_uncertainty_cpb_trust_radius=args.q_uncertainty_cpb_trust_radius,
        q_uncertainty_cpb_action_bound=args.q_uncertainty_cpb_action_bound,
        q_uncertainty_cpb_grad_norm_max=(
            args.q_uncertainty_cpb_grad_norm_max
        ),
        q_uncertainty_cpb_ascent_steps=args.q_uncertainty_cpb_ascent_steps,
        q_uncertainty_cpb_backtracks=args.q_uncertainty_cpb_backtracks,
        q_uncertainty_cpb_backtrack_factor=(
            args.q_uncertainty_cpb_backtrack_factor
        ),
        q_uncertainty_cpb_min_score_improvement=(
            args.q_uncertainty_cpb_min_score_improvement
        ),
        q_uncertainty_cpb_score_tolerance=(
            args.q_uncertainty_cpb_score_tolerance
        ),
        q_uncertainty_cpb_smooth_abs_eps=(
            args.q_uncertainty_cpb_smooth_abs_eps
        ),
        q_uncertainty_cpb_seed=args.q_uncertainty_cpb_seed,
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
            "inference_protocol": (
                "original_temporal_action_kv"
                if original_protocol
                else "state_conditioned_flat_action"
            ),
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
        "--libero",
        action="store_true",
        help="Use the original LIBERO temporal action/KV protocol.",
    )
    parser.add_argument(
        "--frame_chunk_size_override",
        type=int,
        default=None,
        help="Task-local original-LIBERO inference chunk override.",
    )
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
            "adaptive_cpb_gradcos",
            "adaptive_cpb_gradcos_centered",
            "adaptive_cpb_gradcos_centered_fixed_floor",
            "adaptive_cpb_gradcos_qdelta_gated",
            "adaptive_cpb_gradcos_qdelta_smooth",
            "adaptive_cpb_gradcos_multibudget_probe",
            "adaptive_cpb_gradcos_fixed_anchor_probe",
            "adaptive_cpb_gradcos_fixed_anchor_increase_only",
            "adaptive_cpb_gradcos_shuffled",
            "adaptive_cpb_trust_region",
            "local_uncertainty_cpb",
        ),
        help=(
            "How to turn q_guidance_scale into per-step guidance weight. "
            "'auto' preserves legacy behavior; 'fixed' uses q_guidance_scale "
            "directly at every guided denoising step; 'fixed_cpb' applies a "
            "constant task budget under the CPB denoising envelope; adaptive "
            "variants use twin-Q action-gradient cosine similarity; "
            "'local_uncertainty_cpb' backtracks a fixed CPB candidate using "
            "paired local twin-Q disagreement."
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
    parser.add_argument("--q_replay_noise_seed", type=int, default=None)
    parser.add_argument("--q_replay_manifest_path", type=str, default=None)
    parser.add_argument("--q_log_replay_noise_hashes", action="store_true")
    parser.add_argument(
        "--q_deterministic_replay",
        action="store_true",
        help="Require deterministic PyTorch kernels for exact rollout replay.",
    )
    parser.add_argument(
        "--q_log_local_guidance_delta",
        action="store_true",
        help=(
            "Observation-only logging of twin-Q values before and after the "
            "same-step clean-action correction implied by Q guidance."
        ),
    )
    parser.add_argument(
        "--q_log_second_order_cosine",
        action="store_true",
        help=(
            "Log observation-only cosine similarity between central-difference "
            "twin-Q Hessian-vector estimates along the actual guidance direction."
        ),
    )
    parser.add_argument(
        "--q_second_order_finite_difference_epsilon",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--q_second_order_observation_steps",
        type=str,
        default="25,35,40",
    )
    parser.add_argument(
        "--q_log_observation_noise_sensitivity",
        action="store_true",
        help=(
            "Probe detached critic feature-context sensitivity to small "
            "RMS-relative Gaussian noise on history-video tokens only."
        ),
    )
    parser.add_argument(
        "--q_observation_noise_levels",
        type=str,
        default="0.001,0.003,0.01,0.03",
    )
    parser.add_argument("--q_observation_noise_draws", type=int, default=4)
    parser.add_argument(
        "--q_observation_noise_steps",
        type=str,
        default="25,35,40",
    )
    parser.add_argument(
        "--q_observation_noise_request_indices",
        type=str,
        default="2,5,10",
    )
    parser.add_argument(
        "--q_observation_noise_seed",
        type=int,
        default=2026082501,
    )
    parser.add_argument(
        "--q_log_action_gradient_distribution",
        action="store_true",
        help=(
            "Observation-only local action sampling that compares the Q1/Q2 "
            "action-gradient distributions without changing policy actions."
        ),
    )
    parser.add_argument(
        "--q_action_gradient_noise_scale",
        type=float,
        default=0.03,
    )
    parser.add_argument("--q_action_gradient_draws", type=int, default=4)
    parser.add_argument(
        "--q_action_gradient_steps",
        type=str,
        default="35",
    )
    parser.add_argument(
        "--q_action_gradient_request_indices",
        type=str,
        default="1,3,5,7,9",
    )
    parser.add_argument(
        "--q_action_gradient_seed",
        type=int,
        default=2026082601,
    )
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
    parser.add_argument(
        "--q_log_jepa_progress",
        action="store_true",
        help=(
            "Compute and log JEPA goal progress at action-request boundaries "
            "without changing Q-guidance weights."
        ),
    )
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
    parser.add_argument(
        "--q_progress_jepa_offload",
        action="store_true",
        help="Offload the JEPA encoder to CPU between progress observations.",
    )
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
    parser.add_argument("--q_adaptive_cpb_budget_min", type=float, default=0.02)
    parser.add_argument("--q_adaptive_cpb_budget_max", type=float, default=0.10)
    parser.add_argument("--q_adaptive_cpb_cosine_tau", type=float, default=0.3)
    parser.add_argument(
        "--q_adaptive_cpb_cosine_temperature",
        type=float,
        default=5.0,
    )
    parser.add_argument("--q_adaptive_cpb_cosine_pool", type=str, default=None)
    parser.add_argument(
        "--q_adaptive_cpb_cosine_pool_offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--q_adaptive_cpb_gate_steps",
        type=str,
        default="25,30,35,40",
    )
    parser.add_argument(
        "--q_adaptive_cpb_q_relative_low",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--q_adaptive_cpb_q_relative_high",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--q_adaptive_cpb_grad_cosine_low",
        type=float,
        default=-0.15,
    )
    parser.add_argument(
        "--q_adaptive_cpb_grad_cosine_high",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--q_adaptive_cpb_delta_relative_margin",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--q_adaptive_cpb_effect_consistency_floor",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--q_adaptive_cpb_smooth_probe_budget",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--q_adaptive_cpb_delta_gain_center",
        type=float,
        default=14.0,
    )
    parser.add_argument(
        "--q_adaptive_cpb_delta_gain_scale",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--q_adaptive_cpb_delta_score_softness",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--q_adaptive_cpb_delta_logit_strength",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--q_adaptive_cpb_request_center",
        type=float,
        default=1.75,
    )
    parser.add_argument(
        "--q_adaptive_cpb_request_softness",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--q_adaptive_cpb_step_center",
        type=float,
        default=32.5,
    )
    parser.add_argument(
        "--q_adaptive_cpb_step_softness",
        type=float,
        default=2.5,
    )
    parser.add_argument(
        "--q_adaptive_cpb_q_penalty_strength",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--q_adaptive_cpb_probe_subdivisions",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--q_adaptive_cpb_probe_score_tolerance",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--q_adaptive_cpb_probe_noise_multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--q_adaptive_cpb_probe_fixed_budget",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_warmup_requests",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_target_ratio",
        type=float,
        default=0.0,
        help="Zero auto-calibrates the target from anchor-budget warmup requests.",
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_feedback_gain",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_log_deadband",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_max_log_step",
        type=float,
        default=0.22314355131420976,
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_reliability_floor",
        type=float,
        default=0.45,
    )
    parser.add_argument(
        "--q_adaptive_cpb_trust_hysteresis_requests",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--q_adaptive_cpb_collapse_reference_step",
        type=int,
        default=35,
    )
    parser.add_argument(
        "--q_adaptive_cpb_collapse_threshold",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--q_adaptive_cpb_collapse_patience",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--q_adaptive_cpb_recovery_threshold",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_steps",
        type=str,
        default="25,30,35,40",
    )
    parser.add_argument("--q_uncertainty_cpb_samples", type=int, default=4)
    parser.add_argument(
        "--q_uncertainty_cpb_noise_scale",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_uncertainty_max",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_uncertainty_gradient_ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_trust_penalty",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_barrier_strength",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_gradient_cosine_min",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_trust_radius",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_action_bound",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_grad_norm_max",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_ascent_steps",
        type=int,
        default=1,
    )
    parser.add_argument("--q_uncertainty_cpb_backtracks", type=int, default=3)
    parser.add_argument(
        "--q_uncertainty_cpb_backtrack_factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_min_score_improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_score_tolerance",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_smooth_abs_eps",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--q_uncertainty_cpb_seed",
        type=int,
        default=2026082701,
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
