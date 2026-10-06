from __future__ import annotations

import os
from pathlib import Path
from typing import Any


class WandbRunLogger:
    """Small fail-open wrapper around optional W&B logging."""

    def __init__(
        self,
        config: Any,
        *,
        rank: int,
        run_name_prefix: str,
    ) -> None:
        self.config = config
        self.rank = int(rank)
        self.enabled = bool(config.enable_wandb) and self.rank == 0
        self._wandb = None
        self._run = None
        self._failed = False
        if self.enabled:
            self._init(run_name_prefix=run_name_prefix)

    @property
    def active(self) -> bool:
        return self._run is not None and not self._failed

    def _init(self, *, run_name_prefix: str) -> None:
        try:
            if self.config.wandb_dir:
                Path(self.config.wandb_dir).mkdir(parents=True, exist_ok=True)
                os.environ["WANDB_DIR"] = str(self.config.wandb_dir)
            os.environ["WANDB_MODE"] = str(self.config.wandb_mode)
            import wandb

            kwargs: dict[str, Any] = {
                "project": self.config.wandb_project,
                "entity": self.config.wandb_entity,
                "name": self.config.wandb_run_name
                or self._default_run_name(run_name_prefix),
                "tags": list(self.config.wandb_tags),
                "config": self.config.to_dict(),
                "mode": self.config.wandb_mode,
                "dir": self.config.wandb_dir,
            }
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            try:
                kwargs["settings"] = wandb.Settings(
                    init_timeout=int(self.config.wandb_init_timeout)
                )
            except TypeError:
                pass
            self._wandb = wandb
            self._run = wandb.init(**kwargs)
            print(
                "[wandb] enabled "
                f"mode={self.config.wandb_mode} "
                f"dir={self.config.wandb_dir} "
                f"project={self.config.wandb_project}",
                flush=True,
            )
        except Exception as exc:
            self._handle_failure("init", exc)

    def _default_run_name(self, prefix: str) -> str:
        output_name = Path(self.config.output_dir).name
        return (
            f"{prefix}-{self.config.rl_objective}-"
            f"{self.config.critic_type}-{output_name}"
        )

    def _handle_failure(self, action: str, exc: Exception) -> None:
        if self.config.wandb_fail_open:
            self._failed = True
            self._run = None
            print(
                f"[wandb] {action} failed; disabling W&B for this run: {exc!r}",
                flush=True,
            )
            return
        raise exc

    def log(self, payload: dict[str, Any], *, step: int) -> None:
        if not self.active:
            return
        try:
            assert self._wandb is not None
            self._wandb.log(payload, step=int(step))
        except Exception as exc:
            self._handle_failure("log", exc)

    def finish(self) -> None:
        if not self.active:
            return
        try:
            assert self._wandb is not None
            self._wandb.finish()
        except Exception as exc:
            self._handle_failure("finish", exc)


def critic_wandb_payload(
    metrics: dict[str, float],
    *,
    config: Any,
    param_norm: float | None,
    step_time: float | None = None,
) -> dict[str, float]:
    payload: dict[str, float] = {}

    total_loss = metrics.get("total_loss", metrics.get("loss"))
    if total_loss is not None:
        payload["loss/total"] = float(total_loss)
    for source, target in (
        ("q_loss", "loss/q"),
        ("value_loss", "loss/value"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])
    if (
        "q_loss" in metrics
        and "value_loss" in metrics
        and abs(float(metrics["value_loss"])) > 1e-12
    ):
        payload["loss/q_to_value_ratio"] = (
            float(metrics["q_loss"]) / float(metrics["value_loss"])
        )

    for source, target in (
        ("q1_mean", "critic/q1_mean"),
        ("q2_mean", "critic/q2_mean"),
        ("value_mean", "critic/value_mean"),
        ("next_value_mean", "critic/next_value_mean"),
        ("q_disagreement", "critic/q_disagreement_mean"),
        ("q_velocity_mean", "critic/q_velocity_mean"),
        ("value_velocity_mean", "critic/value_velocity_mean"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])
    if "q1_mean" in metrics and "q2_mean" in metrics:
        q1 = float(metrics["q1_mean"])
        q2 = float(metrics["q2_mean"])
        payload["critic/q_min_mean"] = min(q1, q2)
        payload["critic/q_max_mean"] = max(q1, q2)
        if "value_mean" in metrics:
            payload["critic/advantage_mean"] = min(q1, q2) - float(
                metrics["value_mean"]
            )
    for source, target in (
        ("advantage_mean", "critic/advantage_mean"),
        ("advantage_std", "critic/advantage_std"),
        ("positive_advantage_frac", "critic/positive_advantage_frac"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])

    for source, target in (
        ("q_target_mean", "target/q_target_mean"),
        ("q_target_std", "target/q_target_std"),
        ("v_target_mean", "target/value_target_mean"),
        ("v_target_std", "target/value_target_std"),
        ("mc_return_target_mean", "target/mc_return_mean"),
        ("mc_return_target_std", "target/mc_return_std"),
        ("scalar_flow_timestep_mean", "target/scalar_flow_timestep_mean"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])

    for source, target in (
        ("reward_mean", "reward/mean"),
        ("reward_std", "reward/std"),
        ("reward_min", "reward/min"),
        ("reward_max", "reward/max"),
        ("jepa_distance", "reward/jepa_distance_mean"),
        ("cache_hit", "cache/hit_rate"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])

    for key, value in metrics.items():
        if key.startswith("reward_term/"):
            payload[key.replace("reward_term/", "reward_terms/")] = float(value)

    if "grad_norm" in metrics:
        payload["optim/grad_norm"] = float(metrics["grad_norm"])
    if param_norm is not None:
        payload["optim/param_norm"] = float(param_norm)
        if "grad_norm" in metrics and abs(float(param_norm)) > 1e-12:
            payload["optim/grad_to_param_norm"] = (
                float(metrics["grad_norm"]) / float(param_norm)
            )
    payload["optim/lr"] = float(metrics.get("learning_rate", config.learning_rate))
    payload["optim/target_ema_rate"] = float(config.target_ema_rate)

    for source, target in (
        ("data_wait_time", "time/data_wait"),
        ("cache_load_time", "time/cache_load"),
        ("video_inference_time", "time/video_inference"),
        ("feature_extract_time", "time/feature_extraction"),
        ("jepa_reward_time", "time/jepa_reward"),
        ("critic_backward_time", "time/critic_backward"),
        ("optimizer_time", "time/optimizer"),
        ("microbatch_time", "time/microbatch"),
        ("checkpoint_time", "time/checkpoint"),
        ("step_wall_time", "time/step_wall"),
        ("rank_data_wait_time_skew", "time/rank_data_wait_skew"),
        ("rank_microbatch_time_skew", "time/rank_microbatch_skew"),
        ("rank_optimizer_time_skew", "time/rank_optimizer_skew"),
        ("optimizer/all_reduce_time", "time/all_reduce"),
        ("optimizer/grad_clip_time", "time/grad_clip"),
        ("optimizer/adam_step_time", "time/adam_step"),
        ("optimizer/target_update_time", "time/target_update"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])
    if step_time is not None:
        payload["time/step"] = float(step_time)

    payload["data/configured_local_batch_size"] = float(config.batch_size)
    payload["data/gradient_accumulation_steps"] = float(
        config.gradient_accumulation_steps
    )
    for source, target in (
        ("local_microbatch_size", "data/local_microbatch_size"),
        (
            "effective_samples_per_optimizer_step",
            "data/local_effective_batch_size",
        ),
        ("estimated_global_batch_size", "data/global_batch_size"),
        ("rank_batch_size_min", "data/rank_batch_size_min"),
        ("rank_batch_size_max", "data/rank_batch_size_max"),
        ("samples_per_second", "throughput/samples_per_second"),
        ("state_valid_frac", "data/state_valid_frac"),
        ("done_frac", "data/done_frac"),
        ("episode_success_frac", "data/episode_success_frac"),
        ("rl_chunk_idx_mean", "data/rl_chunk_idx_mean"),
        ("episode_chunk_count_mean", "data/episode_chunk_count_mean"),
        ("clean_dataset_frac", "data/clean_dataset_frac"),
        ("augmented_dataset_frac", "data/augmented_dataset_frac"),
        ("other_dataset_frac", "data/other_dataset_frac"),
        ("value_context_token_count_mean", "tokens/value_mean"),
        ("value_context_token_count_max", "tokens/value_max"),
        ("value_context_valid_token_frac", "tokens/value_valid_frac"),
        ("q_context_token_count_mean", "tokens/q_mean"),
        ("q_context_token_count_max", "tokens/q_max"),
        ("q_context_valid_token_frac", "tokens/q_valid_frac"),
        (
            "target_value_context_token_count_mean",
            "tokens/target_value_mean",
        ),
        (
            "target_value_context_token_count_max",
            "tokens/target_value_max",
        ),
        (
            "target_value_context_valid_token_frac",
            "tokens/target_value_valid_frac",
        ),
        ("gpu_memory_allocated_gib", "gpu/allocated_gib"),
        ("gpu_memory_reserved_gib", "gpu/reserved_gib"),
        ("gpu_memory_peak_allocated_gib", "gpu/peak_allocated_gib"),
        ("gpu_memory_peak_reserved_gib", "gpu/peak_reserved_gib"),
        (
            "optimizer/all_reduce_bucket_count",
            "distributed/all_reduce_bucket_count",
        ),
        ("optimizer/all_reduce_bytes", "distributed/gradient_bytes"),
        ("optimizer/q1_grad_norm", "optim/q1_grad_norm"),
        ("optimizer/q2_grad_norm", "optim/q2_grad_norm"),
        ("optimizer/value_grad_norm", "optim/value_grad_norm"),
    ):
        if source in metrics:
            payload[target] = float(metrics[source])

    return payload
