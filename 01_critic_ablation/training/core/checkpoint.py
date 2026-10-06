from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import CriticTrainingConfig
from .critics import CriticBundle


CHECKPOINT_SCHEMA_VERSION = 3


def _json_compatible(value: Any) -> Any:
    return json.loads(json.dumps(value))


def save_critic_checkpoint(
    directory: str | Path,
    bundle: CriticBundle,
    optimizer: torch.optim.Optimizer,
    config: CriticTrainingConfig,
    step: int,
    manifest: dict[str, Any],
) -> Path:
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "critic": bundle.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        output / "training_state.pt",
    )
    (output / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n"
    )
    full_manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": config.algorithm,
        "rl_objective": config.rl_objective,
        "critic_type": config.critic_type,
        "feature_dim": config.feature_dim,
        "infer_latent_chunk_size": config.infer_latent_chunk_size,
        "feature_layers": list(config.feature_layers),
        "critic_feature_layers": list(config.critic_feature_layers),
        "feature_aggregation": config.feature_aggregation,
        "feature_normalization": config.feature_normalization,
        **manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(full_manifest, indent=2, sort_keys=True) + "\n"
    )
    return output


def load_critic_checkpoint(
    directory: str | Path,
    bundle: CriticBundle,
    optimizer: torch.optim.Optimizer | None = None,
    expected_manifest: dict[str, Any] | None = None,
) -> int:
    checkpoint = Path(directory)
    with (checkpoint / "manifest.json").open() as handle:
        manifest = json.load(handle)
    checkpoint_config: dict[str, Any] = {}
    config_path = checkpoint / "config.json"
    if config_path.exists():
        with config_path.open() as handle:
            checkpoint_config = json.load(handle)
    if manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported checkpoint schema: {manifest.get('schema_version')}"
        )
    for key, expected in (expected_manifest or {}).items():
        actual = manifest.get(key)
        if actual is None and key in checkpoint_config:
            actual = checkpoint_config[key]
        if key in {"algorithm", "rl_objective"}:
            from .config import _normalize_rl_objective

            actual_value = manifest.get("rl_objective", manifest.get("algorithm"))
            if actual_value is None:
                actual_value = checkpoint_config.get(
                    "rl_objective",
                    checkpoint_config.get("algorithm"),
                )
            if actual_value is None:
                raise ValueError("Checkpoint manifest has no RL objective")
            actual = _normalize_rl_objective(str(actual_value), key=key)
            expected = _normalize_rl_objective(str(expected), key=key)
        elif key == "phase3_video_prediction_context" and actual is None:
            actual = "episode_prefix"
        if actual != _json_compatible(expected):
            raise ValueError(
                f"Incompatible checkpoint {key}: "
                f"expected {expected!r}, found {actual!r}"
            )
    state = torch.load(
        checkpoint / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )
    bundle.load_state_dict(state["critic"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state["step"])
