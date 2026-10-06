#!/usr/bin/env python3
"""Read-only boundary checks for the isolated Track-A orientation-fix run.

The only write performed by this program is the requested evidence file
``preflight.json`` inside the new stage.  Dataset, export, and old-stage
artifacts are opened/read but never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


STAGE = Path(
    "/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/"
    "A_line2/orientationfix_va2_25k_20260916"
)
DATASET = Path(
    "/path/to/storage/datasets/"
    "libero10_va2_success_mix_orientationfix_gripperfix_frame2_final_20260912"
)
EXPORT = Path(
    "/path/to/storage/checkpoints/va2_libero10_cosmos_orientationfix_gripperfix_"
    "frame2_repro/export_step25000"
)
CACHE_ROOT = DATASET / "inference_state_cache" / (
    "orientationfix_va2_step25000_server_prefix_8layer_20260916"
)
CRITIC = Path(
    "/path/to/storage/checkpoints/rl/"
    "libero10_cosmos_joint_critic2_orientationfix_gripperfix_20260916"
)
OPEN_CONFIG = "orientationfix_gripperfix_va2_open500_frame2_cache_20260916"
COSMOS_CONFIG = "orientationfix_gripperfix_va2_cosmos_success376_frame2_cache_20260916"
MODEL = str(EXPORT)
CAMERAS = ["observation.images.front", "observation.images.wrist"]
Q01 = [
    -0.637499988079071,
    -0.768750011920929,
    -0.7473214268684387,
    -0.1039285734295845,
    -0.15214285254478455,
    -0.2786892834305762,
    -1.0,
]
Q99 = [
    0.7633928656578064,
    0.84375,
    0.9375,
    0.14785714447498322,
    0.15964286029338837,
    0.32035714387893677,
    1.0,
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def inspect_dataset(
    label: str,
    root: Path,
    expected_episodes: int,
    errors: list[str],
) -> dict:
    observed: dict = {"label": label, "root": str(root)}
    require(root.is_dir(), f"{label}: missing dataset root {root}", errors)
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    view_path = root / "view_manifest.json"
    if not info_path.is_file():
        errors.append(f"{label}: missing {info_path}")
        return observed
    info = load_json(info_path)
    view = load_json(view_path) if view_path.is_file() else {}
    observed.update(
        {
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "total_tasks": info.get("total_tasks"),
            "fps": info.get("fps"),
            "latent_fps": info.get("lingbotva_latent_fps"),
            "action_downsample_factor": info.get("lingbotva_action_downsample_factor"),
            "view_format": view.get("format"),
            "view_frame_chunk_size": view.get("frame_chunk_size"),
            "view_action_per_frame": view.get("action_per_frame"),
        }
    )
    require(info.get("total_episodes") == expected_episodes,
            f"{label}: total_episodes={info.get('total_episodes')!r} != {expected_episodes}", errors)
    require(info.get("total_tasks") == 10,
            f"{label}: total_tasks={info.get('total_tasks')!r} != 10", errors)
    require(info.get("fps") == 20,
            f"{label}: info.fps={info.get('fps')!r} != 20", errors)
    require(info.get("lingbotva_latent_fps") == 10.0,
            f"{label}: latent fps is not 10.0", errors)
    require(info.get("lingbotva_action_downsample_factor") == 2,
            f"{label}: action downsample factor is not 2", errors)
    require(view.get("source_fps") == 20 and view.get("target_fps") == 10,
            f"{label}: frame frequency contract is not 20->10", errors)
    require(view.get("frame_chunk_size") == 2 and view.get("action_per_frame") == 4,
            f"{label}: frame2/action4 contract is absent", errors)
    require(view.get("temporal_downsample_factor") == 2 and view.get("action_downsample_factor") == 2,
            f"{label}: temporal/action stride contract is absent", errors)
    require(view.get("action_downsample_policy") == "stride_2",
            f"{label}: action policy is not stride_2", errors)
    orientation_provenance = " ".join(
        str(view.get(key, ""))
        for key in ("image_orientation", "source", "cosmos_source")
    ).lower()
    require(
        "orientation-fixed" in orientation_provenance
        or "orientationfix" in orientation_provenance,
        f"{label}: orientation-fix provenance is absent",
        errors,
    )
    require("-1=open, +1=close" in str(view.get("gripper_convention", "")),
            f"{label}: corrected gripper convention is absent", errors)

    features = info.get("features", {})
    require(features.get("action", {}).get("shape") == [7],
            f"{label}: action shape is not [7]", errors)
    for camera in CAMERAS:
        feature = features.get(camera)
        require(isinstance(feature, dict), f"{label}: missing camera {camera}", errors)
        if isinstance(feature, dict):
            require(feature.get("dtype") == "video", f"{label}: {camera} is not video", errors)
            require(feature.get("info", {}).get("video.fps") == 20,
                    f"{label}: {camera} video fps is not 20", errors)

    rows = []
    if episodes_path.is_file():
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    observed["episodes_jsonl_count"] = len(rows)
    require(len(rows) == expected_episodes,
            f"{label}: episodes.jsonl count={len(rows)} != {expected_episodes}", errors)
    indices = [int(row["episode_index"]) for row in rows]
    require(indices == list(range(expected_episodes)),
            f"{label}: episode indices are not contiguous 0..{expected_episodes - 1}", errors)

    latent_counts = {}
    latent_indices = {}
    for camera in CAMERAS:
        files = sorted((root / "latents").glob(f"chunk-*/{camera}/episode_*.pth"))
        ids = []
        for path in files:
            match = re.match(r"episode_(\d+)_", path.name)
            if match:
                ids.append(int(match.group(1)))
        latent_counts[camera] = len(files)
        latent_indices[camera] = ids
        require(len(files) == expected_episodes,
                f"{label}: {camera} latent file count={len(files)} != {expected_episodes}", errors)
        require(sorted(ids) == list(range(expected_episodes)),
                f"{label}: {camera} latent episode ids are incomplete", errors)
        require(all(path.is_file() for path in files),
                f"{label}: {camera} contains a broken latent link", errors)
    observed["latent_counts"] = latent_counts

    jepa_files = sorted((root / "jepa").glob("chunk-*/episode_*.pt"))
    jepa_ids = [int(m.group(1)) for p in jepa_files if (m := re.match(r"episode_(\d+)\.pt$", p.name))]
    observed["jepa_count"] = len(jepa_files)
    require(len(jepa_files) == expected_episodes,
            f"{label}: JEPA file count={len(jepa_files)} != {expected_episodes}", errors)
    require(sorted(jepa_ids) == list(range(expected_episodes)),
            f"{label}: JEPA episode ids are incomplete", errors)
    return observed


def inspect_config(path: Path, expected_name: str, cache_dir: Path, errors: list[str]) -> dict:
    config = load_json(path)
    observed = {
        "path": str(path),
        "base_config_name": config.get("base_config_name"),
        "transformer_path": config.get("transformer_path"),
        "output_dir": config.get("output_dir"),
        "phase3_precomputed_cache_dir": config.get("phase3_precomputed_cache_dir"),
        "phase3_precompute_mode": config.get("phase3_precompute_mode"),
        "critic_type": config.get("critic_type"),
        "critic_feature_layers": config.get("critic_feature_layers"),
        "reward_terms": config.get("reward_terms"),
    }
    require(config.get("base_config_name") == expected_name,
            f"{path}: unexpected base_config_name", errors)
    require(config.get("transformer_path") == f"{MODEL}/transformer",
            f"{path}: target transformer path mismatch", errors)
    require(config.get("output_dir") == str(cache_dir),
            f"{path}: output_dir mismatch", errors)
    require(config.get("phase3_precomputed_cache_dir") == str(cache_dir),
            f"{path}: phase3_precomputed_cache_dir mismatch", errors)
    require(config.get("phase3_precompute_mode") == "server_prefix",
            f"{path}: phase3 precompute mode mismatch", errors)
    require(config.get("infer_latent_chunk_size") == 2,
            f"{path}: infer_latent_chunk_size mismatch", errors)
    require(config.get("critic_type") == "twin_query_transformer_v1",
            f"{path}: cache critic type mismatch", errors)
    require(config.get("critic_feature_layers") == {"start": 0, "stop": 30, "step": 4},
            f"{path}: cache critic layer range mismatch", errors)
    return observed


def main() -> int:
    errors: list[str] = []
    require(STAGE.is_dir(), f"stage directory missing: {STAGE}", errors)
    complete = EXPORT / "COMPLETE"
    manifest_path = EXPORT / "conversion_manifest.json"
    transformer_config = EXPORT / "transformer" / "config.json"
    weights = EXPORT / "transformer" / "diffusion_pytorch_model.safetensors"
    require(complete.is_file() and complete.stat().st_size > 0,
            f"missing export COMPLETE: {complete}", errors)
    require(manifest_path.is_file(), f"missing conversion manifest: {manifest_path}", errors)
    conversion = load_json(manifest_path) if manifest_path.is_file() else {}
    require(conversion.get("source_step") == 25000,
            f"conversion manifest source_step={conversion.get('source_step')!r} != 25000", errors)
    require(conversion.get("source_global_steps") == 25000,
            "conversion manifest source_global_steps is not 25000", errors)
    require(str(conversion.get("source_checkpoint", "")).endswith("/checkpoints/step_25000"),
            "conversion manifest source_checkpoint is not step_25000", errors)
    require(transformer_config.is_file(), f"missing transformer config: {transformer_config}", errors)
    require(weights.is_file() and weights.stat().st_size > 1_000_000_000,
            f"missing/undersized transformer safetensors: {weights}", errors)

    datasets = [
        inspect_dataset("open500", DATASET / "open500", 500, errors),
        inspect_dataset("cosmos_success376", DATASET / "cosmos_success376", 376, errors),
    ]
    require(not (CACHE_ROOT / "open500").exists(),
            f"refusing pre-existing target cache: {CACHE_ROOT / 'open500'}", errors)
    require(not (CACHE_ROOT / "cosmos_success376").exists(),
            f"refusing pre-existing target cache: {CACHE_ROOT / 'cosmos_success376'}", errors)
    require(not CRITIC.exists() and not Path(str(CRITIC) + "_smoke").exists(),
            f"refusing pre-existing critic output: {CRITIC}", errors)

    configs = []
    for filename, name, cache_name in [
        ("cache_open.json", OPEN_CONFIG, "open500"),
        ("cache_cosmos.json", COSMOS_CONFIG, "cosmos_success376"),
    ]:
        configs.append(inspect_config(STAGE / filename, name, CACHE_ROOT / cache_name, errors))

    # The explicit q01/q99 contract is copied from the verified old stats,
    # but is asserted here independently so a stale config cannot slip in.
    expected_q01, expected_q99 = Q01 + [0.0] * 23, Q99 + [0.0] * 23
    require(len(expected_q01) == 30 and len(expected_q99) == 30,
            "normalization contract is not 30D", errors)
    py_path = os.environ.get("PYTHONPATH", "")
    old_only_tokens = (
        "libero_10_original_500_gripperfix_20260909",
        "LIBERO-Cosmos-Policy-lingbotva-20260906-all2000",
        "/gripperfix_20260909/A_line2/federated",
    )
    serialized = "\n".join(path.read_text(errors="ignore") for path in STAGE.glob("*.json"))
    for token in old_only_tokens:
        require(token not in serialized, f"new stage references old artifact token {token}", errors)

    registration = {}
    try:
        from wan_va.configs import VA_CONFIGS

        for name, root in [
            (OPEN_CONFIG, DATASET / "open500"),
            (COSMOS_CONFIG, DATASET / "cosmos_success376"),
        ]:
            cfg = VA_CONFIGS[name]
            registration[name] = {
                "dataset_path": str(cfg.dataset_path),
                "wan22_pretrained_model_name_or_path": str(cfg.wan22_pretrained_model_name_or_path),
                "obs_cam_keys": list(cfg.obs_cam_keys),
                "latent_only_video_keys": list(cfg.latent_only_video_keys),
                "frame_chunk_size": int(cfg.frame_chunk_size),
                "source_fps": int(cfg.source_fps),
                "target_fps": int(cfg.target_fps),
                "action_downsample_factor": int(cfg.action_downsample_factor),
                "action_downsample_policy": cfg.action_downsample_policy,
                "action_per_frame": int(cfg.action_per_frame),
                "q01": list(cfg.norm_stat["q01"]),
                "q99": list(cfg.norm_stat["q99"]),
            }
            require(Path(cfg.dataset_path) == root, f"{name}: dataset root mismatch", errors)
            require(str(cfg.wan22_pretrained_model_name_or_path) == MODEL,
                    f"{name}: model root mismatch", errors)
            require(list(cfg.obs_cam_keys) == CAMERAS and list(cfg.latent_only_video_keys) == CAMERAS,
                    f"{name}: camera key contract mismatch", errors)
            require(int(cfg.frame_chunk_size) == 2 and int(cfg.source_fps) == 20 and int(cfg.target_fps) == 10,
                    f"{name}: frame/frequency contract mismatch", errors)
            require(int(cfg.action_downsample_factor) == 2 and cfg.action_downsample_policy == "stride_2",
                    f"{name}: action stride contract mismatch", errors)
            require(int(cfg.action_per_frame) == 4, f"{name}: action_per_frame mismatch", errors)
            require(list(cfg.norm_stat["q01"]) == expected_q01 and list(cfg.norm_stat["q99"]) == expected_q99,
                    f"{name}: q01/q99 contract mismatch", errors)
    except Exception as exc:
        errors.append(f"VA_CONFIGS registration/import failed: {exc!r}")

    payload = {
        "status": "pass" if not errors else "fail",
        "read_only_checks": True,
        "host": os.uname().nodename,
        "stage": str(STAGE),
        "export": {
            "root": str(EXPORT),
            "complete": str(complete),
            "conversion_manifest": str(manifest_path),
            "conversion_manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
            "source_step": conversion.get("source_step"),
            "transformer_config": str(transformer_config),
            "transformer_safetensors": str(weights),
            "transformer_safetensors_bytes": weights.stat().st_size if weights.is_file() else None,
        },
        "datasets": datasets,
        "target_cache_root": str(CACHE_ROOT),
        "target_critic_output": str(CRITIC),
        "configs": configs,
        "registrations": registration,
        "q01_30": expected_q01,
        "q99_30": expected_q99,
        "errors": errors,
    }
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "preflight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    sys.exit(main())
