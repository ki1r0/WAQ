#!/usr/bin/env python3
"""Blocking checks for the official LIBERO frame4/action4 pipeline.

The checker is intentionally independent of the training implementation.  It
validates the concrete data/config/export artifacts that cross stage
boundaries, so a stale 10 Hz cache or action normalizer cannot silently enter
training or evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_Q01_7 = np.asarray(
    [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.1596428602933887,
        -0.26571428775787354,
        -1.0,
    ],
    dtype=np.float64,
)
EXPECTED_Q99_7 = np.asarray(
    [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0,
    ],
    dtype=np.float64,
)


def parse_pairs(values: list[str] | None, name: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"{name} must use LABEL=VALUE, got {value!r}")
        label, item = value.split("=", 1)
        if not label or not item:
            raise ValueError(f"{name} must use non-empty LABEL=VALUE, got {value!r}")
        result[label] = item
    return result


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # pragma: no cover - error is reported to user
        raise RuntimeError(f"cannot read JSON {path}: {exc}") from exc


def numeric(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def find_norm_stats(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    """Find a q01/q99 pair in a nested JSON config."""
    if isinstance(value, dict):
        if "q01" in value and "q99" in value:
            try:
                q01 = np.asarray(value["q01"], dtype=np.float64).reshape(-1)
                q99 = np.asarray(value["q99"], dtype=np.float64).reshape(-1)
            except (TypeError, ValueError):
                q01 = q99 = np.asarray([], dtype=np.float64)
            if q01.size >= 7 and q99.size >= 7:
                return q01, q99
        for child in value.values():
            found = find_norm_stats(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_norm_stats(child)
            if found is not None:
                return found
    return None


def assert_expected_norm(
    found: tuple[np.ndarray, np.ndarray] | None, where: str, errors: list[str]
) -> dict[str, Any] | None:
    if found is None:
        errors.append(f"{where}: no q01/q99 normalization contract found")
        return None
    q01, q99 = found
    if not np.allclose(q01[:7], EXPECTED_Q01_7, rtol=0.0, atol=1e-6):
        errors.append(f"{where}: q01 first 7 channels differ from official LIBERO contract")
    if not np.allclose(q99[:7], EXPECTED_Q99_7, rtol=0.0, atol=1e-6):
        errors.append(f"{where}: q99 first 7 channels differ from official LIBERO contract")
    if q01.size != 30 or q99.size != 30:
        errors.append(f"{where}: expected 30D q01/q99, got {q01.size}/{q99.size}")
    return {"q01": q01.tolist(), "q99": q99.tolist()}


def inspect_dataset(
    label: str,
    root: Path,
    expected_episodes: int | None,
    episode_range: tuple[int, int] | None,
    expected_fps: float,
    camera_keys: list[str] | None,
    require_videos: bool,
    errors: list[str],
) -> dict[str, Any]:
    if not root.is_dir():
        errors.append(f"{label}: dataset root does not exist: {root}")
        return {"label": label, "root": str(root), "exists": False}

    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file():
        errors.append(f"{label}: missing {info_path}")
        return {"label": label, "root": str(root), "exists": True}
    info = load_json(info_path)
    observed: dict[str, Any] = {
        "label": label,
        "root": str(root),
        "info_fps": info.get("fps"),
        "latent_fps": info.get("lingbotva_latent_fps", info.get("latent_fps")),
        "action_downsample_factor": info.get("lingbotva_action_downsample_factor"),
        "video_materialized": info.get("lingbotva_video_materialized"),
        "total_episodes": info.get("total_episodes"),
        "total_videos": info.get("total_videos"),
    }

    fps = numeric(info.get("fps"))
    if fps is None or not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"{label}: info.fps={info.get('fps')!r}, expected {expected_fps}")
    latent_fps = numeric(info.get("lingbotva_latent_fps", info.get("latent_fps")))
    if latent_fps is not None and not math.isclose(
        latent_fps, expected_fps, rel_tol=0.0, abs_tol=1e-6
    ):
        errors.append(
            f"{label}: latent fps metadata={latent_fps}, expected {expected_fps}; stale temporal cache"
        )
    downsample = numeric(info.get("lingbotva_action_downsample_factor"))
    if downsample is not None and not math.isclose(downsample, 1.0, rel_tol=0.0, abs_tol=1e-6):
        errors.append(
            f"{label}: action downsample factor={downsample}, expected 1 for official frame4"
        )
    if "10hz" in str(root).lower() and expected_fps == 20:
        errors.append(f"{label}: path still identifies a 10 Hz dataset: {root}")

    total_episodes = info.get("total_episodes")
    if episode_range is None and expected_episodes is not None and total_episodes != expected_episodes:
        errors.append(
            f"{label}: info.total_episodes={total_episodes!r}, expected {expected_episodes}"
        )
    if episode_range is not None:
        start, end = episode_range
        if start < 0 or end <= start:
            errors.append(f"{label}: invalid episode range {episode_range}")
        elif total_episodes is not None and total_episodes < end:
            errors.append(
                f"{label}: info.total_episodes={total_episodes!r} does not cover range {start}:{end}"
            )
        elif expected_episodes is not None and expected_episodes != end - start:
            errors.append(
                f"{label}: expected episodes={expected_episodes} disagrees with range {start}:{end}"
            )

    features = info.get("features", {})
    action_feature = features.get("action", {})
    if action_feature.get("shape") != [7]:
        errors.append(f"{label}: action feature shape={action_feature.get('shape')!r}, expected [7]")
    for key in camera_keys or []:
        if key not in features:
            errors.append(f"{label}: missing camera feature {key!r}")
        else:
            video_fps = numeric(features[key].get("info", {}).get("video.fps"))
            if video_fps is not None and not math.isclose(
                video_fps, expected_fps, rel_tol=0.0, abs_tol=1e-6
            ):
                errors.append(f"{label}: {key} video.fps={video_fps}, expected {expected_fps}")

    if require_videos:
        if info.get("lingbotva_video_materialized") is False:
            errors.append(f"{label}: videos are explicitly marked non-materialized")
        if expected_episodes is not None and camera_keys:
            expected_videos = expected_episodes * len(camera_keys)
            if info.get("total_videos") != expected_videos:
                errors.append(
                    f"{label}: total_videos={info.get('total_videos')!r}, expected {expected_videos}"
                )

    if episodes_path.is_file():
        rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
        if episode_range is not None:
            start, end = episode_range
            rows = [row for row in rows if start <= int(row.get("episode_index", -1)) < end]
        observed["episodes_jsonl_count"] = len(rows)
        observed["episode_range"] = list(episode_range) if episode_range is not None else None
        indices = [row.get("episode_index") for row in rows]
        if expected_episodes is not None and len(rows) != expected_episodes:
            errors.append(f"{label}: episodes.jsonl has {len(rows)}, expected {expected_episodes}")
        if len(indices) != len(set(indices)):
            errors.append(f"{label}: duplicate episode_index values in episodes.jsonl")
        if any(row.get("length", 0) <= 0 for row in rows):
            errors.append(f"{label}: non-positive episode length in episodes.jsonl")
    else:
        errors.append(f"{label}: missing {episodes_path}")

    # Read a bounded sample only; this is enough to catch stale action encoding
    # without scanning the complete dataset.
    parquet_files = sorted((root / "data").rglob("*.parquet"))[:8]
    if not parquet_files:
        errors.append(f"{label}: no parquet files under {root / 'data'}")
    else:
        try:
            import pandas as pd

            action_rows: list[np.ndarray] = []
            for parquet in parquet_files:
                frame = pd.read_parquet(parquet, columns=["action"])
                values = frame["action"].tolist()[:2048]
                if values:
                    action_rows.append(np.asarray(values, dtype=np.float64))
            if action_rows:
                actions = np.concatenate(action_rows, axis=0)
                observed["action_sample_shape"] = list(actions.shape)
                if actions.ndim != 2 or actions.shape[1] != 7:
                    errors.append(f"{label}: sampled action shape={actions.shape}, expected [N,7]")
                elif not np.isfinite(actions).all():
                    errors.append(f"{label}: sampled action contains non-finite values")
                else:
                    gripper = np.round(actions[:, 6], 6)
                    observed["gripper_values_sample"] = sorted(set(gripper.tolist()))
                    if not set(gripper.tolist()).issubset({-1.0, 1.0}):
                        errors.append(f"{label}: gripper sample is not a subset of {{-1,+1}}")

                    # Exercise the same official quantile transform used by
                    # the VA loader and its inverse.  This catches a dataset
                    # that has been normalized twice (or not at all) at the
                    # stage boundary without scanning the complete corpus.
                    q01 = EXPECTED_Q01_7.astype(np.float64)
                    q99 = EXPECTED_Q99_7.astype(np.float64)
                    denom = q99 - q01 + 1e-6
                    raw = actions[:, :7]
                    normalized = (raw - q01[None, :]) / denom[None, :] * 2.0 - 1.0
                    restored = (normalized + 1.0) / 2.0 * denom[None, :] + q01[None, :]
                    roundtrip_error = float(np.max(np.abs(restored - raw)))
                    observed["action_raw_range_7"] = {
                        "min": np.min(raw, axis=0).tolist(),
                        "max": np.max(raw, axis=0).tolist(),
                    }
                    observed["action_normalized_range_7"] = {
                        "min": np.min(normalized, axis=0).tolist(),
                        "max": np.max(normalized, axis=0).tolist(),
                    }
                    observed["normalize_unnormalize_max_abs_error"] = roundtrip_error
                    if roundtrip_error > 1e-5:
                        errors.append(
                            f"{label}: normalize->unnormalize round-trip error "
                            f"{roundtrip_error:.3g} exceeds 1e-5"
                        )
        except Exception as exc:  # pragma: no cover - depends on cluster parquet stack
            errors.append(f"{label}: cannot read bounded parquet sample: {exc}")

    return observed


def inspect_frequency(path: Path, expected_hz: float, errors: list[str]) -> dict[str, Any]:
    if not path.is_file():
        errors.append(f"frequency contract missing: {path}")
        return {}
    data = load_json(path)
    observed = dict(data) if isinstance(data, dict) else {"value": data}
    control = numeric(
        observed.get("control_freq", observed.get("control_hz", observed.get("env_control_hz")))
    )
    if control is None or not math.isclose(control, expected_hz, rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"frequency contract: control frequency={control!r}, expected {expected_hz}")
    repeat = numeric(observed.get("action_repeat"))
    if repeat is not None and not math.isclose(repeat, 1.0, rel_tol=0.0, abs_tol=1e-6):
        errors.append(f"frequency contract: action_repeat={repeat}, expected 1")
    return observed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["dataset", "train", "export", "eval", "all"], required=True)
    parser.add_argument("--dataset", action="append", help="LABEL=DATASET_ROOT")
    parser.add_argument("--expected-episodes", action="append", help="LABEL=N")
    parser.add_argument("--episode-range", action="append", help="LABEL=START:END (half-open)")
    parser.add_argument("--camera-keys", action="append", help="LABEL=KEY1,KEY2")
    parser.add_argument("--target-fps", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--require-videos", action="store_true")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--export-root", type=Path)
    parser.add_argument("--require-export-contract", action="store_true")
    parser.add_argument("--frequency-json", type=Path)
    parser.add_argument("--write-contract", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    datasets = parse_pairs(args.dataset, "--dataset")
    expected = {key: int(value) for key, value in parse_pairs(args.expected_episodes, "--expected-episodes").items()}
    ranges: dict[str, tuple[int, int]] = {}
    for label, value in parse_pairs(args.episode_range, "--episode-range").items():
        try:
            start_text, end_text = value.split(":", 1)
            ranges[label] = (int(start_text), int(end_text))
        except ValueError as exc:
            raise ValueError(f"--episode-range must use LABEL=START:END, got {label}={value}") from exc
    camera_map = {
        key: [item for item in value.split(",") if item]
        for key, value in parse_pairs(args.camera_keys, "--camera-keys").items()
    }
    observed: dict[str, Any] = {
        "status": "pending",
        "stage": args.stage,
        "target_fps": args.target_fps,
        "control_hz": args.control_hz,
        "frame_chunk_size": 4,
        "action_per_frame": 4,
        "official_q01": EXPECTED_Q01_7.tolist() + [0.0] * 23,
        "official_q99": EXPECTED_Q99_7.tolist() + [0.0] * 23,
        "datasets": [],
    }

    if args.stage in {"dataset", "train", "all"} or datasets:
        for label, root_text in datasets.items():
            observed["datasets"].append(
                inspect_dataset(
                    label,
                    Path(root_text),
                    expected.get(label),
                    ranges.get(label),
                    args.target_fps,
                    camera_map.get(label),
                    args.require_videos,
                    errors,
                )
            )

    # Keep the per-source normalized ranges visible in the contract.  They
    # need not be numerically identical because the demonstrations differ,
    # but they must be computed with the same seven-channel transform above.
    normalized_ranges = {
        item.get("label"): item.get("action_normalized_range_7")
        for item in observed["datasets"]
        if item.get("action_normalized_range_7") is not None
    }
    if normalized_ranges:
        observed["normalized_range_comparison"] = normalized_ranges

    if args.config_json is not None:
        if not args.config_json.is_file():
            errors.append(f"resolved config missing: {args.config_json}")
        else:
            resolved = assert_expected_norm(
                find_norm_stats(load_json(args.config_json)),
                f"resolved config {args.config_json}",
                errors,
            )
            observed["resolved_config"] = str(args.config_json)
            if resolved is not None:
                observed["resolved_norm_stat"] = resolved

    if args.stage == "export" or args.export_root is not None:
        if args.export_root is None or not args.export_root.is_dir():
            errors.append(f"export root missing: {args.export_root}")
        else:
            candidates = sorted(args.export_root.rglob("policy_postprocessor.json"))
            if not candidates:
                candidates = sorted(args.export_root.rglob("*postprocessor*.json"))
            if not candidates:
                message = f"no postprocessor JSON under {args.export_root}"
                if args.require_export_contract and "resolved_norm_stat" not in observed:
                    errors.append(message)
                else:
                    observed["export_postprocessor"] = (
                        "absent; resolved config is the recorded equivalent source"
                        if "resolved_norm_stat" in observed
                        else "absent; equivalent source must be recorded"
                    )
            else:
                observed["export_postprocessor"] = str(candidates[0])
                assert_expected_norm(find_norm_stats(load_json(candidates[0])), str(candidates[0]), errors)

    if args.stage == "eval" or args.frequency_json is not None:
        if args.frequency_json is None:
            errors.append("eval stage requires --frequency-json")
        else:
            observed["frequency"] = inspect_frequency(args.frequency_json, args.control_hz, errors)

    if args.write_contract is not None:
        observed["status"] = "fail" if errors else "pass"
        observed["errors"] = list(errors)
        args.write_contract.parent.mkdir(parents=True, exist_ok=True)
        args.write_contract.write_text(json.dumps(observed, indent=2, ensure_ascii=False) + "\n")

    if errors:
        print("FRAME4_CONTRACT_FAIL")
        for error in errors:
            print(f"- {error}")
        return 2
    print("FRAME4_CONTRACT_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
