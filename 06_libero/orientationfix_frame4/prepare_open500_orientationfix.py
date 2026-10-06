#!/usr/bin/env python3
"""Build an orientation-corrected LeRobot v2.1 dataset from LIBERO open500.

The raw open500 HDF5 files are read-only. Both RGB streams are rotated 180
degrees before they are written, matching the official LIBERO conversion
(``image[::-1, ::-1]``). Actions use the existing LIBERO convention for the
first six dimensions and map the gripper to {-1, +1}.

This implementation writes one final LeRobot dataset directly. It does not
depend on the generic any4lerobot aggregation module, because the installed
LeRobot v2.1 package on the cluster exposes ``LeRobotDataset`` from
``lerobot.datasets.lerobot_dataset`` rather than re-exporting it from the
namespace package.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CAMERA_KEYS = {
    "agentview_rgb": "observation.images.image",
    "eye_in_hand_rgb": "observation.images.wrist_image",
}


FEATURES: dict[str, Any] = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.images.wrist_image": {
        "dtype": "video",
        "shape": (128, 128, 3),
        "names": ["height", "width", "rgb"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "motors": [
                "x",
                "y",
                "z",
                "axis_angle1",
                "axis_angle2",
                "axis_angle3",
                "gripper",
                "gripper",
            ]
        },
    },
    "observation.states.ee_state": {
        "dtype": "float32",
        "shape": (6,),
        "names": {
            "motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3"]
        },
    },
    "observation.states.joint_state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {"motors": [f"joint_{i}" for i in range(7)]},
    },
    "observation.states.gripper_state": {
        "dtype": "float32",
        "shape": (2,),
        "names": {"motors": ["gripper", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": ["x", "y", "z", "axis_angle1", "axis_angle2", "axis_angle3", "gripper"]
        },
    },
}


def rotate180(frames: np.ndarray) -> np.ndarray:
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [T,H,W,3] RGB frames, got {frames.shape}")
    return np.ascontiguousarray(frames[:, ::-1, ::-1])


def task_instruction(path: Path) -> str:
    name = path.stem
    if name.endswith("_demo"):
        name = name[:-5]
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] == "KITCHEN" and parts[1].startswith("SCENE"):
        parts = parts[2:]
    elif len(parts) >= 3 and parts[0] == "LIVING" and parts[1] == "ROOM":
        parts = parts[3:]
    elif len(parts) >= 3 and parts[0] == "STUDY" and parts[1].startswith("SCENE"):
        parts = parts[2:]
    return " ".join(parts)


def episode_sources(source_root: Path) -> list[tuple[Path, str, str]]:
    sources: list[tuple[Path, str, str]] = []
    for path in sorted(source_root.glob("*.hdf5")):
        instruction = task_instruction(path)
        with h5py.File(path, "r") as handle:
            for demo_name in sorted(handle["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                sources.append((path, demo_name, instruction))
    return sources


def map_action(raw_action: np.ndarray) -> np.ndarray:
    if raw_action.ndim != 2 or raw_action.shape[1] < 7:
        raise ValueError(f"invalid action shape {raw_action.shape}")
    # Existing converter: {-1,+1} -> {1,0}; then gripperfix: {0,1} -> {-1,+1}.
    g_after_converter = 1.0 - np.clip(raw_action[:, -1], 0.0, 1.0)
    gripper_pm1 = 1.0 - 2.0 * g_after_converter
    return np.concatenate([raw_action[:, :6], gripper_pm1[:, None]], axis=1).astype(np.float32)


def verify(output_root: Path, sources: list[tuple[Path, str, str]]) -> None:
    meta = output_root / "meta" / "episodes.jsonl"
    if not meta.exists():
        raise RuntimeError(f"missing {meta}")
    rows = [json.loads(line) for line in meta.read_text().splitlines() if line.strip()]
    if len(rows) != 500:
        raise RuntimeError(f"expected 500 output episodes, got {len(rows)}")
    videos = sorted((output_root / "videos").rglob("*.mp4"))
    if len(videos) != 1000:
        raise RuntimeError(f"expected 1000 videos (2 cameras x 500), got {len(videos)}")
    links = [p for p in output_root.rglob("*") if p.is_symlink()]
    if links:
        raise RuntimeError(f"output contains symlinks, first={links[0]}")

    import pandas as pd

    bad_actions: list[str] = []
    for parquet in sorted((output_root / "data").rglob("*.parquet")):
        values = pd.read_parquet(parquet, columns=["action"])["action"].to_numpy()
        actions = np.stack(values).astype(np.float32)
        gripper = np.round(actions[:, 6], 3)
        if not set(np.unique(gripper).tolist()).issubset({-1.0, 1.0}):
            bad_actions.append(str(parquet))
    if bad_actions:
        raise RuntimeError(f"non +/-1 gripper values in {bad_actions[:2]}")

    # A small decoded probe proves the encoded output is the rotated source,
    # rather than merely checking that files exist.
    try:
        import cv2
        import imageio.v3 as iio

        sample_indices = [0, len(sources) // 2, len(sources) - 1]
        for output_key, source_key in CAMERA_KEYS.items():
            camera_videos = sorted(
                (output_root / "videos" / "chunk-000" / output_key).glob("*.mp4")
            )
            if len(camera_videos) != 500:
                raise RuntimeError(f"{output_key}: expected 500 videos, got {len(camera_videos)}")
            for index in sample_indices:
                source_path, demo_name, _ = sources[index]
                with h5py.File(source_path, "r") as handle:
                    raw = np.asarray(handle[f"data/{demo_name}/obs/{source_key}"][0])
                decoded = np.asarray(iio.imread(camera_videos[index], index=0))
                if decoded.shape != raw.shape:
                    decoded = cv2.resize(decoded, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_AREA)
                rotated = raw[::-1, ::-1]
                raw_err = float(np.mean(np.abs(decoded.astype(np.float32) - raw.astype(np.float32))))
                rot_err = float(np.mean(np.abs(decoded.astype(np.float32) - rotated.astype(np.float32))))
                if not rot_err < raw_err:
                    raise RuntimeError(
                        f"orientation probe failed camera={source_key} episode={index}: "
                        f"rotated={rot_err:.3f} raw={raw_err:.3f}"
                    )
    except ImportError as exc:
        raise RuntimeError("orientation verification needs imageio and opencv") from exc

    manifest = output_root / "orientation_manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "episode": index,
                    "source_h5": str(path),
                    "source_demo": demo,
                    "task": instruction,
                    "camera_keys": list(CAMERA_KEYS.values()),
                    "rotate": "[::-1,::-1]",
                    "rotate_k": 2,
                    "gripper": "g_new=1-2*g_after_converter",
                },
                ensure_ascii=False,
            )
            + "\n"
            for index, (path, demo, instruction) in enumerate(sources)
        )
    )
    print(f"wrote {manifest} ({len(sources)} episodes)")
    print("ORIENTATION_FIX_OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-h5-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rotate-cameras", nargs="+", required=True)
    parser.add_argument("--rotate-k", type=int, default=2)
    parser.add_argument("--fix-gripper-01-to-pm1", action="store_true")
    parser.add_argument("--source-fps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8, help="image-writer processes")
    args = parser.parse_args()
    if not args.fix_gripper_01_to_pm1:
        raise SystemExit("--fix-gripper-01-to-pm1 is required")
    if args.source_fps != 20:
        raise SystemExit("open500 source fps must be 20")
    if sorted(args.rotate_cameras) != sorted(CAMERA_KEYS):
        raise SystemExit(f"must rotate both cameras: {list(CAMERA_KEYS)}")
    if args.rotate_k != 2:
        raise SystemExit("LIBERO orientation fix requires rotate_k=2 (180 degrees)")

    source_root = args.source_h5_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise SystemExit(f"missing source {source_root}")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing non-empty output {output_root}")
    sources = episode_sources(source_root)
    if len(sources) != 500:
        raise SystemExit(f"expected 500 source episodes, got {len(sources)}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset.create(
        repo_id="local/libero_open500_orientationfix_20260911",
        root=output_root,
        fps=20,
        robot_type="franka",
        features=FEATURES,
        use_videos=True,
        image_writer_processes=args.workers,
        image_writer_threads=max(1, args.workers * 2),
        video_backend="pyav",
    )

    for episode_index, (source_path, demo_name, instruction) in enumerate(sources):
        with h5py.File(source_path, "r") as handle:
            demo = handle[f"data/{demo_name}"]
            agentview = rotate180(np.asarray(demo["obs/agentview_rgb"]))
            wrist = rotate180(np.asarray(demo["obs/eye_in_hand_rgb"]))
            action = map_action(np.asarray(demo["actions"]))
            ee_state = np.asarray(demo["obs/ee_states"], dtype=np.float32)
            gripper_state = np.asarray(demo["obs/gripper_states"], dtype=np.float32)
            joint_state = np.asarray(demo["obs/joint_states"], dtype=np.float32)
            state = np.concatenate([ee_state, gripper_state], axis=1)
            length = len(agentview)
            arrays = [wrist, action, state, ee_state, gripper_state, joint_state]
            if any(len(array) != length for array in arrays):
                raise ValueError(f"inconsistent demo lengths in {source_path}:{demo_name}")
            for frame_index in range(length):
                dataset.add_frame(
                    {
                        "observation.images.image": agentview[frame_index],
                        "observation.images.wrist_image": wrist[frame_index],
                        "observation.state": state[frame_index],
                        "observation.states.ee_state": ee_state[frame_index],
                        "observation.states.joint_state": joint_state[frame_index],
                        "observation.states.gripper_state": gripper_state[frame_index],
                        "action": action[frame_index],
                    },
                    task=instruction,
                )
            dataset.save_episode()
        if episode_index % 25 == 0 or episode_index == len(sources) - 1:
            print(f"[convert] episode={episode_index + 1}/500 task={instruction}", flush=True)

    verify(output_root, sources)

    # Do not let a converted dataset silently proceed to feature extraction or
    # training.  The contract checker is kept as a separate, reusable script,
    # but this producer invokes it at the stage boundary so the check is part
    # of the executable pipeline rather than only a runbook instruction.
    validator = Path(__file__).with_name("validate_frame4_contract.py")
    contract = Path(__file__).with_name("normalization_contract.json")
    if not validator.is_file():
        raise SystemExit(f"missing frame4 contract validator: {validator}")
    subprocess.run(
        [
            sys.executable,
            str(validator),
            "--stage",
            "dataset",
            "--dataset",
            f"open={output_root}",
            "--expected-episodes",
            "open=500",
            "--camera-keys",
            "open=observation.images.image,observation.images.wrist_image",
            "--target-fps",
            str(args.source_fps),
            "--require-videos",
            "--write-contract",
            str(contract),
        ],
        check=True,
    )
    print(f"FRAME4_CONTRACT_OK {contract}", flush=True)


if __name__ == "__main__":
    main()
