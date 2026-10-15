#!/usr/bin/env python3
"""Build the read-only 20-Hz/frame4 VA2 success mix.

The source datasets are never copied or modified.  The view contains the
complete 500-episode open set and the successful episodes among Cosmos
episodes 0..499 (expected 376), with Cosmos episode indices compacted only in
the view metadata and symlink filenames.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() == src.resolve():
            return
        raise RuntimeError(f"refusing to replace existing symlink: {dst}")
    if dst.exists():
        raise RuntimeError(f"refusing to overwrite existing path: {dst}")
    dst.symlink_to(src, target_is_directory=src.is_dir())


def build_open(root: Path, source: Path) -> None:
    for name in ("data", "videos", "meta", "jepa"):
        link(root / name, source / name)
    # Keep open's native sidecar keys in the source view.  The training
    # catalog records this mapping explicitly; mutating the source jepa tree
    # to add aliases would violate the read-only-source contract.
    link(root / "latents/chunk-000/observation.images.front",
         source / "latents/chunk-000/observation.images.image")
    link(root / "latents/chunk-000/observation.images.wrist",
         source / "latents/chunk-000/observation.images.wrist_image")


def build_cosmos(root: Path, source: Path, start: int, end: int, expected: int) -> int:
    rows = [r for r in read_jsonl(source / "meta/episodes.jsonl")
            if start <= int(r["episode_index"]) < end and bool(r.get("success", False))]
    rows.sort(key=lambda r: int(r["episode_index"]))
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} Cosmos successes in {start}:{end}, got {len(rows)}")
    original = [int(r["episode_index"]) for r in rows]
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)
    info = json.loads((source / "meta/info.json").read_text(encoding="utf-8"))
    info.update({
        "total_episodes": len(rows),
        "total_frames": sum(int(r["length"]) for r in rows),
        "total_chunks": 1,
        "splits": {"train": f"0:{len(rows)}"},
    })
    (root / "meta/info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (root / "meta/episodes.jsonl").open("w", encoding="utf-8") as f:
        for new_idx, row in enumerate(rows):
            out = dict(row)
            out["source_episode_index"] = original[new_idx]
            out["episode_index"] = new_idx
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    for name in ("tasks.jsonl",):
        if (source / "meta" / name).exists():
            link(root / "meta" / name, source / "meta" / name)
    for new_idx, old_idx in enumerate(original):
        link(root / f"data/chunk-000/episode_{new_idx:06d}.parquet",
             source / f"data/chunk-000/episode_{old_idx:06d}.parquet")
        # VAE latents are stored per camera and may contain multiple temporal
        # windows.  JEPA is a single per-episode file whose payload contains
        # both camera keys (the layout produced by the frame4 converter).
        for key in ("observation.images.front", "observation.images.wrist"):
            src_dir = source / "latents" / "chunk-000" / key
            if not src_dir.is_dir():
                raise RuntimeError(f"missing Cosmos latent camera directory: {src_dir}")
            files = sorted(src_dir.glob(f"episode_{old_idx:06d}_*.pth"))
            if not files:
                raise RuntimeError(f"missing Cosmos latent sidecar for episode {old_idx}: {src_dir}")
            for src in files:
                suffix = src.name[len(f"episode_{old_idx:06d}"):]
                link(root / "latents" / "chunk-000" / key / f"episode_{new_idx:06d}{suffix}", src)
        jepa_src = source / "jepa" / "chunk-000" / f"episode_{old_idx:06d}.pt"
        if not jepa_src.is_file():
            raise RuntimeError(f"missing Cosmos JEPA sidecar for episode {old_idx}: {jepa_src}")
        link(root / "jepa" / "chunk-000" / f"episode_{new_idx:06d}.pt", jepa_src)
    return len(rows)


def check_actions(root: Path, label: str, expected: int) -> None:
    import numpy as np
    import pandas as pd
    files = sorted((root / "data").glob("chunk-*/*.parquet"))
    if len(files) != expected:
        raise RuntimeError(f"{label}: expected {expected} parquet files, got {len(files)}")
    vals = []
    for p in files[: min(64, len(files))]:
        arr = np.stack(pd.read_parquet(p, columns=["action"])["action"].to_numpy())
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise RuntimeError(f"{label}: bad action shape {arr.shape} in {p}")
        vals.append(arr[:, 6])
    gripper = set(np.unique(np.concatenate(vals)).round(6).tolist())
    if not gripper.issubset({-1.0, 1.0}):
        raise RuntimeError(f"{label}: gripper values {sorted(gripper)}")
    print(f"{label}: episodes={len(files)} action_dim=7 gripper={sorted(gripper)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--open-root", type=Path, required=True)
    ap.add_argument("--cosmos-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--cosmos-start", type=int, default=0)
    ap.add_argument("--cosmos-end", type=int, default=500)
    ap.add_argument("--expected-open", type=int, default=500)
    ap.add_argument("--expected-cosmos-success", type=int, default=376)
    args = ap.parse_args()
    for source in (args.open_root, args.cosmos_root):
        if not source.is_dir():
            raise FileNotFoundError(source)
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    build_open(args.output_root / "open_source", args.open_root)
    cosmos_n = build_cosmos(
        args.output_root / "cosmos_success", args.cosmos_root,
        args.cosmos_start, args.cosmos_end, args.expected_cosmos_success,
    )
    check_actions(args.output_root / "open_source", "open_source", args.expected_open)
    check_actions(args.output_root / "cosmos_success", "cosmos_success", cosmos_n)

    manifest = {
        "format": "libero10_va2_success_mix_orientationfix_frame4_v1",
        "frame_chunk_size": 4,
        "action_per_frame": 4,
        "target_fps": 20,
        "action_downsample_factor": 1,
        "open_source": str(args.open_root),
        "open_source_episodes": args.expected_open,
        "cosmos_source": str(args.cosmos_root),
        "cosmos_source_range": f"{args.cosmos_start}:{args.cosmos_end}",
        "cosmos_success_episodes": cosmos_n,
        "camera_mapping": {
            "open_source.observation.images.image": "observation.images.front",
            "open_source.observation.images.wrist_image": "observation.images.wrist",
            "cosmos_success.observation.images.front": "observation.images.front",
            "cosmos_success.observation.images.wrist": "observation.images.wrist",
        },
        "source_datasets_untouched": True,
    }
    (args.output_root / "view_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)
    print(f"MIX_REBUILD_OK -> {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
