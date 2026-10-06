#!/usr/bin/env python3
"""Build a compact, auditable Cosmos-success view for the frame2 run.

The full Cosmos conversion is kept read-only.  The view reindexes only the
successes in the canonical first-500 episode range and links their newly
generated 10-Hz sidecars; it never links the old frame4 cache.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def safe_link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-full", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=500)
    ap.add_argument("--expected", type=int, default=376)
    args = ap.parse_args()

    src = args.cosmos_full.resolve()
    out = args.output_root.resolve()
    rows = read_jsonl(src / "meta" / "episodes.jsonl")
    selected = [r for r in rows if args.start <= int(r["episode_index"]) < args.end and bool(r.get("success", False))]
    selected.sort(key=lambda r: int(r["episode_index"]))
    if len(selected) != args.expected:
        raise RuntimeError(f"expected {args.expected} successes in {args.start}:{args.end}, got {len(selected)}")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True)

    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    info.update({
        "total_episodes": len(selected),
        "total_frames": sum(int(r["length"]) for r in selected),
        "total_chunks": 1,
        "splits": {"train": f"0:{len(selected)}"},
        "lingbotva_latent_fps": 10.0,
        "lingbotva_action_downsample_factor": 2,
        "lingbotva_video_materialized": False,
    })
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    stats_rows = read_jsonl(src / "meta" / "episodes_stats.jsonl") if (src / "meta" / "episodes_stats.jsonl").is_file() else []
    stats_by_ep = {int(r.get("episode_index", -1)): r for r in stats_rows}
    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f_ep, (out / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as f_st:
        for new_idx, row in enumerate(selected):
            old_idx = int(row["episode_index"])
            item = dict(row)
            item["source_episode_index"] = old_idx
            item["episode_index"] = new_idx
            f_ep.write(json.dumps(item, ensure_ascii=False) + "\n")
            if old_idx in stats_by_ep:
                stat = dict(stats_by_ep[old_idx])
                stat["episode_index"] = new_idx
                f_st.write(json.dumps(stat, ensure_ascii=False) + "\n")

            safe_link(out / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet", src / "data" / "chunk-000" / f"episode_{old_idx:06d}.parquet")
            for key in ("observation.images.front", "observation.images.wrist"):
                src_dir = src / "latents" / "chunk-000" / key
                files = sorted(src_dir.glob(f"episode_{old_idx:06d}_*.pth"))
                if not files:
                    raise RuntimeError(f"missing latent for episode {old_idx}: {src_dir}")
                for source_file in files:
                    suffix = source_file.name[len(f"episode_{old_idx:06d}"):]
                    safe_link(out / "latents" / "chunk-000" / key / f"episode_{new_idx:06d}{suffix}", source_file)
            jepa = src / "jepa" / "chunk-000" / f"episode_{old_idx:06d}.pt"
            if not jepa.is_file():
                raise RuntimeError(f"missing JEPA sidecar: {jepa}")
            safe_link(out / "jepa" / "chunk-000" / f"episode_{new_idx:06d}.pt", jepa)

    tasks = src / "meta" / "tasks.jsonl"
    if tasks.is_file():
        safe_link(out / "meta" / "tasks.jsonl", tasks)

    manifest = {
        "format": "libero10_va2_success_mix_orientationfix_gripperfix_frame2_v1",
        "target_fps": 10,
        "source_fps": 20,
        "frame_chunk_size": 2,
        "action_per_frame": 4,
        "temporal_downsample_factor": 2,
        "action_downsample_factor": 2,
        "action_downsample_policy": "stride_2",
        "cosmos_source": str(src),
        "cosmos_source_range": f"{args.start}:{args.end}",
        "cosmos_success_episodes": len(selected),
        "gripper_convention": "-1=open, +1=close (source verified)",
        "source_datasets_untouched": True,
    }
    (out / "view_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"COSMOS_FRAME2_SUCCESS_VIEW_OK episodes={len(selected)}")


if __name__ == "__main__":
    main()
