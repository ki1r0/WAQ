#!/usr/bin/env python3
"""Build a camera-key-normalized view of the fixed open500 frame2 cache.

The VAE/JEPA extraction itself is unchanged.  This view only maps the open
converter's ``image``/``wrist_image`` keys to the official LIBERO-plus
``front``/``wrist`` keys so the open and Cosmos repositories can be loaded by
one VA2 config without mixing stale sidecars.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def link(dst: Path, src: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise FileExistsError(dst)
    dst.symlink_to(src)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    src = args.source.resolve()
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    # Metadata is copied so the source repository remains untouched.
    shutil.copytree(src / "meta", out / "meta")
    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    renames = {
        "observation.images.image": "observation.images.front",
        "observation.images.wrist_image": "observation.images.wrist",
    }
    info["features"] = {renames.get(k, k): v for k, v in features.items()}
    info.update({
        "camera_key_normalization": renames,
        "source_dataset": str(src),
        "lingbotva_latent_fps": 10.0,
        "lingbotva_action_downsample_factor": 2,
    })
    info_path.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Parquet actions are camera-independent; link the whole data tree.
    link(out / "data", src / "data")

    # Present videos and VAE latent directories with official camera names.
    for parent_name in ("videos", "latents"):
        for old, new in (("observation.images.image", "observation.images.front"),
                         ("observation.images.wrist_image", "observation.images.wrist")):
            old_root = src / parent_name / "chunk-000" / old
            if old_root.exists():
                link(out / parent_name / "chunk-000" / new, old_root)

    # JEPA files contain camera keys in their dictionaries, so rewrite the
    # small sidecars while retaining tensors and frame_ids exactly.
    for src_file in sorted((src / "jepa").glob("chunk-*/*.pt")):
        rel = src_file.relative_to(src / "jepa")
        dst_file = out / "jepa" / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        payload = torch.load(src_file, map_location="cpu", weights_only=False)
        renamed = {renames.get(k, k): v for k, v in payload.items()}
        torch.save(renamed, dst_file)

    manifest = {
        "format": "libero10_va2_success_mix_orientationfix_gripperfix_frame2_open_normalized_v1",
        "source": str(src),
        "camera_key_map": renames,
        "target_fps": 10,
        "source_fps": 20,
        "frame_chunk_size": 2,
        "action_per_frame": 4,
        "temporal_downsample_factor": 2,
        "action_downsample_factor": 2,
        "action_downsample_policy": "stride_2",
        "gripper_convention": "-1=open, +1=close (source verified)",
        "image_orientation": "source is orientation-fixed; no additional rotation applied",
        "stale_sidecars_reused": False,
    }
    (out / "view_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
