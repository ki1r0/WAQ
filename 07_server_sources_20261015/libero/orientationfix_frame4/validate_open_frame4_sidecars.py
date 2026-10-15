#!/usr/bin/env python3
"""Full coverage/shape/frame-id validation for the open500 frame4 sidecars."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch


def ints(x):
    return [int(v) for v in (x.tolist() if hasattr(x, "tolist") else x)]


def main(root: str, report_path: str) -> None:
    r = Path(root)
    out = {
        "dataset_root": str(r),
        "jepa_files": 0,
        "jepa_episodes": 0,
        "jepa_missing_keys": 0,
        "jepa_bad_shapes": 0,
        "jepa_bad_frame_ids": 0,
        "jepa_load_errors": 0,
        "vae_files": 0,
        "vae_load_errors": 0,
        "vae_frame_ids_not_in_jepa": 0,
        "vae_latent_bad_shapes": 0,
        "vae_fps_bad": 0,
        "vae_frame4_bad": 0,
        "episodes_jsonl": 0,
        "episodes_with_empty_action_config": 0,
        "examples": [],
    }
    jmap = {}
    for p in sorted((r / "jepa").glob("chunk-*/episode_*.pt")):
        out["jepa_files"] += 1
        m = re.search(r"episode_(\d+)\.pt$", p.name)
        if not m:
            continue
        ep = int(m.group(1))
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            required = ("frame_ids", "observation.images.image", "observation.images.wrist_image")
            if any(k not in d for k in required):
                out["jepa_missing_keys"] += 1
                if len(out["examples"]) < 10:
                    out["examples"].append({"kind": "jepa_missing_keys", "file": str(p), "keys": sorted(d)})
                continue
            fids = ints(d["frame_ids"])
            jmap[ep] = set(fids)
            if not fids or fids != sorted(set(fids)):
                out["jepa_bad_frame_ids"] += 1
            for k in ("observation.images.image", "observation.images.wrist_image"):
                shape = tuple(d[k].shape) if hasattr(d[k], "shape") else ()
                if len(shape) != 4 or shape[1:] != (8, 8, 1664):
                    out["jepa_bad_shapes"] += 1
                    if len(out["examples"]) < 10:
                        out["examples"].append({"kind": "jepa_bad_shape", "file": str(p), "key": k, "shape": shape})
        except Exception as exc:
            out["jepa_load_errors"] += 1
            if len(out["examples"]) < 10:
                out["examples"].append({"kind": "jepa_load_error", "file": str(p), "error": repr(exc)})
    out["jepa_episodes"] = len(jmap)

    for p in sorted((r / "latents").glob("chunk-*/observation.images.*/episode_*.pth")):
        out["vae_files"] += 1
        try:
            d = torch.load(p, map_location="cpu", weights_only=False)
            m = re.search(r"episode_(\d+)_", p.name)
            ep = int(m.group(1)) if m else -1
            fids = ints(d.get("frame_ids", []))
            if ep not in jmap or not fids or not all(v in jmap[ep] for v in fids):
                out["vae_frame_ids_not_in_jepa"] += 1
            shape = tuple(d["latent"].shape) if hasattr(d.get("latent"), "shape") else ()
            if len(shape) != 2 or shape[1] != 48:
                out["vae_latent_bad_shapes"] += 1
            if float(d.get("fps", -1)) != 20.0 or float(d.get("ori_fps", -1)) != 20.0:
                out["vae_fps_bad"] += 1
            if (len(fids) - 1) % 4 != 0:
                out["vae_frame4_bad"] += 1
        except Exception as exc:
            out["vae_load_errors"] += 1
            if len(out["examples"]) < 10:
                out["examples"].append({"kind": "vae_load_error", "file": str(p), "error": repr(exc)})

    ep_path = r / "meta" / "episodes.jsonl"
    if ep_path.exists():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            out["episodes_jsonl"] += 1
            row = json.loads(line)
            if not row.get("action_config"):
                out["episodes_with_empty_action_config"] += 1

    out["status"] = "pass" if all(
        out[k] == 0 for k in (
            "jepa_missing_keys", "jepa_bad_shapes", "jepa_bad_frame_ids", "jepa_load_errors",
            "vae_load_errors", "vae_frame_ids_not_in_jepa", "vae_latent_bad_shapes", "vae_fps_bad",
            "vae_frame4_bad", "episodes_with_empty_action_config",
        )
    ) and out["jepa_files"] == 500 and out["jepa_episodes"] == 500 and out["vae_files"] == 1030 and out["episodes_jsonl"] == 500 else "fail"
    Path(report_path).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
