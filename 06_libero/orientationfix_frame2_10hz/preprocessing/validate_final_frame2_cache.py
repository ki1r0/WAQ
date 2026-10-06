#!/usr/bin/env python3
"""Strict validation for the final frame2/10Hz open+Cosmos views."""
from __future__ import annotations

import json
from pathlib import Path

import torch


def main() -> int:
    root = Path(__import__("sys").argv[1])
    expected = {"open500": 500, "cosmos_success376": 376}
    errors: list[str] = []
    reports: dict[str, dict] = {}
    for name, n_expected in expected.items():
        p = root / name
        info = json.loads((p / "meta" / "info.json").read_text(encoding="utf-8"))
        episodes = (p / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        jfiles = sorted((p / "jepa").glob("chunk-*/*.pt"))
        front = sorted((p / "latents").glob("chunk-*/observation.images.front/*.pth"))
        wrist = sorted((p / "latents").glob("chunk-*/observation.images.wrist/*.pth"))
        feats = sorted(k for k in info.get("features", {}) if k.startswith("observation.images."))
        if len(episodes) != n_expected:
            errors.append(f"{name}: episodes={len(episodes)} expected={n_expected}")
        if len(jfiles) != n_expected:
            errors.append(f"{name}: jepa={len(jfiles)} expected={n_expected}")
        if len(front) != n_expected or len(wrist) != n_expected or [x.name for x in front] != [x.name for x in wrist]:
            errors.append(f"{name}: latent sets front={len(front)} wrist={len(wrist)}")
        if feats != ["observation.images.front", "observation.images.wrist"]:
            errors.append(f"{name}: camera features={feats}")
        checked = 0
        for jf in jfiles:
            jo = torch.load(jf, map_location="cpu", weights_only=False)
            keys = sorted(k for k in jo if k.startswith("observation.images."))
            if keys != ["observation.images.front", "observation.images.wrist"]:
                errors.append(f"{name}/{jf.name}: jepa keys={keys}")
                continue
            ids = [int(x) for x in jo["frame_ids"]]
            if len(ids) < 2 or any(b - a != 2 for a, b in zip(ids, ids[1:])):
                errors.append(f"{name}/{jf.name}: jepa frame_ids not contiguous stride2")
            ep = jf.stem
            for cam, files in (("front", front), ("wrist", wrist)):
                matches = [x for x in files if x.name.startswith(ep + "_")]
                if len(matches) != 1:
                    errors.append(f"{name}/{jf.name}: {cam} latent matches={len(matches)}")
                    continue
                lo = torch.load(matches[0], map_location="cpu", weights_only=False)
                lids = [int(x) for x in lo["frame_ids"]]
                if ids != lids:
                    errors.append(f"{name}/{jf.name}: {cam} JEPA/VAE frame ids differ ({len(ids)} vs {len(lids)})")
                if float(lo.get("fps", -1)) != 10.0 or float(lo.get("ori_fps", -1)) != 20.0:
                    errors.append(f"{name}/{matches[0].name}: fps metadata={lo.get('fps')}/{lo.get('ori_fps')}")
            checked += 1
        reports[name] = {"episodes": len(episodes), "jepa": len(jfiles), "vae_front": len(front), "vae_wrist": len(wrist), "checked": checked, "camera_features": feats}
    result = {"status": "pass" if not errors else "fail", "root": str(root), "reports": reports, "errors": errors}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
