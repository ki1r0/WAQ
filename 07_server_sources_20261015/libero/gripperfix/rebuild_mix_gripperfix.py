#!/usr/bin/env python3
"""Step 2: rebuild the VA2 training mix on top of the gripper-fixed open
dataset. cosmos_success is reused untouched (its convention was already
correct). Layout mirrors libero10_va2_success_mix_20260906 exactly.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

OLD_MIX = "/mnt/zengqixin/datasets/libero10_va2_success_mix_20260906"
PATCHED_OPEN = "/mnt/zengqixin/datasets/libero_10_original_500_gripperfix_20260909/lerobot_v21_10hz_vae128"
NEW_MIX = "/mnt/zengqixin/datasets/libero10_va2_success_mix_gripperfix_20260909"
GRIPPER_DIM = 6

os.makedirs(f"{NEW_MIX}/open_source", exist_ok=True)
# cosmos side: reuse the existing (correct) view wholesale.
if not os.path.lexists(f"{NEW_MIX}/cosmos_success"):
    os.symlink(f"{OLD_MIX}/cosmos_success", f"{NEW_MIX}/cosmos_success")
# open side: data -> patched parquets; latents -> old mix's materialized
# front/wrist rename (latents are unaffected by the action fix); the rest ->
# patched dataset (which itself symlinks the original).
links = {
    "data": f"{PATCHED_OPEN}/data",
    "latents": f"{OLD_MIX}/open_source/latents",
    "jepa": f"{PATCHED_OPEN}/jepa",
    "meta": f"{PATCHED_OPEN}/meta",
    "videos": f"{PATCHED_OPEN}/videos",
}
for name, target in links.items():
    dst = f"{NEW_MIX}/open_source/{name}"
    if not os.path.lexists(dst):
        os.symlink(target, dst)

manifest = json.load(open(f"{OLD_MIX}/view_manifest.json"))
manifest["format"] = "libero10_va2_success_mix_v2_gripperfix"
manifest["gripper_convention"] = "-1=open, +1=close (both sources, verified)"
manifest["provenance"] = {
    "open_source": PATCHED_OPEN,
    "cosmos_success": f"{OLD_MIX}/cosmos_success (untouched)",
    "fix": "open gripper remapped g_new = 1 - 2*g on 2026-09-09",
}
json.dump(manifest, open(f"{NEW_MIX}/view_manifest.json", "w"),
          indent=2, ensure_ascii=False)

# ---- assertions over BOTH sources ----
def check(pat, name):
    fs = sorted(glob.glob(pat))
    assert fs, f"no parquets under {pat}"
    g_all, firsts = [], []
    for f in fs[:60]:
        a = np.stack(pd.read_parquet(f, columns=["action"])["action"].to_numpy())
        g_all.append(a[:, GRIPPER_DIM])
        firsts.append(a[:5, GRIPPER_DIM].mean())
    g = np.concatenate(g_all)
    u = set(np.unique(np.round(g, 3)).tolist())
    m = float(np.mean(firsts))
    print(f"{name}: unique={sorted(u)} start-mean={m:+.3f} n={len(g)}")
    assert u.issubset({-1.0, 1.0}), f"{name}: gripper values {sorted(u)} not in {{-1,1}}"
    assert m < -0.9, f"{name}: episode start not open (mean {m})"

check(f"{NEW_MIX}/open_source/data/chunk-*/*.parquet", "open_source")
check(f"{NEW_MIX}/cosmos_success/data/chunk-*/*.parquet", "cosmos_success")
print("MIX REBUILD OK ->", NEW_MIX)
