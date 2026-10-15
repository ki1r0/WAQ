#!/usr/bin/env python3
"""Step 1: patched copy of the open_500 lerobot dataset with the gripper
column remapped from {0,1} (1=open) to {-1,+1} (-1=open): g_new = 1 - 2*g.

The original dataset is NEVER modified (it is provenance for historical
experiments). Output: a sibling dataset dir where everything except data/
is a symlink to the original, and data/ parquets are rewritten.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

SRC = "/mnt4/zengqixin/dataset/libero_10_original_500_20260729/lerobot_v21_10hz_vae128"
DST = "/mnt/zengqixin/datasets/libero_10_original_500_gripperfix_20260909/lerobot_v21_10hz_vae128"
GRIPPER_DIM = 6

os.makedirs(DST, exist_ok=True)
# symlink everything except data/ and inference_state_cache (stale-action cache
# must NOT be carried over silently).
for name in sorted(os.listdir(SRC)):
    if name in ("data", "inference_state_cache"):
        continue
    dst = os.path.join(DST, name)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(SRC, name), dst)

files = sorted(glob.glob(f"{SRC}/data/chunk-*/*.parquet"))
assert len(files) == 500, f"expected 500 episode parquets, got {len(files)}"
n_done = 0
for f in files:
    rel = os.path.relpath(f, SRC)
    out = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df = pd.read_parquet(f)
    acts = np.stack(df["action"].to_numpy()).astype(np.float32)
    g = acts[:, GRIPPER_DIM]
    u = set(np.unique(np.round(g, 3)).tolist())
    if not u.issubset({0.0, 1.0}):
        raise RuntimeError(f"{f}: unexpected source gripper values {sorted(u)}")
    acts[:, GRIPPER_DIM] = 1.0 - 2.0 * g
    df["action"] = list(acts)
    df.to_parquet(out, index=False)
    n_done += 1
print(f"patched {n_done} parquets -> {DST}/data")

# ---- verification ----
bad = 0
firsts = []
for f in sorted(glob.glob(f"{DST}/data/chunk-*/*.parquet")):
    a = np.stack(pd.read_parquet(f, columns=["action"])["action"].to_numpy())
    g = a[:, GRIPPER_DIM]
    if not set(np.unique(np.round(g, 3)).tolist()).issubset({-1.0, 1.0}):
        bad += 1
    firsts.append(g[:5].mean())
m = float(np.mean(firsts))
print(f"verify: non-conforming files={bad}; episode-start gripper mean={m:+.3f} "
      f"(must be ~ -1.0 = open)")
if bad or m > -0.9:
    sys.exit("VERIFICATION FAILED")
print("VERIFICATION OK")
