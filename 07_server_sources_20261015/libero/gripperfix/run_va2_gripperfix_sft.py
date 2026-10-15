#!/usr/bin/env python3
"""Step 3 launcher: rerun the exact VA2 SFT with the gripper-fixed mix.

Wraps siliang's original from-scratch script (config untouched) and only
rewrites the DATASET constant + save root, following the same source-patch
pattern as run_va2_libero10_cosmos_bs1_accum2_resume7500.py.
Launch it exactly the way the original run_va2_libero10_cosmos_sft.py was
launched (same deepspeed/torchrun invocation, same GPUs), e.g.:
  VA2_SAVE_ROOT=/mnt/zengqixin/checkpoints/va2_libero10_cosmos_gripperfix_20260909/bf16_bs1_accum2_lr1e-5_25k \
  <same launcher> run_va2_gripperfix_sft.py
"""
from pathlib import Path

SOURCE = Path(
    "/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/run_va2_libero10_cosmos_sft.py"
)
NEW_DATASET = "/mnt/zengqixin/datasets/libero10_va2_success_mix_gripperfix_20260909"
NEW_SAVE = ("/mnt/zengqixin/checkpoints/va2_libero10_cosmos_gripperfix_20260909/"
            "bf16_bs1_accum2_lr1e-5_25k")


def main() -> None:
    src = SOURCE.read_text()
    old_ds = 'DATASET = "/mnt/zengqixin/datasets/libero10_va2_success_mix_20260906"'
    assert old_ds in src, "DATASET constant not found; source script changed"
    src = src.replace(old_ds, f'DATASET = "{NEW_DATASET}"')
    old_sv = ('SAVE_ROOT = "/mnt/zengqixin/checkpoints/va2_libero10_cosmos_20260906/'
              'bf16_6gpu_bs1_accum2_lr1e-5_25k"')
    assert old_sv in src, "SAVE_ROOT constant not found; source script changed"
    src = src.replace(old_sv, f'SAVE_ROOT = "{NEW_SAVE}"')
    ns = {"__name__": "__main__", "__file__": str(SOURCE)}
    exec(compile(src, str(SOURCE), "exec"), ns)


if __name__ == "__main__":
    main()
