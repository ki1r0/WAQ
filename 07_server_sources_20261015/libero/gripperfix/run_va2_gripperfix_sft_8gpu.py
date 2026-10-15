#!/usr/bin/env python3
"""Step 3 launcher: exact gripper-fixed VA2 SFT, pinned to eight GPUs.

This is a sibling of ``run_va2_gripperfix_sft.py``.  Its training source,
dataset, model initialization, output root, and all training hyperparameters
remain identical; only the distributed launch contract is checked here.
Launch with DeepSpeed on physical GPUs 0 through 7.
"""
from pathlib import Path
import os


SOURCE = Path(
    "/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/run_va2_libero10_cosmos_sft.py"
)
NEW_DATASET = "/mnt/zengqixin/datasets/libero10_va2_success_mix_gripperfix_20260909"
NEW_SAVE = ("/mnt/zengqixin/checkpoints/va2_libero10_cosmos_gripperfix_20260909/"
            "bf16_bs1_accum2_lr1e-5_25k")

EXPECTED_CUDA_VISIBLE_DEVICES = "0,1,2,3,4,5,6,7"
EXPECTED_WORLD_SIZE = 8


def main() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    world_size = os.environ.get("WORLD_SIZE")
    if visible_devices != EXPECTED_CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            "Refusing to run: expected CUDA_VISIBLE_DEVICES="
            f"{EXPECTED_CUDA_VISIBLE_DEVICES!r}, got {visible_devices!r}"
        )
    if world_size != str(EXPECTED_WORLD_SIZE):
        raise RuntimeError(
            f"Refusing to run: expected WORLD_SIZE={EXPECTED_WORLD_SIZE}, got {world_size!r}"
        )

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
