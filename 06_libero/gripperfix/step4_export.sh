#!/usr/bin/env bash
# Track A / step 4: assemble the diffusers export from the step-25000 checkpoint.
# The gripper-fix run writes clean-resume-v2 checkpoints, whose transformer is
# already consolidated -- this only verifies the key set and links the rest.
set -Eeuo pipefail
CKPT="${CKPT:-/path/to/storage/checkpoints/va2_libero10_cosmos_gripperfix_20260909/bf16_bs1_accum2_lr1e-5_25k/checkpoints/checkpoint_step_25000}"
OUT="/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
REF="/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_libero10_cosmos_20260907_step25000_diffusers"
LOG=/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/logs/export.log
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate lingbot
mkdir -p "$(dirname "${LOG}")"
[[ -f "${CKPT}/COMPLETE" ]] || { echo "checkpoint not COMPLETE yet: ${CKPT}"; exit 2; }
export PYTHONUNBUFFERED=1
exec /path/to/lingbot-env/bin/python \
  /path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/lib/export_diffusers.py \
  --checkpoint "${CKPT}" --reference-export "${REF}" --out "${OUT}" 2>&1 | tee -a "${LOG}"
