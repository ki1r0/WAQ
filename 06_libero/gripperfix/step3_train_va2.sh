#!/usr/bin/env bash
# Track A / step 3: retrain VA2 SFT on the gripper-fixed mix (~17h on 6 GPUs).
# Same launcher, GPUs and deepspeed config as the 2026-09-06 original run.
set -Eeuo pipefail
DEVICES="${DEVICES:-2,3,4,5,6,7}"
SCRIPT="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/run_va2_gripperfix_sft.py"
SAVE="/path/to/storage/checkpoints/va2_libero10_cosmos_gripperfix_20260909/bf16_bs1_accum2_lr1e-5_25k"
LOG="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/logs/va2_sft.log"
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate lingbot
cd "/path/to/lingbot-va"
mkdir -p "$(dirname "${LOG}")"
[[ -d /path/to/storage/datasets/libero10_va2_success_mix_gripperfix_20260909 ]] || { echo "missing patched mix"; exit 2; }
[[ ! -e "${SAVE}" ]] || { echo "refusing pre-existing save root: ${SAVE}"; exit 3; }
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec deepspeed --include "localhost:${DEVICES}" --master_port 29849 "${SCRIPT}" 2>&1 | tee -a "${LOG}"
