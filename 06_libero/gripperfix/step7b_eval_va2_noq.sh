#!/usr/bin/env bash
# Track A / step 7b: VA2-GF WITHOUT guidance -- the paired control for step 7.
# Uses the guided launcher with guidance weight 0 (exactly zero action delta),
# so protocol/client/task-split are identical to the guided arm.
set -Eeuo pipefail
export MODEL="/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
export POLICY_ID="va2-gripperfix-20260909-noq"
export Q_CHECKPOINT="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/critic2_v1_compat"
export OUTPUT_ROOT="/path/to/storage/eval_libero10_va2gf_NOQ_baseline_20260910"
export RUN_ID="EVAL-LIBERO10-VA2GF-NOQ-BASELINE-20260910"
[[ -s "${MODEL}/transformer/diffusion_pytorch_model.safetensors" ]] || { echo "missing policy"; exit 2; }
exec bash /path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/lib/launch_eval_libero_noq_baseline.sh
