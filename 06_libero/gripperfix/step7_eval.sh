#!/usr/bin/env bash
# Track A / step 7: final line-2 evaluation (VA2-GF policy + Critic2-GF guidance).
set -Eeuo pipefail
export MODEL="/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
export POLICY_ID="va2-gripperfix-20260909"
export Q_CHECKPOINT="/path/to/storage/checkpoints/rl/libero10_cosmos_joint_critic2_gripperfix_20260909/checkpoints/checkpoint_00025000"
export OUTPUT_ROOT="/path/to/storage/eval_libero10_cosmos_joint_va2critic2_gripperfix_b008_20260909"
export RUN_ID="EVAL-LIBERO10-COSMOS-JOINT-GRIPPERFIX-B008-20260909"
[[ -s "${Q_CHECKPOINT}/training_state.pt" ]] || { echo "missing critic checkpoint"; exit 2; }
[[ -s "${MODEL}/transformer/diffusion_pytorch_model.safetensors" ]] || { echo "missing policy"; exit 2; }
exec bash /path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/launch_eval_A_libero_guided_b008_pi06qema_critic2_compat.sh
