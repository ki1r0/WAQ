#!/usr/bin/env bash
# Track B / step B3: line-1 evaluation (unchanged 486 policy + Critic1-GF guidance).
set -Eeuo pipefail
export Q_CHECKPOINT="/path/to/mnt/checkpoints/rl/libero10_cosmos_solo_critic1_gripperfix_20260909/checkpoints/checkpoint_00025000"
export OUTPUT_ROOT="/path/to/mnt/eval_libero10_cosmos_solo_critic1_gripperfix_b008_20260909"
export RUN_ID="EVAL-LIBERO10-COSMOS-SOLO-CRITIC1-GRIPPERFIX-B008-20260909"
[[ -s "${Q_CHECKPOINT}/training_state.pt" ]] || { echo "missing critic checkpoint"; exit 2; }
exec bash /path/to/mnt/ops/libero_pi06_qema_20260827/launch_eval_A_libero_guided_b008_pi06qema.sh
