#!/usr/bin/env bash
# Start only after the explicit stage preflight and cache-init smoke pass.
# Step 5 uses GPUs 1-6; step 6 switches to all GPUs 0-7 after step5 and its
# federated assembly have completed.
set -Eeuo pipefail

STAGE="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/orientationfix_va2_25k_20260916"
STEP5="${STAGE}/step5_features_6gpu_20260916.sh"
STEP6="${STAGE}/step6_train_critic2_20260916.sh"
LOG="${STAGE}/logs/steps5_6_chain.log"

mkdir -p "${STAGE}/logs"
exec > >(tee -a "${LOG}") 2>&1

echo "[$(date --iso-8601=seconds)] CHAIN_START"
echo "[$(date --iso-8601=seconds)] STEP5_COMMAND DEVICES=1,2,3,4,5,6 bash ${STEP5}"
DEVICES=1,2,3,4,5,6 bash "${STEP5}"
[[ -s "${STAGE}/STEP5_DONE" ]] || { echo "STEP5_ABORT marker missing"; exit 40; }
echo "[$(date --iso-8601=seconds)] STEP5_CONFIRMED marker=${STAGE}/STEP5_DONE"

echo "[$(date --iso-8601=seconds)] STEP6_COMMAND DEVICES=0,1,2,3,4,5,6,7 bash ${STEP6}"
DEVICES=0,1,2,3,4,5,6,7 bash "${STEP6}"
[[ -s "${STAGE}/CRITIC_COMPLETE" ]] || { echo "STEP6_ABORT completion marker missing"; exit 41; }
echo "[$(date --iso-8601=seconds)] CHAIN_DONE"
