#!/usr/bin/env bash
# Start the language-conditioned RobotWin-4 ablation without JEPA prediction
# reward. The shared launcher performs the complete preflight.
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG="/path/to/shared-storage/datasets/reward_ablation/configs/robotwin4_qtranl_6layer_no_jepa_prediction.json"
export MASTER_PORT="${MASTER_PORT:-29691}"
exec "${SCRIPT_DIR}/launch_robotwin4_reward_ablation_8gpu.sh" \
  "${CONFIG}" \
  "no_jepa_prediction"
