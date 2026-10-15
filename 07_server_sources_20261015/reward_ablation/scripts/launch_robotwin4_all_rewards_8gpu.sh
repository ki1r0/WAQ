#!/usr/bin/env bash
# Start the language-conditioned RobotWin-4 reference with all three rewards.
# The shared launcher performs the complete preflight.
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CONFIG="/mnt/zengqixin/datasets/reward_ablation/configs/robotwin4_qtranl_6layer_all_rewards.json"
export MASTER_PORT="${MASTER_PORT:-29697}"
exec "${SCRIPT_DIR}/launch_robotwin4_reward_ablation_8gpu.sh" \
  "${CONFIG}" \
  "all_rewards"
