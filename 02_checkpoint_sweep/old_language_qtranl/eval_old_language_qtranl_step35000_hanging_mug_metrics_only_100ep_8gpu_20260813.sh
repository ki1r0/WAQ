#!/usr/bin/env bash
# Old Language QTran-L checkpoint 35k: formal Hanging Mug metrics-only evaluation.
# Matched 100-episode replay seeds; no rollout PKL/HDF5/MP4 or debug tensors.
set -Eeuo pipefail

readonly LINGBOT_ROOT="/path/to/home/wy/lingbot-va-original-rl-r6-eval-source-20260811"
readonly ROBOTWIN_ROOT="/path/to/home/wy/robotwin-r6-eval-runtime-20260812"
readonly COLLECTOR="/path/to/home/wy/lingbot-va-original-rl/scripts/collect_robotwin_qguided_metrics_only_r6_20260813.sh"
readonly Q_SERVER_SCRIPT="/path/to/home/wy/lingbot-va-original-rl/scripts/wan_va_server_q_guiding_metrics_only_old_language_20260813.py"
readonly Q_CHECKPOINT="/path/to/mnt2/checkpoints/rl/robotwin_rw50_language_qtranl_everylayer_precontext_fixed1e5_50k_resume33000_20260804_rs5_r6/checkpoints/checkpoint_00035000"
readonly MODEL_PATH="/path/to/home/wy/giga/weight/lingbot-va-posttrain-robotwin"
readonly REPLAY_MANIFEST_ROOT="/path/to/mnt2/robotwin_eval_protocols/hanging_mug_replay100_bintd20260812_rebalanced8_20260813"
readonly OUTPUT_ROOT="/path/to/mnt2/robotwin_eval_results/EVAL-RW50-OLD-LANGUAGE-QTRANL-STEP35000-HANGINGMUG-100EP-GUIDED-METRICS-ONLY-NOARTIFACT-8GPU-20260813-R1"

for path in "${LINGBOT_ROOT}" "${ROBOTWIN_ROOT}" "${Q_CHECKPOINT}" "${MODEL_PATH}" "${REPLAY_MANIFEST_ROOT}"; do
  [[ -d "${path}" ]] || { echo "missing directory: ${path}" >&2; exit 2; }
done
for file in "${COLLECTOR}" "${Q_SERVER_SCRIPT}"; do
  [[ -f "${file}" ]] || { echo "missing file: ${file}" >&2; exit 2; }
done
for rank in 0 1 2 3 4 5 6 7; do
  [[ -s "${REPLAY_MANIFEST_ROOT}/worker-${rank}.jsonl" ]] || { echo "missing replay manifest for worker ${rank}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" && ! -e "${OUTPUT_ROOT}.launcher.log" ]] || {
  echo "refusing existing output root/log: ${OUTPUT_ROOT}" >&2
  exit 3
}

exec > >(tee -a "${OUTPUT_ROOT}.launcher.log") 2>&1
printf 'Old Language QTran-L formal metrics-only evaluation\n'
printf 'host=%s checkpoint=%s task=hanging_mug/demo_clean episodes=100\n' "$(hostname)" "${Q_CHECKPOINT}"
printf 'q_guidance=fixed_scale_0.1 interval_5 objective_min grad_normalize_true history_10\n'
printf 'replay_manifests=%s\n' "${REPLAY_MANIFEST_ROOT}"
printf 'artifacts=metrics/timing/Q only; no rollout PKL/HDF5/MP4/debug tensors\n'

env \
  LINGBOT_ROOT="${LINGBOT_ROOT}" \
  ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" \
  MODEL_PATH="${MODEL_PATH}" \
  Q_CHECKPOINT="${Q_CHECKPOINT}" \
  Q_SERVER_SCRIPT="${Q_SERVER_SCRIPT}" \
  TASK=hanging_mug TASK_CONFIG=demo_clean \
  SERVER_CONFIG=robotwin_train EVAL_CONFIG=policy/ACT/deploy_policy.yml \
  OUTPUT_ROOT="${OUTPUT_ROOT}" APP_PORT_BASE=57300 MASTER_PORT_BASE=58300 \
  QGF_RUN_MODE=guided QGF_GUIDED_VARIANT=guided \
  QGF_REPLAY_MANIFEST_ROOT="${REPLAY_MANIFEST_ROOT}" \
  Q_GUIDANCE_WEIGHT_SCHEDULER=fixed Q_GUIDANCE_SCALE=0.1 Q_GUIDANCE_BETA=2.0 \
  Q_GUIDANCE_START_STEP=0 Q_GUIDANCE_END_STEP=-1 Q_GUIDANCE_INTERVAL=5 \
  Q_OBJECTIVE=min Q_GRAD_CLIP=0.0 Q_GRAD_NORMALIZE=true Q_HISTORY_CHUNKS=10 \
  Q_LOG_DENOISING_STEPS=false Q_PROGRESS_GUIDANCE=false \
  RECORD_PREDICTION_REWARD=false \
  SERVER_TIMEOUT_SECONDS=900 PROGRESS_INTERVAL_SECONDS=60 \
  "${COLLECTOR}" 0,1,2,3,4,5,6,7 100

printf 'evaluation COMPLETE: %s\n' "${OUTPUT_ROOT}"
