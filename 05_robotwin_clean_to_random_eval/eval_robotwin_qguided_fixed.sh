#!/usr/bin/env bash
#
# Single entrypoint for RobotWin QTran-L evaluation with the current fixed
# Q-guidance protocol. This script launches no job until all arguments and
# checkpoint/manifest bindings pass validation.
#
# Fixed protocol:
#   scheduler=fixed_cpb, guidance scale=0.08, budget=0.08,
#   sigma_peak=0.25 (P25), sigma_concentration=8 (C8), history=10 (H10),
#   interval=5, objective=min, normalized gradient, clip=0,
#   progress/confidence/OOD/calibration disabled.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  eval_robotwin_qguided_fixed.sh \
    --q-checkpoint PATH \
    --task TASK \
    --device-ids CSV \
    --episodes N \
    --output-root PATH \
    [options]

Required arguments:
  --q-checkpoint PATH   QTran-L checkpoint directory.
  --task TASK           RobotWin task, e.g. place_can_basket.
  --device-ids CSV      CUDA device ids, e.g. 0 or 0,1,2,3.
  --episodes N          Episodes per arm; paired mode runs baseline and guided.
  --output-root PATH    New output directory; existing roots are refused.

Evaluation options:
  --mode MODE           paired, guided, or baseline (default: paired).
  --task-config NAME    RobotWin task config (default: demo_clean).
  --server-config NAME  LingBot server config (default: robotwin_train).
  --eval-config PATH    RobotWin policy config (default: policy/ACT/deploy_policy.yml).
  --manifest-mode MODE  generate or read (default: generate).
  --manifest-root PATH  Existing worker-partitioned replay manifest root.
  --model-path PATH     Frozen LingBot policy directory.
  --lingbot-root PATH   LingBot repository (default: /path/to/lingbot-va-original-rl).
  --robotwin-root PATH  RobotWin repository (default: /path/to/robotwin).
  --collector PATH      Direct fixed-protocol collector override.
  --q-server-script PATH Task-local fixed-protocol server override.
  --app-port-base N     First per-worker HTTP port (default: 54000).
  --master-port-base N  First per-worker torch port (default: 55000).
  --skip-smoke          Skip the automatic 1-episode admission smoke.
  --no-prediction-reward Do not write prediction-reward logs.
  -h, --help            Show this help.

Manifest modes:
  generate: paired mode writes a fresh baseline replay manifest, then the
            guided arm reads that exact manifest. This is the normal fresh
            paired evaluation path.
  read:     both arms read an existing manifest root. The root must contain
            worker-0.jsonl ... worker-(N-1).jsonl, where N is the number of
            selected device ids, and the total rows must equal --episodes.
EOF
}

OPS_ROOT="/path/to/robotwin-evaluation-runtime"
LINGBOT_ROOT="/path/to/lingbot-va-original-rl"
ROBOTWIN_ROOT="/path/to/robotwin"
LINGBOT_PYTHON="/path/to/lingbot-python"
ROBOTWIN_PYTHON="/path/to/robotwin-python"
MODEL_PATH="/path/to/model-weights/lingbot-va-posttrain-robotwin"
COLLECTOR="${OPS_ROOT}/collect_lingbot_qguided_rollouts_multi_gpu.direct_fixed_cpb_20260801.sh"
Q_SERVER_SCRIPT="${OPS_ROOT}/wan_va_server_q_guiding_fixed_cpb.py"
TASK_CONFIG="demo_clean"
SERVER_CONFIG="robotwin_train"
EVAL_CONFIG="policy/ACT/deploy_policy.yml"
MODE="paired"
MANIFEST_MODE="generate"
MANIFEST_ROOT=""
DEVICE_IDS=""
Q_CHECKPOINT=""
TASK=""
EPISODES=""
OUTPUT_ROOT=""
APP_PORT_BASE="54000"
MASTER_PORT_BASE="55000"
SERVER_TIMEOUT_SECONDS="300"
PROGRESS_INTERVAL_SECONDS="20"
ROLLOUT_MODE="all"
RECORD_PREDICTION_REWARD="true"
SKIP_SMOKE="false"

while (($#)); do
    case "$1" in
        --q-checkpoint) Q_CHECKPOINT="${2:?missing value for --q-checkpoint}"; shift 2 ;;
        --task) TASK="${2:?missing value for --task}"; shift 2 ;;
        --device-ids) DEVICE_IDS="${2:?missing value for --device-ids}"; shift 2 ;;
        --episodes) EPISODES="${2:?missing value for --episodes}"; shift 2 ;;
        --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
        --mode) MODE="${2:?missing value for --mode}"; shift 2 ;;
        --task-config) TASK_CONFIG="${2:?missing value for --task-config}"; shift 2 ;;
        --server-config) SERVER_CONFIG="${2:?missing value for --server-config}"; shift 2 ;;
        --eval-config) EVAL_CONFIG="${2:?missing value for --eval-config}"; shift 2 ;;
        --manifest-mode) MANIFEST_MODE="${2:?missing value for --manifest-mode}"; shift 2 ;;
        --manifest-root) MANIFEST_ROOT="${2:?missing value for --manifest-root}"; shift 2 ;;
        --model-path) MODEL_PATH="${2:?missing value for --model-path}"; shift 2 ;;
        --lingbot-root) LINGBOT_ROOT="${2:?missing value for --lingbot-root}"; shift 2 ;;
        --robotwin-root) ROBOTWIN_ROOT="${2:?missing value for --robotwin-root}"; shift 2 ;;
        --collector) COLLECTOR="${2:?missing value for --collector}"; shift 2 ;;
        --q-server-script) Q_SERVER_SCRIPT="${2:?missing value for --q-server-script}"; shift 2 ;;
        --app-port-base) APP_PORT_BASE="${2:?missing value for --app-port-base}"; shift 2 ;;
        --master-port-base) MASTER_PORT_BASE="${2:?missing value for --master-port-base}"; shift 2 ;;
        --skip-smoke) SKIP_SMOKE="true"; shift ;;
        --no-prediction-reward) RECORD_PREDICTION_REWARD="false"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 64 ;;
    esac
done

for required in Q_CHECKPOINT TASK DEVICE_IDS EPISODES OUTPUT_ROOT; do
    [[ -n "${!required}" ]] || { echo "missing required argument: ${required}" >&2; usage >&2; exit 64; }
done
[[ "${MODE}" == paired || "${MODE}" == guided || "${MODE}" == baseline ]] || { echo "--mode must be paired, guided, or baseline" >&2; exit 64; }
[[ "${MANIFEST_MODE}" == generate || "${MANIFEST_MODE}" == read ]] || { echo "--manifest-mode must be generate or read" >&2; exit 64; }
[[ "${EPISODES}" =~ ^[1-9][0-9]*$ ]] || { echo "--episodes must be a positive integer" >&2; exit 64; }
[[ "${APP_PORT_BASE}" =~ ^[1-9][0-9]*$ && "${MASTER_PORT_BASE}" =~ ^[1-9][0-9]*$ ]] || { echo "ports must be positive integers" >&2; exit 64; }
[[ "${SKIP_SMOKE}" == true || "${SKIP_SMOKE}" == false ]] || { echo "invalid smoke setting" >&2; exit 64; }
[[ -d "${LINGBOT_ROOT}" && -d "${ROBOTWIN_ROOT}" && -d "${MODEL_PATH}" && -d "${Q_CHECKPOINT}" ]] || { echo "missing LingBot/RobotWin/model/Q checkpoint directory" >&2; exit 2; }
[[ -x "${COLLECTOR}" && -f "${Q_SERVER_SCRIPT}" ]] || { echo "missing collector or fixed-protocol server" >&2; exit 2; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing existing output root: ${OUTPUT_ROOT}" >&2; exit 3; }

IFS=',' read -r -a GPU_LIST <<< "${DEVICE_IDS}"
(( ${#GPU_LIST[@]} > 0 )) || { echo "--device-ids is empty" >&2; exit 64; }
for gpu in "${GPU_LIST[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "invalid CUDA device id: ${gpu}" >&2; exit 64; }
done

if [[ "${MANIFEST_MODE}" == read ]]; then
    [[ -n "${MANIFEST_ROOT}" ]] || { echo "--manifest-root is required with --manifest-mode read" >&2; exit 64; }
    [[ -d "${MANIFEST_ROOT}" ]] || { echo "manifest root does not exist: ${MANIFEST_ROOT}" >&2; exit 2; }
    "${LINGBOT_PYTHON}" - "${MANIFEST_ROOT}" "${TASK}" "${TASK_CONFIG}" "${EPISODES}" "${#GPU_LIST[@]}" <<'PY'
import json, sys
from pathlib import Path
root, task, task_config, episodes, workers = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
rows = []
for worker in range(workers):
    path = root / f"worker-{worker}.jsonl"
    if not path.is_file():
        raise SystemExit(f"missing manifest shard: {path}")
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("task") != task or row.get("task_config") != task_config:
                raise SystemExit(f"task binding mismatch in {path}")
            if "seed" not in row or "episode_info" not in row:
                raise SystemExit(f"missing seed/episode_info in {path}")
            rows.append(row)
if len(rows) != episodes:
    raise SystemExit(f"manifest rows={len(rows)} but episodes={episodes}; repartition for this device count")
if len({int(row["seed"]) for row in rows}) != len(rows):
    raise SystemExit("manifest contains duplicate seeds")
print(f"manifest PASS: workers={workers} rows={len(rows)} task={task} task_config={task_config}")
PY
fi

mkdir -p "${OUTPUT_ROOT}/run_metadata"
printf '%s\n' \
    "task=${TASK}" "task_config=${TASK_CONFIG}" "server_config=${SERVER_CONFIG}" "eval_config=${EVAL_CONFIG}" \
    "model_path=${MODEL_PATH}" "q_checkpoint=${Q_CHECKPOINT}" "device_ids=${DEVICE_IDS}" "episodes=${EPISODES}" \
    "mode=${MODE}" "manifest_mode=${MANIFEST_MODE}" "manifest_root=${MANIFEST_ROOT}" \
    "q_guidance_weight_scheduler=fixed_cpb" "q_guidance_scale=0.08" "q_cpb_fixed_budget=0.08" \
    "q_cpb_sigma_peak=0.25" "q_cpb_sigma_concentration=8.0" "q_guidance_beta=2.0" \
    "q_guidance_interval=5" "q_objective=min" "q_grad_clip=0.0" "q_grad_normalize=true" \
    "q_history_chunks=10" "q_log_denoising_steps=true" "q_progress_guidance=false" \
    "q_cpb_confidence=0.0" "q_cpb_ood_margin=null" "q_calibration=false" \
    "record_prediction_reward=${RECORD_PREDICTION_REWARD}" \
    > "${OUTPUT_ROOT}/run_metadata/protocol_manifest.txt"

run_one() {
    local root="$1" episodes="$2" app="$3" master="$4"
    local -a manifest_env=(QGF_REPLAY_MANIFEST_MODE=write)
    [[ "${MANIFEST_MODE}" == read ]] && manifest_env=(QGF_REPLAY_MANIFEST_ROOT="${MANIFEST_ROOT}" QGF_REPLAY_MANIFEST_MODE=read)
    local -a envs=(
        TASK="${TASK}" TASK_CONFIG="${TASK_CONFIG}" SERVER_CONFIG="${SERVER_CONFIG}" EVAL_CONFIG="${EVAL_CONFIG}"
        MODEL_PATH="${MODEL_PATH}" LINGBOT_ROOT="${LINGBOT_ROOT}" ROBOTWIN_ROOT="${ROBOTWIN_ROOT}"
        LINGBOT_PYTHON="${LINGBOT_PYTHON}" ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}"
        Q_CHECKPOINT="${Q_CHECKPOINT}" Q_SERVER_SCRIPT="${Q_SERVER_SCRIPT}"
        QGF_RUN_MODE="${MODE}" QGF_BASELINE_VARIANT=baseline QGF_GUIDED_VARIANT=guided
        Q_GUIDANCE_WEIGHT_SCHEDULER=fixed_cpb Q_GUIDANCE_SCALE=0.08 Q_GUIDANCE_BETA=2.0
        Q_GUIDANCE_START_STEP=0 Q_GUIDANCE_END_STEP=-1 Q_GUIDANCE_INTERVAL=5
        Q_OBJECTIVE=min Q_GRAD_CLIP=0.0 Q_GRAD_NORMALIZE=true Q_HISTORY_CHUNKS=10 Q_LOG_DENOISING_STEPS=true
        RECORD_PREDICTION_REWARD="${RECORD_PREDICTION_REWARD}" ROLLOUT_MODE=all
        Q_PROGRESS_GUIDANCE=false Q_CPB_CONFIDENCE=0.0 Q_CPB_FIXED_BUDGET=0.08
        Q_CPB_SIGMA_PEAK=0.25 Q_CPB_SIGMA_CONCENTRATION=8.0
        SERVER_TIMEOUT_SECONDS=300 PROGRESS_INTERVAL_SECONDS=20
    )
    [[ -n "${ROBOTWIN_RUNTIME_ROOT:-}" ]] && envs+=(ROBOTWIN_RUNTIME_ROOT="${ROBOTWIN_RUNTIME_ROOT}")
    env "${envs[@]}" "${manifest_env[@]}" OUTPUT_ROOT="${root}" APP_PORT_BASE="${app}" MASTER_PORT_BASE="${master}" \
        "${COLLECTOR}" "${DEVICE_IDS}" "${episodes}" > "${root}.log" 2>&1
}

echo "fixed RobotWin QTran-L evaluation: task=${TASK} mode=${MODE} device_ids=${DEVICE_IDS} episodes=${EPISODES}"
if [[ "${EPISODES}" -gt 1 && "${SKIP_SMOKE}" == false ]]; then
    run_one "${OUTPUT_ROOT}/smoke_1ep" 1 "${APP_PORT_BASE}" "${MASTER_PORT_BASE}"
    echo "smoke PASS; starting requested evaluation"
fi
run_one "${OUTPUT_ROOT}/full_${EPISODES}ep" "${EPISODES}" "$((APP_PORT_BASE + 10))" "$((MASTER_PORT_BASE + 10))"
echo "evaluation COMPLETE: ${OUTPUT_ROOT}/full_${EPISODES}ep"
