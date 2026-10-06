#!/usr/bin/env bash
# R6 metrics-only evaluation: no rollout PKL/HDF5/MP4 and no eval visualization.
# Policy, Q-guidance, seeds, horizon, timing, and Q summaries are unchanged.
#
# Q-guided variant of RobotWin's collect_lingbot_rollouts_multi_gpu.sh.
# Example:
#   cd /path/to/home/wy/robotwin
#   LINGBOT_ROOT=/path/to/home/wy/lingbot-va-original-rl \
#   Q_CHECKPOINT=/path/to/home/wy/lingbot-va-original-rl/checkpoints/phase3_critic_iql_place_can_basket_official_online_jepa_posttrain_robotwin/checkpoints/checkpoint_00005000 \
#   TOTAL_EPISODES=4 \
#   bash /path/to/home/wy/lingbot-va-original-rl/script/collect_lingbot_qguided_rollouts_multi_gpu.sh 0,1,2,3 4

set -Eeuo pipefail

GPUS_CSV="${1:-${GPUS:-0}}"
TOTAL_EPISODES="${2:-${TOTAL_EPISODES:-4}}"

LINGBOT_ROOT="${LINGBOT_ROOT:-/path/to/home/wy/lingbot-va-original-rl}"
ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/path/to/home/wy/robotwin}"
LINGBOT_PYTHON="${LINGBOT_PYTHON:-/path/to/home/miniconda3/envs/lingbot/bin/python}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/path/to/home/miniconda3/envs/robotwin/bin/python}"
MODEL_PATH="${MODEL_PATH:-/path/to/home/wy/giga/weight/lingbot-va-posttrain-robotwin}"
SERVER_CONFIG="${SERVER_CONFIG:-robotwin_train}"
EVAL_CONFIG="${EVAL_CONFIG:-policy/ACT/deploy_policy.yml}"
TASK="${TASK:-place_can_basket}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
ROLLOUT_MODE="${ROLLOUT_MODE:-all}"
APP_PORT_BASE="${APP_PORT_BASE:-9101}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"
SERVER_TIMEOUT_SECONDS="${SERVER_TIMEOUT_SECONDS:-240}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-20}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/eval_output/qguided_phase3_smoke/${RUN_ID}}"
EVAL_SEED_OFFSET="${EVAL_SEED_OFFSET:-0}"

Q_CHECKPOINT="${Q_CHECKPOINT:-/path/to/home/wy/lingbot-va-original-rl/checkpoints/phase3_critic_iql_place_can_basket_official_online_jepa_posttrain_robotwin/checkpoints/checkpoint_00005000}"
Q_SERVER_SCRIPT="${Q_SERVER_SCRIPT:-}"
Q_GUIDANCE_WEIGHT_SCHEDULER="${Q_GUIDANCE_WEIGHT_SCHEDULER:-fixed}"
Q_GUIDANCE_SCALE="${Q_GUIDANCE_SCALE:-0.1}"
Q_GUIDANCE_BETA="${Q_GUIDANCE_BETA:-2.0}"
Q_GUIDANCE_START_STEP="${Q_GUIDANCE_START_STEP:-0}"
Q_GUIDANCE_END_STEP="${Q_GUIDANCE_END_STEP:--1}"
Q_GUIDANCE_INTERVAL="${Q_GUIDANCE_INTERVAL:-5}"
Q_OBJECTIVE="${Q_OBJECTIVE:-min}"
Q_GRAD_CLIP="${Q_GRAD_CLIP:-0.0}"
Q_GRAD_NORMALIZE="${Q_GRAD_NORMALIZE:-true}"
Q_HISTORY_CHUNKS="${Q_HISTORY_CHUNKS:-10}"
Q_LOG_DENOISING_STEPS="${Q_LOG_DENOISING_STEPS:-false}"
QGF_COMPARE_BASELINE="${QGF_COMPARE_BASELINE:-false}"
QGF_BASELINE_VARIANT="${QGF_BASELINE_VARIANT:-baseline}"
QGF_GUIDED_VARIANT="${QGF_GUIDED_VARIANT:-qgf}"
QGF_RUN_MODE="${QGF_RUN_MODE:-}"
QGF_REPLAY_MANIFEST_ROOT="${QGF_REPLAY_MANIFEST_ROOT:-}"
RECORD_PREDICTION_REWARD="${RECORD_PREDICTION_REWARD:-false}"
Q_PROGRESS_GUIDANCE="${Q_PROGRESS_GUIDANCE:-false}"
Q_PROGRESS_GOAL_EMBEDDING="${Q_PROGRESS_GOAL_EMBEDDING:-}"
Q_PROGRESS_JEPA_CHECKPOINT="${Q_PROGRESS_JEPA_CHECKPOINT:-/path/to/home/wy/giga/weight/vjepa2.1/vjepa2_1_vitG_384.pt}"
Q_PROGRESS_BETA_MAX="${Q_PROGRESS_BETA_MAX:-}"
Q_PROGRESS_BETA_MIN="${Q_PROGRESS_BETA_MIN:-0.0}"
Q_PROGRESS_KAPPA="${Q_PROGRESS_KAPPA:-1.0}"
Q_PROGRESS_RHO="${Q_PROGRESS_RHO:-1.0}"
Q_PROGRESS_EPS="${Q_PROGRESS_EPS:-1e-6}"
Q_PROGRESS_SCHEDULER="${Q_PROGRESS_SCHEDULER:-progress_decay}"
Q_PROGRESS_EXP_BETA="${Q_PROGRESS_EXP_BETA:-10.0}"
Q_CPB_FIXED_BUDGET="${Q_CPB_FIXED_BUDGET:-}"
Q_CPB_SIGMA_PEAK="${Q_CPB_SIGMA_PEAK:-}"
Q_CPB_SIGMA_CONCENTRATION="${Q_CPB_SIGMA_CONCENTRATION:-}"
Q_CPB_CONFIDENCE="${Q_CPB_CONFIDENCE:-0.0}"

IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"
if (( ${#GPUS[@]} == 0 )); then
    echo "No GPUs were provided" >&2
    exit 2
fi
if ! [[ "${TOTAL_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "TOTAL_EPISODES must be a positive integer, got: ${TOTAL_EPISODES}" >&2
    exit 2
fi
if ! [[ "${EVAL_SEED_OFFSET}" =~ ^-?[0-9]+$ ]]; then
    echo "EVAL_SEED_OFFSET must be an integer, got: ${EVAL_SEED_OFFSET}" >&2
    exit 2
fi
if [[ -z "${QGF_RUN_MODE}" ]]; then
    if [[ "${QGF_COMPARE_BASELINE}" == "true" ]]; then
        QGF_RUN_MODE="paired"
    else
        QGF_RUN_MODE="guided"
    fi
fi
if [[ "${QGF_RUN_MODE}" != "paired" && "${QGF_RUN_MODE}" != "baseline" && "${QGF_RUN_MODE}" != "guided" ]]; then
    echo "QGF_RUN_MODE must be one of: paired, baseline, guided; got ${QGF_RUN_MODE}" >&2
    exit 2
fi
if [[ "${QGF_RUN_MODE}" == "paired" ]]; then
    QGF_COMPARE_BASELINE="true"
else
    QGF_COMPARE_BASELINE="false"
fi
if [[ -n "${QGF_REPLAY_MANIFEST_ROOT}" && ! -d "${QGF_REPLAY_MANIFEST_ROOT}" ]]; then
    echo "QGF_REPLAY_MANIFEST_ROOT does not exist: ${QGF_REPLAY_MANIFEST_ROOT}" >&2
    exit 2
fi
if [[ ! -d "${LINGBOT_ROOT}" || ! -d "${ROBOTWIN_ROOT}" ]]; then
    echo "LINGBOT_ROOT or ROBOTWIN_ROOT does not exist" >&2
    exit 2
fi
if [[ ! -d "${Q_CHECKPOINT}" ]]; then
    echo "Q_CHECKPOINT does not exist: ${Q_CHECKPOINT}" >&2
    exit 2
fi
if [[ -n "${Q_SERVER_SCRIPT}" && ! -f "${Q_SERVER_SCRIPT}" ]]; then
    echo "Q_SERVER_SCRIPT does not exist: ${Q_SERVER_SCRIPT}" >&2
    exit 2
fi
if [[ "${Q_GUIDANCE_WEIGHT_SCHEDULER}" == "fixed_cpb" ]]; then
    for value in Q_CPB_FIXED_BUDGET Q_CPB_SIGMA_PEAK Q_CPB_SIGMA_CONCENTRATION; do
        if [[ -z "${!value}" ]]; then
            echo "${value} is required for the fixed_cpb scheduler" >&2
            exit 2
        fi
    done
fi
if [[ "${Q_PROGRESS_GUIDANCE}" == "true" && ! -f "${Q_PROGRESS_GOAL_EMBEDDING}" ]]; then
    echo "Q_PROGRESS_GOAL_EMBEDDING is required for progress guidance: ${Q_PROGRESS_GOAL_EMBEDDING}" >&2
    exit 2
fi
if [[ ! -x "${LINGBOT_PYTHON}" || ! -x "${ROBOTWIN_PYTHON}" ]]; then
    echo "LINGBOT_PYTHON or ROBOTWIN_PYTHON is not executable" >&2
    exit 2
fi
if ! "${LINGBOT_PYTHON}" -c "import easydict, torch" >/dev/null 2>&1; then
    echo "LingBot dependencies are unavailable in ${LINGBOT_PYTHON}" >&2
    exit 2
fi
if ! "${ROBOTWIN_PYTHON}" -c "import sapien, h5py" >/dev/null 2>&1; then
    echo "RoboTwin dependencies are unavailable in ${ROBOTWIN_PYTHON}" >&2
    exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
    FFMPEG_BIN="$("${ROBOTWIN_PYTHON}" - <<'PY'
try:
    import imageio_ffmpeg
    print(imageio_ffmpeg.get_ffmpeg_exe())
except Exception:
    pass
PY
)"
    if [[ -z "${FFMPEG_BIN}" || ! -x "${FFMPEG_BIN}" ]]; then
        echo "ffmpeg is unavailable and imageio-ffmpeg did not provide an executable" >&2
        exit 2
    fi
    FFMPEG_SHIM_DIR="${OUTPUT_ROOT}/run_metadata/bin"
    mkdir -p "${FFMPEG_SHIM_DIR}"
    ln -sf "${FFMPEG_BIN}" "${FFMPEG_SHIM_DIR}/ffmpeg"
    export PATH="${FFMPEG_SHIM_DIR}:${PATH}"
fi

write_shell_words() {
    printf '%q ' "$@"
    printf '\n'
}

write_run_metadata() {
    local metadata_dir="${OUTPUT_ROOT}/run_metadata"
    local eval_config_source=""

    mkdir -p "${metadata_dir}"
    cat >"${metadata_dir}/evaluation_config.env" <<EOF
GPUS_CSV=${GPUS_CSV}
TOTAL_EPISODES=${TOTAL_EPISODES}
LINGBOT_ROOT=${LINGBOT_ROOT}
ROBOTWIN_ROOT=${ROBOTWIN_ROOT}
LINGBOT_PYTHON=${LINGBOT_PYTHON}
ROBOTWIN_PYTHON=${ROBOTWIN_PYTHON}
MODEL_PATH=${MODEL_PATH}
SERVER_CONFIG=${SERVER_CONFIG}
EVAL_CONFIG=${EVAL_CONFIG}
TASK=${TASK}
TASK_CONFIG=${TASK_CONFIG}
ROLLOUT_MODE=${ROLLOUT_MODE}
APP_PORT_BASE=${APP_PORT_BASE}
MASTER_PORT_BASE=${MASTER_PORT_BASE}
SERVER_TIMEOUT_SECONDS=${SERVER_TIMEOUT_SECONDS}
PROGRESS_INTERVAL_SECONDS=${PROGRESS_INTERVAL_SECONDS}
RUN_ID=${RUN_ID}
OUTPUT_ROOT=${OUTPUT_ROOT}
EVAL_SEED_OFFSET=${EVAL_SEED_OFFSET}
Q_CHECKPOINT=${Q_CHECKPOINT}
Q_GUIDANCE_SCALE=${Q_GUIDANCE_SCALE}
Q_GUIDANCE_BETA=${Q_GUIDANCE_BETA}
Q_GUIDANCE_START_STEP=${Q_GUIDANCE_START_STEP}
Q_GUIDANCE_END_STEP=${Q_GUIDANCE_END_STEP}
Q_GUIDANCE_INTERVAL=${Q_GUIDANCE_INTERVAL}
Q_OBJECTIVE=${Q_OBJECTIVE}
Q_GRAD_CLIP=${Q_GRAD_CLIP}
Q_GRAD_NORMALIZE=${Q_GRAD_NORMALIZE}
Q_HISTORY_CHUNKS=${Q_HISTORY_CHUNKS}
Q_LOG_DENOISING_STEPS=${Q_LOG_DENOISING_STEPS}
QGF_COMPARE_BASELINE=${QGF_COMPARE_BASELINE}
QGF_BASELINE_VARIANT=${QGF_BASELINE_VARIANT}
QGF_GUIDED_VARIANT=${QGF_GUIDED_VARIANT}
QGF_RUN_MODE=${QGF_RUN_MODE}
QGF_REPLAY_MANIFEST_ROOT=${QGF_REPLAY_MANIFEST_ROOT}
RECORD_PREDICTION_REWARD=${RECORD_PREDICTION_REWARD}
Q_PROGRESS_GUIDANCE=${Q_PROGRESS_GUIDANCE}
Q_PROGRESS_GOAL_EMBEDDING=${Q_PROGRESS_GOAL_EMBEDDING}
Q_PROGRESS_JEPA_CHECKPOINT=${Q_PROGRESS_JEPA_CHECKPOINT}
Q_PROGRESS_BETA_MAX=${Q_PROGRESS_BETA_MAX}
Q_PROGRESS_BETA_MIN=${Q_PROGRESS_BETA_MIN}
Q_PROGRESS_KAPPA=${Q_PROGRESS_KAPPA}
Q_PROGRESS_RHO=${Q_PROGRESS_RHO}
Q_PROGRESS_EPS=${Q_PROGRESS_EPS}
Q_PROGRESS_SCHEDULER=${Q_PROGRESS_SCHEDULER}
Q_PROGRESS_EXP_BETA=${Q_PROGRESS_EXP_BETA}
VISUALIZATION_ROOT=${OUTPUT_ROOT}/evaluation
METRICS_ROOT=${OUTPUT_ROOT}/metrics
ROLLOUT_LAYOUT=per-worker
EOF

    {
        echo '#!/usr/bin/env bash'
        printf 'cd %q\n' "${ROBOTWIN_ROOT}"
        printf 'LINGBOT_ROOT=%q \\\n' "${LINGBOT_ROOT}"
        printf 'ROBOTWIN_ROOT=%q \\\n' "${ROBOTWIN_ROOT}"
        printf 'LINGBOT_PYTHON=%q \\\n' "${LINGBOT_PYTHON}"
        printf 'ROBOTWIN_PYTHON=%q \\\n' "${ROBOTWIN_PYTHON}"
        printf 'MODEL_PATH=%q \\\n' "${MODEL_PATH}"
        printf 'SERVER_CONFIG=%q \\\n' "${SERVER_CONFIG}"
        printf 'EVAL_CONFIG=%q \\\n' "${EVAL_CONFIG}"
        printf 'TASK=%q \\\n' "${TASK}"
        printf 'TASK_CONFIG=%q \\\n' "${TASK_CONFIG}"
        printf 'ROLLOUT_MODE=%q \\\n' "${ROLLOUT_MODE}"
        printf 'OUTPUT_ROOT=%q \\\n' "${OUTPUT_ROOT}"
        printf 'EVAL_SEED_OFFSET=%q \\\n' "${EVAL_SEED_OFFSET}"
        printf 'Q_CHECKPOINT=%q \\\n' "${Q_CHECKPOINT}"
        printf 'Q_GUIDANCE_SCALE=%q \\\n' "${Q_GUIDANCE_SCALE}"
        printf 'Q_GUIDANCE_BETA=%q \\\n' "${Q_GUIDANCE_BETA}"
        printf 'Q_GUIDANCE_START_STEP=%q \\\n' "${Q_GUIDANCE_START_STEP}"
        printf 'Q_GUIDANCE_END_STEP=%q \\\n' "${Q_GUIDANCE_END_STEP}"
        printf 'Q_GUIDANCE_INTERVAL=%q \\\n' "${Q_GUIDANCE_INTERVAL}"
        printf 'Q_OBJECTIVE=%q \\\n' "${Q_OBJECTIVE}"
        printf 'Q_GRAD_CLIP=%q \\\n' "${Q_GRAD_CLIP}"
        printf 'Q_GRAD_NORMALIZE=%q \\\n' "${Q_GRAD_NORMALIZE}"
        printf 'Q_LOG_DENOISING_STEPS=%q \\\n' "${Q_LOG_DENOISING_STEPS}"
        printf 'QGF_COMPARE_BASELINE=%q \\\n' "${QGF_COMPARE_BASELINE}"
        printf 'QGF_BASELINE_VARIANT=%q \\\n' "${QGF_BASELINE_VARIANT}"
        printf 'QGF_GUIDED_VARIANT=%q \\\n' "${QGF_GUIDED_VARIANT}"
        printf 'QGF_RUN_MODE=%q \\\n' "${QGF_RUN_MODE}"
        printf 'QGF_REPLAY_MANIFEST_ROOT=%q \\\n' "${QGF_REPLAY_MANIFEST_ROOT}"
        printf 'RECORD_PREDICTION_REWARD=%q \\\n' "${RECORD_PREDICTION_REWARD}"
        printf 'Q_PROGRESS_GUIDANCE=%q \\\n' "${Q_PROGRESS_GUIDANCE}"
        printf 'Q_PROGRESS_GOAL_EMBEDDING=%q \\\n' "${Q_PROGRESS_GOAL_EMBEDDING}"
        printf 'Q_PROGRESS_JEPA_CHECKPOINT=%q \\\n' "${Q_PROGRESS_JEPA_CHECKPOINT}"
        printf 'Q_PROGRESS_BETA_MAX=%q \\\n' "${Q_PROGRESS_BETA_MAX}"
        printf 'Q_PROGRESS_BETA_MIN=%q \\\n' "${Q_PROGRESS_BETA_MIN}"
        printf 'Q_PROGRESS_KAPPA=%q \\\n' "${Q_PROGRESS_KAPPA}"
        printf 'Q_PROGRESS_RHO=%q \\\n' "${Q_PROGRESS_RHO}"
        printf 'Q_PROGRESS_EPS=%q \\\n' "${Q_PROGRESS_EPS}"
        printf 'Q_PROGRESS_SCHEDULER=%q \\\n' "${Q_PROGRESS_SCHEDULER}"
        printf 'Q_PROGRESS_EXP_BETA=%q \\\n' "${Q_PROGRESS_EXP_BETA}"
        printf 'PROGRESS_INTERVAL_SECONDS=%q \\\n' "${PROGRESS_INTERVAL_SECONDS}"
        write_shell_words bash "$0" "${GPUS_CSV}" "${TOTAL_EPISODES}"
    } >"${metadata_dir}/start_command.sh"
    chmod +x "${metadata_dir}/start_command.sh"

    for candidate in "${ROBOTWIN_ROOT}/${EVAL_CONFIG}" "${LINGBOT_ROOT}/${EVAL_CONFIG}" "${EVAL_CONFIG}"; do
        if [[ -f "${candidate}" ]]; then
            eval_config_source="${candidate}"
            cp "${candidate}" "${metadata_dir}/eval_config.yml"
            break
        fi
    done
    if [[ -n "${eval_config_source}" ]]; then
        printf '%s\n' "${eval_config_source}" >"${metadata_dir}/eval_config_source.txt"
    else
        printf 'not found: %s\n' "${EVAL_CONFIG}" >"${metadata_dir}/eval_config_source.txt"
    fi
}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/rollouts" "${OUTPUT_ROOT}/evaluation" "${OUTPUT_ROOT}/q_values" "${OUTPUT_ROOT}/metrics"
mkdir -p "${OUTPUT_ROOT}/replay_manifests"
write_run_metadata
echo "output_root=${OUTPUT_ROOT}"
echo "q_checkpoint=${Q_CHECKPOINT}"
echo "q_guidance_scale=${Q_GUIDANCE_SCALE} q_interval=${Q_GUIDANCE_INTERVAL} q_grad_normalize=${Q_GRAD_NORMALIZE}"

wait_for_server() {
    local server_pid="$1"
    local port="$2"
    local log_file="$3"
    local waited=0

    while (( waited < SERVER_TIMEOUT_SECONDS )); do
        if ! kill -0 "${server_pid}" 2>/dev/null; then
            echo "Server on port ${port} exited during startup. Log: ${log_file}" >&2
            tail -n 120 "${log_file}" >&2 || true
            return 1
        fi
        if timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done

    echo "Timed out waiting for server on port ${port}. Log: ${log_file}" >&2
    tail -n 120 "${log_file}" >&2 || true
    return 1
}

update_metrics() {
    "${ROBOTWIN_PYTHON}" - "${OUTPUT_ROOT}" "${TOTAL_EPISODES}" <<'PY' || true
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

output = Path(sys.argv[1])
target_episodes = int(sys.argv[2])
metrics_dir = output / "metrics"
metrics_dir.mkdir(parents=True, exist_ok=True)

ansi_re = re.compile(r"\x1b\[[0-9;]*m")
rate_re = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+).*?current seed:\s*(\d+)")
worker_re = re.compile(r"eval-worker-(\d+)\.log$")

def variant_for_path(path: Path) -> str:
    rel = path.relative_to(output)
    parts = rel.parts
    if len(parts) >= 3 and parts[0] == "variants":
        return parts[1]
    return "default"

rows = []
for log_path in sorted(output.rglob("eval-worker-*.log")):
    variant = variant_for_path(log_path)
    worker_match = worker_re.search(log_path.name)
    worker = int(worker_match.group(1)) if worker_match else None
    prev_successes = 0
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except OSError:
        continue
    for line in lines:
        clean = ansi_re.sub("", line)
        match = rate_re.search(clean)
        if not match:
            continue
        cumulative_successes = int(match.group(1))
        cumulative_episodes = int(match.group(2))
        seed = int(match.group(3))
        episode_successes = cumulative_successes - prev_successes
        rows.append(
            {
                "variant": variant,
                "worker": worker,
                "worker_episode": cumulative_episodes - 1,
                "seed": seed,
                "success": episode_successes > 0,
                "cumulative_successes": cumulative_successes,
                "cumulative_episodes": cumulative_episodes,
                "log": str(log_path),
            }
        )
        prev_successes = cumulative_successes

rows.sort(
    key=lambda row: (
        row["variant"],
        row["worker"] if row["worker"] is not None else -1,
        row["worker_episode"],
    )
)
completed = len(rows)
successes = sum(1 for row in rows if row["success"])
by_variant = {}
for variant in sorted({row["variant"] for row in rows} or {"default"}):
    variant_rows = [row for row in rows if row["variant"] == variant]
    variant_successes = sum(1 for row in variant_rows if row["success"])
    variant_root = output / "variants" / variant if (output / "variants" / variant).exists() else output
    by_variant[variant] = {
        "target_episodes": target_episodes,
        "completed": len(variant_rows),
        "successes": variant_successes,
        "success_rate": (variant_successes / len(variant_rows)) if variant_rows else None,
        "visualizations": len(list((variant_root / "evaluation").rglob("*.mp4"))),
        "rollouts": len(list((variant_root / "rollouts").rglob("*.hdf5"))),
        "prediction_reward_records": sum(
            1
            for path in (variant_root / "rewards").rglob("prediction_rewards_worker-*.jsonl")
            for _ in path.open(errors="ignore")
        )
        if (variant_root / "rewards").exists()
        else 0,
        "timing_records": sum(
            1
            for path in (variant_root / "timing").rglob("rollout_timing_worker-*.jsonl")
            for _ in path.open(errors="ignore")
        )
        if (variant_root / "timing").exists()
        else 0,
    }

pairs_by_key = defaultdict(dict)
for row in rows:
    pairs_by_key[(row["worker"], row["worker_episode"], row["seed"])][
        row["variant"]
    ] = row
paired_rows = []
for (worker, worker_episode, seed), variants in sorted(pairs_by_key.items()):
    record = {
        "worker": worker,
        "worker_episode": worker_episode,
        "seed": seed,
        "variants": sorted(variants),
    }
    for variant, row in sorted(variants.items()):
        record[f"{variant}_success"] = row["success"]
        record[f"{variant}_log"] = row["log"]
    paired_rows.append(record)

summary = {
    "target_episodes": target_episodes,
    "completed": completed,
    "successes": successes,
    "success_rate": (successes / completed) if completed else None,
    "by_variant": by_variant,
    "paired_records": len(paired_rows),
    "visualizations": len(list(output.rglob("*.mp4"))),
    "rollouts": len(list(output.rglob("*.hdf5"))),
}

episodes_tmp = metrics_dir / "episodes.jsonl.tmp"
episodes_path = metrics_dir / "episodes.jsonl"
with episodes_tmp.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
episodes_tmp.replace(episodes_path)

paired_tmp = metrics_dir / "paired_episodes.jsonl.tmp"
paired_path = metrics_dir / "paired_episodes.jsonl"
with paired_tmp.open("w") as f:
    for row in paired_rows:
        f.write(json.dumps(row, sort_keys=True) + "\n")
paired_tmp.replace(paired_path)

summary_tmp = metrics_dir / "summary.json.tmp"
summary_path = metrics_dir / "summary.json"
summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
summary_tmp.replace(summary_path)
PY
}

run_worker_variant() {
    local rank="$1"
    local gpu="$2"
    local episode_count="$3"
    local variant="$4"
    local variant_scale="$5"
    local variant_root="$6"
    local app_port=$((APP_PORT_BASE + rank))
    local master_port=$((MASTER_PORT_BASE + rank))
    local worker_root="${variant_root}/rollouts/worker-${rank}"
    local eval_root="${variant_root}/evaluation"
    local q_log_root="${variant_root}/q_values/worker-${rank}"
    local reward_log="${variant_root}/rewards/prediction_rewards_worker-${rank}.jsonl"
    local timing_log="${variant_root}/timing/rollout_timing_worker-${rank}.jsonl"
    local replay_manifest="${OUTPUT_ROOT}/replay_manifests/worker-${rank}.jsonl"
    local read_replay_manifest="${replay_manifest}"
    local server_log="${variant_root}/logs/server-worker-${rank}.log"
    local eval_log="${variant_root}/logs/eval-worker-${rank}.log"
    local server_pid=""
    local is_baseline="false"
    local eval_seed=$((rank + EVAL_SEED_OFFSET))

    if [[ "${variant}" == "${QGF_BASELINE_VARIANT}" ]]; then
        is_baseline="true"
    fi
    if [[ -n "${QGF_REPLAY_MANIFEST_ROOT}" ]]; then
        read_replay_manifest="${QGF_REPLAY_MANIFEST_ROOT}/worker-${rank}.jsonl"
    fi

    mkdir -p "${worker_root}" "${eval_root}" "${q_log_root}" "${variant_root}/logs" "${variant_root}/rewards" "${variant_root}/timing"
    echo "worker=${rank} variant=${variant} gpu=${gpu} episodes=${episode_count} app_port=${app_port} q_guidance_scale=${variant_scale}"

    server_args=(
        -m torch.distributed.run
        --nproc_per_node=1
        --master_port="${master_port}"
    )
    if [[ -n "${Q_SERVER_SCRIPT}" ]]; then
        server_args+=("${Q_SERVER_SCRIPT}")
    else
        server_args+=(-m wan_va.wan_va_server_q_guiding)
    fi
    server_args+=(
        --config-name "${SERVER_CONFIG}"
        --port "${app_port}"
        --eval_model_path "${MODEL_PATH}"
        --robotwin
        --q_checkpoint "${Q_CHECKPOINT}"
        --q_guidance_scale "${variant_scale}"
        --q_guidance_weight_scheduler "${Q_GUIDANCE_WEIGHT_SCHEDULER}"
        --q_guidance_beta "${Q_GUIDANCE_BETA}"
        --q_guidance_start_step "${Q_GUIDANCE_START_STEP}"
        --q_guidance_end_step "${Q_GUIDANCE_END_STEP}"
        --q_guidance_interval "${Q_GUIDANCE_INTERVAL}"
        --q_objective "${Q_OBJECTIVE}"
        --q_grad_clip "${Q_GRAD_CLIP}"
        --q_history_chunks "${Q_HISTORY_CHUNKS}"
        --q_log_dir "${q_log_root}"
    )
    if [[ "${Q_GRAD_NORMALIZE}" == "true" ]]; then
        server_args+=(--q_grad_normalize)
    fi
    if [[ "${Q_LOG_DENOISING_STEPS}" == "true" ]]; then
        server_args+=(--q_log_denoising_steps)
    fi
    if [[ "${Q_GUIDANCE_WEIGHT_SCHEDULER}" == "fixed_cpb" ]]; then
        server_args+=(
            --q_cpb_fixed_budget "${Q_CPB_FIXED_BUDGET}"
            --q_cpb_sigma_peak "${Q_CPB_SIGMA_PEAK}"
            --q_cpb_sigma_concentration "${Q_CPB_SIGMA_CONCENTRATION}"
            --q_cpb_confidence "${Q_CPB_CONFIDENCE}"
        )
    fi
    if [[ "${Q_PROGRESS_GUIDANCE}" == "true" && "${is_baseline}" != "true" ]]; then
        server_args+=(
            --q_progress_guidance
            --q_progress_goal_embedding "${Q_PROGRESS_GOAL_EMBEDDING}"
            --q_progress_jepa_checkpoint "${Q_PROGRESS_JEPA_CHECKPOINT}"
            --q_progress_beta_min "${Q_PROGRESS_BETA_MIN}"
            --q_progress_kappa "${Q_PROGRESS_KAPPA}"
            --q_progress_rho "${Q_PROGRESS_RHO}"
            --q_progress_eps "${Q_PROGRESS_EPS}"
            --q_progress_scheduler "${Q_PROGRESS_SCHEDULER}"
            --q_progress_exp_beta "${Q_PROGRESS_EXP_BETA}"
        )
        if [[ -n "${Q_PROGRESS_BETA_MAX}" ]]; then
            server_args+=(--q_progress_beta_max "${Q_PROGRESS_BETA_MAX}")
        fi
    fi

    eval_args=(
        -m evaluation.robotwin.eval_polict_client_openpi_novideo_r6_20260812
        --config "${EVAL_CONFIG}"
        --server_host 127.0.0.1
        --port "${app_port}"
        --test_num "${episode_count}"
        --save_root "${eval_root}"
        --eval_variant "${variant}"
        --timing_log_path "${timing_log}"
    )
    if [[ "${QGF_RUN_MODE}" == "paired" && "${is_baseline}" == "true" ]]; then
        eval_args+=(--replay_manifest_write_path "${replay_manifest}")
    fi
    if [[ "${QGF_RUN_MODE}" == "baseline" && "${is_baseline}" == "true" ]]; then
        eval_args+=(--replay_manifest_write_path "${replay_manifest}")
    fi
    if [[ "${QGF_RUN_MODE}" == "paired" && "${is_baseline}" != "true" ]]; then
        eval_args+=(--replay_manifest_read_path "${read_replay_manifest}")
    fi
    if [[ "${QGF_RUN_MODE}" == "guided" && -n "${QGF_REPLAY_MANIFEST_ROOT}" ]]; then
        if [[ ! -f "${read_replay_manifest}" ]]; then
            echo "Replay manifest for worker ${rank} does not exist: ${read_replay_manifest}" >&2
            return 2
        fi
        eval_args+=(--replay_manifest_read_path "${read_replay_manifest}")
    fi
    if [[ "${RECORD_PREDICTION_REWARD}" == "true" ]]; then
        eval_args+=(
            --record_prediction_reward
            --prediction_reward_log_path "${reward_log}"
        )
    fi
    eval_args+=(
        --overrides
        --task_name "${TASK}"
        --task_config "${TASK_CONFIG}"
        --eval_video_log False
        --seed "${eval_seed}"
    )

    {
        echo '#!/usr/bin/env bash'
        echo '# Server command'
        printf 'cd %q\n' "${LINGBOT_ROOT}"
        printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
        write_shell_words "${LINGBOT_PYTHON}" "${server_args[@]}"
        echo
        echo '# Evaluation client command'
        printf 'cd %q\n' "${LINGBOT_ROOT}"
        printf 'CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ROBOTWIN_ROOT=%q ' "${gpu}" "${ROBOTWIN_ROOT}"
        write_shell_words "${ROBOTWIN_PYTHON}" "${eval_args[@]}"
    } >"${OUTPUT_ROOT}/run_metadata/worker-${rank}-${variant}-commands.sh"
    chmod +x "${OUTPUT_ROOT}/run_metadata/worker-${rank}-${variant}-commands.sh"

    (
        cd "${LINGBOT_ROOT}"
        exec env CUDA_VISIBLE_DEVICES="${gpu}" "${LINGBOT_PYTHON}" "${server_args[@]}"
    ) >"${server_log}" 2>&1 &
    server_pid=$!

    cleanup_worker() {
        if [[ -n "${server_pid:-}" ]]; then
            kill "${server_pid}" 2>/dev/null || true
            wait "${server_pid}" 2>/dev/null || true
            server_pid=""
        fi
    }
    trap cleanup_worker EXIT
    trap 'exit 130' INT TERM

    wait_for_server "${server_pid}" "${app_port}" "${server_log}"

    (
        cd "${LINGBOT_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 ROBOTWIN_ROOT="${ROBOTWIN_ROOT}" "${ROBOTWIN_PYTHON}" "${eval_args[@]}"
    ) >"${eval_log}" 2>&1

    cleanup_worker
}

run_worker() {
    local rank="$1"
    local gpu="$2"
    local episode_count="$3"

    if [[ "${QGF_COMPARE_BASELINE}" == "true" ]]; then
        run_worker_variant "${rank}" "${gpu}" "${episode_count}" "${QGF_BASELINE_VARIANT}" "0.0" "${OUTPUT_ROOT}/variants/${QGF_BASELINE_VARIANT}"
        run_worker_variant "${rank}" "${gpu}" "${episode_count}" "${QGF_GUIDED_VARIANT}" "${Q_GUIDANCE_SCALE}" "${OUTPUT_ROOT}/variants/${QGF_GUIDED_VARIANT}"
    elif [[ "${QGF_RUN_MODE}" == "baseline" ]]; then
        run_worker_variant "${rank}" "${gpu}" "${episode_count}" "${QGF_BASELINE_VARIANT}" "0.0" "${OUTPUT_ROOT}"
    else
        run_worker_variant "${rank}" "${gpu}" "${episode_count}" "${QGF_GUIDED_VARIANT}" "${Q_GUIDANCE_SCALE}" "${OUTPUT_ROOT}"
    fi
}

worker_count=${#GPUS[@]}
base_count=$((TOTAL_EPISODES / worker_count))
remainder=$((TOTAL_EPISODES % worker_count))
worker_pids=()
PROGRESS_PID=""

monitor_progress() {
    while true; do
        sleep "${PROGRESS_INTERVAL_SECONDS}"
        local alive=0
        local pid
        for pid in "${worker_pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                alive=$((alive + 1))
            fi
        done
        local rollout_count
        local recording_frames
        local visualization_count
        local q_value_records
        rollout_count=$(find "${OUTPUT_ROOT}" -type f -path '*/rollouts/*' -name '*.hdf5' 2>/dev/null | wc -l)
        recording_frames=$(find "${OUTPUT_ROOT}" -type f -path '*/rollouts/*/.cache/*.pkl' 2>/dev/null | wc -l)
        visualization_count=$(find "${OUTPUT_ROOT}" -type f -path '*/evaluation/*' -name '*.mp4' 2>/dev/null | wc -l)
        q_value_records=$(find "${OUTPUT_ROOT}" -type f -name 'q_guidance_summary_rank*.jsonl' -exec cat {} + 2>/dev/null | wc -l)
        update_metrics >/dev/null 2>&1 || true
        echo "progress workers_alive=${alive} rollouts=${rollout_count} recording_frames=${recording_frames} visualizations=${visualization_count} q_value_records=${q_value_records}"
        if (( alive == 0 )); then
            return
        fi
    done
}

cleanup_all() {
    for pid in "${worker_pids[@]}"; do
        kill "${pid}" 2>/dev/null || true
    done
    if [[ -n "${PROGRESS_PID}" ]]; then
        kill "${PROGRESS_PID}" 2>/dev/null || true
    fi
}
trap cleanup_all INT TERM

for rank in "${!GPUS[@]}"; do
    gpu="${GPUS[$rank]//[[:space:]]/}"
    if ! [[ "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU identifier: ${GPUS[$rank]}" >&2
        exit 2
    fi
    episode_count=${base_count}
    if (( rank < remainder )); then
        episode_count=$((episode_count + 1))
    fi
    if (( episode_count == 0 )); then
        continue
    fi
    run_worker "${rank}" "${gpu}" "${episode_count}" &
    worker_pids+=("$!")
done

monitor_progress &
PROGRESS_PID=$!

status=0
for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
        status=1
    fi
done

kill "${PROGRESS_PID}" 2>/dev/null || true
wait "${PROGRESS_PID}" 2>/dev/null || true
PROGRESS_PID=""
update_metrics >/dev/null 2>&1 || true

if (( status != 0 )); then
    echo "At least one worker failed. Inspect ${OUTPUT_ROOT}/logs" >&2
    echo "Unified metrics: ${OUTPUT_ROOT}/metrics/summary.json"
    exit "${status}"
fi

echo "Collection complete: ${OUTPUT_ROOT}"
if [[ "${QGF_RUN_MODE}" == "paired" ]]; then
    echo "Baseline visualizations: ${OUTPUT_ROOT}/variants/${QGF_BASELINE_VARIANT}/evaluation"
    echo "Guided visualizations: ${OUTPUT_ROOT}/variants/${QGF_GUIDED_VARIANT}/evaluation"
    echo "Prediction reward logs: ${OUTPUT_ROOT}/variants/*/rewards/prediction_rewards_worker-*.jsonl"
    echo "Timing logs: ${OUTPUT_ROOT}/variants/*/timing/rollout_timing_worker-*.jsonl"
    echo "Replay manifests: ${OUTPUT_ROOT}/replay_manifests/worker-*.jsonl"
    echo "Raw worker datasets: ${OUTPUT_ROOT}/variants/*/rollouts/worker-*/${TASK}/${TASK_CONFIG}"
    echo "Q/V JSONL logs: ${OUTPUT_ROOT}/variants/*/q_values/worker-*/q_guidance_summary_rank*.jsonl"
elif [[ "${QGF_RUN_MODE}" == "baseline" ]]; then
    echo "Baseline visualizations: ${OUTPUT_ROOT}/evaluation"
    echo "Replay manifests: ${OUTPUT_ROOT}/replay_manifests/worker-*.jsonl"
    echo "Raw worker datasets: ${OUTPUT_ROOT}/rollouts/worker-*/${TASK}/${TASK_CONFIG}"
else
    echo "Unified visualizations: ${OUTPUT_ROOT}/evaluation"
    if [[ -n "${QGF_REPLAY_MANIFEST_ROOT}" ]]; then
        echo "Replay manifests read from: ${QGF_REPLAY_MANIFEST_ROOT}/worker-*.jsonl"
    fi
    echo "Prediction reward logs: ${OUTPUT_ROOT}/rewards/prediction_rewards_worker-*.jsonl"
    echo "Timing logs: ${OUTPUT_ROOT}/timing/rollout_timing_worker-*.jsonl"
    echo "Raw worker datasets: ${OUTPUT_ROOT}/rollouts/worker-*/${TASK}/${TASK_CONFIG}"
    echo "Q/V JSONL logs: ${OUTPUT_ROOT}/q_values/worker-*/q_guidance_summary_rank*.jsonl"
fi
echo "Unified metrics: ${OUTPUT_ROOT}/metrics/summary.json"
