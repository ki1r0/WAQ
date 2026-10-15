#!/usr/bin/env bash
# Fifth-generation four-node R2 VA+critic resume launcher (32 ranks).
# Resumes the full-manifest window-sequential joint pretraining from the latest
# complete checkpoint: prefers this run's own latest, else the r4 step_400 root.
# Fixed node/rank layout: 2033:0-7, 2004:8-15, 2113:16-23, 2078:24-31.
#
# Usage:
#   bash launch_r2_full_mixed_4node_bs10_r5_resume.sh 1                       # PREPARE (once, any node)
#   bash launch_r2_full_mixed_4node_bs10_r5_resume.sh 0 <BASE_RANK> [GPUS]    # launch (each node)
# Crash-restart: just re-run the PREPARE=0 command on every node (idempotent:
# it re-detects the newest complete checkpoint and resumes from it).
set -euo pipefail

PREPARE="${1:-0}"
BASE_RANK="${2:-0}"
GPU_LIST="${3:-0,1,2,3,4,5,6,7}"

CONTROL="/mnt/zengqixin/experiments/robocoin_droid_critic_controls_20260826/revised_r3_20260826"
SRC="/media/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb"
PY="/home/zengqixin/miniconda3/envs/lingbot/bin/python"
PREV_ROOT="/mnt/zengqixin/experiments/robocoin_droid_critic_controls_20260826/revised_r3_20260918_4node_bs10_r4"
PREV_SCRIPT="${PREV_ROOT}/r2_revised_r4_patched.py"
RUN_ROOT="/mnt/zengqixin/experiments/robocoin_droid_critic_controls_20260826/revised_r3_20260918_4node_bs10_r5"
SCRIPT="${RUN_ROOT}/r2_revised_r5_patched.py"
CRITIC_CONFIG="${CONTROL}/critic_timecost.json"
DS_CONFIG="${SRC}/configs/deepspeed_zero2_bf16.json"
MANIFEST_PATH="${CONTROL}/manifests/episode_full_seed42.json"
MASTER_ADDR="10.30.40.33"
MASTER_PORT="29702"
WORLD_SIZE=32
BATCH_SIZE=10
SAVE_INTERVAL=100

# DeepSpeed tag dir contract for this model: 1 model states (exact size) + 32 optim shards.
MODEL_STATES_EXPECTED=10177994317
TAG_FILES_EXPECTED=33
MIN_FREE_KIB=209715200   # 200 GiB in KiB; one save writes ~69 GB, keep headroom for two.

verify_step_dir() {  # $1 = step_<N> dir -> 0 if complete
  local d="$1" n ms sz
  [[ -d "${d}" ]] || return 1
  n="$(ls -1 "${d}" | wc -l)"
  [[ "${n}" -eq "${TAG_FILES_EXPECTED}" ]] || return 1
  ms="${d}/mp_rank_00_model_states.pt"
  [[ -f "${ms}" ]] || return 1
  sz="$(stat -c %s "${ms}")"
  [[ "${sz}" -eq "${MODEL_STATES_EXPECTED}" ]] || return 1
  return 0
}

verify_critic_dir() {  # $1 = checkpoint_step_<N>/critic dir
  local d="$1"
  [[ -r "${d}/training_state.pt" && -r "${d}/manifest.json" && -r "${d}/config.json" ]]
}

if [[ "${PREPARE}" != "0" && "${PREPARE}" != "1" ]]; then
  echo "PREPARE must be 0 or 1" >&2; exit 2
fi

if [[ "${PREPARE}" == "1" ]]; then
  [[ -r "${PREV_SCRIPT}" ]] || { echo "missing patched script: ${PREV_SCRIPT}" >&2; exit 2; }
  [[ -r "${CRITIC_CONFIG}" ]] || { echo "missing critic config" >&2; exit 2; }
  [[ -r "${MANIFEST_PATH}" ]] || { echo "missing full manifest" >&2; exit 2; }
  [[ -r "${DS_CONFIG}" ]] || { echo "missing DeepSpeed config" >&2; exit 2; }
  [[ -r "${PREV_ROOT}/checkpoints/latest" ]] || { echo "missing predecessor latest" >&2; exit 2; }
  tag="$(<"${PREV_ROOT}/checkpoints/latest")"
  verify_step_dir "${PREV_ROOT}/checkpoints/${tag}" \
    || { echo "predecessor checkpoint incomplete: ${tag} (see runbook triage)" >&2; exit 2; }
  verify_critic_dir "${PREV_ROOT}/checkpoints/checkpoint_${tag}/critic" \
    || { echo "predecessor critic checkpoint incomplete: ${tag}" >&2; exit 2; }
  mkdir -p "${RUN_ROOT}/logs"
  cp "${PREV_SCRIPT}" "${SCRIPT}"
  cat > "${RUN_ROOT}/run_provenance.txt" <<EOF
initialization=r4 checkpoint resume (auto: own latest if complete, else predecessor)
dataset_loader=robocoin_droid_canonical30
manifest=${MANIFEST_PATH}
batch_per_gpu=${BATCH_SIZE}
world_size=${WORLD_SIZE}
checkpoint_interval_steps=${SAVE_INTERVAL}
predecessor=${PREV_ROOT}
predecessor_latest=${tag}
schedule=max_steps=0 (computed from the full manifest; 2826 VA-window steps)
node_order=lyg2033(rank0-7),lyg2004(rank8-15),lyg2113(rank16-23),lyg2078(rank24-31)
master=${MASTER_ADDR}:${MASTER_PORT}
launcher=launch_r2_full_mixed_4node_bs10_r5_resume.sh (auto-resume; disk guard ${MIN_FREE_KIB} KiB)
EOF
  echo "prepared ${RUN_ROOT} (resume seed: predecessor ${tag})"
  exit 0
fi

# ---- PREPARE=0: launch this node's 8 ranks ----
HOST_SHORT="$(hostname -s)"
case "${HOST_SHORT}:${BASE_RANK}" in
  lyg2033:0|lyg2004:8|lyg2113:16|lyg2078:24) ;;
  *) echo "host/base-rank mismatch: ${HOST_SHORT}:${BASE_RANK}" >&2; exit 2 ;;
esac
[[ -d "${RUN_ROOT}" ]] || { echo "missing output root (run PREPARE=1 first): ${RUN_ROOT}" >&2; exit 2; }
[[ -r "${SCRIPT}" ]] || { echo "missing persistent patched script: ${SCRIPT}" >&2; exit 2; }
[[ -r "${CRITIC_CONFIG}" && -r "${MANIFEST_PATH}" && -r "${DS_CONFIG}" ]] || { echo "missing config inputs" >&2; exit 2; }

# Resume pointer: own latest if complete, else predecessor latest (must be complete).
RESUME_BACKBONE="" RESUME_CRITIC=""
if [[ -r "${RUN_ROOT}/checkpoints/latest" ]]; then
  tag="$(<"${RUN_ROOT}/checkpoints/latest")"
  if verify_step_dir "${RUN_ROOT}/checkpoints/${tag}" \
     && verify_critic_dir "${RUN_ROOT}/checkpoints/checkpoint_${tag}/critic"; then
    RESUME_BACKBONE="${RUN_ROOT}/checkpoints"
    RESUME_CRITIC="${RUN_ROOT}/checkpoints/checkpoint_${tag}/critic"
  fi
fi
if [[ -z "${RESUME_BACKBONE}" ]]; then
  [[ -r "${PREV_ROOT}/checkpoints/latest" ]] || { echo "no usable resume checkpoint" >&2; exit 2; }
  tag="$(<"${PREV_ROOT}/checkpoints/latest")"
  verify_step_dir "${PREV_ROOT}/checkpoints/${tag}" \
    || { echo "predecessor checkpoint incomplete: ${tag}" >&2; exit 2; }
  verify_critic_dir "${PREV_ROOT}/checkpoints/checkpoint_${tag}/critic" \
    || { echo "predecessor critic checkpoint incomplete: ${tag}" >&2; exit 2; }
  RESUME_BACKBONE="${PREV_ROOT}/checkpoints"
  RESUME_CRITIC="${PREV_ROOT}/checkpoints/checkpoint_${tag}/critic"
fi
echo "resume backbone=${RESUME_BACKBONE} (latest=${tag}) critic=${RESUME_CRITIC}"

# Disk guard: CephFS at ~98% has already killed one run mid-save (incidents #2).
if [[ "${SKIP_DISK_CHECK:-0}" != "1" ]]; then
  free_kib="$(df -Pk /mnt | awk 'NR==2 {print $4}')"
  if [[ "${free_kib}" -lt "${MIN_FREE_KIB}" ]]; then
    echo "refusing to start: /mnt free ${free_kib} KiB < ${MIN_FREE_KIB} KiB" >&2
    echo "clean old step_* checkpoints first (runbook section 6), or set SKIP_DISK_CHECK=1 to override" >&2
    exit 3
  fi
fi
if [[ "${BASE_RANK}" == "0" ]] && { ss -ltn 2>/dev/null || true; } | grep -q ":${MASTER_PORT} "; then
  echo "master port ${MASTER_PORT} already in use on ${HOST_SHORT}" >&2; exit 2
fi

mkdir -p "${RUN_ROOT}/logs"
export PYTHONPATH="${SRC}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${RUN_ROOT}/wandb"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=bond0.45
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TRACE_BUFFER_SIZE=2048

pids=()
IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
for gpu in "${GPUS[@]}"; do
  rank=$((BASE_RANK + gpu))
  CUDA_VISIBLE_DEVICES="${gpu}" \
    RANK="${rank}" WORLD_SIZE="${WORLD_SIZE}" LOCAL_RANK=0 LOCAL_WORLD_SIZE=1 \
    MASTER_ADDR="${MASTER_ADDR}" MASTER_PORT="${MASTER_PORT}" \
    "${PY}" -u "${SCRIPT}" \
      --mode sequential_r2 \
      --va-config-name robocoin_droid_joint_actionaligned_r11_v1 \
      --critic-config "${CRITIC_CONFIG}" \
      --episode-subset-manifest "${MANIFEST_PATH}" \
      --save-root "${RUN_ROOT}" \
      --deepspeed --deepspeed-config "${DS_CONFIG}" \
      --batch-size "${BATCH_SIZE}" \
      --save-interval "${SAVE_INTERVAL}" \
      --log-interval 10 --seed 42 --max-steps 0 \
      --resume-from "${RESUME_BACKBONE}" \
      --critic-resume-from "${RESUME_CRITIC}" \
      > "${RUN_ROOT}/logs/rank_${rank}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
