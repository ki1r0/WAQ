#!/usr/bin/env bash
set -Eeuo pipefail

: "${NODE_LABEL:?NODE_LABEL is required}"
: "${GPU_IDS:?GPU_IDS is required, comma-separated}"
: "${PORT_BASE:?PORT_BASE is required}"
: "${RESULT_ROOT:?RESULT_ROOT is required}"
: "${QUEUE_ROOT:?QUEUE_ROOT is required}"

SERVER_ROOT=/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb
CLIENT_ROOT=/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb
OVERLAY=/home/zengqixin/wy/lingbot-eval-deps/lib/python3.10/site-packages
LIBERO_ROOT=/home/zengqixin/wy/LIBERO
VJEPA_ROOT=/home/zengqixin/wy/vjepa2_source_restore
LINGBOT_PYTHON=/home/zengqixin/miniconda3/envs/lingbot/bin/python
CLIENT_PYTHON=/home/zengqixin/miniconda3/envs/liberoplus/bin/python
STATE_PY=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/libero_node_queue_state_20260910.py
CLIENT_ENTRY=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/libero_client_no_video_20260910.py
Q_SERVER=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/wan_va_server_q_guiding_critic2_v1_libero.py
Q_CHECKPOINT=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/critic2_v1_compat
MODEL=/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers
MANIFEST=/home/zengqixin/wy/lingbot-va-original-rl-pi0.6/remote_configs/rs1/libero_10_original_20260729/libero_10_eval_manifest_500.json
POLICY_ID=va2-gripperfix-20260909
RUN_ID=EVAL-LIBERO10-COSMOS-JOINT-GRIPPERFIX-B008-20260910-NODEQUEUE

mkdir -p "$QUEUE_ROOT/claims" "$QUEUE_ROOT/logs" "$QUEUE_ROOT/assignments"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
GPU_COUNT=${#GPUS[@]}
(( GPU_COUNT > 0 )) || { echo "no GPUs" >&2; exit 64; }

require_file() { [[ -s "$1" ]] || { echo "missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing dir: $1" >&2; exit 2; }; }
require_file "$STATE_PY"
require_file "$CLIENT_ENTRY"
require_file "$Q_SERVER"
require_file "$Q_CHECKPOINT/training_state.pt"
require_file "$Q_CHECKPOINT/config.json"
require_file "$Q_CHECKPOINT/manifest.json"
require_file "$MANIFEST"
require_dir "$MODEL"
require_dir "$LIBERO_ROOT"

now() { date --iso-8601=seconds; }
log() { echo "[$(now)] $*" | tee -a "$QUEUE_ROOT/logs/node-$NODE_LABEL.log"; }
json_field() {
  "$LINGBOT_PYTHON" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

claim_next_task() {
  local task state completed claim
  for task in 0 1 2 3 4 5 6 7 8 9; do
    state=$("$LINGBOT_PYTHON" "$STATE_PY" --root "$RESULT_ROOT" --task "$task")
    completed=$(printf '%s' "$state" | json_field completed)
    if (( completed >= 50 )); then
      touch "$QUEUE_ROOT/claims/task_${task}.DONE"
      continue
    fi
    claim="$QUEUE_ROOT/claims/task_${task}"
    if mkdir "$claim" 2>/dev/null; then
      printf 'node=%s\nclaimed_at=%s\ntask=%s\n' "$NODE_LABEL" "$(now)" "$task" > "$claim/owner"
      printf '%s node=%s task=%s completed_before=%s remaining_before=%s\n' "$(now)" "$NODE_LABEL" "$task" "$completed" "$((50-completed))" >> "$QUEUE_ROOT/assignments.log"
      printf '%s\n' "$task"
      return 0
    fi
  done
  return 1
}

wait_for_server() {
  local pid="$1" log_path="$2" attempt
  for ((attempt=1; attempt<=300; attempt++)); do
    grep -q "server listening on" "$log_path" 2>/dev/null && return 0
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "server exited pid=$pid" >&2
      tail -n 80 "$log_path" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "server timeout pid=$pid" >&2
  tail -n 80 "$log_path" >&2 || true
  return 1
}

run_task() {
  local task="$1" task_root="$QUEUE_ROOT/$NODE_LABEL/task_${task}"
  local assign_file="$QUEUE_ROOT/assignments/${NODE_LABEL}.task_${task}.txt"
  local -a specs server_pids client_pids
  local i gpu port count csv out slog clog
  mkdir -p "$task_root" "$task_root/logs"
  cleanup_task_pids() {
    local p
    for p in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
      [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true
    done
    sleep 2
    for p in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
      [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null || true
    done
  }
  mapfile -t specs < <("$LINGBOT_PYTHON" "$STATE_PY" --root "$RESULT_ROOT" --task "$task" --gpu-count "$GPU_COUNT" --assign-csv)
  : > "$assign_file"
  for ((i=0; i<GPU_COUNT; i++)); do
    count="${specs[$i]%%|*}"
    csv="${specs[$i]#*|}"
    gpu="${GPUS[$i]}"
    printf '%s node=%s task=%s gpu=%s count=%s trials=%s\n' "$(now)" "$NODE_LABEL" "$task" "$gpu" "$count" "$csv" | tee -a "$assign_file" >> "$QUEUE_ROOT/assignments.log"
  done

  server_pids=()
  for ((i=0; i<GPU_COUNT; i++)); do
    count="${specs[$i]%%|*}"
    (( count > 0 )) || continue
    gpu="${GPUS[$i]}"
    port=$((PORT_BASE+i))
    slog="$task_root/logs/server-gpu${gpu}.log"
    log "START_SERVER task=$task gpu=$gpu port=$port count=$count"
    CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYTHONPATH="$SERVER_ROOT:$OVERLAY" \
      "$LINGBOT_PYTHON" -u "$Q_SERVER" --config-name libero --libero --frame_chunk_size_override 2 --attn_mode torch \
      --eval_model_path "$MODEL" --port "$port" --save_root "$task_root/server-gpu${gpu}" \
      --q_checkpoint "$Q_CHECKPOINT" --q_guidance_scale 0.08 --q_guidance_weight_scheduler fixed_cpb \
      --q_guidance_start_step 0 --q_guidance_end_step -1 --q_guidance_interval 5 --q_objective min \
      --q_grad_clip 0.0 --q_grad_normalize --q_history_chunks 10 --q_guidance_activation_checkpointing --q_log_denoising_steps \
      --q_cpb_confidence 0.0 --q_cpb_fixed_budget 0.08 --q_cpb_sigma_peak 0.25 --q_cpb_sigma_concentration 8.0 \
      --q_log_dir "$task_root/q_values-gpu${gpu}" > "$slog" 2>&1 &
    server_pids+=("$!")
  done
  local server_index=0
  for ((i=0; i<GPU_COUNT; i++)); do
    count="${specs[$i]%%|*}"
    (( count > 0 )) || continue
    if ! wait_for_server "${server_pids[$server_index]}" "$task_root/logs/server-gpu${GPUS[$i]}.log"; then
      cleanup_task_pids
      return 1
    fi
    ((server_index++))
  done

  client_pids=()
  for ((i=0; i<GPU_COUNT; i++)); do
    count="${specs[$i]%%|*}"
    csv="${specs[$i]#*|}"
    (( count > 0 )) || continue
    gpu="${GPUS[$i]}"
    port=$((PORT_BASE+i))
    out="$task_root/client-gpu${gpu}"
    clog="$task_root/logs/client-gpu${gpu}.log"
    IFS=',' read -r -a trial_args <<< "$csv"
    log "START_CLIENT task=$task gpu=$gpu port=$port count=$count trials=$csv"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 LIBERO_NO_VIDEO=1 MUJOCO_GL=egl PYTHONPATH="$OVERLAY:$CLIENT_ROOT:$LIBERO_ROOT:$VJEPA_ROOT:$SERVER_ROOT" \
      "$CLIENT_PYTHON" -u "$CLIENT_ENTRY" --libero-benchmark libero_10 --task-indices "$task" \
      --port "$port" --test-num "$count" --trial-offset 0 --trial-indices "${trial_args[@]}" --out-dir "$out" \
      --policy-id "$POLICY_ID" --policy-checkpoint "$MODEL" --inference-config libero-long-frame2-attn-torch \
      --skip-existing --run-id "$RUN_ID-node-$NODE_LABEL-task-$task-gpu-$gpu" > "$clog" 2>&1 &
    client_pids+=("$!")
  done

  local failed=0 p state completed duplicates
  for p in "${client_pids[@]}"; do wait "$p" || failed=1; done
  client_pids=()
  for p in "${server_pids[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  server_pids=()
  if (( failed )); then
    log "TASK_CLIENT_FAILED task=$task"
    cleanup_task_pids
    return 1
  fi
  state=$("$LINGBOT_PYTHON" "$STATE_PY" --root "$RESULT_ROOT" --task "$task")
  completed=$(printf '%s' "$state" | json_field completed)
  duplicates=$(printf '%s' "$state" | json_field duplicates)
  if (( completed < 50 || duplicates > 0 )); then
    log "TASK_INCOMPLETE task=$task completed=$completed duplicates=$duplicates"
    return 1
  fi
  printf '%s\n' "$(now) node=$NODE_LABEL task=$task completed=50" > "$QUEUE_ROOT/claims/task_${task}/DONE"
  log "TASK_COMPLETE task=$task completed=50"
  return 0
}

log "QUEUE_START gpus=$GPU_IDS gpu_count=$GPU_COUNT result_root=$RESULT_ROOT"
while true; do
  task="$(claim_next_task || true)"
  if [[ -z "$task" ]]; then
    all_done=1
    for t in 0 1 2 3 4 5 6 7 8 9; do
      st=$("$LINGBOT_PYTHON" "$STATE_PY" --root "$RESULT_ROOT" --task "$t")
      c=$(printf '%s' "$st" | json_field completed)
      if (( c < 50 )); then all_done=0; fi
    done
    if (( all_done )); then
      log "QUEUE_COMPLETE"
      exit 0
    fi
    log "QUEUE_WAIT another node holds remaining tasks"
    sleep 30
    continue
  fi
  log "TASK_CLAIMED task=$task"
  if run_task "$task"; then
    rmdir "$QUEUE_ROOT/claims/task_${task}" 2>/dev/null || true
    touch "$QUEUE_ROOT/claims/task_${task}.DONE"
  else
    log "TASK_RETRY task=$task"
    rm -rf "$QUEUE_ROOT/claims/task_${task}" 2>/dev/null || true
    sleep 5
  fi
done
