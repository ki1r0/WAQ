#!/usr/bin/env bash
set -Eeuo pipefail
: "${HOST_LABEL:?HOST_LABEL required}"
: "${GPU_IDS:?GPU_IDS required, comma-separated}"
: "${TASK_IDS:?TASK_IDS required, comma-separated}"
: "${WORKER_OFFSET:?WORKER_OFFSET required}"
: "${PORT_BASE:?PORT_BASE required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT required}"
SERVER_ROOT=/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb
CLIENT_ROOT=/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb
OVERLAY=/home/zengqixin/wy/lingbot-eval-deps/lib/python3.10/site-packages
LIBERO_ROOT=/home/zengqixin/wy/LIBERO
VJEPA_ROOT=/home/zengqixin/wy/vjepa2_source_restore
LINGBOT_PYTHON=/home/zengqixin/miniconda3/envs/lingbot/bin/python
CLIENT_PYTHON=/home/zengqixin/miniconda3/envs/liberoplus/bin/python
Q_SERVER=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/wan_va_server_q_guiding_critic2_v1_libero.py
Q_CHECKPOINT=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/critic2_v1_compat
MODEL=/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers
MANIFEST=/home/zengqixin/wy/lingbot-va-original-rl-pi0.6/remote_configs/rs1/libero_10_original_20260729/libero_10_eval_manifest_500.json
POLICY_ID=va2-gripperfix-20260909
RUN_ID=EVAL-LIBERO10-COSMOS-JOINT-GRIPPERFIX-B008-20260910-10GPU
FULL_ROOT="$OUTPUT_ROOT/full/guided"
mkdir -p "$FULL_ROOT/logs" "$FULL_ROOT/server" "$FULL_ROOT/client" "$FULL_ROOT/q_values"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
IFS=',' read -r -a TASKS <<< "$TASK_IDS"
if [[ ${#GPUS[@]} -ne ${#TASKS[@]} ]]; then echo "GPU/TASK count mismatch" >&2; exit 64; fi
server_pids=(); client_pids=(); worker_ids=()
cleanup() {
  trap - EXIT INT TERM
  local p
  for p in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "$p" ]] && kill -TERM "$p" 2>/dev/null || true
  done
  sleep 3
  for p in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    [[ -n "$p" ]] && kill -KILL "$p" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM
require_file() { [[ -s "$1" ]] || { echo "missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing dir: $1" >&2; exit 2; }; }
require_file "$Q_SERVER"; require_file "$Q_CHECKPOINT/training_state.pt"; require_file "$Q_CHECKPOINT/config.json"; require_file "$Q_CHECKPOINT/manifest.json"; require_file "$MANIFEST"; require_dir "$MODEL"; require_dir "$LIBERO_ROOT"
wait_for_server() {
  local pid="$1" log="$2" attempt
  for ((attempt=1; attempt<=600; attempt++)); do
    grep -q "server listening on" "$log" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || { echo "server exited: $pid" >&2; tail -n 120 "$log" >&2 || true; return 1; }
    sleep 1
  done
  echo "server timeout: $pid" >&2; tail -n 120 "$log" >&2 || true; return 1
}
for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"; task="${TASKS[$i]}"; worker=$((WORKER_OFFSET+i)); port=$((PORT_BASE+i))
  worker_ids+=("$worker")
  log="$FULL_ROOT/logs/server-$worker.log"
  echo "[$(date --iso-8601=seconds)] START_SERVER host=$HOST_LABEL worker=$worker gpu=$gpu task=$task port=$port" | tee -a "$FULL_ROOT/logs/host-$HOST_LABEL.log"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYTHONPATH="$SERVER_ROOT:$OVERLAY" \
    "$LINGBOT_PYTHON" -u "$Q_SERVER" --config-name libero --libero --frame_chunk_size_override 2 --attn_mode torch \
      --eval_model_path "$MODEL" --port "$port" --save_root "$FULL_ROOT/server/worker-$worker" \
      --q_checkpoint "$Q_CHECKPOINT" --q_guidance_scale 0.08 --q_guidance_weight_scheduler fixed_cpb \
      --q_guidance_start_step 0 --q_guidance_end_step -1 --q_guidance_interval 5 --q_objective min \
      --q_grad_clip 0.0 --q_grad_normalize --q_history_chunks 10 --q_guidance_activation_checkpointing --q_log_denoising_steps \
      --q_cpb_confidence 0.0 --q_cpb_fixed_budget 0.08 --q_cpb_sigma_peak 0.25 --q_cpb_sigma_concentration 8.0 \
      --q_log_dir "$FULL_ROOT/q_values/worker-$worker" > "$log" 2>&1 &
  server_pids+=("$!")
done
for i in "${!server_pids[@]}"; do wait_for_server "${server_pids[$i]}" "$FULL_ROOT/logs/server-${worker_ids[$i]}.log"; done
for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"; task="${TASKS[$i]}"; worker=$((WORKER_OFFSET+i)); port=$((PORT_BASE+i))
  log="$FULL_ROOT/logs/client-$worker.log"
  echo "[$(date --iso-8601=seconds)] START_CLIENT host=$HOST_LABEL worker=$worker gpu=$gpu task=$task port=$port trials=50" | tee -a "$FULL_ROOT/logs/host-$HOST_LABEL.log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH="$OVERLAY:$CLIENT_ROOT:$LIBERO_ROOT:$VJEPA_ROOT:$SERVER_ROOT" \
    "$CLIENT_PYTHON" -u "$CLIENT_ROOT/evaluation/libero/client.py" --libero-benchmark libero_10 --task-indices "$task" \
      --port "$port" --test-num 50 --trial-offset 0 --out-dir "$FULL_ROOT/client/worker-$worker" \
      --policy-id "$POLICY_ID" --policy-checkpoint "$MODEL" --inference-config libero-long-frame2-attn-torch \
      --run-id "$RUN_ID-full-worker-$worker" > "$log" 2>&1 &
  client_pids+=("$!")
done
failed=0
for p in "${client_pids[@]}"; do wait "$p" || failed=1; done
if [[ "$failed" -ne 0 ]]; then echo "[$(date --iso-8601=seconds)] HOST_FAILED host=$HOST_LABEL" | tee -a "$FULL_ROOT/logs/host-$HOST_LABEL.log"; exit 1; fi
for ((i=0; i<${#GPUS[@]}; i++)); do worker=$((WORKER_OFFSET+i)); printf '%s host=%s worker=%s gpu=%s task=%s trials=50\n' "$(date --iso-8601=seconds)" "$HOST_LABEL" "$worker" "${GPUS[$i]}" "${TASKS[$i]}" > "$FULL_ROOT/worker-$worker.COMPLETE"; done
echo "[$(date --iso-8601=seconds)] HOST_COMPLETE host=$HOST_LABEL workers=${#GPUS[@]}" | tee -a "$FULL_ROOT/logs/host-$HOST_LABEL.log"