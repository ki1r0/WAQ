#!/usr/bin/env bash
set -Eeuo pipefail

# Continuation launcher for the formal VA2 + Critic2 LIBERO-10 run.  It keeps
# the original output root and worker mapping, but uses a private client that
# skips immutable trial records and never writes MP4 files.
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
NO_VIDEO_CLIENT=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/libero_client_no_video_resume_20260910.py
POLICY_ID=va2-gripperfix-20260909
RUN_ID=EVAL-LIBERO10-COSMOS-JOINT-GRIPPERFIX-B008-20260910-10GPU
TRIALS_PER_TASK=50
FULL_ROOT="$OUTPUT_ROOT/full/guided"
RESUME_MAP_DIR="$FULL_ROOT/novideo_resume_maps_20260910"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
IFS=',' read -r -a TASKS <<< "$TASK_IDS"
if [[ ${#GPUS[@]} -ne ${#TASKS[@]} ]]; then
  echo "GPU/TASK count mismatch" >&2
  exit 64
fi

require_file() { [[ -s "$1" ]] || { echo "missing file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing dir: $1" >&2; exit 2; }; }
require_file "$Q_SERVER"
require_file "$Q_CHECKPOINT/training_state.pt"
require_file "$Q_CHECKPOINT/config.json"
require_file "$Q_CHECKPOINT/manifest.json"
require_file "$MANIFEST"
require_file "$NO_VIDEO_CLIENT"
require_dir "$MODEL"
require_dir "$LIBERO_ROOT"
require_dir "$RESUME_MAP_DIR"
grep -q 'NO_VIDEO' "$NO_VIDEO_CLIENT" || { echo "private client is not marked NO_VIDEO" >&2; exit 2; }

mkdir -p "$FULL_ROOT/logs" "$FULL_ROOT/server" "$FULL_ROOT/client" "$FULL_ROOT/q_values"

server_pids=()
client_pids=()
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

timestamp() { date --iso-8601=seconds; }
host_log="$FULL_ROOT/logs/novideo-host-${HOST_LABEL}.log"
echo "[$(timestamp)] NOVVIDEO_START host=$HOST_LABEL mapping_gpu_ids=$GPU_IDS mapping_task_ids=$TASK_IDS worker_offset=$WORKER_OFFSET port_base=$PORT_BASE output_root=$OUTPUT_ROOT manifest=$MANIFEST client=$NO_VIDEO_CLIENT" | tee -a "$host_log"

# Validate each map against the post-stop records immediately before starting
# servers.  A stale map aborts before any target GPU is used.
for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"
  task="${TASKS[$i]}"
  worker=$((WORKER_OFFSET+i))
  map="$RESUME_MAP_DIR/worker-$worker.json"
  record_root="$FULL_ROOT/client/worker-$worker/task_records/libero_10/task_$(printf '%05d' "$task")"
  require_file "$map"
  map_info="$($LINGBOT_PYTHON - "$map" "$task" "$TRIALS_PER_TASK" "$record_root" <<'PY'
import json
import sys
from pathlib import Path

map_path, task, trials, record_root = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
raw = json.loads(Path(map_path).read_text())
assert set(raw) == {str(task)}, (map_path, sorted(raw))
remaining = [int(x) for x in raw[str(task)]]
assert len(remaining) == len(set(remaining)), (map_path, remaining)
assert all(0 <= x < trials for x in remaining), (map_path, remaining)
existing = {}
for p in sorted(record_root.glob('trial_*.json')):
    try:
        row = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    idx = row.get('trial_index')
    if row.get('status') == 'completed' and isinstance(idx, int) and 0 <= idx < trials:
        existing[idx] = row
expected = sorted(set(range(trials)) - set(existing))
assert sorted(remaining) == expected, (map_path, sorted(remaining), expected)
print(json.dumps({'skip_existing': len(existing), 'remaining': len(remaining), 'existing_indices': sorted(existing), 'remaining_indices': sorted(remaining)}, sort_keys=True))
PY
  )"
  echo "[$(timestamp)] RESUME_MAP host=$HOST_LABEL worker=$worker gpu=$gpu task=$task map=$map $map_info" | tee -a "$host_log"
done

wait_for_server() {
  local pid="$1" log="$2" attempt
  for ((attempt=1; attempt<=600; attempt++)); do
    grep -q "server listening on" "$log" 2>/dev/null && return 0
    kill -0 "$pid" 2>/dev/null || { echo "server exited: $pid" >&2; tail -n 120 "$log" >&2 || true; return 1; }
    sleep 1
  done
  echo "server timeout: $pid" >&2
  tail -n 120 "$log" >&2 || true
  return 1
}

for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"
  task="${TASKS[$i]}"
  worker=$((WORKER_OFFSET+i))
  port=$((PORT_BASE+i))
  log="$FULL_ROOT/logs/novideo-server-$worker.log"
  echo "[$(timestamp)] START_SERVER host=$HOST_LABEL worker=$worker gpu=$gpu task=$task port=$port log=$log" | tee -a "$host_log"
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

for i in "${!server_pids[@]}"; do
  worker=$((WORKER_OFFSET+i))
  wait_for_server "${server_pids[$i]}" "$FULL_ROOT/logs/novideo-server-$worker.log"
done

for ((i=0; i<${#GPUS[@]}; i++)); do
  gpu="${GPUS[$i]}"
  task="${TASKS[$i]}"
  worker=$((WORKER_OFFSET+i))
  port=$((PORT_BASE+i))
  map="$RESUME_MAP_DIR/worker-$worker.json"
  log="$FULL_ROOT/logs/novideo-client-$worker.log"
  echo "[$(timestamp)] START_CLIENT host=$HOST_LABEL worker=$worker gpu=$gpu task=$task port=$port trials=$TRIALS_PER_TASK resume_map=$map no_video=1 log=$log" | tee -a "$host_log"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH="$OVERLAY:$CLIENT_ROOT:$LIBERO_ROOT:$VJEPA_ROOT:$SERVER_ROOT" \
    "$CLIENT_PYTHON" -u "$NO_VIDEO_CLIENT" --libero-benchmark libero_10 --task-indices "$task" \
      --port "$port" --test-num "$TRIALS_PER_TASK" --trial-offset 0 --out-dir "$FULL_ROOT/client/worker-$worker" \
      --policy-id "$POLICY_ID" --policy-checkpoint "$MODEL" --inference-config libero-long-frame2-attn-torch \
      --run-id "$RUN_ID-full-worker-$worker" --resume-map "$map" > "$log" 2>&1 &
  client_pids+=("$!")
done

failed=0
for p in "${client_pids[@]}"; do
  wait "$p" || failed=1
done
for p in "${server_pids[@]}"; do
  kill -TERM "$p" 2>/dev/null || true
done
sleep 3
for p in "${server_pids[@]}"; do
  kill -KILL "$p" 2>/dev/null || true
done
server_pids=()
client_pids=()

if [[ "$failed" -ne 0 ]]; then
  echo "[$(timestamp)] NOVVIDEO_FAILED host=$HOST_LABEL" | tee -a "$host_log"
  exit 1
fi
for ((i=0; i<${#GPUS[@]}; i++)); do
  worker=$((WORKER_OFFSET+i))
  printf '%s host=%s worker=%s gpu=%s task=%s trials=%s no_video=1\n' "$(timestamp)" "$HOST_LABEL" "$worker" "${GPUS[$i]}" "${TASKS[$i]}" "$TRIALS_PER_TASK" > "$FULL_ROOT/novideo-worker-$worker.COMPLETE"
done
echo "[$(timestamp)] NOVVIDEO_COMPLETE host=$HOST_LABEL workers=${#GPUS[@]} output_root=$OUTPUT_ROOT" | tee -a "$host_log"
