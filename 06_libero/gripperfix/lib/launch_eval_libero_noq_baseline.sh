#!/usr/bin/env bash
# Guided-only official LIBERO-10 evaluation template.
#
# NO-GUIDANCE BASELINE. Byte-identical to the guided launcher except the
# guidance weight is 0 (guidance = weight * grad, so the action delta is
# exactly zero) -- same server, same protocol, same client, same task split,
# so it is a true paired control for the B=.08 guided arm. It performs a
# one-episode smoke before the 500-episode run by default. The checkpoint,
# identity manifest, policy, and output root are explicit and can be replaced
# through environment variables for a new immutable evaluation lineage.
set -Eeuo pipefail

umask 007

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly ROOT="${ROOT:-/path/to/home/wy/lingbot-va-original-rl-pi0.6}"
readonly OPS="${OPS:-/path/to/home/wy/lingbot-va-original-rl/remote_configs/rs1/libero_10_original_20260729}"
readonly Q_SERVER="${Q_SERVER:-/path/to/mnt/ops/libero_pi06_qema_20260827/wan_va_server_q_guiding_libero_fixed_cpb_language_20260828.py}"
readonly MANIFEST="${MANIFEST:-${OPS}/libero_10_eval_manifest_500.json}"
readonly LINGBOT_PYTHON="${LINGBOT_PYTHON:-/path/to/home/miniconda3/envs/lingbot/bin/python}"
readonly CLIENT_PYTHON="${CLIENT_PYTHON:-/path/to/home/miniconda3/envs/liberoplus/bin/python}"
readonly OVERLAY="${OVERLAY:-/path/to/home/wy/lingbot-eval-deps/lib/python3.10/site-packages}"
readonly LIBERO_ROOT="${LIBERO_ROOT:-/path/to/home/wy/LIBERO}"
readonly VJEPA_ROOT="${VJEPA_ROOT:-/path/to/home/wy/vjepa2_source_restore}"
readonly MODEL="${MODEL:-/path/to/mnt4/checkpoints/lingbot-va-posttrain-libero-long}"
readonly Q_CHECKPOINT="${Q_CHECKPOINT:-/path/to/mnt/checkpoints/rl/libero_long_original_language_qtranl_jepa2_pi06_targetqema_25k_20260827/checkpoints/checkpoint_00025000}"
readonly OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/mnt/eval_libero_10_original_pi06qema_fixed_b008_guided_only_20260828}"
readonly RUN_ID="${RUN_ID:-EVAL-LIBERO10-PI06QEMA-FIXED-B008-GUIDED-ONLY-20260828}"
readonly POLICY_ID="${POLICY_ID:-Robbyant/lingbot-va-posttrain-libero-long}"
readonly INFERENCE_CONFIG="${INFERENCE_CONFIG:-libero-long-frame2-attn-torch}"
readonly PORT_BASE="${PORT_BASE:-32600}"
readonly WORKERS=8
readonly TRIALS_PER_TASK=50
readonly -a WORKER_TASKS=("0 1" "2 3" "4" "5" "6" "7" "8" "9")
readonly MODE="${MODE:-full}"

case "${MODE}" in
  preflight|smoke|full) ;;
  *) echo "MODE must be preflight, smoke, or full" >&2; exit 64 ;;
esac

server_pids=()
client_pids=()

cleanup() {
  local pid
  for pid in "${client_pids[@]:-}" "${server_pids[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

require_file() { [[ -s "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }; }
require_dir() { [[ -d "$1" ]] || { echo "missing required directory: $1" >&2; exit 2; }; }

require_file "${Q_SERVER}"
require_file "${MANIFEST}"
require_file "${Q_CHECKPOINT}/training_state.pt"
require_file "${Q_CHECKPOINT}/config.json"
require_file "${Q_CHECKPOINT}/manifest.json"
require_dir "${MODEL}"
require_dir "${LIBERO_ROOT}"

"${LINGBOT_PYTHON}" - "${MANIFEST}" "${Q_CHECKPOINT}/config.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
config = json.loads(Path(sys.argv[2]).read_text())
expected_workers = [[0, 1], [2, 3], [4], [5], [6], [7], [8], [9]]
assert manifest["benchmark"] == "libero_10", manifest.get("benchmark")
assert manifest["trials_per_task"] == 50, manifest.get("trials_per_task")
assert manifest["worker_task_indices"] == expected_workers, manifest["worker_task_indices"]
assert [row["task_index"] for row in manifest["tasks"]] == list(range(10))
assert config["critic_type"] == "twin_language_query_transformer_target_q_v1"
assert config["infer_latent_chunk_size"] == 2
assert config["critic_feature_layers"] in (
    {"start": 0, "stop": 30, "step": 4},
    [0, 4, 8, 12, 16, 20, 24, 28],
)
print("official LIBERO guided-only manifest/checkpoint preflight: PASS")
PY

if [[ "${MODE}" == "preflight" ]]; then
  exit 0
fi

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "refusing pre-existing output root: ${OUTPUT_ROOT}" >&2
  exit 3
fi

mkdir -p "${OUTPUT_ROOT}/run_metadata"
cp "${MANIFEST}" "${OUTPUT_ROOT}/run_metadata/identity_manifest.json"
sha256sum "${MANIFEST}" "${Q_SERVER}" "${Q_CHECKPOINT}/training_state.pt" \
  "${Q_CHECKPOINT}/config.json" "${Q_CHECKPOINT}/manifest.json" \
  > "${OUTPUT_ROOT}/run_metadata/sha256sums.txt"

"${LINGBOT_PYTHON}" - "${OUTPUT_ROOT}/run_metadata/guidance_binding.json" \
  "${MANIFEST}" "${Q_CHECKPOINT}" "${MODEL}" "${RUN_ID}" <<'PY'
import json
import sys
from pathlib import Path

binding = {
    "benchmark": "official_libero_10",
    "identity_manifest": sys.argv[2],
    "critic_checkpoint": sys.argv[3],
    "policy_checkpoint": sys.argv[4],
    "run_id": sys.argv[5],
    "guided_only": False,
    "protocol": "original_temporal_action_kv",
    "frame_chunk_size": 2,
    "action_per_frame": 4,
    "q_guidance_weight_scheduler": "fixed_cpb",
    "q_guidance_scale": 0.0,
    "q_cpb_fixed_budget": 0.0,
    "q_cpb_sigma_peak": 0.25,
    "q_cpb_sigma_concentration": 8.0,
    "q_cpb_confidence": 0.0,
    "q_progress_guidance": False,
    "q_guidance_interval": 5,
    "q_objective": "min",
    "q_gradient_normalized": True,
}
Path(sys.argv[1]).write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n")
PY

wait_for_server() {
  local pid="$1" log="$2" attempt
  for ((attempt = 1; attempt <= 600; attempt++)); do
    if grep -q "server listening on" "${log}" 2>/dev/null; then return 0; fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -n 120 "${log}" >&2 || true
      return 1
    fi
    sleep 1
  done
  tail -n 120 "${log}" >&2 || true
  return 1
}

validate_phase() {
  local phase_root="$1" expected="$2"
  "${LINGBOT_PYTHON}" - "${phase_root}" "${expected}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
records = [json.loads(p.read_text()) for p in sorted((root / "client").glob("worker-*/task_records/libero_10/task_*/trial_*.json"))]
videos = list((root / "client").glob("worker-*/libero_10/**/*.mp4"))
assert len(records) == expected, len(records)
assert len(videos) == expected, len(videos)
assert len({(r["task_index"], r["trial_index"]) for r in records}) == expected
assert all(r["status"] == "completed" for r in records)
assert all(r["server_metadata"]["inference_protocol"] == "original_temporal_action_kv" for r in records)
assert all(r["server_metadata"]["frame_chunk_size"] == 2 for r in records)
assert all(r["server_metadata"]["action_per_frame"] == 4 for r in records)

steps = []
for p in sorted((root / "q_values").glob("worker-*/q_guidance_steps_rank0.jsonl")):
    steps.extend(json.loads(line) for line in p.read_text().splitlines() if line.strip())
assert steps
for row in steps:
    assert row["guidance_weight_scheduler"] == "fixed_cpb"
    assert abs(float(row["cpb_fixed_budget"]) - 0.0) < 1e-9
    assert abs(float(row["guidance_weight"])) < 1e-12, row["guidance_weight"]
    assert abs(float(row["cpb_sigma_peak"]) - 0.25) < 1e-9
    assert abs(float(row["cpb_sigma_concentration"]) - 8.0) < 1e-9
    assert row["guidance_coordinate_scaling"] == "none"
    assert row["guidance_gradient_normalized"] is True
    assert all(math.isfinite(float(v)) for v in row.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
print(json.dumps({"records": len(records), "videos": len(videos), "successes": sum(bool(r["success"]) for r in records)}))
PY
}

run_guided_phase() {
  local phase="$1" trials="$2" worker_count="$3" phase_port_base="$4"
  local phase_root="${OUTPUT_ROOT}/${phase}/guided"
  local worker gpu port server_log client_log
  mkdir -p "${phase_root}/logs" "${phase_root}/server" "${phase_root}/client" "${phase_root}/q_values"
  server_pids=()
  client_pids=()

  for ((worker = 0; worker < worker_count; worker++)); do
    gpu="${worker}"
    port=$((phase_port_base + worker))
    server_log="${phase_root}/logs/server-${worker}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      TOKENIZERS_PARALLELISM=false MUJOCO_GL=egl PYTHONPATH="${OVERLAY}:${ROOT}" \
      "${LINGBOT_PYTHON}" -u "${Q_SERVER}" \
        --config-name libero --libero --frame_chunk_size_override 2 --attn_mode torch \
        --eval_model_path "${MODEL}" --port "${port}" \
        --save_root "${phase_root}/server/worker-${worker}" \
        --q_checkpoint "${Q_CHECKPOINT}" --q_guidance_scale 0.0 \
        --q_guidance_weight_scheduler fixed_cpb --q_guidance_start_step 0 \
        --q_guidance_end_step -1 --q_guidance_interval 5 --q_objective min \
        --q_grad_clip 0.0 --q_grad_normalize --q_history_chunks 10 \
        --q_guidance_activation_checkpointing --q_log_denoising_steps \
        --q_cpb_confidence 0.0 --q_cpb_fixed_budget 0.0 \
        --q_cpb_sigma_peak 0.25 --q_cpb_sigma_concentration 8.0 \
        --q_log_dir "${phase_root}/q_values/worker-${worker}" \
        > "${server_log}" 2>&1 &
    server_pids+=("$!")
  done

  for ((worker = 0; worker < worker_count; worker++)); do
    wait_for_server "${server_pids[$worker]}" "${phase_root}/logs/server-${worker}.log"
  done

  for ((worker = 0; worker < worker_count; worker++)); do
    gpu="${worker}"
    port=$((phase_port_base + worker))
    client_log="${phase_root}/logs/client-${worker}.log"
    local -a tasks
    read -r -a tasks <<< "${WORKER_TASKS[$worker]}"
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
      PYTHONPATH="${OVERLAY}:${LIBERO_ROOT}:${VJEPA_ROOT}:${ROOT}" \
      "${CLIENT_PYTHON}" -u "${ROOT}/evaluation/libero/client.py" \
        --libero-benchmark libero_10 --task-indices "${tasks[@]}" \
        --port "${port}" --test-num "${trials}" --trial-offset 0 \
        --out-dir "${phase_root}/client/worker-${worker}" \
        --policy-id "${POLICY_ID}" --policy-checkpoint "${MODEL}" \
        --inference-config "${INFERENCE_CONFIG}" \
        --run-id "${RUN_ID}-${phase}-worker-${worker}" \
        > "${client_log}" 2>&1 &
    client_pids+=("$!")
  done

  local failed=0 pid
  for pid in "${client_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  for pid in "${server_pids[@]}"; do kill -TERM "${pid}" 2>/dev/null || true; done
  wait 2>/dev/null || true
  if [[ "${failed}" -ne 0 ]]; then
    for client_log in "${phase_root}"/logs/client-*.log; do tail -n 120 "${client_log}" >&2 || true; done
    return 1
  fi
  validate_phase "${phase_root}" "$((worker_count == 1 ? 2 : 500))"
  server_pids=()
  client_pids=()
}

echo "[$(date --iso-8601=seconds)] official LIBERO guided-only fixed-B008 smoke"
run_guided_phase smoke 1 1 "${PORT_BASE}"
if [[ "${MODE}" == "full" ]]; then
  echo "[$(date --iso-8601=seconds)] official LIBERO guided-only fixed-B008 500EP"
  run_guided_phase full "${TRIALS_PER_TASK}" "${WORKERS}" "$((PORT_BASE + 100))"
fi

date --iso-8601=seconds > "${OUTPUT_ROOT}/COMPLETE"
echo "official LIBERO guided-only fixed-B008 evaluation complete: ${OUTPUT_ROOT}"
