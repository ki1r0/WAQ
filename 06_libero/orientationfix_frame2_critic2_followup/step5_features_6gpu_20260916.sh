#!/usr/bin/env bash
# Isolated Track-A step 5: orientation+gripper-fixed VA2 feature extraction.
# This is a fresh copy of the old Track-A flow with only target roots, model,
# config names, camera keys, and output paths changed.
set -Eeuo pipefail

STAGE="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/orientationfix_va2_25k_20260916"
LIB="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/lib"
PY="/path/to/lingbot-env/bin/python"
REPO="/path/to/lingbot-va"
MODEL="/path/to/storage/checkpoints/va2_libero10_cosmos_orientationfix_gripperfix_frame2_repro/export_step25000"
DEVICES="${DEVICES:-1,2,3,4,5,6}"
NPROC="$(awk -F, '{print NF}' <<<"${DEVICES}")"
TARGET_DATASET="/path/to/storage/datasets/libero10_va2_success_mix_orientationfix_gripperfix_frame2_final_20260912"
OUT_OPEN="${TARGET_DATASET}/inference_state_cache/orientationfix_va2_step25000_server_prefix_8layer_20260916/open500"
OUT_COSMOS="${TARGET_DATASET}/inference_state_cache/orientationfix_va2_step25000_server_prefix_8layer_20260916/cosmos_success376"
OPEN_ROOT="${TARGET_DATASET}/open500"
COSMOS_ROOT="${TARGET_DATASET}/cosmos_success376"
LOG_ROOT="${STAGE}/logs/step5"

source /path/to/miniconda/etc/profile.d/conda.sh
conda activate lingbot
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${STAGE}:${REPO}:/path/to/vjepa2-source${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1

echo "[$(date --iso-8601=seconds)] STEP5_START devices=${DEVICES} nproc=${NPROC}"
[[ -s "${STAGE}/preflight.json" ]] || { echo "missing preflight evidence"; exit 20; }
"${PY}" - "${STAGE}/preflight.json" <<'PYCHK'
import json, sys
payload = json.load(open(sys.argv[1]))
if payload.get("status") != "pass" or payload.get("read_only_checks") is not True:
    raise SystemExit(f"preflight did not pass: {payload.get('status')}")
print("PREFLIGHT_PASS", payload["stage"])
PYCHK

[[ -s "${MODEL}/COMPLETE" ]] || { echo "missing export COMPLETE"; exit 21; }
[[ -s "${MODEL}/transformer/diffusion_pytorch_model.safetensors" ]] || { echo "missing target transformer"; exit 21; }
[[ -d "${OPEN_ROOT}/data" && -d "${COSMOS_ROOT}/data" ]] || { echo "missing target data roots"; exit 21; }
for p in "${OUT_OPEN}" "${OUT_COSMOS}" "${STAGE}/catalog_open.json" "${STAGE}/catalog_cosmos.json" "${STAGE}/overlay_open.json" "${STAGE}/overlay_cosmos.json" "${STAGE}/sidecar_open" "${STAGE}/sidecar_cosmos" "${STAGE}/federated" "${STAGE}/STEP5_DONE"; do
  [[ ! -e "${p}" ]] || { echo "refusing pre-existing step5 output: ${p}"; exit 22; }
done

run_cache() {
  local name="$1" config="$2" output="$3" port="$4" dataset_root="$5"
  mkdir -p "${LOG_ROOT}/${name}"
  echo "[$(date --iso-8601=seconds)] CACHE_START ${name} dataset=${dataset_root} output=${output}"
  "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    --master_port="${port}" --log_dir="${LOG_ROOT}/${name}/torchrun" --tee 3 \
    -m wan_va.rl.precompute_phase3_cache \
    --config "${config}" --output-dir "${output}" \
    --cache-feature-layers "0,4,8,12,16,20,24,28" \
    --batch-size 1 --num-workers 0 --dataset-init-workers 0 \
    --log-interval 1 --progress --standalone-ranks
  [[ -s "${output}/manifest.json" ]] || { echo "cache manifest missing: ${output}"; exit 24; }
  "${PY}" - "${output}/manifest.json" "${dataset_root}" "${name}" <<'PYMAN'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text())
payload["target_dataset_root"] = sys.argv[2]
payload["target_protocol"] = {
    "source_fps": 20,
    "target_fps": 10,
    "frame_chunk_size": 2,
    "action_per_frame": 4,
    "action_downsample_factor": 2,
    "action_downsample_policy": "stride_2",
    "camera_keys": ["observation.images.front", "observation.images.wrist"],
}
payload["target_source_name"] = sys.argv[3]
path.write_text(json.dumps(payload, indent=2) + "\n")
print("CACHE_MANIFEST_TARGET_BOUND", path, sys.argv[2])
PYMAN
  echo "[$(date --iso-8601=seconds)] CACHE_DONE ${name}"
}

run_cache open "${STAGE}/cache_open.json" "${OUT_OPEN}" 31451 "${OPEN_ROOT}"
run_cache cosmos "${STAGE}/cache_cosmos.json" "${OUT_COSMOS}" 31452 "${COSMOS_ROOT}"

echo "[$(date --iso-8601=seconds)] CATALOG_START"
"${PY}" "${LIB}/build_source_catalog.py" \
  --cache-dir "${OUT_OPEN}" --source-root "${OPEN_ROOT}" \
  --base-config "orientationfix_gripperfix_va2_open500_frame2_cache_20260916" \
  --task-name official_libero_long_10_tasks --source-name official_libero \
  --contract-name "orientationfix_va2_25k_20260916_open500" \
  --note "orientation-fixed + gripper-fixed open500; frame2/10Hz/action stride2" \
  --catalog-out "${STAGE}/catalog_open.json" --overlay-out "${STAGE}/overlay_open.json"
"${PY}" "${LIB}/build_source_catalog.py" \
  --cache-dir "${OUT_COSMOS}" --source-root "${COSMOS_ROOT}" \
  --base-config "orientationfix_gripperfix_va2_cosmos_success376_frame2_cache_20260916" \
  --task-name cosmos_policy_libero_10 --source-name cosmos_policy \
  --contract-name "orientationfix_va2_25k_20260916_cosmos_success376" \
  --note "orientation-fixed + gripper-fixed Cosmos success376; frame2/10Hz/action stride2" \
  --catalog-out "${STAGE}/catalog_cosmos.json" --overlay-out "${STAGE}/overlay_cosmos.json"

echo "[$(date --iso-8601=seconds)] SIDECAR_START"
"${PY}" /path/to/storage/ops/libero_pi06_qema_20260827/build_single_source_language_sidecar.py \
  --catalog "${STAGE}/catalog_open.json" --source-root "${OPEN_ROOT}" \
  --camera-key observation.images.wrist --model-root "${MODEL}" \
  --output-dir "${STAGE}/sidecar_open"
"${PY}" /path/to/storage/ops/libero_pi06_qema_20260827/build_single_source_language_sidecar.py \
  --catalog "${STAGE}/catalog_cosmos.json" --source-root "${COSMOS_ROOT}" \
  --camera-key observation.images.front --model-root "${MODEL}" \
  --output-dir "${STAGE}/sidecar_cosmos"

"${PY}" "${LIB}/build_source_catalog.py" --validate-only \
  --catalog-out "${STAGE}/catalog_open.json" --overlay-out "${STAGE}/overlay_open.json" \
  --sidecar-dir "${STAGE}/sidecar_open"
"${PY}" "${LIB}/build_source_catalog.py" --validate-only \
  --catalog-out "${STAGE}/catalog_cosmos.json" --overlay-out "${STAGE}/overlay_cosmos.json" \
  --sidecar-dir "${STAGE}/sidecar_cosmos"

echo "[$(date --iso-8601=seconds)] ASSEMBLE_START"
"${PY}" "${LIB}/assemble_federated.py" \
  --catalog "${STAGE}/catalog_open.json" "${STAGE}/catalog_cosmos.json" \
  --overlay "${STAGE}/overlay_open.json" "${STAGE}/overlay_cosmos.json" \
  --sidecar "${STAGE}/sidecar_open" "${STAGE}/sidecar_cosmos" \
  --out-dir "${STAGE}/federated" --name "orientationfix_va2_25k_20260916" \
  --note "orientation-fixed + gripper-fixed target data; frame2/10Hz/action stride2"

[[ -s "${STAGE}/federated/catalog.json" ]] || { echo "federated catalog missing"; exit 25; }
[[ -s "${STAGE}/federated/success_overlay.json" ]] || { echo "federated overlay missing"; exit 25; }
[[ -s "${STAGE}/federated/sidecar/manifest.json" && -s "${STAGE}/federated/sidecar/COMPLETE" ]] || { echo "federated sidecar incomplete"; exit 25; }
echo "STEP5_DONE" > "${STAGE}/STEP5_DONE"
echo "[$(date --iso-8601=seconds)] STEP5_DONE federated=${STAGE}/federated"
