#!/usr/bin/env bash
# Track A / step 5: VA2-GF feature extraction + federated inputs
# Extract phase-3 features for BOTH sources, build single-source catalogs and
# language sidecars, then assemble the federated inputs the critic consumes.
set -Eeuo pipefail

STAGE="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2"
LIB="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/lib"
PY="/path/to/lingbot-env/bin/python"
REPO="/path/to/lingbot-va"
MODEL="/path/to/storage/experiments/robotwin_clean_sft_20260903/export/from_va2_gripperfix_20260909_step25000_diffusers"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="$(awk -F, '{print NF}' <<<"${DEVICES}")"
OUT_OPEN="/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/lerobot_v21_10hz_vae128/inference_state_cache/va2gf_step25000_server_prefix_8layer_20260909"
OUT_COSMOS="/path/to/storage/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000/inference_state_cache/va2gf_step25000_server_prefix_8layer_20260909"
OPEN_ROOT="/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/lerobot_v21_10hz_vae128"
COSMOS_ROOT="/path/to/storage/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000"
LOG_ROOT="${STAGE}/logs"

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

# ---- preflight -------------------------------------------------------------
[[ -s "/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/action_q01_q99.json" ]] || { echo "missing corrected normalization stats"; exit 2; }
"${PY}" - <<'PYCHK'
import json
s = json.load(open("/path/to/storage/datasets/libero_10_original_500_gripperfix_20260909/action_q01_q99.json"))
assert s["q01"][6] == -1.0 and s["q99"][6] == 1.0, s
print("PREFLIGHT stats gripper identity OK")
PYCHK
[[ -s "${MODEL}/transformer/diffusion_pytorch_model.safetensors" ]] || { echo "missing model: ${MODEL}"; exit 2; }
[[ -d "${OPEN_ROOT}/data" ]] || { echo "missing patched open dataset"; exit 2; }
[[ ! -e "${OUT_OPEN}" ]] || { echo "refusing pre-existing cache: ${OUT_OPEN}"; exit 3; }
[[ ! -e "${OUT_COSMOS}" ]] || { echo "refusing pre-existing cache: ${OUT_COSMOS}"; exit 3; }

run_cache() {
  local name="$1" config="$2" output="$3" port="$4"
  mkdir -p "${LOG_ROOT}/${name}"
  echo "[$(date --iso-8601=seconds)] CACHE_START ${name} devices=${DEVICES}"
  "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    --master_port="${port}" --log_dir="${LOG_ROOT}/${name}" --tee 3 \
    -m wan_va.rl.precompute_phase3_cache \
    --config "${config}" --output-dir "${output}" \
    --cache-feature-layers "0,4,8,12,16,20,24,28" \
    --batch-size 1 --num-workers 0 --dataset-init-workers 8 \
    --log-interval 1 --progress --standalone-ranks
  [[ -s "${output}/manifest.json" ]] || { echo "cache manifest missing"; exit 4; }
  echo "[$(date --iso-8601=seconds)] CACHE_DONE ${name}"
}

run_cache open   "${STAGE}/cache_open.json"   "${OUT_OPEN}"   31421
run_cache cosmos "${STAGE}/cache_cosmos.json" "${OUT_COSMOS}" 31422

echo "[$(date --iso-8601=seconds)] CATALOG_START"
"${PY}" "${LIB}/build_source_catalog.py" \
  --cache-dir "${OUT_OPEN}" --source-root "${OPEN_ROOT}" \
  --base-config "libero_gf_va2_open_500_cache_20260909" --task-name official_libero_long_10_tasks \
  --source-name official_libero --contract-name "va2gf_1000_20260909_open_500" \
  --note "gripper-fixed open LIBERO-10 500" \
  --catalog-out "${STAGE}/catalog_open.json" --overlay-out "${STAGE}/overlay_open.json"
"${PY}" "${LIB}/build_source_catalog.py" \
  --cache-dir "${OUT_COSMOS}" --source-root "${COSMOS_ROOT}" \
  --base-config "libero_gf_va2_cosmos_500_cache_20260909" --task-name cosmos_policy_libero_10 \
  --source-name cosmos_policy --contract-name "va2gf_1000_20260909_cosmos_500" \
  --max-episodes 500 --note "Cosmos-Policy episodes 0-499 (124 real failures)" \
  --catalog-out "${STAGE}/catalog_cosmos.json" --overlay-out "${STAGE}/overlay_cosmos.json"

echo "[$(date --iso-8601=seconds)] SIDECAR_START"
[[ ! -e "${STAGE}/sidecar_open" && ! -e "${STAGE}/sidecar_cosmos" ]] || { echo "refusing pre-existing sidecar"; exit 5; }
"${PY}" /path/to/storage/ops/libero_pi06_qema_20260827/build_single_source_language_sidecar.py \
  --catalog "${STAGE}/catalog_open.json" --source-root "${OPEN_ROOT}" \
  --camera-key observation.images.wrist_image --model-root "${MODEL}" \
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
  --out-dir "${STAGE}/federated" --name "va2gf_1000_20260909" \
  --note "gripper-fixed data + identity gripper normalization"

echo "[$(date --iso-8601=seconds)] STEP5_DONE"
