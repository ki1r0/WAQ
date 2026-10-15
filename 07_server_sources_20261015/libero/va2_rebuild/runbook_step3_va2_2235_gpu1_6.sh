#!/usr/bin/env bash
set -Eeuo pipefail

OPS="/mnt/zengqixin/ops/cosmos_policy_analysis_20260906"
STAGE="${OPS}/va2_rebuild_20260907"
REPO="/home/zengqixin/wy/lingbot-va-original-rl"
PY="/home/zengqixin/miniconda3/envs/lingbot/bin/python"
DEVICES="1,2,3,4,5,6"
NPROC=6
MODEL="/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/export/from_va2_libero10_cosmos_20260907_step25000_diffusers"
OVERLAY="${STAGE}"
OUT_OPEN="/mnt4/zengqixin/dataset/libero_10_original_500_20260729/lerobot_v21_10hz_vae128/inference_state_cache/va2_step25000_server_prefix_8layer_20260907"
OUT_COSMOS="/mnt/zengqixin/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000/inference_state_cache/va2_step25000_server_prefix_8layer_20260907"
OPEN_CONFIG="${STAGE}/va2_open_500_cache_config.json"
COSMOS_CONFIG="${STAGE}/va2_cosmos_500_cache_config.json"
LOG_ROOT="${STAGE}/logs_gpu1_6"

source /home/zengqixin/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd "${REPO}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${STAGE}:${REPO}:/home/zengqixin/wy/vjepa2_source_restore${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/step3_launcher.log") 2>&1

[[ -s "${MODEL}/transformer/diffusion_pytorch_model.safetensors" ]] || { echo "missing VA2 transformer"; exit 2; }
[[ -s "${OPEN_CONFIG}" && -s "${COSMOS_CONFIG}" ]] || { echo "missing VA2 cache config"; exit 2; }
[[ ! -e "${OUT_OPEN}" ]] || { echo "refusing pre-existing open cache: ${OUT_OPEN}"; exit 3; }
[[ ! -e "${OUT_COSMOS}" ]] || { echo "refusing pre-existing cosmos cache: ${OUT_COSMOS}"; exit 3; }

run_cache() {
  local name="$1" config="$2" output="$3" port="$4"
  local log_dir="${LOG_ROOT}/${name}"
  mkdir -p "${log_dir}"
  echo "[$(date --iso-8601=seconds)] CACHE_START name=${name} devices=${DEVICES} output=${output}"
  "${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" \
    --master_port="${port}" --log_dir="${log_dir}" --tee 3 \
    -m wan_va.rl.precompute_phase3_cache \
    --config "${config}" --output-dir "${output}" \
    --cache-feature-layers "0,4,8,12,16,20,24,28" \
    --batch-size 1 --num-workers 0 --dataset-init-workers 8 \
    --log-interval 1 --progress --standalone-ranks
  [[ -s "${output}/manifest.json" ]] || { echo "cache manifest missing: ${output}"; exit 4; }
  echo "[$(date --iso-8601=seconds)] CACHE_DONE name=${name}"
}

run_cache open "${OPEN_CONFIG}" "${OUT_OPEN}" 31421
run_cache cosmos "${COSMOS_CONFIG}" "${OUT_COSMOS}" 31422

echo "[$(date --iso-8601=seconds)] METADATA_START"
"${PY}" "${STAGE}/build_va2_step3_catalogs.py" \
  --kind open \
  --cache-dir "${OUT_OPEN}" \
  --source-root /mnt4/zengqixin/dataset/libero_10_original_500_20260729/lerobot_v21_10hz_vae128 \
  --catalog-out "${STAGE}/catalog_va2_open_500_20260907.json" \
  --overlay-out "${STAGE}/success_overlay_va2_open_500_20260907.json"
"${PY}" "${STAGE}/build_va2_step3_catalogs.py" \
  --kind cosmos \
  --cache-dir "${OUT_COSMOS}" \
  --source-root /mnt/zengqixin/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000 \
  --catalog-out "${STAGE}/catalog_va2_cosmos_500_20260907.json" \
  --overlay-out "${STAGE}/success_overlay_va2_cosmos_500_20260907.json"

echo "[$(date --iso-8601=seconds)] SIDECAR_START"
[[ ! -e "${STAGE}/sidecar_va2_open_500" ]] || { echo "refusing pre-existing sidecar: ${STAGE}/sidecar_va2_open_500"; exit 5; }
[[ ! -e "${STAGE}/sidecar_va2_cosmos_500" ]] || { echo "refusing pre-existing sidecar: ${STAGE}/sidecar_va2_cosmos_500"; exit 5; }
"${PY}" /mnt/zengqixin/ops/libero_pi06_qema_20260827/build_single_source_language_sidecar.py \
  --catalog "${STAGE}/catalog_va2_open_500_20260907.json" \
  --source-root /mnt4/zengqixin/dataset/libero_10_original_500_20260729/lerobot_v21_10hz_vae128 \
  --camera-key observation.images.wrist_image \
  --model-root "${MODEL}" \
  --output-dir "${STAGE}/sidecar_va2_open_500"
"${PY}" /mnt/zengqixin/ops/libero_pi06_qema_20260827/build_single_source_language_sidecar.py \
  --catalog "${STAGE}/catalog_va2_cosmos_500_20260907.json" \
  --source-root /mnt/zengqixin/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-all2000 \
  --camera-key observation.images.front \
  --model-root "${MODEL}" \
  --output-dir "${STAGE}/sidecar_va2_cosmos_500"
echo "[$(date --iso-8601=seconds)] SIDECAR_DONE"

"${PY}" "${STAGE}/build_va2_step3_catalogs.py" --validate-only \
  --catalog-out "${STAGE}/catalog_va2_open_500_20260907.json" \
  --overlay-out "${STAGE}/success_overlay_va2_open_500_20260907.json" \
  --sidecar-dir "${STAGE}/sidecar_va2_open_500"
"${PY}" "${STAGE}/build_va2_step3_catalogs.py" --validate-only \
  --catalog-out "${STAGE}/catalog_va2_cosmos_500_20260907.json" \
  --overlay-out "${STAGE}/success_overlay_va2_cosmos_500_20260907.json" \
  --sidecar-dir "${STAGE}/sidecar_va2_cosmos_500"
echo "[$(date --iso-8601=seconds)] STEP3_DONE"
