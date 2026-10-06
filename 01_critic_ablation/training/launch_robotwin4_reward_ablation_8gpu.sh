#!/usr/bin/env bash
# Shared 8-GPU launcher for the RobotWin-4 language-conditioned QTran-L
# reward-ablation runs.
#
# This launcher only starts the already-declared offline critic experiment.
# It does not build or rewrite caches, change reward terms, or resume an
# existing output. The config owns the scientific settings; this script
# supplies the runtime and performs a bounded preflight.
set -Eeuo pipefail

readonly CONFIG_INPUT="${1:?usage: $0 CONFIG.json VARIANT}"
readonly VARIANT="${2:?usage: $0 CONFIG.json VARIANT}"
readonly REPO_ROOT="${REPO_ROOT:-/path/to/lingbot-va-original-rl}"
readonly PYTHON_BIN="${PYTHON_BIN:-/path/to/lingbot-python}"
readonly NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
readonly CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
readonly MASTER_PORT="${MASTER_PORT:-29691}"
readonly VJEPA_ROOT="${VJEPA_ROOT:-/path/to/lingbot-workspace/vjepa2_source_restore}"
readonly LOG_ROOT="${LOG_ROOT:-/path/to/shared-storage/datasets/reward_ablation/logs/${VARIANT}}"

if [[ "${CONFIG_INPUT}" = /* ]]; then
  readonly CONFIG="${CONFIG_INPUT}"
else
  readonly CONFIG="$(cd "$(dirname "${CONFIG_INPUT}")" && pwd)/$(basename "${CONFIG_INPUT}")"
fi

[[ -d "${REPO_ROOT}" ]] || {
  echo "missing canonical repository: ${REPO_ROOT}" >&2
  exit 2
}
[[ -x "${PYTHON_BIN}" ]] || {
  echo "missing Python interpreter: ${PYTHON_BIN}" >&2
  exit 2
}
[[ -s "${CONFIG}" ]] || {
  echo "missing config: ${CONFIG}" >&2
  exit 2
}

if [[ "${NPROC_PER_NODE}" != "8" ]]; then
  echo "this launcher is fixed to 8 GPUs; set NPROC_PER_NODE=8" >&2
  exit 2
fi
IFS=',' read -r -a _gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_gpu_ids[@]} != NPROC_PER_NODE )); then
  echo "CUDA_VISIBLE_DEVICES must contain exactly 8 entries: ${CUDA_VISIBLE_DEVICES}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${VJEPA_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

readonly OUTPUT_DIR="$(${PYTHON_BIN} - "${CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

from wan_va.rl.config import CriticTrainingConfig

config_path = Path(sys.argv[1])
cfg = CriticTrainingConfig.from_json(config_path)

if cfg.critic_type != "twin_language_query_transformer_v1":
    raise SystemExit(f"expected language-conditioned critic, got {cfg.critic_type!r}")
if int(cfg.critic_text_embed_dim) != 4096:
    raise SystemExit(f"expected 4096-dim language sidecar, got {cfg.critic_text_embed_dim}")
if cfg.phase3_cache_format != "federated_canonical_episode":
    raise SystemExit(f"unexpected cache format: {cfg.phase3_cache_format!r}")
if not cfg.phase3_require_precomputed_cache:
    raise SystemExit("precomputed cache is required for this launcher")
if cfg.phase3_loader_mode != "episode":
    raise SystemExit(f"expected episode loader, got {cfg.phase3_loader_mode!r}")
if cfg.phase3_cache_residency != "bounded_group_prefetching":
    raise SystemExit(
        "expected bounded_group_prefetching; refusing to start with a "
        f"different loader: {cfg.phase3_cache_residency!r}"
    )
if int(cfg.phase3_episodes_per_batch) != 3:
    raise SystemExit(f"expected episodes_per_batch=3, got {cfg.phase3_episodes_per_batch}")
if int(cfg.batch_size) != 2:
    raise SystemExit(f"expected per-GPU batch_size=2, got {cfg.batch_size}")
if int(cfg.num_steps) != 15000:
    raise SystemExit(f"expected num_steps=15000, got {cfg.num_steps}")
if int(cfg.phase3_cache_group_size) != 5:
    raise SystemExit(f"expected cache_group_size=5, got {cfg.phase3_cache_group_size}")
if int(cfg.phase3_cache_group_prefetch_depth) != 2:
    raise SystemExit(
        "expected cache_group_prefetch_depth=2, got "
        f"{cfg.phase3_cache_group_prefetch_depth}"
    )
if not cfg.phase3_preserve_episode_group_order:
    raise SystemExit("bounded group prefetching requires preserved episode-group order")
if not cfg.phase3_global_balanced_batching or not cfg.phase3_drop_last:
    raise SystemExit("expected global balanced batching with drop_last=true")
if cfg.phase3_federated_source_balanced_sampling:
    raise SystemExit("source-balanced sampling must remain disabled for group locality")
if not cfg.phase3_cuda_prefetch or int(cfg.phase3_cuda_prefetch_depth) != 2:
    raise SystemExit("expected CUDA prefetch depth 2 for the bounded loader")
if cfg.phase3_federated_cache_catalog is None:
    raise SystemExit("missing federated cache catalog")
if cfg.phase3_episode_success_overlay is None:
    raise SystemExit("missing required episode-success overlay")
if cfg.phase3_language_sidecar_catalog is None:
    raise SystemExit("missing required language sidecar catalog")

paths = {
    "catalog": Path(cfg.phase3_federated_cache_catalog),
    "overlay": Path(cfg.phase3_episode_success_overlay),
    "language_sidecar": Path(cfg.phase3_language_sidecar_catalog),
    "transformer": Path(cfg.transformer_path),
}
for name, path in paths.items():
    if not path.exists():
        raise SystemExit(f"missing {name}: {path}")

catalog = json.loads(paths["catalog"].read_text())
for source in catalog.get("sources", []):
    cache_dir = Path(source["cache_dir"])
    if not cache_dir.is_dir():
        raise SystemExit(f"missing cache directory for {source.get('task_name')}: {cache_dir}")
    if not (cache_dir / "manifest.json").is_file():
        raise SystemExit(f"missing cache manifest: {cache_dir / 'manifest.json'}")

print(json.dumps({
    "config": str(config_path),
    "output_dir": str(cfg.output_dir),
    "variant": config_path.stem,
    "reward_terms": cfg.reward_terms,
    "critic_type": cfg.critic_type,
    "critic_feature_layers": list(cfg.critic_feature_layers),
    "batch_size_per_gpu": cfg.batch_size,
    "global_batch_size": cfg.batch_size * 8,
    "num_steps": cfg.num_steps,
    "loader_mode": cfg.phase3_loader_mode,
    "cache_residency": cfg.phase3_cache_residency,
    "episodes_per_batch": cfg.phase3_episodes_per_batch,
    "cache_group_size": cfg.phase3_cache_group_size,
    "cache_group_prefetch_depth": cfg.phase3_cache_group_prefetch_depth,
    "cache_max_resident_gib": cfg.phase3_cache_max_resident_gib,
    "cuda_prefetch_depth": cfg.phase3_cuda_prefetch_depth,
}, sort_keys=True), file=sys.stderr)
print(cfg.output_dir)
PY
)"

if [[ -e "${OUTPUT_DIR}" && "${ALLOW_EXISTING_OUTPUT:-0}" != "1" ]]; then
  echo "refusing existing output directory: ${OUTPUT_DIR}" >&2
  echo "set ALLOW_EXISTING_OUTPUT=1 only when deliberately managing that run" >&2
  exit 3
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  readonly GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
  if (( GPU_COUNT < NPROC_PER_NODE )); then
    echo "only ${GPU_COUNT} GPUs are visible; 8 are required" >&2
    exit 2
  fi
else
  echo "nvidia-smi is unavailable; cannot verify an 8-GPU host" >&2
  exit 2
fi

if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :${MASTER_PORT}" | grep -q .; then
  echo "master port ${MASTER_PORT} is already in use" >&2
  exit 2
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "preflight only; training was not started"
  exit 0
fi

mkdir -p "${LOG_ROOT}/torchrun"
source /path/to/conda/etc/profile.d/conda.sh
conda activate lingbot
export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1

echo "starting ${VARIANT} on ${NPROC_PER_NODE} GPUs"
echo "config=${CONFIG}"
echo "output=${OUTPUT_DIR}"
echo "logs=${LOG_ROOT}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  --log_dir="${LOG_ROOT}/torchrun" \
  --tee 3 \
  -m wan_va.rl.train_critic_phase3 \
  --config "${CONFIG}"
