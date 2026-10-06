#!/usr/bin/env bash
# Reproducible RobotWin-50 reward-only R5 launcher.
# Usage: launch_robotwin50_rewardonly_r5_8gpu.sh pi06_timecost|success01_progress smoke|full
set -Eeuo pipefail

readonly EXPERIMENT="${1:?usage: $0 pi06_timecost|success01_progress smoke|full}"
readonly MODE="${2:?usage: $0 pi06_timecost|success01_progress smoke|full}"
readonly REPO_ROOT="${REPO_ROOT:-/path/to/home/wy/lingbot-va-original-rl}"
readonly CONFIG_ROOT="${CONFIG_ROOT:-${REPO_ROOT}/wan_va/rl/configs}"
readonly STAGE="${STAGE:-/path/to/mnt4/ops/robotwin50_rewardonly_r5_20260811}"
readonly SOURCE="${SOURCE:-/path/to/mnt4/ops/robotwin50_success_failure_language_qtranl_fromscratch_terminal_outcome_cost_20260806_rs1_r3_serialsource/source}"
readonly MANIFEST="${STAGE}/immutable_manifest.json"

case "${EXPERIMENT}:${MODE}" in
  pi06_timecost:smoke)
    readonly CONFIG_NAME="robotwin50_pi06_timecost_rewardonly_r5_smoke1_20260811.json"
    readonly MASTER_PORT="30851"
    ;;
  pi06_timecost:full)
    readonly CONFIG_NAME="robotwin50_pi06_timecost_rewardonly_r5_full50k_20260811.json"
    readonly MASTER_PORT="30853"
    ;;
  success01_progress:smoke)
    readonly CONFIG_NAME="robotwin50_success01_progress_rewardonly_r5_smoke1_20260811.json"
    readonly MASTER_PORT="30861"
    ;;
  success01_progress:full)
    readonly CONFIG_NAME="robotwin50_success01_progress_rewardonly_r5_full50k_20260811.json"
    readonly MASTER_PORT="30863"
    ;;
  *)
    echo "invalid experiment/mode: ${EXPERIMENT} ${MODE}" >&2
    exit 64
    ;;
esac

readonly CONFIG="${CONFIG_ROOT}/${CONFIG_NAME}"
readonly STAGED_CONFIG="${STAGE}/configs/${CONFIG_NAME}"
readonly OUTPUT="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_dir"])' "${CONFIG}")"
readonly LOG_ROOT="/path/to/mnt2/training_logs/robotwin50_rewardonly_r5_20260811/${EXPERIMENT}/${MODE}"

[[ -s "${CONFIG}" && -s "${STAGED_CONFIG}" && -s "${MANIFEST}" && -d "${SOURCE}" ]] || {
  echo "missing immutable binding component" >&2
  exit 2
}
cmp -s "${CONFIG}" "${STAGED_CONFIG}" || {
  echo "repository config differs from immutable staged config" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing output root: ${OUTPUT}" >&2
  exit 3
}
if ss -H -ltn "sport = :${MASTER_PORT}" | grep -q .; then
  echo "master port ${MASTER_PORT} is already in use" >&2
  exit 2
fi

python3 - "${MANIFEST}" "${CONFIG}" "${SOURCE}" "${EXPERIMENT}" "${MODE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, config_path, source_path = map(Path, sys.argv[1:4])
experiment, mode = sys.argv[4:6]
manifest = json.loads(manifest_path.read_text())
config = json.loads(config_path.read_text())
cell = manifest["experiments"][experiment]
expected = cell["configs"][mode]["sha256"]
actual = hashlib.sha256(config_path.read_bytes()).hexdigest()
assert actual == expected, (actual, expected)
assert config["reward_terms"] == cell["reward_terms"]
common = manifest["common_contract"]
assert config["critic_type"] == common["critic_type"]
assert config["rl_objective"] == common["rl_objective"]
assert config["gamma"] == common["gamma"]
assert config["seed"] == common["seed"]
assert config["batch_size"] == 2 and config["gradient_accumulation_steps"] == 1
assert config["resume_from"] is None and config["initialize_from"] is None
assert config["phase3_global_balanced_batching"] is True
assert config["phase3_federated_source_balanced_sampling"] is True
assert config["final_reward_normalization"] == "none"
assert config["num_steps"] == (1 if mode == "smoke" else common["steps"])
assert config["save_interval"] == (1 if mode == "smoke" else 2500)
for relative, expected_hash in manifest["source_files"].items():
    actual_hash = hashlib.sha256((source_path / relative).read_bytes()).hexdigest()
    assert actual_hash == expected_hash, (relative, actual_hash, expected_hash)
print("immutable preflight PASS", json.dumps({
    "experiment": experiment,
    "mode": mode,
    "steps": config["num_steps"],
    "global_batch": common["global_batch_size"],
    "reward_terms": config["reward_terms"],
}, sort_keys=True))
PY

mkdir -p "${LOG_ROOT}/train"
source /path/to/home/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
cd "${SOURCE}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${SOURCE}:/path/to/home/wy/vjepa2_source_restore${PYTHONPATH:+:${PYTHONPATH}}"
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1
exec /path/to/home/miniconda3/envs/lingbot/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=8 --master_port="${MASTER_PORT}" \
  --log_dir="${LOG_ROOT}/train" --tee 3 \
  -m wan_va.rl.train_critic_phase3 --config "${CONFIG}"
