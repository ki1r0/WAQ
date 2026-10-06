#!/usr/bin/env bash
# Isolated Track-A step 6: Critic2 smoke, then full 25k training.
set -Eeuo pipefail

STAGE="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2/orientationfix_va2_25k_20260916"
ROOT="/path/to/lingbot-va"
PY="/path/to/lingbot-env/bin/python"
OUT="/path/to/storage/checkpoints/rl/libero10_cosmos_joint_critic2_orientationfix_gripperfix_20260916"
LOG_ROOT="${STAGE}/logs/step6"
DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="$(awk -F, '{print NF}' <<<"${DEVICES}")"

source /path/to/miniconda/etc/profile.d/conda.sh
conda activate lingbot
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${DEVICES}"
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${STAGE}:${ROOT}:/path/to/vjepa2-source${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1

echo "[$(date --iso-8601=seconds)] STEP6_START devices=${DEVICES} nproc=${NPROC}"
[[ -s "${STAGE}/STEP5_DONE" ]] || { echo "step5 marker missing"; exit 30; }
for f in "${STAGE}/config_smoke.json" "${STAGE}/config_full.json" \
         "${STAGE}/federated/catalog.json" "${STAGE}/federated/success_overlay.json" \
         "${STAGE}/federated/sidecar/manifest.json" "${STAGE}/federated/sidecar/COMPLETE"; do
  [[ -s "${f}" ]] || { echo "missing required step6 input: ${f}"; exit 31; }
done
[[ ! -e "${OUT}" && ! -e "${OUT}_smoke" ]] || { echo "refusing pre-existing critic outputs"; exit 32; }

echo "[$(date --iso-8601=seconds)] SMOKE_START 2_steps all_8_gpus"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port=29792 \
  --log_dir="${LOG_ROOT}/smoke_torchrun" --tee 3 \
  -m wan_va.rl.train_critic_phase3 --config "${STAGE}/config_smoke.json"
"${PY}" - "${OUT}_smoke" <<'PYCHK'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1])
cps = sorted((root / "checkpoints").glob("checkpoint_*") )
assert cps, f"no checkpoints under {root}"
cp = cps[-1]
for name in ("training_state.pt", "config.json", "manifest.json"):
    assert (cp / name).is_file() and (cp / name).stat().st_size > 0, (cp, name)
cfg = json.loads((cp / "config.json").read_text())
assert cfg["critic_type"] == "twin_language_query_transformer_v1", cfg["critic_type"]
loss_path = root / "checkpoints" / "loss.jsonl"
assert loss_path.is_file() and loss_path.stat().st_size > 0, loss_path
loss = json.loads(loss_path.read_text().splitlines()[-1])
for key in ("total_loss", "q_loss", "value_loss"):
    assert math.isfinite(float(loss[key])), (key, loss[key])
print("SMOKE_PASS", cp, {key: loss[key] for key in ("total_loss", "q_loss", "value_loss")})
PYCHK
echo "[$(date --iso-8601=seconds)] SMOKE_PASS output=${OUT}_smoke"

echo "[$(date --iso-8601=seconds)] FULL_START steps=25000 save_interval=2500 all_8_gpus"
"${PY}" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port=29793 \
  --log_dir="${LOG_ROOT}/full_torchrun" --tee 3 \
  -m wan_va.rl.train_critic_phase3 --config "${STAGE}/config_full.json"
"${PY}" - "${OUT}" <<'PYFULL'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
cp = root / "checkpoints" / "checkpoint_00025000"
assert cp.is_dir(), cp
for name in ("training_state.pt", "config.json", "manifest.json"):
    assert (cp / name).is_file() and (cp / name).stat().st_size > 0, (cp, name)
cfg = json.loads((cp / "config.json").read_text())
assert cfg["num_steps"] == 25000 and cfg["save_interval"] == 2500, cfg
print("FULL_PASS", cp)
PYFULL
echo "CRITIC_COMPLETE" > "${STAGE}/CRITIC_COMPLETE"
echo "[$(date --iso-8601=seconds)] STEP6_DONE output=${OUT}"
