#!/usr/bin/env bash
# Track A / step 6: retrain Critic2 on gripper-fixed VA2 features
set -Eeuo pipefail
readonly ROOT="/path/to/lingbot-va"
readonly PY="/path/to/lingbot-env/bin/python"
readonly STAGE="/path/to/storage/ops/cosmos_policy_analysis_20260906/gripperfix_20260909/A_line2"
readonly OUT="/path/to/storage/checkpoints/rl/libero10_cosmos_joint_critic2_gripperfix_20260909"
readonly LOG_ROOT="/path/to/storage/training_logs/libero10_cosmos_joint_critic2_gripperfix_20260909"
source /path/to/miniconda/etc/profile.d/conda.sh
conda activate lingbot
cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")"
export TOKENIZERS_PARALLELISM=false PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${ROOT}:/path/to/vjepa2-source${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${LOG_ROOT}"
exec > >(tee -a "${LOG_ROOT}/launcher.log") 2>&1

for f in "${STAGE}/config_smoke.json" "${STAGE}/config_full.json" \
         "${STAGE}/federated/catalog.json" "${STAGE}/federated/success_overlay.json" \
         "${STAGE}/federated/sidecar/manifest.json"; do
  [[ -s "$f" ]] || { echo "missing required input: $f" >&2; exit 2; }
done
[[ ! -e "${OUT}" && ! -e "${OUT}_smoke" ]] || { echo "refusing pre-existing output roots" >&2; exit 3; }

echo "[$(date --iso-8601=seconds)] SMOKE 2 steps on ${NPROC} ranks"
"$PY" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port=29790 \
  --log_dir="${LOG_ROOT}/smoke" --tee 3 -m wan_va.rl.train_critic_phase3 --config "${STAGE}/config_smoke.json"
"$PY" - "${OUT}_smoke" <<'PYCHK'
import json, math, sys
from pathlib import Path
root = Path(sys.argv[1]); cps = sorted((root / 'checkpoints').glob('checkpoint_*'))
assert cps, f'no checkpoints under {root}'
cp = cps[-1]
for name in ('training_state.pt', 'config.json', 'manifest.json'):
    assert (cp / name).is_file() and (cp / name).stat().st_size > 0, (cp, name)
cfg = json.loads((cp / 'config.json').read_text())
assert cfg['critic_type'] == 'twin_language_query_transformer_v1', cfg['critic_type']
loss = json.loads((root / 'checkpoints' / 'loss.jsonl').read_text().splitlines()[-1])
for k in ('total_loss', 'q_loss', 'value_loss'):
    assert math.isfinite(float(loss[k])), (k, loss[k])
print('SMOKE_PASS', cp, {k: loss[k] for k in ('total_loss', 'q_loss', 'value_loss')})
PYCHK

echo "[$(date --iso-8601=seconds)] FULL 25000 steps"
exec "$PY" -m torch.distributed.run --standalone --nproc_per_node="${NPROC}" --master_port=29791 \
  --log_dir="${LOG_ROOT}/train" --tee 3 -m wan_va.rl.train_critic_phase3 --config "${STAGE}/config_full.json"
