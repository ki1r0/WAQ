#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_HOST="worker-c"
readonly FAMILY="success01_goalprogress_overlayfixed"
readonly STAMP="20260814"
readonly EPISODES_PER_CHECKPOINT=8
readonly GPUS_CSV="0,1,2,3,4,5,6,7"
readonly SOURCE_ROOT="/path/to/shared-storage/checkpoints/rl/robotwin50_success01_goalprogress_overlayfixed_scalar_iql_50k_20260813/checkpoints"
readonly ADAPTER_ROOT_LOW="/path/to/shared-storage/robotwin_eval_adapters/STAGE-CKPT-SWEEP-PCB-DEMO-CLEAN-20260813T172044/${FAMILY}"
readonly ADAPTER_ROOT_HIGH="/path/to/shared-storage/robotwin_eval_adapters/STAGE-CKPT-SWEEP-PCB-DEMO-CLEAN-20260814T003300-CONT30K50K/${FAMILY}"
readonly RESULT_ROOT="/path/to/shared-storage/robotwin_eval_results/GOALPROGRESS-OVERLAYFIXED-HANGINGMUG-CKPT-SWEEP-5K50K-8EP-METRICS-ONLY-DQ-${STAMP}-R1"
readonly REPLAY_ROOT="/path/to/shared-storage/robotwin_eval_protocols/hanging_mug_replay100_bintd20260812_rebalanced8_20260813"
readonly EVAL_LINGBOT_ROOT="/path/to/lingbot-va-original-rl-r6-eval-source-20260811"
readonly EVAL_ROBOTWIN_ROOT="/path/to/robotwin-r6-eval-runtime-20260812"
readonly COLLECTOR="/path/to/lingbot-va-original-rl/scripts/collect_robotwin_qguided_metrics_only_dynamic_queue_20260813.sh"
readonly EVAL_Q_SERVER_SCRIPT="/path/to/lingbot-va-original-rl/scripts/wan_va_server_q_guiding_metrics_only_old_language_20260813.py"
readonly STATUS_FILE="${RESULT_ROOT}/sweep_status.tsv"
readonly LOCK_FILE="/tmp/user-${FAMILY}-hanging-mug-sweep-${STAMP}.lock"
readonly -a STEPS=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)

fail() {
    echo "REFUSING TO START: $*" >&2
    exit 2
}

adapter_for_step() {
    local step="$1"
    local step8
    step8=$(printf '%08d' "${step}")
    if (( step <= 25000 )); then
        printf '%s/checkpoint_%s\n' "${ADAPTER_ROOT_LOW}" "${step8}"
    else
        printf '%s/checkpoint_%s\n' "${ADAPTER_ROOT_HIGH}" "${step8}"
    fi
}

gpu_process_count() {
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | wc -l
}

port_is_free() {
    local port="$1"
    ! ss -H -ltn "sport = :${port}" 2>/dev/null | grep -q .
}

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || fail "expected ${EXPECTED_HOST}, got $(hostname)"
[[ ! -e "${RESULT_ROOT}" ]] || fail "result root already exists: ${RESULT_ROOT}"
[[ -x "${COLLECTOR}" ]] || fail "collector is unavailable: ${COLLECTOR}"
[[ -f "${EVAL_Q_SERVER_SCRIPT}" ]] || fail "Q server script is unavailable: ${EVAL_Q_SERVER_SCRIPT}"
[[ -d "${EVAL_LINGBOT_ROOT}" && -d "${EVAL_ROBOTWIN_ROOT}" ]] || fail "evaluation runtimes are unavailable"
[[ -d "${REPLAY_ROOT}" ]] || fail "replay protocol is unavailable: ${REPLAY_ROOT}"

for path in "${SOURCE_ROOT}" "${ADAPTER_ROOT_LOW}" "${ADAPTER_ROOT_HIGH}" "${RESULT_ROOT}" \
            "${REPLAY_ROOT}" "${EVAL_LINGBOT_ROOT}" "${EVAL_ROBOTWIN_ROOT}" "${COLLECTOR}" \
            "${EVAL_Q_SERVER_SCRIPT}"; do
    [[ "${path}" != /mnt4* ]] || fail "path uses unavailable /mnt4: ${path}"
done

for rank in 0 1 2 3 4 5 6 7; do
    [[ -s "${REPLAY_ROOT}/worker-${rank}.jsonl" ]] || fail "missing replay manifest worker-${rank}"
done

/path/to/robotwin-python - "${REPLAY_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("worker-*.jsonl")):
    rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
assert len(rows) == 100, len(rows)
assert len({int(row["seed"]) for row in rows}) == 100
for row in rows:
    assert row.get("instruction") and row.get("episode_info") and "seed" in row, row
print("replay protocol PASS rows=100 unique_seeds=100")
PY

expected_config_sha="bfa903d9c3ab4b9128aac78b2f599a686bd09635e1d3b705b9e7e092583ecb62"
expected_manifest_sha="d5e7029477c7f43a5af43bd387a9714c6e0f34c0dd3c2e3096edbdd821edccc0"
for step in "${STEPS[@]}"; do
    step8=$(printf '%08d' "${step}")
    source_checkpoint="${SOURCE_ROOT}/checkpoint_${step8}"
    adapter=$(adapter_for_step "${step}")
    for required in training_state.pt config.json manifest.json; do
        [[ -s "${source_checkpoint}/${required}" ]] || fail "source checkpoint ${step} lacks ${required}"
        [[ -s "${adapter}/${required}" ]] || fail "adapter ${step} lacks ${required}"
    done
    [[ "$(readlink -f "${adapter}/training_state.pt")" == "${source_checkpoint}/training_state.pt" ]] || \
        fail "adapter state does not point at strict checkpoint ${step}"
    [[ "$(sha256sum "${adapter}/config.json" | awk '{print $1}')" == "${expected_config_sha}" ]] || \
        fail "adapter config hash mismatch at ${step}"
    [[ "$(sha256sum "${adapter}/manifest.json" | awk '{print $1}')" == "${expected_manifest_sha}" ]] || \
        fail "adapter manifest hash mismatch at ${step}"
    ! grep -Iq '/mnt4' "${adapter}/config.json" || fail "adapter config references /mnt4 at ${step}"
done

exec 9>"${LOCK_FILE}"
flock -n 9 || fail "sweep lock is already held: ${LOCK_FILE}"
[[ "$(gpu_process_count)" == 0 ]] || fail "GPU processes already exist on ${EXPECTED_HOST}"

for index in "${!STEPS[@]}"; do
    app_base=$((59400 + index * 20))
    master_base=$((60400 + index * 20))
    for offset in 0 1 2 3 4 5 6 7; do
        port_is_free $((app_base + offset)) || fail "application port $((app_base + offset)) is occupied"
        port_is_free $((master_base + offset)) || fail "master port $((master_base + offset)) is occupied"
    done
done

mkdir -p "${RESULT_ROOT}"
printf 'timestamp\tevent\tcheckpoint\tdetail\n' >"${STATUS_FILE}"
cat >"${RESULT_ROOT}/sweep_protocol.json" <<EOF
{
  "family": "${FAMILY}",
  "task": "hanging_mug",
  "task_config": "demo_clean",
  "checkpoints": [5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
  "episodes_per_checkpoint": ${EPISODES_PER_CHECKPOINT},
  "replay_manifest_root": "${REPLAY_ROOT}",
  "replay_selection": "first 8 rows of the immutable 100-seed protocol; identical for every checkpoint",
  "rendering": false,
  "dynamic_queue": true,
  "q_guidance_weight_scheduler": "fixed",
  "q_guidance_scale": 0.1,
  "q_guidance_beta": 2.0,
  "q_guidance_interval": 5,
  "q_objective": "min",
  "q_grad_normalize": true,
  "ranking_contract": "rank by 8-episode success rate; ties retain the same rank; Q statistics are descriptive and never break ties",
  "comparability_warning": "not directly comparable to the earlier place_can_basket rendered/static-worker sweep"
}
EOF

current_step="preflight"
on_error() {
    local rc=$?
    printf '%s\tfailed\t%s\texit=%s line=%s\n' "$(date -Is)" "${current_step}" "${rc}" "${BASH_LINENO[0]}" >>"${STATUS_FILE}" || true
    exit "${rc}"
}
trap on_error ERR

for index in "${!STEPS[@]}"; do
    step="${STEPS[$index]}"
    current_step="${step}"
    step8=$(printf '%08d' "${step}")
    adapter=$(adapter_for_step "${step}")
    step_output="${RESULT_ROOT}/step_${step8}"
    app_base=$((59400 + index * 20))
    master_base=$((60400 + index * 20))

    [[ ! -e "${step_output}" ]] || fail "step output already exists: ${step_output}"
    [[ "$(gpu_process_count)" == 0 ]] || fail "GPU process appeared before checkpoint ${step}"
    printf '%s\tstarted\t%s\t%s\n' "$(date -Is)" "${step}" "${step_output}" >>"${STATUS_FILE}"

    cd "${EVAL_ROBOTWIN_ROOT}"
    LINGBOT_ROOT="${EVAL_LINGBOT_ROOT}" \
    ROBOTWIN_ROOT="${EVAL_ROBOTWIN_ROOT}" \
    LINGBOT_PYTHON=/path/to/lingbot-python \
    ROBOTWIN_PYTHON=/path/to/robotwin-python \
    MODEL_PATH=/path/to/model-weights/lingbot-va-posttrain-robotwin \
    SERVER_CONFIG=robotwin_train \
    EVAL_CONFIG=policy/ACT/deploy_policy.yml \
    TASK=hanging_mug TASK_CONFIG=demo_clean \
    OUTPUT_ROOT="${step_output}" \
    RUN_ID="${FAMILY}-hanging-mug-step${step8}-8ep-${STAMP}" \
    Q_CHECKPOINT="${adapter}" \
    Q_SERVER_SCRIPT="${EVAL_Q_SERVER_SCRIPT}" \
    Q_CLIENT_MODULE=evaluation.robotwin.eval_polict_client_openpi_novideo_dynamic_queue_20260813 \
    QGF_REPLAY_MANIFEST_ROOT="${REPLAY_ROOT}" \
    QGF_COMPARE_BASELINE=false QGF_RUN_MODE=guided \
    Q_GUIDANCE_WEIGHT_SCHEDULER=fixed Q_GUIDANCE_SCALE=0.1 Q_GUIDANCE_BETA=2.0 \
    Q_GUIDANCE_START_STEP=0 Q_GUIDANCE_END_STEP=-1 Q_GUIDANCE_INTERVAL=5 \
    Q_OBJECTIVE=min Q_GRAD_CLIP=0.0 Q_GRAD_NORMALIZE=true Q_HISTORY_CHUNKS=10 \
    Q_LOG_DENOISING_STEPS=false Q_PROGRESS_GUIDANCE=false \
    RECORD_PREDICTION_REWARD=false DYNAMIC_QUEUE_RESUME=false \
    APP_PORT_BASE="${app_base}" MASTER_PORT_BASE="${master_base}" \
    SERVER_TIMEOUT_SECONDS=900 PROGRESS_INTERVAL_SECONDS=20 \
    bash "${COLLECTOR}" "${GPUS_CSV}" "${EPISODES_PER_CHECKPOINT}"

    /path/to/robotwin-python - "${step_output}" "${EPISODES_PER_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = int(sys.argv[2])
summary = json.loads((root / "metrics" / "summary.json").read_text())
episodes = [json.loads(line) for line in (root / "metrics" / "episodes.jsonl").read_text().splitlines() if line.strip()]
queue = json.loads((root / "queue" / "index.json").read_text())
assert summary["target_episodes"] == target, summary
assert summary["completed"] == target, summary
assert len(episodes) == target, len(episodes)
assert len({int(row["seed"]) for row in episodes}) == target
assert queue["selected_total"] == target and queue["unique_selected_seeds"] == target, queue
assert len(list((root / "queue" / "done").glob("job-*.json"))) == target
assert not list(root.rglob("*.hdf5"))
assert not list(root.rglob("*.mp4"))
assert not list(root.rglob("*.pkl"))
q_records = 0
for path in root.rglob("q_guidance_summary_rank*.jsonl"):
    q_records += sum(1 for line in path.open(errors="ignore") if line.strip())
assert q_records > 0
print(json.dumps({"event": "step_audit_pass", "completed": target, "q_records": q_records, "summary": summary}, sort_keys=True))
PY
    printf '%s\tcompleted\t%s\t%s\n' "$(date -Is)" "${step}" "${step_output}/metrics/summary.json" >>"${STATUS_FILE}"
done

current_step="ranking"
/path/to/robotwin-python - "${RESULT_ROOT}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for step_dir in sorted(root.glob("step_*")):
    step = int(step_dir.name.split("_")[-1])
    summary = json.loads((step_dir / "metrics" / "summary.json").read_text())
    episodes = [json.loads(line) for line in (step_dir / "metrics" / "episodes.jsonl").read_text().splitlines() if line.strip()]
    q_values = []
    for path in step_dir.rglob("q_guidance_summary_rank*.jsonl"):
        for line in path.read_text(errors="ignore").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            value = record.get("q_min")
            if isinstance(value, list):
                q_values.extend(float(item) for item in value if math.isfinite(float(item)))
            elif value is not None and math.isfinite(float(value)):
                q_values.append(float(value))
    artifacts = {
        "hdf5": len(list(step_dir.rglob("*.hdf5"))),
        "mp4": len(list(step_dir.rglob("*.mp4"))),
        "pkl": len(list(step_dir.rglob("*.pkl"))),
    }
    complete = (
        summary.get("target_episodes") == 8
        and summary.get("completed") == 8
        and len(episodes) == 8
        and len({int(item["seed"]) for item in episodes}) == 8
        and len(q_values) > 0
        and not any(artifacts.values())
    )
    rows.append({
        "checkpoint": step,
        "complete": complete,
        "completed": summary.get("completed"),
        "successes": summary.get("successes"),
        "success_rate": summary.get("success_rate"),
        "unique_episode_seeds": len({int(item["seed"]) for item in episodes}),
        "q_records": len(q_values),
        "q_min_mean": statistics.fmean(q_values) if q_values else None,
        "q_min_median": statistics.median(q_values) if q_values else None,
        "q_min_std": statistics.pstdev(q_values) if len(q_values) > 1 else 0.0 if q_values else None,
        "artifacts": artifacts,
        "metrics_path": str(step_dir / "metrics" / "summary.json"),
    })

assert len(rows) == 10, len(rows)
assert all(row["complete"] for row in rows), rows
rates = sorted({row["success_rate"] for row in rows}, reverse=True)
rank_for_rate = {rate: index + 1 for index, rate in enumerate(rates)}
for row in rows:
    row["rank"] = rank_for_rate[row["success_rate"]]
rows.sort(key=lambda row: (-row["success_rate"], row["checkpoint"]))
payload = {
    "schema_version": 1,
    "ranking_basis": "8-episode hanging_mug success_rate only; equal rates share rank",
    "q_value_policy": "descriptive only; absolute Q scale does not break ranking ties",
    "comparability": {
        "within_this_sweep": True,
        "same_task": "hanging_mug/demo_clean",
        "same_eight_replay_rows": True,
        "same_guidance": "fixed scale=0.1 beta=2 interval=5 min grad-normalized",
        "not_comparable_to_prior_place_can_basket_sweep": True,
        "sampling_uncertainty": "8 episodes per checkpoint; one episode changes success rate by 12.5 percentage points",
    },
    "rows": rows,
}
(root / "checkpoint_ranking.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
with (root / "checkpoint_ranking.tsv").open("w") as handle:
    handle.write("rank\tcheckpoint\tcompleted\tsuccesses\tsuccess_rate\tq_records\tq_min_mean\tq_min_median\tq_min_std\tcomplete\n")
    for row in rows:
        handle.write("\t".join(str(row[key]) for key in ("rank", "checkpoint", "completed", "successes", "success_rate", "q_records", "q_min_mean", "q_min_median", "q_min_std", "complete")) + "\n")
print(json.dumps({"event": "ranking_complete", "path": str(root / "checkpoint_ranking.json"), "rows": len(rows)}, sort_keys=True))
PY

printf '%s\tranking_completed\tall\t%s\n' "$(date -Is)" "${RESULT_ROOT}/checkpoint_ranking.json" >>"${STATUS_FILE}"
touch "${RESULT_ROOT}/RANKING_COMPLETE_READY_FOR_100EP"
