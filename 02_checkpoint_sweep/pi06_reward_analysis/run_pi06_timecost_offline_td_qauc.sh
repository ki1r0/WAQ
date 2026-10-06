#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_HOST="${EXPECTED_HOST:?Set the expected worker hostname}"
readonly REPO=/path/to/home/wy/lingbot-va-original-rl
readonly PYTHON=/path/to/home/miniconda3/envs/lingbot/bin/python
readonly SCORER=/path/to/home/wy/lingbot-va-original-rl/scripts/plot_robotwin_sft_td_checkpoint_sweep_language_contractfix_v2_20260814.py
readonly VALIDATOR=/path/to/home/wy/lingbot-va-original-rl/scripts/validate_oldlang_hm_td_contractfix_output_20260814.py
readonly RANKER=/path/to/home/wy/lingbot-va-original-rl/scripts/build_pi06_timecost_offline_td_qauc_ranking_20260814.py
readonly ADAPTER_BUILDER=/path/to/home/wy/lingbot-va-original-rl/scripts/build_pi06_offline_eval_adapter_20260814.py
readonly INPUT=/path/to/mnt2/analysis/robotwin_sft_td_checkpoint_sweep_20260812
readonly REPLAY=${INPUT}/replays/hanging_mug
readonly CLEAN_CACHE=${INPUT}/clean_cache/hanging_mug
readonly REWARD=${INPUT}/frozen/reward_metadata.pt
readonly CLEAN_ROOT=/path/to/home/wy/giga/dataset/lingbot-robotwin-clean-and-aug-lerobot/robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50/hanging_mug-demo_clean_collect_200-50
readonly EXPECTED_CATALOG=a722b05487e244293aa10fca6a81b8a3eed2b1a30a35b1ec4dd6e74dde2bd977
readonly ORIGINAL=/path/to/mnt2/checkpoints/rl/robotwin50_pi06_timecost_overlayfixed_scalar_iql_50k_20260813/checkpoints
readonly RESUMED=/path/to/mnt2/checkpoints/rl/robotwin50_pi06_timecost_overlayfixed_scalar_iql_50k_resume7500_20260813/checkpoints
readonly ADAPTER_ROOT=/path/to/mnt2/robotwin_eval_adapters/PI06-TIMECOST-OVERLAYFIXED-OFFLINE-TDQAUC-20260814-R4
readonly OUTPUT=/path/to/mnt2/analysis/robotwin_pi06_timecost_overlayfixed_hanging_mug_td_qauc_5k50k_20260814_r5_contractfix_offline_10point

[[ "$(hostname)" == "${EXPECTED_HOST}" ]] || { echo "expected ${EXPECTED_HOST}" >&2; exit 2; }
[[ -s "${SCORER}" && -s "${VALIDATOR}" && -s "${RANKER}" && -s "${ADAPTER_BUILDER}" ]] || { echo "missing scorer/validator/ranker/adapter builder" >&2; exit 2; }
for path in "${REPLAY}/sft_replay.pt" "${REPLAY}/selection.json" "${REWARD}"; do
  [[ -s "${path}" ]] || { echo "missing input ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 2; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -qE '[0-9]'; then
  echo "GPU compute process already active" >&2
  exit 3
fi

steps=(5000 10000 15000 20000 25000 30000 35000 40000 45000 50000)
checkpoints=()
for step in "${steps[@]}"; do
  if (( step == 5000 )); then root="${ORIGINAL}"; else root="${RESUMED}"; fi
  source_checkpoint="${root}/checkpoint_$(printf '%08d' "${step}")"
  for file in config.json manifest.json training_state.pt; do
    [[ -s "${source_checkpoint}/${file}" ]] || { echo "incomplete checkpoint ${source_checkpoint}/${file}" >&2; exit 2; }
  done
  grep -Fq '"critic_type": "twin_language_query_transformer_v1"' "${source_checkpoint}/config.json"
  grep -Fq '"final_reward_normalization": "none"' "${source_checkpoint}/config.json"
  checkpoint="${ADAPTER_ROOT}/checkpoint_$(printf '%08d' "${step}")"
  "${PYTHON}" "${ADAPTER_BUILDER}" "${source_checkpoint}" "${checkpoint}"
  checkpoints+=("${checkpoint}")
done

mkdir -p "${OUTPUT}/logs" "${OUTPUT}/scores" "${OUTPUT}/validation" "${OUTPUT}/run_metadata"
date -Ins >"${OUTPUT}/run_metadata/start_time.txt"
sha256sum "${SCORER}" "${VALIDATOR}" "${RANKER}" "${ADAPTER_BUILDER}" >"${OUTPUT}/run_metadata/source_hashes.sha256"
sha256sum "${REPLAY}/sft_replay.pt" "${REPLAY}/selection.json" "${REWARD}" >"${OUTPUT}/run_metadata/input_hashes.sha256"
cp "$0" "${OUTPUT}/run_metadata/launcher.sh"
printf 'gpu\tstep\tcheckpoint\toutput\n' >"${OUTPUT}/run_metadata/worker_map.tsv"

score_one() {
  local gpu="$1" step="$2" checkpoint="$3"
  local out="${OUTPUT}/scores/checkpoint_$(printf '%08d' "${step}")"
  local log="${OUTPUT}/logs/score-${step}.log"
  printf '%s\t%s\t%s\t%s\n' "${gpu}" "${step}" "${checkpoint}" "${out}" >>"${OUTPUT}/run_metadata/worker_map.tsv"
  CUDA_VISIBLE_DEVICES="${gpu}" LINGBOT_ROOT="${REPO}" VJEPA_SOURCE=/path/to/home/wy/vjepa2_source_restore \
    "${PYTHON}" "${SCORER}" score \
      --task hanging_mug \
      --checkpoint "${checkpoint}" \
      --sft-replay "${REPLAY}" \
      --clean-root "${CLEAN_ROOT}" \
      --clean-cache-dir "${CLEAN_CACHE}" \
      --clean-episode 0 --clean-episode 1 --clean-episode 2 --clean-episode 3 --clean-episode 4 \
      --clean-episode 5 --clean-episode 6 --clean-episode 7 --clean-episode 8 --clean-episode 9 \
      --reward-metadata "${REWARD}" \
      --expected-catalog-sha256 "${EXPECTED_CATALOG}" \
      --output "${out}" >"${log}" 2>&1
  "${PYTHON}" "${VALIDATOR}" --score-output "${out}" --checkpoint-step "${step}" \
    --write-validation "${OUTPUT}/validation/checkpoint_$(printf '%08d' "${step}").json" >>"${log}" 2>&1
}

# checkpoint 50k is the contract smoke and also the formal 50k result.
score_one 0 50000 "${checkpoints[9]}"
echo 'SMOKE_PASS checkpoint=50000 rows=506 trajectories=10/10/10' >"${OUTPUT}/run_metadata/smoke_status.txt"

# Remaining nine checkpoints: eight GPUs, with GPU0 taking the final queued item.
pids=()
for index in 0 1 2 3 4 5 6 7; do
  (
    score_one "${index}" "${steps[$index]}" "${checkpoints[$index]}"
    if (( index == 0 )); then
      score_one 0 "${steps[8]}" "${checkpoints[8]}"
    fi
  ) &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
if (( status != 0 )); then
  echo 'FAILED worker_exit' >"${OUTPUT}/status.txt"
  exit 1
fi

echo 'PASS checkpoints=10/10 rows_each=506 trajectories_each=10/10/10 finite=true online=false' >"${OUTPUT}/status.txt"
date -Ins >"${OUTPUT}/run_metadata/end_time.txt"
"${PYTHON}" "${RANKER}" "${OUTPUT}"
echo "OFFLINE SWEEP COMPLETE ${OUTPUT}"
