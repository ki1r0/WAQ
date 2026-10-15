"""Launch the task-4 video eval from the known-good no-Q launcher.

The remote launcher is kept as the source of truth; this wrapper only changes
the requested policy checkpoint, output lineage, video flag, Q scale/budget,
and explicit non-overlapping trial shards.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


SOURCE = Path(
    "/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/"
    "gripperfix_20260909/A_line2/launch_trackA_step7_va2critic2_novideo_20260910.sh"
)


def replace_once(script: str, old: str, new: str) -> str:
    count = script.count(old)
    if count != 1:
        raise RuntimeError(f"expected one occurrence of {old!r}, found {count}")
    return script.replace(old, new)


def build_script(step: int, model: str, policy_id: str, run_id: str) -> str:
    script = SOURCE.read_text()
    script = replace_once(script, "LIBERO_NO_VIDEO=1", "LIBERO_NO_VIDEO=0")
    script = replace_once(
        script,
        "MODEL=/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/"
        "export/from_va2_gripperfix_20260909_step25000_diffusers",
        f"MODEL={model}",
    )
    script = replace_once(
        script, 'FULL_ROOT="$OUTPUT_ROOT/full/guided"',
        'FULL_ROOT="$OUTPUT_ROOT/full/noq"',
    )
    script = replace_once(script, "--q_guidance_scale 0.08", "--q_guidance_scale 0.0")
    script = replace_once(script, "--q_cpb_fixed_budget 0.08", "--q_cpb_fixed_budget 0.0")
    script = replace_once(script, "POLICY_ID=va2-gripperfix-20260909", f"POLICY_ID={policy_id}")
    script = replace_once(
        script,
        "RUN_ID=EVAL-LIBERO10-COSMOS-JOINT-GRIPPERFIX-B008-20260910-10GPU",
        f"RUN_ID={run_id}",
    )

    arrays = (
        'IFS=\',\' read -r -a GPUS <<< "$GPU_IDS"\n'
        'IFS=\',\' read -r -a TASKS <<< "$TASK_IDS"\n'
        'IFS=\',\' read -r -a TRIAL_COUNTS <<< "$TRIAL_COUNTS_CSV"\n'
        'IFS=\',\' read -r -a TRIAL_OFFSETS <<< "$TRIAL_OFFSETS_CSV"\n'
        'if [[ ${#GPUS[@]} -ne ${#TASKS[@]} || ${#GPUS[@]} -ne ${#TRIAL_COUNTS[@]} || '
        '${#GPUS[@]} -ne ${#TRIAL_OFFSETS[@]} ]]; then '
        'echo "GPU/TASK/trial count mismatch" >&2; exit 64; fi'
    )
    old_arrays = (
        'IFS=\',\' read -r -a GPUS <<< "$GPU_IDS"\n'
        'IFS=\',\' read -r -a TASKS <<< "$TASK_IDS"\n'
        'if [[ ${#GPUS[@]} -ne ${#TASKS[@]} ]]; then echo "GPU/TASK count mismatch" >&2; exit 64; fi'
    )
    script = replace_once(script, old_arrays, arrays)
    script = replace_once(
        script,
        'echo "[$(date --iso-8601=seconds)] START_CLIENT host=$HOST_LABEL worker=$worker '
        'gpu=$gpu task=$task port=$port trials=50"',
        'echo "[$(date --iso-8601=seconds)] START_CLIENT host=$HOST_LABEL worker=$worker '
        'gpu=$gpu task=$task port=$port trials=${TRIAL_COUNTS[$i]} '
        'trial_offset=${TRIAL_OFFSETS[$i]}"',
    )
    script = replace_once(
        script,
        '--port "$port" --test-num 50 --trial-offset 0 --out-dir "$FULL_ROOT/client/worker-$worker"',
        '--port "$port" --test-num "${TRIAL_COUNTS[$i]}" '
        '--trial-offset "${TRIAL_OFFSETS[$i]}" --out-dir "$FULL_ROOT/client/worker-$worker"',
    )
    script = replace_once(
        script,
        'printf \'%s host=%s worker=%s gpu=%s task=%s trials=50\\n\' '
        '"$(date --iso-8601=seconds)" "$HOST_LABEL" "$worker" "${GPUS[$i]}" "${TASKS[$i]}"',
        'printf \'%s host=%s worker=%s gpu=%s task=%s trials=%s trial_offset=%s\\n\' '
        '"$(date --iso-8601=seconds)" "$HOST_LABEL" "$worker" "${GPUS[$i]}" '
        '"${TASKS[$i]}" "${TRIAL_COUNTS[$i]}" "${TRIAL_OFFSETS[$i]}"',
    )
    return script


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=(2500, 5000, 7500, 20000), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--trial-counts", required=True)
    parser.add_argument("--trial-offsets", required=True)
    parser.add_argument("--port-base", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    gpu_ids = [item for item in args.gpus.split(",") if item]
    counts = [item for item in args.trial_counts.split(",") if item]
    offsets = [item for item in args.trial_offsets.split(",") if item]
    if not (len(gpu_ids) == len(counts) == len(offsets)):
        raise SystemExit("GPU/trial shard lengths must match")
    if sum(map(int, counts)) != 50:
        raise SystemExit("trial shards must cover exactly 50 trials")
    if sorted({x for start, count in zip(map(int, offsets), map(int, counts)) for x in range(start, start + count)}) != list(range(50)):
        raise SystemExit("trial shards must cover each trial 0..49 exactly once")

    policy_id = f"va2-gripperfix-20260909-step{args.step}-task4-video"
    run_id = f"EVAL-LIBERO10-GRIPPERFIX-STEP{args.step}-TASK4-VIDEO-20260910-{args.host_label}"
    script = build_script(args.step, args.model, policy_id, run_id)
    if args.dry_run:
        print(script)
        return
    os.environ.update(
        HOST_LABEL=args.host_label,
        GPU_IDS=args.gpus,
        TASK_IDS=",".join(["4"] * len(gpu_ids)),
        WORKER_OFFSET="0",
        PORT_BASE=str(args.port_base),
        OUTPUT_ROOT=args.output_root,
        TRIAL_COUNTS_CSV=args.trial_counts,
        TRIAL_OFFSETS_CSV=args.trial_offsets,
    )
    os.execv("/bin/bash", ["bash", "-c", script])


if __name__ == "__main__":
    main()
