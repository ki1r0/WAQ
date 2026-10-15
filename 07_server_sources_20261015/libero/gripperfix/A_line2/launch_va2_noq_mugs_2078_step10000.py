from pathlib import Path
import os


source = Path(
    "/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/"
    "gripperfix_20260909/A_line2/launch_trackA_step7_va2critic2_novideo_20260910.sh"
)
script = source.read_text()
changes = {
    "LIBERO_NO_VIDEO=1": "LIBERO_NO_VIDEO=0",
    "MODEL=/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/"
    "export/from_va2_gripperfix_20260909_step25000_diffusers":
    "MODEL=/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/"
    "export/from_va2_gripperfix_20260909_step10000_diffusers_eval2078_20260910",
    'FULL_ROOT="$OUTPUT_ROOT/full/guided"': 'FULL_ROOT="$OUTPUT_ROOT/full/noq"',
    "--q_guidance_scale 0.08": "--q_guidance_scale 0.0",
    "--q_cpb_fixed_budget 0.08": "--q_cpb_fixed_budget 0.0",
    "POLICY_ID=va2-gripperfix-20260909": "POLICY_ID=va2-gripperfix-20260909-step10000-noq",
    "RUN_ID=EVAL-LIBERO10-COSMOS-JOINT-GRIPPERFIX-B008-20260910": (
        "RUN_ID=EVAL-LIBERO10-GRIPPERFIX-STEP10000-NOQ-MUGS-2078-20260910"
    ),
    "--test-num 50 --trial-offset 0": (
        '--test-num "$((i == 7 ? 8 : 6))" --trial-offset "$((i*6))"'
    ),
}
for old, new in changes.items():
    assert script.count(old) == 1, old
    script = script.replace(old, new)
script = script.replace("trials=50", "trials=sharded")

os.environ.update(
    HOST_LABEL="lyg2078-va2-step10000-noq",
    GPU_IDS="0,1,2,3,4,5,6,7",
    TASK_IDS="4,4,4,4,4,4,4,4",
    WORKER_OFFSET="0",
    PORT_BASE="34720",
    OUTPUT_ROOT="/mnt/zengqixin/eval_libero10_va2gf_noq_mugs_20260910_2078_step10000",
)
os.execv("/bin/bash", ["bash", "-c", script])
