#!/usr/bin/env bash
set -euo pipefail

rank="$1"
label="$2"

root=/path/to/mnt/datasets/libero10_va2_success_mix_orientationfix_gripperfix_frame2_20260912
logroot=/path/to/mnt/ops/cosmos_policy_analysis_20260906/frame2_10hz_fix_20260912/open_jepa_4node_r2
session="frame2-open-jepa-4node-n${rank}-${label}-20260912"
mkdir -p "$logroot/torchrun_node${rank}"

tmux new-session -d -s "$session" \
  "cd /path/to/home/wy/lingbot-va-original-rl && \
   export PYTHONPATH=/path/to/home/wy/vjepa2_source_restore:/path/to/home/wy/lingbot-va-original-rl && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
   export TOKENIZERS_PARALLELISM=false && \
   exec /path/to/home/miniconda3/envs/lingbot/bin/python -m torch.distributed.run \
     --nnodes=4 --nproc_per_node=8 --node_rank=${rank} \
     --master_addr=192.168.4.33 --master_port=31316 --max_restarts=0 \
     --log_dir=${logroot}/torchrun_node${rank} --tee 3 \
     /path/to/home/wy/lingbot-va-original-rl/preprocessing/extract_jepa_features.py \
     --dataset-root ${root}/open_source \
     --checkpoint /path/to/mnt/weight/vjepa2.1/vjepa2_1_vitG_384.pt \
     --camera-keys observation.images.image observation.images.wrist_image \
     --target-fps 10 --height 128 --width 128 --batch-size 1 \
     > ${logroot}/node${rank}.log 2>&1"

echo "launched ${session}"
