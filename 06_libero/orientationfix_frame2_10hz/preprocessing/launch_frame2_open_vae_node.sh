#!/usr/bin/env bash
set -euo pipefail

rank="$1"
label="$2"

root=/path/to/mnt/datasets/libero10_va2_success_mix_orientationfix_gripperfix_frame2_20260912
logroot=/path/to/mnt/ops/cosmos_policy_analysis_20260906/frame2_10hz_fix_20260912/open_vae_4node
session="frame2-open-vae-4node-n${rank}-${label}-20260912"
mkdir -p "$logroot/torchrun_node${rank}"

tmux new-session -d -s "$session" \
  "cd /path/to/home/wy/lingbot-va-original-rl && \
   export PYTHONPATH=/path/to/home/wy/lingbot-va-original-rl && \
   export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
   export TOKENIZERS_PARALLELISM=false && \
   exec /path/to/home/miniconda3/envs/lingbot/bin/python -m torch.distributed.run \
     --nnodes=4 --nproc_per_node=8 --node_rank=${rank} \
     --master_addr=192.168.4.33 --master_port=31314 --max_restarts=0 \
     --log_dir=${logroot}/torchrun_node${rank} --tee 3 \
     /path/to/mnt/ops/cosmos_policy_analysis_20260906/frame2_10hz_fix_20260912/extract_latent_vae_frame2_distributed.py \
     --config-name libero_plus_critic_cache_lingbot_va \
     --dataset-root ${root}/open_source \
     --camera-keys image wrist_image --target-fps 10 --height 128 --width 128 --chunk-size 450 \
     > ${logroot}/node${rank}.log 2>&1"

echo "launched ${session}"
