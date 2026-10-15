#!/usr/bin/env bash
set -euo pipefail

G=/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/libero_open500_orientationfix_va2_critic2_official_20260911
V=/mnt/zengqixin/datasets/libero_10_original_500_orientationfix_20260911/lerobot_v21_official_frame4_vae128
J=$G/open_frame4_jepa_exact.py
LOG=$G/open_frame4_jepa_2124.log
TORCHLOG=$G/open_frame4_jepa_torchrun_logs

source /home/zengqixin/miniconda3/etc/profile.d/conda.sh
conda activate lingbot
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/zengqixin/wy/lingbot-va-original-rl:/home/zengqixin/wy/vjepa2
cd /home/zengqixin/wy/lingbot-va-original-rl

exec python -m torch.distributed.run \
  --standalone --nproc_per_node=6 --master_port=31582 \
  --log_dir="$TORCHLOG" --tee 3 \
  "$J" \
  --dataset-root "$V" \
  --checkpoint /mnt/zengqixin/weight/vjepa2.1/vjepa2_1_vitG_384.pt \
  --camera-keys observation.images.image observation.images.wrist_image \
  --target-fps 20 --height 128 --width 128 --batch-size 1 --skip-existing
