#!/usr/bin/env python3
"""Run VA2 SFT on the prepared LIBERO-10 success-only mix.

This wrapper reuses the repository's official ``wan_va.train.Trainer`` and
only overrides paths/shape settings required by the two LIBERO source views.
"""

from __future__ import annotations

import copy
import os

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.train import Trainer, init_logger


DATASET = "/path/to/mnt/datasets/libero10_va2_success_mix_20260906"
INIT_MODEL = "/path/to/mnt/experiments/robotwin_clean_sft_20260903/init_model_joint1185"
SAVE_ROOT = "/path/to/mnt/checkpoints/va2_libero10_cosmos_20260906/bf16_6gpu_bs1_accum2_lr1e-5_25k"
DS_CONFIG = "/path/to/home/wy/lingbot-va-original-rl-adaptive-cpb/configs/deepspeed_zero2_bf16.json"


def main() -> None:
    config = copy.deepcopy(VA_CONFIGS["libero_plus_sft_lingbot_va"])
    config.dataset_path = DATASET
    config.empty_emb_path = "/path/to/home/wy/giga/dataset/empty_emb.pt"
    config.wan22_pretrained_model_name_or_path = INIT_MODEL
    config.save_root = os.environ.get("VA2_SAVE_ROOT", SAVE_ROOT)
    config.env_type = "libero_plus_panda"
    config.height = 128
    config.width = 128
    config.source_fps = 20
    config.target_fps = 10
    config.temporal_downsample_factor = 2
    config.action_fps = 10
    config.action_downsample_policy = "stride_2"
    config.action_downsample_factor = 2
    config.frame_chunk_size = 2
    config.obs_cam_keys = [
        "observation.images.front",
        "observation.images.wrist",
    ]
    # No MP4 decode is needed: both views have precomputed VAE latents.
    config.latent_only_video_keys = [
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.image",
        "observation.images.wrist_image",
    ]
    config.action_dim = 30
    config.action_per_frame = 4
    config.batch_size = 1
    config.gradient_accumulation_steps = 2
    config.num_steps = int(os.environ.get("VA2_NUM_STEPS", "25000"))
    config.save_interval = 2_500
    config.load_worker = 8
    config.enable_wandb = True
    config.wandb_mode = "offline"
    config.wandb_dir = "/path/to/mnt/wandb"
    config.wandb_project = "libero10-cosmos-va2-sft"
    config.wandb_run_name = "va2_libero10_cosmos_20260906"
    config.wandb_fail_open = True
    config.use_deepspeed = True
    config.deepspeed_config = DS_CONFIG

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    init_distributed(world_size, local_rank, rank)
    if rank == 0:
        print(
            f"VA2 SFT launch: world_size={world_size} dataset={DATASET} "
            f"save_root={SAVE_ROOT}"
        )
    Trainer(config).train()


if __name__ == "__main__":
    init_logger()
    main()
