#!/usr/bin/env python3
"""Official LingBot-VA LIBERO SFT with the frame4/orientationfix mix.

Only data/model/output paths and the explicitly audited frame4 settings differ
from the released ``libero_train`` configuration.  The wrapper runs the
dataset contract gate before constructing the Trainer so a stale 10-Hz cache
or normalization contract cannot silently enter training.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.train import Trainer, _config_to_dict, init_logger


G = Path("/mnt/zengqixin/ops/cosmos_policy_analysis_20260906/libero_open500_orientationfix_va2_critic2_official_20260911")
DATASET = "/mnt/zengqixin/datasets/libero10_va2_success_mix_orientationfix_frame4_20260911"
OPEN = "/mnt/zengqixin/datasets/libero_10_original_500_orientationfix_20260911/lerobot_v21_official_frame4_vae128"
COSMOS = "/mnt/zengqixin/datasets/LIBERO-Cosmos-Policy-lingbotva-20260906-frame4_500"
INIT_MODEL = "/mnt/zengqixin/experiments/robotwin_clean_sft_20260903/init_model_joint1185"
SAVE_ROOT = "/mnt/zengqixin/checkpoints/va2_libero10_cosmos_orientationfix_frame4_20260911/bf16_bs1_accum10_lr1e-5_5k"
EMPTY_EMB = "/home/zengqixin/wy/giga/dataset/empty_emb.pt"
DS_CONFIG = "/home/zengqixin/wy/lingbot-va-original-rl-adaptive-cpb/configs/deepspeed_zero2_bf16.json"


def _run_contract(config_path: Path) -> None:
    validator = G / "validate_frame4_contract.py"
    contract = G / "normalization_contract.json"
    if not validator.is_file():
        raise FileNotFoundError(validator)
    cmd = [
        sys.executable, str(validator), "--stage", "train",
        "--dataset", f"open={OPEN}", "--expected-episodes", "open=500",
        "--camera-keys", "open=observation.images.image,observation.images.wrist_image",
        "--dataset", f"cosmos={COSMOS}", "--episode-range", "cosmos=0:500",
        "--expected-episodes", "cosmos=500",
        "--target-fps", "20", "--config-json", str(config_path),
        "--write-contract", str(contract),
    ]
    subprocess.run(cmd, check=True)


def _make_config():
    # This is the official VA LIBERO-plus SFT config; all overrides below are
    # explicit frame4/path changes required by the current runbook.
    config = copy.deepcopy(VA_CONFIGS["libero_plus_sft_lingbot_va"])
    config.dataset_path = DATASET
    config.empty_emb_path = EMPTY_EMB
    config.wan22_pretrained_model_name_or_path = INIT_MODEL
    config.save_root = os.environ.get("VA2_SAVE_ROOT", SAVE_ROOT)
    config.env_type = "libero_plus_panda"
    config.dataset_format = "lerobot_v21"
    config.height = 128
    config.width = 128
    config.source_fps = 20
    config.target_fps = 20
    config.temporal_downsample_factor = 1
    config.action_fps = 20
    config.action_downsample_policy = "contiguous"
    config.action_downsample_factor = 1
    config.action_downsample_reduction = "stride"
    config.frame_chunk_size = 4
    config.action_per_frame = 4
    config.obs_cam_keys = [
        "observation.images.front",
        "observation.images.wrist",
    ]
    config.latent_only_video_keys = [
        "observation.images.front",
        "observation.images.wrist",
        "observation.images.image",
        "observation.images.wrist_image",
    ]
    config.action_dim = 30
    config.used_action_channel_ids = list(range(7))
    config.inverse_used_action_channel_ids = [
        i if i < 7 else 7 for i in range(30)
    ]
    config.action_norm_method = "quantiles"
    config.norm_stat = {
        "q01": [
            -0.6589285731315613, -0.84375, -0.9375,
            -0.12107142806053162, -0.1596428602933887,
            -0.26571428775787354, -1.0,
        ] + [0.0] * 23,
        "q99": [
            0.8999999761581421, 0.8544642925262451, 0.9375,
            0.17142857611179352, 0.1842857152223587,
            0.34392857551574707, 1.0,
        ] + [0.0] * 23,
    }
    config.attn_mode = "flex"
    config.attn_window = 30
    config.batch_size = 1
    config.gradient_accumulation_steps = 10
    config.learning_rate = 1e-5
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.1
    config.warmup_steps = 10
    config.num_steps = int(os.environ.get("VA2_NUM_STEPS", "5000"))
    config.save_interval = 200
    config.gc_interval = 50
    config.load_worker = 16
    config.cfg_prob = 0.1
    config.log_freq = 10
    config.enable_wandb = True
    config.wandb_mode = "offline"
    config.wandb_dir = "/mnt/zengqixin/wandb"
    config.wandb_project = "lingbot-va-libero-orientationfix-frame4"
    config.wandb_run_name = "va2_libero10_cosmos_orientationfix_frame4_5k"
    config.wandb_fail_open = True
    config.use_deepspeed = True
    config.deepspeed_config = DS_CONFIG
    config.jepa_loss_enabled = False
    config.load_jepa_targets = False
    config.dataset_init_workers = 0
    config.rank = int(os.environ.get("RANK", 0))
    config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    return config


def main() -> None:
    config = _make_config()
    G.mkdir(parents=True, exist_ok=True)
    resolved = G / "va2_resolved_config.json"
    if config.rank == 0:
        # Reuse the official Trainer's serializer: the config contains
        # torch.dtype and other non-JSON-native values.
        resolved.write_text(json.dumps(_config_to_dict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Ensure every rank observes the resolved config before the gate.  The
    # validator itself is read-only with respect to the datasets.
    if config.world_size > 1:
        import torch.distributed as dist
        if not dist.is_initialized():
            init_distributed(config.world_size, config.local_rank, config.rank)
        dist.barrier()
    # Only rank 0 writes the shared contract.  Running the validator on every
    # rank would race on normalization_contract.json and could make otherwise
    # valid multi-GPU startup nondeterministic.
    if config.rank == 0:
        _run_contract(resolved)
    if config.world_size > 1:
        import torch.distributed as dist
        dist.barrier()
    if config.rank == 0:
        print(
            f"VA2 frame4 SFT: world_size={config.world_size} "
            f"dataset={DATASET} save_root={config.save_root}",
            flush=True,
        )
    Trainer(config).train()


if __name__ == "__main__":
    init_logger()
    main()
