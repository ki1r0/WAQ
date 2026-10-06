#!/usr/bin/env python3
"""Frame-2/10-Hz LIBERO VA-only SFT from the R2 step-1185 transformer.

This is a run-specific launcher wrapper.  It uses the repository's official
``wan_va.train.Trainer`` and only supplies the already materialized 10-Hz
dataset and the requested frame-2 temporal contract.  No critic or Q-guidance
code is imported or enabled.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import torch

from wan_va.configs import VA_CONFIGS
from wan_va.distributed.util import init_distributed
from wan_va.train import Trainer, _config_to_dict, init_logger


DATASET = "/path/to/mnt/datasets/libero10_va2_success_mix_20260906"
INIT_MODEL = "/path/to/mnt/experiments/robotwin_clean_sft_20260903/init_model_joint1185"
EMPTY_EMB = "/path/to/home/wy/giga/dataset/empty_emb.pt"
SAVE_ROOT = "/path/to/mnt/checkpoints/va2_libero10_cosmos_frame2_20260912/bf16_8gpu_bs1_accum10_lr1e-5_5k"
RUN_ROOT = Path("/path/to/mnt/ops/cosmos_policy_analysis_20260906/frame2_10hz_va_sft_2033_20260912")


def _preflight_dataset() -> None:
    root = Path(DATASET)
    expected = {"open_source": 500, "cosmos_success": 376}
    for name, expected_episodes in expected.items():
        droot = root / name
        info_path = droot / "meta" / "info.json"
        episodes_path = droot / "meta" / "episodes.jsonl"
        assert info_path.is_file(), info_path
        assert episodes_path.is_file(), episodes_path
        info = json.loads(info_path.read_text())
        assert int(info["total_episodes"]) == expected_episodes, (name, info)
        assert float(info["fps"]) == 20.0, (name, info.get("fps"))
        latent_dir = droot / "latents" / "chunk-000" / "observation.images.front"
        files = sorted(latent_dir.glob("episode_*.pth"))
        assert len(files) == expected_episodes, (name, len(files))
        sample = torch.load(files[0], map_location="cpu", weights_only=False)
        assert float(sample["fps"]) == 10.0, (name, sample.get("fps"))
        assert float(sample["ori_fps"]) == 20.0, (name, sample.get("ori_fps"))
        frame_ids = list(sample["frame_ids"])
        assert len(frame_ids) >= 3 and frame_ids[1] - frame_ids[0] == 2, (name, frame_ids[:5])
    assert (Path(INIT_MODEL) / "transformer" / "config.json").is_file()
    assert Path(EMPTY_EMB).is_file()
    print(
        "FRAME2_10HZ_PREFLIGHT PASS dataset=open500+cosmos_success376 "
        "ori_fps=20 target_fps=10 stride=2 init=R2_step1185",
        flush=True,
    )


def _make_config():
    config = copy.deepcopy(VA_CONFIGS["libero_plus_sft_lingbot_va"])
    config.dataset_path = DATASET
    config.empty_emb_path = EMPTY_EMB
    config.wan22_pretrained_model_name_or_path = INIT_MODEL
    config.save_root = os.environ.get("VA2_SAVE_ROOT", SAVE_ROOT)

    config.dataset_format = "lerobot_v21"
    config.env_type = "libero_plus_panda"
    config.height = 128
    config.width = 128
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

    # Official VA1 LIBERO temporal/action contract requested for this run.
    config.source_fps = 20
    config.target_fps = 10
    config.temporal_downsample_factor = 2
    config.action_fps = 10
    config.action_downsample_policy = "stride_2"
    config.action_downsample_factor = 2
    config.frame_chunk_size = 2
    config.action_dim = 30
    config.action_per_frame = 4

    # Official public LIBERO post-train defaults.
    config.batch_size = 1
    config.gradient_accumulation_steps = 10
    config.num_steps = int(os.environ.get("VA2_NUM_STEPS", "5000"))
    config.learning_rate = 1e-5
    config.beta1 = 0.9
    config.beta2 = 0.95
    config.weight_decay = 0.1
    config.warmup_steps = 10
    config.save_interval = 200
    config.gc_interval = 50
    config.load_worker = 16
    config.dataset_init_workers = 0
    config.cfg_prob = 0.1
    config.log_freq = 10

    # Explicit VA-only route: no critic, Q guidance, or JEPA auxiliary loss.
    config.jepa_loss_enabled = False
    config.load_jepa_targets = False
    config.distributed_backend = os.environ.get("DISTRIBUTED_BACKEND", "fsdp")
    config.rank = int(os.environ.get("RANK", 0))
    config.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    config.world_size = int(os.environ.get("WORLD_SIZE", 1))
    config.enable_wandb = True
    config.wandb_mode = "offline"
    config.wandb_dir = "/path/to/mnt/wandb"
    config.wandb_project = "libero10-va2-frame2-10hz"
    config.wandb_run_name = "va2_libero10_cosmos_frame2_10hz_5k_r2step1185"
    config.wandb_fail_open = True
    return config


def main() -> None:
    config = _make_config()
    init_distributed(config.world_size, config.local_rank, config.rank)
    if config.rank == 0:
        _preflight_dataset()
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        (RUN_ROOT / "resolved_config.json").write_text(
            json.dumps(_config_to_dict(config), indent=2, ensure_ascii=False) + "\n"
        )
        print(
            "FRAME2_10HZ_CONFIG "
            + json.dumps(
                {
                    "dataset": config.dataset_path,
                    "init_model": config.wan22_pretrained_model_name_or_path,
                    "frame_chunk_size": config.frame_chunk_size,
                    "action_per_frame": config.action_per_frame,
                    "source_fps": config.source_fps,
                    "target_fps": config.target_fps,
                    "temporal_downsample_factor": config.temporal_downsample_factor,
                    "action_downsample_factor": config.action_downsample_factor,
                    "gradient_accumulation_steps": config.gradient_accumulation_steps,
                    "num_steps": config.num_steps,
                    "distributed_backend": config.distributed_backend,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if config.world_size > 1:
        import torch.distributed as dist

        dist.barrier()
    Trainer(config).train()


if __name__ == "__main__":
    init_logger()
    main()
