#!/usr/bin/env python3
"""Five-node ZeRO-2 RobotWin clean SFT, initialized from R5 step 2826 weights."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from argparse import Namespace

REPO_ROOT = "/path/to/shared-storage/software/robotwin_runtime/lingbot-va-original-rl-adaptive-cpb"
HISTORICAL_CONFIG = (
    "/path/to/shared-storage/checkpoints/robotwin50_clean_sft_from_joint1185_20260903/"
    "bf16_8gpu_bs1_accum2_lr1e-5_25k/checkpoints/config.json"
)
SOURCE_CONFIG_NAME = "robotwin50_clean_sft_from_joint1185_20260903"
CONFIG_NAME = "robotwin50_clean_sft_from_full_r5_zero2_5node_20261003_400k_bs2_accum1_retry5"
INIT_MODEL_ROOT = (
    "/path/to/shared-storage/experiments/robotwin_clean_sft_from_full_r5_20261003_5node400k/"
    "init_model_step2826_diffusers_mnt"
)
DATASET_PATH = (
    "/path/to/shared-storage/wy/giga/dataset/lingbot-robotwin-clean-and-aug-lerobot/"
    "robotwin-clean-and-aug-lerobot/lerobot_robotwin_eef_clean_50"
)
EMPTY_EMB_PATH = "/path/to/shared-storage/wy/giga/dataset/empty_emb.pt"
RUN_ROOT = (
    "/path/to/shared-storage/experiments/robotwin_clean_sft_from_full_r5_20261003_5node400k/"
    "zero2_5node_400k_bs2_accum1_retry5"
)
DEEPSPEED_CONFIG = f"{RUN_ROOT}/configs/deepspeed_zero2_bf16_bs2_accum1.json"
OUTPUT_ROOT = (
    "/path/to/shared-storage/checkpoints/robotwin50_clean_sft_from_full_r5_20261003_"
    "zero2_5node400k/bf16_40gpu_bs2_accum1_lr1e-5_5k_save200_retry5"
)
LOG_ROOT = (
    "/path/to/shared-storage/training_logs/robotwin50_clean_sft_from_full_r5_20261003_"
    "zero2_5node400k_bs2_accum1_retry5"
)

for path in (REPO_ROOT, os.path.join(REPO_ROOT, "wan_va")):
    if path not in sys.path:
        sys.path.insert(0, path)

import wan_va.train as official_train  # noqa: E402

_ORIGINAL_TRAIN_DATALOADER = official_train.DataLoader


class _LengthBucketSampler:
    """Yield pairwise length-bucketed indices through PyTorch's normal batching."""

    def __init__(self, dataset, batch_size: int, distributed_sampler):
        if int(batch_size) != 2:
            raise RuntimeError(f"expected per-rank batch_size=2, got {batch_size}")
        if distributed_sampler is None:
            self.rank = 0
            self.world_size = 1
            self.seed = 42
        else:
            self.rank = int(distributed_sampler.rank)
            self.world_size = int(distributed_sampler.num_replicas)
            self.seed = int(getattr(distributed_sampler, "seed", 42))
        if not 0 <= self.rank < self.world_size:
            raise RuntimeError("distributed sampler rank is outside world size")
        self.batch_size = 2
        self.epoch = 0

        if not hasattr(dataset, "_datasets"):
            raise TypeError("expected MultiLatentLeRobotDataset with _datasets")
        frame_counts = []
        for subdataset in dataset._datasets:
            for meta in subdataset.new_metas:
                span = int(meta["end_frame"]) - int(meta["start_frame"])
                if span <= 0:
                    raise ValueError(f"invalid latent segment frame span: {span}")
                # RobotWin-50 cached segments encode one latent frame per 16
                # source frames; verified against samples from all 50 tasks.
                frame_counts.append((span + 15) // 16)
        if len(frame_counts) != len(dataset):
            raise RuntimeError(
                f"frame metadata length {len(frame_counts)} != dataset length {len(dataset)}"
            )
        self.frame_counts = frame_counts

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        indices = list(range(len(self.frame_counts)))
        rng.shuffle(indices)
        buckets = {}
        for index in indices:
            buckets.setdefault(self.frame_counts[index], []).append(index)

        same_shape_batches = []
        for bucket in buckets.values():
            for offset in range(0, len(bucket) - 1, self.batch_size):
                same_shape_batches.append(bucket[offset:offset + self.batch_size])
        rng.shuffle(same_shape_batches)

        # Only complete world-size groups are yielded, keeping exactly two
        # examples per rank on every distributed update.
        usable_batches = (len(same_shape_batches) // self.world_size) * self.world_size
        for offset in range(0, usable_batches, self.world_size):
            # Yield a flat pair of adjacent indices. The ordinary DataLoader
            # BatchSampler then forms a batch of exactly these two examples.
            yield from same_shape_batches[offset + self.rank]

    def __len__(self) -> int:
        counts = {}
        for frame_count in self.frame_counts:
            counts[frame_count] = counts.get(frame_count, 0) + 1
        pair_batches = sum(count // self.batch_size for count in counts.values())
        return (pair_batches // self.world_size) * self.batch_size


def _length_bucketed_dataloader(dataset, batch_size, shuffle, num_workers, sampler):
    if shuffle:
        raise RuntimeError("expected shuffle=False when a sampler is provided")
    length_bucket_sampler = _LengthBucketSampler(dataset, batch_size, sampler)
    return _ORIGINAL_TRAIN_DATALOADER(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sampler=length_bucket_sampler,
    )

RUNTIME_OR_PATH_FIELDS = {
    "__name__", "rank", "local_rank", "world_size", "use_deepspeed",
    "deepspeed_config", "resume_from", "save_root",
    "wan22_pretrained_model_name_or_path", "dataset_path", "empty_emb_path",
    "wandb_dir", "wandb_run_name", "num_steps", "save_interval",
}


def prepare_config() -> tuple[object, dict]:
    if not os.path.isfile(HISTORICAL_CONFIG):
        raise FileNotFoundError(HISTORICAL_CONFIG)
    if SOURCE_CONFIG_NAME not in official_train.VA_CONFIGS:
        raise KeyError(f"required source config is not registered: {SOURCE_CONFIG_NAME}")

    with open(HISTORICAL_CONFIG, "r", encoding="utf-8") as handle:
        saved = json.load(handle)
    reference_batch = int(saved["batch_size"])
    reference_accum = int(saved["gradient_accumulation_steps"])
    reference_world = int(saved["world_size"])
    reference_steps = int(saved["num_steps"])
    reference_samples = reference_batch * reference_accum * reference_world * reference_steps
    if (reference_batch, reference_accum, reference_world, reference_steps) != (1, 2, 8, 25000):
        raise RuntimeError(
            "1185 config changed: expected batch/accum/world/steps=1/2/8/25000, got "
            f"{reference_batch}/{reference_accum}/{reference_world}/{reference_steps}"
        )
    if not saved.get("use_deepspeed"):
        raise RuntimeError("the historical clean-SFT run was not DeepSpeed-based")

    with open(DEEPSPEED_CONFIG, "r", encoding="utf-8") as handle:
        ds_config = json.load(handle)
    if ds_config.get("zero_optimization", {}).get("stage") != 2:
        raise RuntimeError("expected the historical ZeRO stage 2 config")
    if not ds_config.get("bf16", {}).get("enabled"):
        raise RuntimeError("expected bf16 enabled in the historical ZeRO-2 config")
    if ds_config.get("gradient_accumulation_steps") != 1:
        raise RuntimeError("DeepSpeed JSON must retain the historical accumulation=1")
    if ds_config.get("train_micro_batch_size_per_gpu") != 2:
        raise RuntimeError("DeepSpeed JSON must use per-GPU micro batch size 2")

    current_world = 40
    batch = 2
    accumulation = 1
    effective_batch = batch * accumulation * current_world
    if reference_samples % effective_batch:
        raise RuntimeError("historical sample budget is not divisible at world size 40")
    current_steps = reference_samples // effective_batch
    save_interval = 200
    if current_steps % save_interval:
        raise RuntimeError("final training step must also be a save step")

    config = copy.deepcopy(official_train.VA_CONFIGS[SOURCE_CONFIG_NAME])
    for key, value in saved.items():
        if key not in RUNTIME_OR_PATH_FIELDS and key != "param_dtype" and key in config:
            config[key] = value

    config.__name__ = (
        "Config: RobotWin-50 clean SFT from full RoboCOIN/DROID R5 step 2826 "
        "(40-GPU DeepSpeed ZeRO-2, matched 1185 sample budget)"
    )
    config.wan22_pretrained_model_name_or_path = INIT_MODEL_ROOT
    config.dataset_path = DATASET_PATH
    config.empty_emb_path = EMPTY_EMB_PATH
    config.save_root = OUTPUT_ROOT
    config.wandb_dir = LOG_ROOT
    config.wandb_run_name = "rw50_clean_sft_full_r5_zero2_5node_400k_bs2_accum1_retry5_20261003"
    config.batch_size = batch
    config.gradient_accumulation_steps = accumulation
    config.num_steps = current_steps
    config.save_interval = save_interval
    config.resume_from = None

    plan = {
        "config_name": CONFIG_NAME,
        "source_config": SOURCE_CONFIG_NAME,
        "historical_config": HISTORICAL_CONFIG,
        "init_model_weights_only": INIT_MODEL_ROOT,
        "resume_from": None,
        "optimizer_state_restored": False,
        "dataset": DATASET_PATH,
        "world_size": current_world,
        "batch_per_rank": batch,
        "manual_gradient_accumulation_steps": accumulation,
        "effective_global_batch": effective_batch,
        "reference_1185_steps": reference_steps,
        "reference_1185_world_size": reference_world,
        "reference_sample_budget": reference_samples,
        "planned_steps": current_steps,
        "save_interval": save_interval,
        "periodic_checkpoints": current_steps // save_interval,
        "batching_strategy": "per-rank latent-frame-length buckets; batch=2; no padding",
        "deepspeed_zero_stage": ds_config["zero_optimization"]["stage"],
        "deepspeed_json_accumulation_steps": ds_config["gradient_accumulation_steps"],
        "nccl_socket_ifname": "bond0.45",
        "output_root": OUTPUT_ROOT,
        "log_root": LOG_ROOT,
    }
    return config, plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config, plan = prepare_config()
    if args.plan:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return

    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    if world_size != 40:
        raise RuntimeError(f"expected 40 distributed workers, got WORLD_SIZE={world_size}")
    configs = dict(official_train.VA_CONFIGS)
    configs[CONFIG_NAME] = config
    official_train.VA_CONFIGS = configs
    official_train.DataLoader = _length_bucketed_dataloader
    official_train.init_logger()
    official_train.run(
        Namespace(
            config_name=CONFIG_NAME,
            save_root=OUTPUT_ROOT,
            resume_from=None,
            deepspeed=True,
            deepspeed_config=DEEPSPEED_CONFIG,
        )
    )


if __name__ == "__main__":
    main()
