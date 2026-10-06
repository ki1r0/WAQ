from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import default_collate
from tqdm import tqdm

from wan_va.configs import VA_CONFIGS
from wan_va.utils import init_logger

from .config import CriticTrainingConfig
from .train_critic_phase3 import (
    Phase3CriticTrainer,
)
from .canonical_cache import phase3_canonical_episode_cache_file


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(path)


def _runtime(args: argparse.Namespace) -> tuple[int, int, int, torch.device]:
    rank = args.rank if args.rank is not None else _env_int("RANK", 0)
    local_rank = (
        args.local_rank
        if args.local_rank is not None
        else _env_int("LOCAL_RANK", 0)
    )
    world_size = (
        args.world_size
        if args.world_size is not None
        else _env_int("WORLD_SIZE", 1)
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Phase 3 cache precompute requires CUDA")
    torch.cuda.set_device(local_rank)
    if args.standalone_ranks:
        print(
            f"[phase3-cache][rank{rank}] standalone rank mode; "
            "skipping distributed process-group initialization",
            flush=True,
        )
    elif world_size > 1 and not dist.is_initialized():
        print(
            f"[phase3-cache][rank{rank}] initializing NCCL process group",
            flush=True,
        )
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        print(
            f"[phase3-cache][rank{rank}] NCCL process group ready",
            flush=True,
        )
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def _episode_shard(
    episode_keys: list[tuple[int, int]],
    total: int,
    start: int,
    end: int | None,
    rank: int,
    world_size: int,
    max_items: int | None,
) -> list[int]:
    stop = total if end is None else min(end, total)
    if not 0 <= start <= stop <= total:
        raise ValueError(
            f"Invalid index range start={start}, end={end}, total={total}"
        )
    indices = list(range(start, stop))
    if max_items is not None:
        indices = indices[:max_items]
    return [episode_keys[index] for index in indices[rank::world_size]]


def _default_cache_dir(config: CriticTrainingConfig, config_path: str) -> Path:
    if config.base_config_name not in VA_CONFIGS:
        raise KeyError(f"Unknown base config: {config.base_config_name}")
    dataset_path = getattr(VA_CONFIGS[config.base_config_name], "dataset_path", None)
    if not dataset_path:
        raise ValueError(
            "No dataset_path found in the base config; pass --output-dir or "
            "set phase3_precomputed_cache_dir"
        )
    return Path(dataset_path) / "inference_state_cache" / Path(config_path).stem


def _group_episode_indices(
    trainer: Phase3CriticTrainer,
) -> dict[tuple[int, int], list[int]]:
    grouped: dict[tuple[int, int], list[int]] = {}
    for transition_idx, record in enumerate(trainer.dataset.records):
        grouped.setdefault(record.episode_key, []).append(transition_idx)
    for indices in grouped.values():
        indices.sort(
            key=lambda transition_idx: trainer.dataset.records[
                transition_idx
            ].rl_chunk_idx
        )
    return grouped


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _parse_cache_feature_layers(raw_layers: str | None) -> tuple[int, ...] | None:
    if raw_layers is None:
        return None
    layers = tuple(int(value.strip()) for value in raw_layers.split(",") if value.strip())
    if not layers:
        raise ValueError("--cache-feature-layers must contain at least one layer")
    if len(set(layers)) != len(layers):
        raise ValueError("--cache-feature-layers must not contain duplicates")
    if tuple(sorted(layers)) != layers:
        raise ValueError("--cache-feature-layers must be sorted")
    return layers


def _distributed_max_int(value: int, device: torch.device) -> int:
    if not (dist.is_available() and dist.is_initialized()):
        return int(value)
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def run(args: argparse.Namespace) -> None:
    config = CriticTrainingConfig.from_json(args.config)
    output_dir = Path(
        args.output_dir
        or config.phase3_precomputed_cache_dir
        or _default_cache_dir(config, args.config)
    )
    object.__setattr__(config, "phase3_use_packed_cache", False)
    object.__setattr__(config, "phase3_require_precomputed_cache", False)
    object.__setattr__(config, "phase3_cache_format", "canonical_episode")
    object.__setattr__(config, "phase3_exclude_last_n_chunks", 0)
    object.__setattr__(config, "enable_wandb", False)
    object.__setattr__(
        config,
        "reward_terms",
        tuple(
            {**term, "normalize": False}
            for term in config.reward_terms
            if term["name"]
            not in {"goal_jepa_progress", "goal_jepa_potential_shaping"}
        ),
    )
    object.__setattr__(config, "final_reward_normalization", "none")

    rank, local_rank, world_size, device = _runtime(args)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    torch.manual_seed(config.seed + rank)

    trainer = Phase3CriticTrainer(
        config,
        rank=rank,
        local_rank=local_rank,
        world_size=1,
        device=device,
        build_critic=False,
        create_outputs=False,
        dataset_init_workers=args.dataset_init_workers,
    )
    requested_cache_layers = _parse_cache_feature_layers(args.cache_feature_layers)
    if requested_cache_layers is not None:
        available_layers = set(trainer.cached_feature_layers)
        missing_layers = [
            layer for layer in requested_cache_layers if layer not in available_layers
        ]
        if missing_layers:
            raise ValueError(
                "Requested cache feature layers are unavailable: "
                f"{missing_layers}; available={sorted(available_layers)}"
            )
        missing_critic_layers = [
            layer
            for layer in config.critic_feature_layers
            if layer not in requested_cache_layers
        ]
        if missing_critic_layers:
            raise ValueError(
                "Compact cache must include every critic feature layer: "
                f"missing={missing_critic_layers}"
            )
        trainer.cached_feature_layers = requested_cache_layers

    grouped = _group_episode_indices(trainer)
    episode_keys = sorted(grouped)
    assigned_episode_keys = _episode_shard(
        episode_keys,
        total=len(episode_keys),
        start=args.start,
        end=args.end,
        rank=rank,
        world_size=world_size,
        max_items=args.max_items,
    )
    if not args.overwrite:
        assigned_episode_keys = [
            episode_key
            for episode_key in assigned_episode_keys
            if not phase3_canonical_episode_cache_file(output_dir, episode_key).exists()
        ]
    batch_size = args.batch_size or config.batch_size
    local_feature_batch_count = sum(
        math.ceil(len(grouped[episode_key]) / batch_size)
        for episode_key in assigned_episode_keys
    )
    server_prefix_mode = config.phase3_precompute_mode == "server_prefix"
    if server_prefix_mode:
        local_feature_batch_count = sum(
            len(grouped[episode_key]) for episode_key in assigned_episode_keys
        )
    max_feature_batch_count = _distributed_max_int(
        local_feature_batch_count,
        device,
    )

    if rank == 0:
        manifest = trainer.canonical_cache_manifest()
        manifest.update(
            {
                "config_path": str(Path(args.config).resolve()),
                "output_dir": str(output_dir.resolve()),
                "dataset_size": len(trainer.dataset),
                "cache_format": "phase3_canonical_episode",
                "episode_count": len(episode_keys),
                "requested_start": args.start,
                "requested_end": args.end,
                "requested_max_items": args.max_items,
                "world_size": world_size,
            }
        )
        _write_json_atomic(output_dir / "manifest.json", manifest)

    if not assigned_episode_keys and max_feature_batch_count == 0:
        print(f"[phase3-cache][rank{rank}] nothing to precompute", flush=True)
        return

    log_path = output_dir / f"precompute_rank{rank:03d}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    saved_episodes = 0
    saved_chunks = 0
    feature_batch_count = 0
    started = time.perf_counter()
    progress = tqdm(
        assigned_episode_keys,
        disable=not args.progress or rank != 0,
        desc="phase3-episode-cache",
    )
    for episode_index, episode_key in enumerate(progress, start=1):
        episode_start = time.perf_counter()
        chunks: list[dict] = []
        timings_total: dict[str, float] = {}
        reward_means: list[float] = []
        distance_means: list[float] = []
        transition_indices = grouped[episode_key]
        if server_prefix_mode:
            with torch.no_grad():
                chunks = trainer.compute_phase3_server_prefix_episode(
                    transition_indices,
                    timings_total,
                )
            feature_batch_count += len(transition_indices)
            reward_means.extend(
                float(chunk["cached_reward"].detach().float().cpu())
                for chunk in chunks
            )
            distance_means.extend(
                float(chunk["jepa_distance"].detach().float().cpu())
                for chunk in chunks
            )
        else:
            for offset in range(0, len(transition_indices), batch_size):
                batch_indices = transition_indices[offset : offset + batch_size]
                batch = default_collate(
                    [trainer.dataset[index] for index in batch_indices]
                )
                timings: dict[str, float] = {}
                with torch.no_grad():
                    computed = trainer.compute_phase3_canonical_inputs(
                        batch,
                        timings,
                        feature_layers=trainer.cached_feature_layers,
                    )
                feature_batch_count += 1
                chunks.extend(trainer.build_phase3_canonical_chunks(batch, computed))
                reward_means.append(
                    float(computed["reward"].detach().float().mean().cpu())
                )
                distance_means.append(
                    float(computed["jepa_distance"].detach().float().mean().cpu())
                )
                for key, value in timings.items():
                    timings_total[key] = timings_total.get(key, 0.0) + float(value)

        saved = trainer.save_phase3_canonical_episode_cache(
            episode_key=episode_key,
            chunks=chunks,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )
        saved_episodes += int(saved)
        saved_chunks += len(chunks) if saved else 0
        elapsed = time.perf_counter() - episode_start
        entry = {
            "rank": rank,
            "episode_number": episode_index,
            "episode_key": [int(episode_key[0]), int(episode_key[1])],
            "transition_count": len(transition_indices),
            "chunk_count": len(chunks),
            "saved": bool(saved),
            "saved_episodes": saved_episodes,
            "saved_chunks": saved_chunks,
            "elapsed": elapsed,
            "items_per_second": (
                float(len(transition_indices)) / elapsed
                if elapsed > 0
                else 0.0
            ),
            "reward_mean": _mean(reward_means),
            "jepa_distance": _mean(distance_means),
            **timings_total,
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        if rank == 0 and episode_index % args.log_interval == 0:
            print(
                "[phase3-cache][rank0] "
                f"episode={episode_index} saved_episodes={saved_episodes} "
                f"saved_chunks={saved_chunks} "
                f"elapsed={elapsed:.2f}s "
                f"video={timings_total.get('predict_video_time', 0.0):.2f}s "
                f"features={timings_total.get('q_feature_time', 0.0):.2f}s "
                f"jepa={timings_total.get('jepa_reward_time', 0.0):.2f}s",
                flush=True,
            )

    if (not server_prefix_mode) and feature_batch_count < max_feature_batch_count:
        dummy_transition_indices = grouped[episode_keys[0]][:batch_size]
        dummy_batch = default_collate(
            [trainer.dataset[index] for index in dummy_transition_indices]
        )
        dummy_count = max_feature_batch_count - feature_batch_count
        print(
            f"[phase3-cache][rank{rank}] running {dummy_count} dummy "
            "feature batches to keep distributed FSDP ranks aligned",
            flush=True,
        )
        for _ in range(dummy_count):
            with torch.no_grad():
                trainer.compute_phase3_canonical_inputs(
                    dummy_batch,
                    timings=None,
                    feature_layers=trainer.cached_feature_layers,
                )
        feature_batch_count = max_feature_batch_count

    total_elapsed = time.perf_counter() - started
    print(
        f"[phase3-cache][rank{rank}] complete "
        f"saved_episodes={saved_episodes} saved_chunks={saved_chunks} "
        f"feature_batches={feature_batch_count} "
        f"elapsed={total_elapsed:.2f}s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute Phase 3 canonical episode hidden states and JEPA distances"
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        help=(
            "Cache output directory. Defaults to phase3_precomputed_cache_dir, "
            "then <dataset_path>/inference_state_cache/<config_stem>."
        ),
    )
    parser.add_argument("--start", type=int, default=0, help="Start episode index")
    parser.add_argument("--end", type=int, help="End episode index, exclusive")
    parser.add_argument("--max-items", type=int, help="Maximum episodes before rank sharding")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--world-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Unused; kept for launcher compatibility.",
    )
    parser.add_argument("--dataset-init-workers", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--cache-feature-layers",
        help=(
            "Optional comma-separated DiT block indices to write. The model "
            "captures only these layers, and every critic feature layer must "
            "be included. Omit to preserve the existing full-layer behavior."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--standalone-ranks",
        action="store_true",
        help=(
            "When launched with torchrun, use RANK/LOCAL_RANK for task "
            "assignment but do not initialize torch.distributed. Each rank "
            "loads a full eval model on its own GPU, avoiding FSDP/NCCL "
            "collectives during long uneven precompute jobs."
        ),
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    run(args)


if __name__ == "__main__":
    init_logger()
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        raise
