from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset, Sampler


PACKED_CACHE_SCHEMA_VERSION = 1
PACKED_CACHE_FORMAT = "phase3_packed_transition_shards"


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


def _load_index(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _move_to_device(
    value: Any,
    device: torch.device,
    non_blocking: bool = True,
) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, dict):
        return {
            key: _move_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _move_to_device(item, device, non_blocking=non_blocking)
            for item in value
        )
    return value


class Phase3PackedCacheDataset(Dataset):
    """Row-level reader for Phase 3 packed transition shards."""

    def __init__(
        self,
        cache_dir: str | Path,
        critic_feature_layers: tuple[int, ...],
        expected_signature: dict[str, Any] | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        manifest_path = self.cache_dir / "manifest.json"
        index_path = self.cache_dir / "index.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing packed-cache manifest: {manifest_path}")
        if not index_path.exists():
            raise FileNotFoundError(f"Missing packed-cache index: {index_path}")

        self.manifest = _read_json(manifest_path)
        if self.manifest.get("schema_version") != PACKED_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported packed-cache schema: "
                f"{self.manifest.get('schema_version')}"
            )
        if self.manifest.get("cache_format") != PACKED_CACHE_FORMAT:
            raise ValueError(
                "Unsupported packed-cache format: "
                f"{self.manifest.get('cache_format')}"
            )
        if expected_signature is not None:
            actual = self.manifest.get("source_signature")
            if actual != expected_signature:
                raise ValueError(
                    "Packed cache signature mismatch; regenerate packed cache "
                    "for this critic config"
                )

        self.cached_feature_layers = tuple(
            int(layer) for layer in self.manifest["cached_feature_layers"]
        )
        self.critic_feature_layers = tuple(int(layer) for layer in critic_feature_layers)
        layer_to_offset = {
            layer: offset for offset, layer in enumerate(self.cached_feature_layers)
        }
        missing = [
            layer
            for layer in self.critic_feature_layers
            if layer not in layer_to_offset
        ]
        if missing:
            raise ValueError(f"Packed cache does not contain critic layers: {missing}")
        self.layer_offsets = torch.tensor(
            [layer_to_offset[layer] for layer in self.critic_feature_layers],
            dtype=torch.long,
        )
        self.feature_keys = tuple(self.manifest["feature_keys"])
        self.scalar_keys = set(
            self.manifest.get(
                "scalar_keys",
                (
                    "transition_idx",
                    "discount",
                    "state_valid",
                    "jepa_distance",
                    "cached_reward",
                    "done",
                    "dataset_id",
                    "episode_index",
                    "rl_chunk_idx",
                ),
            )
        )
        self.has_episode_metadata = bool(
            self.manifest.get(
                "has_episode_metadata",
                {
                    "episode_success",
                    "episode_chunk_count",
                }.issubset(self.scalar_keys),
            )
        )
        self.index = _load_index(index_path)
        self.shards = {
            int(item["shard_id"]): self.cache_dir / item["file"]
            for item in self.manifest["shards"]
        }
        self.positions_by_shard: dict[int, list[int]] = defaultdict(list)
        for position, row in enumerate(self.index):
            self.positions_by_shard[int(row["shard"])].append(position)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([int(index)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for output_position, index in enumerate(indices):
            row = self.index[int(index)]
            grouped[int(row["shard"])].append((output_position, row))

        samples: list[dict[str, Any] | None] = [None] * len(indices)
        for shard_id, rows in grouped.items():
            path = self.shards[shard_id]
            with safe_open(path, framework="pt", device="cpu") as tensors:
                for output_position, row in rows:
                    row_id = int(row["row"])
                    features: dict[str, torch.Tensor] = {}
                    for key in self.feature_keys:
                        value = tensors.get_slice(key)[row_id]
                        if key.endswith("_tokens") or key.endswith("_mask"):
                            value = value.index_select(0, self.layer_offsets)
                        features[key] = value.contiguous()
                    sample = {
                        "_phase3_packed_cache": True,
                        "_packed_has_episode_metadata": self.has_episode_metadata,
                        "features": features,
                        "transition_idx": tensors.get_slice("transition_idx")[row_id],
                        "discount": tensors.get_slice("discount")[row_id],
                        "state_valid": tensors.get_slice("state_valid")[row_id],
                        "jepa_distance": tensors.get_slice("jepa_distance")[row_id],
                        "cached_reward": tensors.get_slice("cached_reward")[row_id],
                        "done": tensors.get_slice("done")[row_id],
                        "dataset_id": tensors.get_slice("dataset_id")[row_id],
                        "episode_index": tensors.get_slice("episode_index")[row_id],
                        "rl_chunk_idx": tensors.get_slice("rl_chunk_idx")[row_id],
                    }
                    if self.has_episode_metadata:
                        sample["episode_success"] = tensors.get_slice(
                            "episode_success"
                        )[row_id]
                        sample["episode_chunk_count"] = tensors.get_slice(
                            "episode_chunk_count"
                        )[row_id]
                    else:
                        sample["episode_success"] = torch.tensor(False)
                        sample["episode_chunk_count"] = torch.tensor(0)
                    samples[output_position] = sample
        missing = [index for index, sample in enumerate(samples) if sample is None]
        if missing:
            raise RuntimeError(f"Packed cache failed to load rows: {missing}")
        return [sample for sample in samples if sample is not None]


def phase3_packed_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    features: dict[str, torch.Tensor] = {}
    for key in samples[0]["features"]:
        features[key] = torch.stack(
            [sample["features"][key] for sample in samples],
            dim=0,
        )
    batch: dict[str, Any] = {
        "_phase3_packed_cache": True,
        "_packed_has_episode_metadata": all(
            bool(sample.get("_packed_has_episode_metadata", False))
            for sample in samples
        ),
        "features": features,
    }
    for key in (
        "transition_idx",
        "discount",
        "state_valid",
        "jepa_distance",
        "cached_reward",
        "done",
        "dataset_id",
        "episode_index",
        "rl_chunk_idx",
        "episode_success",
        "episode_chunk_count",
    ):
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    return batch


class ShardLocalBatchSampler(Sampler[list[int]]):
    """Emit random batches while preserving locality inside packed shards."""

    def __init__(
        self,
        dataset: Phase3PackedCacheDataset,
        batch_size: int,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        shard_ids = sorted(self.dataset.positions_by_shard)
        rng.shuffle(shard_ids)
        for shard_id in shard_ids:
            positions = list(self.dataset.positions_by_shard[shard_id])
            rng.shuffle(positions)
            for start in range(0, len(positions), self.batch_size):
                batch = positions[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        rng.shuffle(batches)
        yield from batches[self.rank :: self.world_size]

    def __len__(self) -> int:
        shard_batch_count = 0
        for positions in self.dataset.positions_by_shard.values():
            if self.drop_last:
                shard_batch_count += len(positions) // self.batch_size
            else:
                shard_batch_count += math.ceil(len(positions) / self.batch_size)
        return math.ceil(shard_batch_count / max(self.world_size, 1))


class CudaPrefetcher:
    """Move the next DataLoader batch to CUDA on a side stream."""

    def __init__(self, loader, device: torch.device) -> None:
        self.loader = loader
        self.device = device

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.loader)
        next_batch = None

        def preload():
            try:
                batch = next(iterator)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                return _move_to_device(batch, self.device, non_blocking=True)

        next_batch = preload()
        while next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            batch = next_batch
            next_batch = preload()
            yield batch

    def __len__(self) -> int:
        return len(self.loader)
