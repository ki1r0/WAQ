from __future__ import annotations

import json
import hashlib
import math
import mmap
import os
import queue
import random
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset, Sampler


CANONICAL_CACHE_SCHEMA_VERSION = 2
CANONICAL_CACHE_FORMAT = "phase3_canonical_episode"
FEDERATED_CACHE_SCHEMA_VERSION = 1
FEDERATED_INDEX_SCHEMA_VERSION = 1
FEDERATED_REWARD_METADATA_SCHEMA_VERSION = 1
FEDERATED_CACHE_FORMAT = "phase3_federated_canonical_episode"
LANGUAGE_SIDECAR_SCHEMA_VERSION = 1
LANGUAGE_SIDECAR_FORMAT = "robotwin_federated_episode_text_v1"
SUCCESS_OVERLAY_SCHEMA_VERSION = 1
SUCCESS_OVERLAY_FORMAT = "phase3_episode_success_overlay_v1"
CANONICAL_FEATURE_KEYS = (
    "pred_video_tokens",
    "pred_action_tokens",
    "real_video_tokens",
    "real_action_tokens",
)
FEDERATED_REWARD_METADATA_KEYS = (
    "transition_idx",
    "jepa_distance",
    "goal_jepa_distance",
    "episode_goal_jepa_distance_max",
    "cached_reward",
    "dataset_id",
    "episode_index",
    "rl_chunk_idx",
    "done",
    "episode_success",
    "episode_chunk_count",
    "has_episode_metadata",
    "has_goal_jepa_distance",
)


def _load_success_overlay(
    overlay_path: str | Path,
    *,
    catalog_sha256: str,
    catalog: dict[str, Any],
    binding_key: str = "federated_catalog_sha256",
) -> dict[str, Any]:
    path = Path(overlay_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing episode-success overlay: {path}")
    try:
        overlay = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid episode-success overlay JSON: {path}") from error
    if not isinstance(overlay, dict):
        raise TypeError("Episode-success overlay must be a JSON object")
    if overlay.get("schema_version") != SUCCESS_OVERLAY_SCHEMA_VERSION:
        raise ValueError("Unsupported episode-success overlay schema")
    if overlay.get("format") != SUCCESS_OVERLAY_FORMAT:
        raise ValueError("Unsupported episode-success overlay format")
    if overlay.get(binding_key) != catalog_sha256:
        raise ValueError(
            "Episode-success overlay is bound to a different cache catalog"
        )
    source_hashes = {
        str(int(source["source_id"])): str(source.get("manifest_sha256", ""))
        for source in catalog.get("sources", [])
    }
    if any(not value for value in source_hashes.values()):
        raise ValueError("Catalog must bind every source to a manifest SHA256")
    declared_hashes = overlay.get("source_manifest_sha256_by_source", {})
    if declared_hashes != source_hashes:
        raise ValueError(
            "Episode-success overlay source-manifest bindings do not match "
            "the federated catalog"
        )
    if overlay.get("mode") not in {"identity", "replace"}:
        raise ValueError("Episode-success overlay mode must be identity or replace")
    defaults = overlay.get("source_defaults", {})
    if not isinstance(defaults, dict):
        raise TypeError("source_defaults must be an object")
    for source_id, value in defaults.items():
        if str(int(source_id)) not in source_hashes:
            raise ValueError(f"Overlay references unknown source_id={source_id}")
        if not isinstance(value, bool):
            raise TypeError(f"Overlay source default must be boolean: {source_id}")
    overrides = overlay.get("episode_overrides", [])
    if not isinstance(overrides, list):
        raise TypeError("episode_overrides must be a list")
    normalized = []
    seen = set()
    for item in overrides:
        if not isinstance(item, dict):
            raise TypeError("episode_overrides entries must be objects")
        source_id = int(item["source_id"])
        dataset_id = int(item["dataset_id"])
        episode_index = int(item["episode_index"])
        key = (source_id, dataset_id, episode_index)
        if key in seen:
            raise ValueError(f"Duplicate episode overlay override: {key}")
        seen.add(key)
        value = item.get("episode_success")
        if not isinstance(value, bool):
            raise TypeError(f"episode_success override must be boolean: {key}")
        normalized.append({
            "source_id": source_id,
            "dataset_id": dataset_id,
            "episode_index": episode_index,
            "episode_success": value,
        })
    result = dict(overlay)
    result["source_defaults"] = {int(key): value for key, value in defaults.items()}
    result["episode_overrides"] = normalized
    return result


def _apply_success_overlay(
    episode_success: torch.Tensor,
    *,
    source_id: int,
    dataset_id: torch.Tensor,
    episode_index: torch.Tensor,
    overlay: dict[str, Any],
) -> torch.Tensor:
    result = episode_success.bool().clone()
    if overlay["mode"] == "identity":
        return result
    defaults = overlay.get("source_defaults", {})
    default = defaults.get(str(int(source_id)), defaults.get(int(source_id)))
    if default is not None:
        result.fill_(bool(default))
    for item in overlay.get("episode_overrides", []):
        mask = (
            (dataset_id == int(item["dataset_id"]))
            & (episode_index == int(item["episode_index"]))
            & (int(item["source_id"]) == int(source_id))
        )
        result[mask] = bool(item["episode_success"])
    return result


def phase3_canonical_episode_cache_file(
    cache_dir: str | Path,
    episode_key: tuple[int, int],
) -> Path:
    dataset_id, episode_index = (int(episode_key[0]), int(episode_key[1]))
    return (
        Path(cache_dir)
        / f"dataset-{dataset_id:06d}"
        / f"episode_d{dataset_id:06d}_ep{episode_index:09d}.safetensors"
    )


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as handle:
        return json.load(handle)


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


def _pin_memory(value: Any) -> Any:
    if torch.is_tensor(value):
        return value if value.is_pinned() else value.pin_memory()
    if isinstance(value, dict):
        return {key: _pin_memory(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_pin_memory(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_pin_memory(item) for item in value)
    return value


def _record_stream(value: Any, stream: torch.cuda.Stream) -> None:
    if torch.is_tensor(value):
        value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


def _tensor_bytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _safetensors_payload_bytes(
    path: Path,
    *,
    layer_offsets: torch.Tensor | None,
) -> int:
    """Estimate the resident bytes produced by ``load_episode_payload``.

    ``safe_open`` uses the safetensors memory-mapped file backend, but
    ``get_tensor`` still materializes the selected tensors in the loader's
    process.  Disk size is compressed/serialized size and can substantially
    underestimate that resident allocation, so bounded residency decisions
    must use tensor shape*dtype bytes instead of ``stat().st_size``.
    """
    dtype_bytes = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "I16": 2,
        "I32": 4,
        "I64": 8,
        "F16": 2,
        "BF16": 2,
        "F32": 4,
        "F64": 8,
    }
    total = 0
    with safe_open(path, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            view = tensors.get_slice(key)
            shape = list(view.get_shape())
            if (
                key in CANONICAL_FEATURE_KEYS
                and len(shape) == 5
                and layer_offsets is not None
            ):
                shape[1] = int(layer_offsets.numel())
            dtype = view.get_dtype()
            if isinstance(dtype, str):
                try:
                    element_size = dtype_bytes[dtype]
                except KeyError as error:
                    raise ValueError(
                        f"Unsupported safetensors dtype {dtype!r}: {path}"
                    ) from error
            else:
                element_size = torch.empty((), dtype=dtype).element_size()
            total += math.prod(shape) * element_size
    return int(total)


def _read_tensor_range(
    tensors: Any,
    key: str,
    start: int,
    stop: int,
) -> torch.Tensor:
    """Read only the transition range from a safetensors tensor.

    Canonical episode files are laid out with transitions on dimension zero.
    ``safe_open.get_slice`` preserves that layout while avoiding materializing
    the rest of a large episode file.
    """

    value = tensors.get_slice(key)[int(start) : int(stop)]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    return value


def _flatten_video_context(
    tokens: torch.Tensor,
    frame_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 4:
        raise ValueError(f"video tokens must be [L,F,S,D], got {tokens.shape}")
    if frame_mask.ndim != 1:
        raise ValueError(f"frame_mask must be [F], got {frame_mask.shape}")
    mask = frame_mask.bool()[None, :, None].expand(
        tokens.shape[0],
        tokens.shape[1],
        tokens.shape[2],
    )
    return tokens.flatten(1, 2), mask.flatten(1, 2)


def _flatten_action_context(
    tokens: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 4:
        raise ValueError(f"action tokens must be [L,F,N,D], got {tokens.shape}")
    if action_mask.ndim != 4:
        raise ValueError(
            "action_mask must be [C,F,N,1], got "
            f"{tuple(action_mask.shape)}"
        )
    mask = action_mask.bool().squeeze(-1).any(dim=0)
    if tuple(mask.shape) != tuple(tokens.shape[1:3]):
        raise ValueError(
            f"action mask shape {tuple(mask.shape)} does not match "
            f"tokens {tuple(tokens.shape)}"
        )
    mask = mask[None].expand(tokens.shape[0], -1, -1)
    return tokens.flatten(1, 2), mask.flatten(1, 2)


def _cat_context(
    *parts: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.cat([tokens for tokens, _ in parts], dim=1),
        torch.cat([mask for _, mask in parts], dim=1),
    )


def _pad_and_stack_feature(values: list[torch.Tensor]) -> torch.Tensor:
    shapes = [tuple(value.shape) for value in values]
    if all(shape == shapes[0] for shape in shapes):
        return torch.stack(values, dim=0)
    if values[0].ndim == 3:
        layers = values[0].shape[0]
        dim = values[0].shape[2]
        max_tokens = max(value.shape[1] for value in values)
        output = values[0].new_zeros(len(values), layers, max_tokens, dim)
        for index, value in enumerate(values):
            if value.shape[0] != layers or value.shape[2] != dim:
                raise ValueError(
                    "Cannot collate variable query contexts with mismatched "
                    f"layer/feature shapes: {shapes}"
                )
            output[index, :, : value.shape[1]] = value
        return output
    if values[0].ndim == 2:
        layers = values[0].shape[0]
        max_tokens = max(value.shape[1] for value in values)
        output = values[0].new_zeros(len(values), layers, max_tokens)
        for index, value in enumerate(values):
            if value.shape[0] != layers:
                raise ValueError(
                    "Cannot collate variable query masks with mismatched "
                    f"layer shapes: {shapes}"
                )
            output[index, :, : value.shape[1]] = value
        return output
    raise ValueError(f"Cannot collate variable feature shapes: {shapes}")


def _pad_and_stack_text_tokens(values: list[torch.Tensor]) -> torch.Tensor:
    shapes = [tuple(value.shape) for value in values]
    if any(value.ndim != 2 or value.shape[-1] != 4096 for value in values):
        raise ValueError(
            "text_tokens must have shape [tokens, 4096], got "
            f"{shapes}"
        )
    dim = values[0].shape[-1]
    max_tokens = max(value.shape[0] for value in values)
    output = values[0].new_zeros(len(values), max_tokens, dim)
    for index, value in enumerate(values):
        output[index, : value.shape[0]] = value
    return output


def _pad_and_stack_text_masks(values: list[torch.Tensor]) -> torch.Tensor:
    shapes = [tuple(value.shape) for value in values]
    if any(value.ndim != 1 for value in values):
        raise ValueError(f"text_mask must have shape [tokens], got {shapes}")
    max_tokens = max(value.shape[0] for value in values)
    output = values[0].new_zeros(len(values), max_tokens, dtype=torch.bool)
    for index, value in enumerate(values):
        output[index, : value.shape[0]] = value.bool()
    return output


def _select_layers(
    tensor: torch.Tensor,
    offsets: torch.Tensor | None,
) -> torch.Tensor:
    if tensor.ndim == 5:
        if offsets is None:
            return tensor
        if offsets.numel() == tensor.shape[1] and torch.equal(
            offsets,
            torch.arange(
                tensor.shape[1],
                dtype=offsets.dtype,
                device=offsets.device,
            ),
        ):
            return tensor
        return tensor.index_select(1, offsets)
    return tensor


def _concat_layer_features(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 3:
        return tokens
    if tokens.ndim != 4:
        raise ValueError(
            "Expected MLP tokens with shape [F,S,D] or [L,F,S,D], got "
            f"{tuple(tokens.shape)}"
        )
    layers, frames, tokens_per_frame, dim = tokens.shape
    return (
        tokens.permute(1, 2, 0, 3)
        .contiguous()
        .view(frames, tokens_per_frame, layers * dim)
    )


def _resolve_layer_offsets(
    cached_feature_layers: tuple[int, ...],
    requested_layers: tuple[int, ...],
) -> torch.Tensor:
    if not cached_feature_layers:
        raise ValueError("Canonical cache manifest has no cached_feature_layers")
    layer_to_offset = {
        layer: offset for offset, layer in enumerate(cached_feature_layers)
    }
    offsets: list[int] = []
    missing: list[int] = []
    for layer in requested_layers:
        if layer in layer_to_offset:
            offsets.append(layer_to_offset[layer])
        elif layer == -1:
            offsets.append(len(cached_feature_layers) - 1)
        else:
            missing.append(layer)
    if missing:
        raise ValueError(
            f"Canonical cache does not contain critic layers: {missing}"
        )
    return torch.tensor(offsets, dtype=torch.long)


def _ensure_layer_dim(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 3:
        return tokens.unsqueeze(0)
    if tokens.ndim in {4, 5}:
        return tokens
    raise ValueError(
        "Expected [F,S,D], [L,F,S,D], or [T,L,F,S,D], got "
        f"{tuple(tokens.shape)}"
    )


class Phase3CanonicalEpisodeCacheDataset(Dataset):
    """Random-access view over canonical per-episode Phase 3 hidden-state caches."""

    def __init__(
        self,
        cache_dir: str | Path,
        critic_feature_layers: tuple[int, ...],
        *,
        critic_type: str,
        infer_latent_chunk_size: int,
        gamma: float = 0.99,
        expected_signature: dict[str, Any] | None = None,
        lru_size: int = 4,
        exclude_last_n_chunks: int = 0,
        episode_success_overlay: str | Path | None = None,
        window_size: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Missing canonical-cache manifest: {self.manifest_path}"
            )
        self.manifest = _read_json(self.manifest_path)
        if self.manifest.get("schema_version") != CANONICAL_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported canonical cache schema: "
                f"{self.manifest.get('schema_version')}"
            )
        if self.manifest.get("cache_format") != CANONICAL_CACHE_FORMAT:
            raise ValueError(
                "Unsupported canonical cache format: "
                f"{self.manifest.get('cache_format')}"
            )
        if expected_signature is not None:
            actual = self.manifest.get("signature")
            if actual != expected_signature:
                raise ValueError(
                    "Canonical Phase 3 cache signature mismatch; regenerate "
                    "the cache for this critic config"
                )

        self.critic_type = critic_type
        self.chunk_size = int(infer_latent_chunk_size)
        self.gamma = float(gamma)
        self.cached_feature_layers = tuple(
            int(layer) for layer in self.manifest["cached_feature_layers"]
        )
        self.critic_feature_layers = tuple(int(layer) for layer in critic_feature_layers)
        if self._uses_layer_context_features():
            self.layer_offsets: torch.Tensor | None = _resolve_layer_offsets(
                self.cached_feature_layers,
                self.critic_feature_layers,
            )
        else:
            self.layer_offsets = _resolve_layer_offsets(
                self.cached_feature_layers,
                self.critic_feature_layers,
            )

        self.lru_size = max(1, int(lru_size))
        if window_size is not None and int(window_size) <= 0:
            raise ValueError("window_size must be positive when provided")
        # Use the critic's existing temporal window to bound CPU-side context
        # materialization. None preserves legacy full-prefix behavior for
        # standalone dataset callers that do not have a training config.
        self.context_window_chunks = (
            None if window_size is None else int(window_size)
        )
        if exclude_last_n_chunks < 0:
            raise ValueError("exclude_last_n_chunks must be non-negative")
        self.exclude_last_n_chunks = int(exclude_last_n_chunks)
        self.manifest_sha256 = hashlib.sha256(
            self.manifest_path.read_bytes()
        ).hexdigest()
        self.success_overlay = None
        if episode_success_overlay is not None:
            self.success_overlay = _load_success_overlay(
                episode_success_overlay,
                catalog_sha256=self.manifest_sha256,
                catalog={
                    "sources": [
                        {
                            "source_id": 0,
                            "manifest_sha256": self.manifest_sha256,
                        }
                    ]
                },
                binding_key="canonical_cache_manifest_sha256",
            )
        self._episode_cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = (
            OrderedDict()
        )
        self._episode_byte_estimates: dict[tuple[int, int], int] = {}
        self.rows: list[dict[str, Any]] = []
        self.positions_by_episode: dict[tuple[int, int], list[int]] = {}
        self._build_index()

    def _uses_layer_context_features(self) -> bool:
        return self.critic_type in {
            "twin_query_transformer_v1",
            "twin_language_query_transformer_v1",
            "twin_flow_scalar_transformer_v1",
        }

    def _episode_files(self) -> list[Path]:
        return sorted(self.cache_dir.glob("dataset-*/episode_d*_ep*.safetensors"))

    def _build_index(self) -> None:
        files = self._episode_files()
        if not files:
            raise FileNotFoundError(
                f"No canonical episode cache files found under {self.cache_dir}"
            )
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as tensors:
                keys = set(tensors.keys())
                missing = set(CANONICAL_FEATURE_KEYS).difference(keys)
                if missing:
                    raise KeyError(
                        f"Canonical episode cache {path} is missing {sorted(missing)}"
                    )
                transition_idx = tensors.get_tensor("transition_idx")
                dataset_id = tensors.get_tensor("dataset_id")
                episode_index = tensors.get_tensor("episode_index")
                rl_chunk_idx = tensors.get_tensor("rl_chunk_idx")
                episode_chunk_count = tensors.get_tensor("episode_chunk_count")
                count = int(transition_idx.shape[0])
                for row_id in range(count):
                    if (
                        self.exclude_last_n_chunks > 0
                        and int(rl_chunk_idx[row_id])
                        >= max(
                            0,
                            int(episode_chunk_count[row_id])
                            - self.exclude_last_n_chunks,
                        )
                    ):
                        continue
                    episode_key = (
                        int(dataset_id[row_id]),
                        int(episode_index[row_id]),
                    )
                    position = len(self.rows)
                    self.rows.append(
                        {
                            "path": path,
                            "row": row_id,
                            "transition_idx": int(transition_idx[row_id]),
                            "episode_key": episode_key,
                            "rl_chunk_idx": int(rl_chunk_idx[row_id]),
                        }
                    )
                    self.positions_by_episode.setdefault(episode_key, []).append(
                        position
                    )
        if not self.rows:
            raise ValueError(
                "Canonical episode cache has no trainable rows after applying "
                f"exclude_last_n_chunks={self.exclude_last_n_chunks}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def _load_episode(
        self,
        episode_key: tuple[int, int],
        path: Path,
    ) -> dict[str, torch.Tensor]:
        cached = self._episode_cache.get(episode_key)
        if cached is not None:
            self._episode_cache.move_to_end(episode_key)
            return cached
        payload = self.load_episode_payload(path)
        self._episode_cache[episode_key] = payload
        self._episode_cache.move_to_end(episode_key)
        while len(self._episode_cache) > self.lru_size:
            self._episode_cache.popitem(last=False)
        return payload

    def load_episode_payload(
        self,
        path: Path,
        *,
        materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            payload: dict[str, torch.Tensor] = {}
            for key in tensors.keys():
                value = tensors.get_tensor(key)
                if key in CANONICAL_FEATURE_KEYS:
                    value = _select_layers(value, self.layer_offsets)
                    if not value.is_contiguous():
                        value = value.contiguous()
                if materialize:
                    value = value.clone()
                payload[key] = value
        if self.success_overlay is not None:
            payload["episode_success"] = _apply_success_overlay(
                payload.get(
                    "episode_success",
                    torch.zeros_like(payload["done"], dtype=torch.bool),
                ),
                source_id=0,
                dataset_id=payload["dataset_id"],
                episode_index=payload["episode_index"],
                overlay=self.success_overlay,
            )
        return payload

    def load_episode_metadata(self, path: Path) -> dict[str, torch.Tensor]:
        with safe_open(path, framework="pt", device="cpu") as tensors:
            payload = {
                key: tensors.get_tensor(key)
                for key in tensors.keys()
                if key not in CANONICAL_FEATURE_KEYS
            }
        if self.success_overlay is not None:
            payload["episode_success"] = _apply_success_overlay(
                payload.get(
                    "episode_success",
                    torch.zeros_like(payload["done"], dtype=torch.bool),
                ),
                source_id=0,
                dataset_id=payload["dataset_id"],
                episode_index=payload["episode_index"],
                overlay=self.success_overlay,
            )
        return payload

    def load_episode_range_payload(
        self,
        path: Path,
        start: int,
        stop: int,
        *,
        materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load transitions ``[start:stop]`` without reading the full episode."""

        if start < 0 or stop <= start:
            raise ValueError(f"invalid episode range [{start}, {stop})")
        with safe_open(path, framework="pt", device="cpu") as tensors:
            payload: dict[str, torch.Tensor] = {}
            for key in tensors.keys():
                value = _read_tensor_range(tensors, key, start, stop)
                if key in CANONICAL_FEATURE_KEYS:
                    value = _select_layers(value, self.layer_offsets)
                    if not value.is_contiguous():
                        value = value.contiguous()
                if materialize:
                    value = value.clone()
                payload[key] = value
        if self.success_overlay is not None:
            payload["episode_success"] = _apply_success_overlay(
                payload.get(
                    "episode_success",
                    torch.zeros_like(payload["done"], dtype=torch.bool),
                ),
                source_id=0,
                dataset_id=payload["dataset_id"],
                episode_index=payload["episode_index"],
                overlay=self.success_overlay,
            )
        return payload

    def episode_key_for_position(self, position: int) -> tuple[int, int]:
        return self.rows[int(position)]["episode_key"]

    def episode_path(self, episode_key: tuple[int, int]) -> Path:
        positions = self.positions_by_episode[episode_key]
        return self.rows[positions[0]]["path"]

    def episode_payload_bytes(self, episode_key: tuple[int, int]) -> int:
        """Return the materialized CPU payload size for one episode.

        The result is cached because group admission consults the same files
        repeatedly as the sampler advances.  This is metadata-only: it does
        not read the feature payloads.
        """
        cached = self._episode_byte_estimates.get(episode_key)
        if cached is not None:
            return cached
        value = _safetensors_payload_bytes(
            self.episode_path(episode_key),
            layer_offsets=self.layer_offsets,
        )
        self._episode_byte_estimates[episode_key] = value
        return value

    def sample_from_payload(
        self,
        payload: dict[str, torch.Tensor],
        position: int,
    ) -> dict[str, Any]:
        row = self.rows[int(position)]
        return self._sample_from_episode(payload, int(row["row"]))

    def sample_from_payload_range(
        self,
        payload: dict[str, torch.Tensor],
        position: int,
        range_start: int,
    ) -> dict[str, Any]:
        """Sample a row from a payload loaded for ``[range_start, range_end)``."""

        row = self.rows[int(position)]
        local_row = int(row["row"]) - int(range_start)
        if local_row < 0 or local_row >= int(payload["done"].shape[0]):
            raise IndexError(
                f"row {row['row']} is outside loaded range starting at "
                f"{range_start}"
            )
        return self._sample_from_episode(payload, local_row)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.__getitems__([int(index)])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, Any]]:
        samples: list[dict[str, Any]] = []
        for index in indices:
            row = self.rows[int(index)]
            episode = self._load_episode(row["episode_key"], row["path"])
            samples.append(self._sample_from_episode(episode, int(row["row"])))
        return samples

    def _sample_from_episode(
        self,
        episode: dict[str, torch.Tensor],
        row: int,
    ) -> dict[str, Any]:
        pred_action = episode["pred_action_tokens"][row]
        real_video = episode["real_video_tokens"][row]
        current_latent_mask = episode["latents_mask"][row].bool()
        current_action_mask = episode["actions_mask"][row].bool()
        state_valid = episode["state_valid"][row].bool()
        if bool(state_valid):
            previous_row = max(0, row - 1)
            previous_real_video = episode["real_video_tokens"][previous_row]
            previous_latent_mask = episode["latents_mask"][previous_row].bool()
            previous_action_mask = episode["actions_mask"][previous_row].bool()
        else:
            previous_real_video = torch.zeros_like(real_video)
            previous_latent_mask = torch.zeros_like(current_latent_mask)
            previous_action_mask = torch.zeros_like(current_action_mask)

        if self._uses_layer_context_features():
            if bool(state_valid):
                context_start = (
                    0
                    if self.context_window_chunks is None
                    else max(0, row - self.context_window_chunks)
                )
                prefix_video = _ensure_layer_dim(
                    episode["real_video_tokens"][context_start:row]
                )
                prefix_latent_mask = episode["latents_mask"][
                    context_start:row
                ].bool()
            else:
                prefix_video = _ensure_layer_dim(previous_real_video).unsqueeze(0)
                prefix_latent_mask = previous_latent_mask.unsqueeze(0)
            target_context_start = (
                0
                if self.context_window_chunks is None
                else max(0, row + 1 - self.context_window_chunks)
            )
            target_prefix_video = _ensure_layer_dim(
                episode["real_video_tokens"][target_context_start : row + 1]
            )
            target_prefix_latent_mask = episode["latents_mask"][
                target_context_start : row + 1
            ].bool()
            features = self._build_context_features(
                _ensure_layer_dim(pred_action),
                prefix_video,
                target_prefix_video,
                prefix_latent_mask,
                target_prefix_latent_mask,
                current_action_mask,
            )
        else:
            features = self._build_mlp_features(
                pred_action,
                previous_real_video,
                real_video,
            )
        if "text_tokens" in episode:
            features["text_tokens"] = episode["text_tokens"]
            features["text_mask"] = episode["text_mask"]

        done = episode["done"][row].bool()
        discount = torch.tensor(
            0.0 if bool(done) else self.gamma,
            dtype=torch.float32,
        )

        sample: dict[str, Any] = {
            "_phase3_canonical_cache": True,
            "features": features,
            "transition_idx": episode["transition_idx"][row],
            "discount": discount,
            "state_valid": episode["state_valid"][row].bool(),
            "jepa_distance": episode["jepa_distance"][row].float(),
            "goal_jepa_distance": episode.get(
                "goal_jepa_distance",
                torch.zeros_like(episode["jepa_distance"]),
            )[row].float(),
            "episode_goal_jepa_distance_max": episode.get(
                "episode_goal_jepa_distance_max",
                torch.ones_like(episode["jepa_distance"]),
            )[row].float(),
            "cached_reward": episode["cached_reward"][row].float(),
            "done": done,
            "dataset_id": episode["dataset_id"][row],
            "episode_index": episode["episode_index"][row],
            "rl_chunk_idx": episode["rl_chunk_idx"][row],
            "episode_success": episode["episode_success"][row].bool(),
            "episode_chunk_count": episode["episode_chunk_count"][row],
            "previous_latents_mask": previous_latent_mask,
            "latents_mask": current_latent_mask,
            "previous_actions_mask": previous_action_mask,
            "actions_mask": current_action_mask,
        }
        return sample

    def _build_context_features(
        self,
        pred_action: torch.Tensor,
        prefix_real_video: torch.Tensor,
        target_prefix_real_video: torch.Tensor,
        prefix_latent_mask: torch.Tensor,
        target_prefix_latent_mask: torch.Tensor,
        current_action_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        chunk = self.chunk_size
        prefix_layers = prefix_real_video.shape[1]
        prefix_frames = prefix_real_video.shape[0] * prefix_real_video.shape[2]
        value_video = prefix_real_video.permute(1, 0, 2, 3, 4).reshape(
            prefix_layers,
            prefix_frames,
            prefix_real_video.shape[3],
            prefix_real_video.shape[4],
        )
        value_mask = prefix_latent_mask.reshape(-1)
        target_layers = target_prefix_real_video.shape[1]
        target_frames = (
            target_prefix_real_video.shape[0] * target_prefix_real_video.shape[2]
        )
        target_video = target_prefix_real_video.permute(1, 0, 2, 3, 4).reshape(
            target_layers,
            target_frames,
            target_prefix_real_video.shape[3],
            target_prefix_real_video.shape[4],
        )
        target_mask = target_prefix_latent_mask.reshape(-1)
        value_context = _flatten_video_context(value_video, value_mask)
        q_context = _cat_context(
            value_context,
            _flatten_action_context(pred_action[:, -chunk:], current_action_mask),
        )
        target_value_context = _flatten_video_context(target_video, target_mask)
        return {
            "value_context_tokens": value_context[0],
            "value_context_mask": value_context[1],
            "q_context_tokens": q_context[0],
            "q_context_mask": q_context[1],
            "target_value_context_tokens": target_value_context[0],
            "target_value_context_mask": target_value_context[1],
        }

    def _build_mlp_features(
        self,
        pred_action: torch.Tensor,
        previous_real_video: torch.Tensor,
        real_video: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pred_action = _concat_layer_features(pred_action)
        previous_real_video = _concat_layer_features(previous_real_video)
        real_video = _concat_layer_features(real_video)
        chunk = self.chunk_size
        return {
            "previous_video_tokens": previous_real_video,
            "action_tokens": pred_action[-chunk:],
            "current_real_video_tokens": real_video,
        }


@dataclass(frozen=True)
class FederatedCanonicalCacheSource:
    source_id: int
    task_name: str
    cache_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    cached_feature_layers: tuple[int, ...]
    layer_offsets: torch.Tensor
    dataset_id_map: dict[int, int]
    transition_idx_offset: int
    dataset_size: int
    episode_files: tuple[Path, ...]
    artifacts_manifest_path: Path | None = None
    artifacts_root: Path | None = None


def _scan_federated_source_reward_metadata(
    job: tuple[FederatedCanonicalCacheSource, tuple[int, ...]],
) -> dict[str, torch.Tensor]:
    source, indexed_global_offsets = job
    if len(indexed_global_offsets) != len(source.episode_files):
        raise ValueError(
            f"Source {source.source_id} sampler path count mismatch: "
            f"offsets={len(indexed_global_offsets)} "
            f"files={len(source.episode_files)}"
        )

    parts: dict[str, list[torch.Tensor]] = {
        key: [] for key in FEDERATED_REWARD_METADATA_KEYS
    }
    source_row_offset = 0
    for path, indexed_global_offset in zip(
        source.episode_files, indexed_global_offsets
    ):
        expected_global_offset = source.transition_idx_offset + source_row_offset
        if indexed_global_offset != expected_global_offset:
            raise ValueError(
                "Federated sampler offset mismatch for "
                f"{path}: index={indexed_global_offset} "
                f"expected={expected_global_offset}"
            )
        with safe_open(path, framework="pt", device="cpu") as tensors:
            tensor_keys = set(tensors.keys())
            required = {
                "transition_idx",
                "jepa_distance",
                "cached_reward",
                "dataset_id",
                "episode_index",
                "rl_chunk_idx",
                "done",
            }
            missing = required.difference(tensor_keys)
            if missing:
                raise KeyError(
                    f"Canonical episode cache {path} is missing "
                    f"reward metadata {sorted(missing)}"
                )
            transition_idx = tensors.get_tensor("transition_idx")
            jepa_distance = tensors.get_tensor("jepa_distance").float()
            cached_reward = tensors.get_tensor("cached_reward").float()
            dataset_id = tensors.get_tensor("dataset_id")
            episode_index = tensors.get_tensor("episode_index")
            rl_chunk_idx = tensors.get_tensor("rl_chunk_idx")
            done = tensors.get_tensor("done").bool()
            count = int(transition_idx.shape[0])
            has_episode_metadata = (
                "episode_success" in tensor_keys
                and "episode_chunk_count" in tensor_keys
            )
            has_goal_jepa_distance = (
                "goal_jepa_distance" in tensor_keys
                and "episode_goal_jepa_distance_max" in tensor_keys
            )
            if has_goal_jepa_distance:
                goal_jepa_distance = tensors.get_tensor(
                    "goal_jepa_distance"
                ).float()
                goal_jepa_distance_max = tensors.get_tensor(
                    "episode_goal_jepa_distance_max"
                ).float()
            else:
                goal_jepa_distance = torch.zeros_like(jepa_distance)
                goal_jepa_distance_max = torch.ones_like(jepa_distance)
            if has_episode_metadata:
                episode_success = tensors.get_tensor("episode_success").bool()
                episode_chunk_count = tensors.get_tensor("episode_chunk_count")
            else:
                episode_success = torch.zeros_like(done, dtype=torch.bool)
                episode_chunk_count = torch.zeros_like(rl_chunk_idx)

            remapped_dataset_id = torch.empty_like(dataset_id)
            for local_id in dataset_id.unique().tolist():
                local_id = int(local_id)
                if local_id not in source.dataset_id_map:
                    raise KeyError(
                        f"Source {source.source_id} has no mapping for "
                        f"local dataset ID {local_id}"
                    )
                remapped_dataset_id[dataset_id == local_id] = (
                    source.dataset_id_map[local_id]
                )

            metadata_values = {
                "transition_idx": torch.arange(
                    expected_global_offset,
                    expected_global_offset + count,
                    dtype=torch.long,
                ),
                "jepa_distance": jepa_distance,
                "goal_jepa_distance": goal_jepa_distance,
                "episode_goal_jepa_distance_max": goal_jepa_distance_max,
                "cached_reward": cached_reward,
                "dataset_id": remapped_dataset_id,
                "episode_index": episode_index,
                "rl_chunk_idx": rl_chunk_idx,
                "done": done,
                "episode_success": episode_success,
                "episode_chunk_count": episode_chunk_count,
                "has_episode_metadata": torch.full(
                    (count,), has_episode_metadata, dtype=torch.bool
                ),
                "has_goal_jepa_distance": torch.full(
                    (count,), has_goal_jepa_distance, dtype=torch.bool
                ),
            }
            for key, value in metadata_values.items():
                if value.ndim != 1 or int(value.shape[0]) != count:
                    raise ValueError(
                        "Federated reward metadata shape mismatch for "
                        f"{path}: {key}={tuple(value.shape)} count={count}"
                    )
                parts[key].append(value)
        source_row_offset += count

    if source_row_offset != source.dataset_size:
        raise ValueError(
            f"Source {source.source_id} reward row count mismatch: "
            f"files={source_row_offset} manifest={source.dataset_size}"
        )
    return {key: torch.cat(values, dim=0) for key, values in parts.items()}


class Phase3FederatedCanonicalEpisodeCacheDataset(
    Phase3CanonicalEpisodeCacheDataset
):
    """Catalog-backed view over multiple canonical episode-cache roots."""

    def __init__(
        self,
        catalog_path: str | Path,
        critic_feature_layers: tuple[int, ...],
        *,
        critic_type: str,
        infer_latent_chunk_size: int,
        gamma: float = 0.99,
        lru_size: int = 4,
        exclude_last_n_chunks: int = 0,
        episode_success_overlay: str | Path | None = None,
        language_sidecar_catalog: str | Path | None = None,
        artifacts_manifest: str | Path | None = None,
        window_size: int | None = None,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        if not self.catalog_path.is_file():
            raise FileNotFoundError(
                f"Missing federated canonical-cache catalog: {self.catalog_path}"
            )
        catalog_bytes = self.catalog_path.read_bytes()
        self.catalog_sha256 = hashlib.sha256(catalog_bytes).hexdigest()
        self.catalog = json.loads(catalog_bytes)
        self.artifacts_manifest_path = (
            Path(artifacts_manifest).resolve()
            if artifacts_manifest is not None
            else None
        )
        self._artifact_manifest_by_source = self._load_artifact_manifest_overrides()
        self.success_overlay = None
        if episode_success_overlay is not None:
            self.success_overlay = _load_success_overlay(
                episode_success_overlay,
                catalog_sha256=self.catalog_sha256,
                catalog=self.catalog,
            )
        if self.catalog.get("schema_version") != FEDERATED_CACHE_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported federated cache schema: "
                f"{self.catalog.get('schema_version')}"
            )
        if self.catalog.get("cache_format") != FEDERATED_CACHE_FORMAT:
            raise ValueError(
                "Unsupported federated cache format: "
                f"{self.catalog.get('cache_format')}"
            )

        self.cache_dir = self.catalog_path.parent
        self.manifest_path = self.catalog_path
        self.manifest = self.catalog
        self.critic_type = critic_type
        self.chunk_size = int(infer_latent_chunk_size)
        self.gamma = float(gamma)
        self.critic_feature_layers = tuple(
            int(layer) for layer in critic_feature_layers
        )
        self.cached_feature_layers = self.critic_feature_layers
        self.layer_offsets = None
        self.lru_size = max(1, int(lru_size))
        if window_size is not None and int(window_size) <= 0:
            raise ValueError("window_size must be positive when provided")
        self.context_window_chunks = (
            None if window_size is None else int(window_size)
        )
        if exclude_last_n_chunks < 0:
            raise ValueError("exclude_last_n_chunks must be non-negative")
        self.exclude_last_n_chunks = int(exclude_last_n_chunks)
        self._episode_cache: OrderedDict[
            tuple[int, int], dict[str, torch.Tensor]
        ] = OrderedDict()
        self._episode_byte_estimates: dict[tuple[int, int], int] = {}
        self.rows: list[dict[str, Any]] = []
        self.positions_by_episode: dict[tuple[int, int], list[int]] = {}
        self._transition_row_offset_by_path: dict[Path, int] = {}
        self._reward_metadata: dict[str, torch.Tensor] = {}
        self.language_sidecar_catalog_path = (
            Path(language_sidecar_catalog)
            if language_sidecar_catalog is not None
            else None
        )
        self._language_sidecar_by_dataset: dict[tuple[int, int], Path] = {}
        self.language_max_tokens = 0
        self.language_sidecar_catalog_sha256: str | None = None
        self.sources = self._load_sources()
        if self.language_sidecar_catalog_path is not None:
            self._load_language_sidecar_catalog()
        self._source_by_path = {
            path: source
            for source in self.sources
            for path in source.episode_files
        }
        self._load_or_build_sampler_index()
        self._load_or_build_reward_metadata()

    def _load_language_sidecar_catalog(self) -> None:
        assert self.language_sidecar_catalog_path is not None
        if not self.language_sidecar_catalog_path.is_file():
            raise FileNotFoundError(f"Missing language sidecar catalog: {self.language_sidecar_catalog_path}")
        manifest_bytes = self.language_sidecar_catalog_path.read_bytes()
        self.language_sidecar_catalog_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        if manifest.get("schema_version") != LANGUAGE_SIDECAR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported language sidecar schema: {manifest.get('schema_version')}")
        if manifest.get("format") != LANGUAGE_SIDECAR_FORMAT:
            raise ValueError(f"Unsupported language sidecar format: {manifest.get('format')}")
        if manifest.get("federated_catalog_sha256") != self.catalog_sha256:
            raise ValueError("Language sidecar was not built from the configured federated cache catalog")
        self.language_max_tokens = int(manifest.get("max_tokens", 0))
        if self.language_max_tokens <= 0:
            raise ValueError("Language sidecar max_tokens must be positive")
        root = self.language_sidecar_catalog_path.parent
        for source in manifest.get("sources", []):
            source_id = int(source["source_id"])
            for dataset in source.get("datasets", []):
                key = (source_id, int(dataset["local_dataset_id"]))
                path = root / dataset["file"]
                if key in self._language_sidecar_by_dataset:
                    raise ValueError(f"Duplicate language sidecar mapping: {key}")
                if not path.is_file():
                    raise FileNotFoundError(f"Missing language sidecar tensor file: {path}")
                self._language_sidecar_by_dataset[key] = path

    def _load_episode_language(self, source, payload):
        dataset_ids = payload["dataset_id"].unique()
        episode_indices = payload["episode_index"].unique()
        if dataset_ids.numel() != 1 or episode_indices.numel() != 1:
            raise ValueError("Canonical episode payload has mixed episode identity")
        key = (source.source_id, int(dataset_ids.item()))
        try:
            sidecar_path = self._language_sidecar_by_dataset[key]
        except KeyError as error:
            raise KeyError(f"No language sidecar for source/dataset {key}") from error
        with safe_open(sidecar_path, framework="pt", device="cpu") as tensors:
            text_tokens = tensors.get_slice("text_tokens")[int(episode_indices.item())]
            text_mask = tensors.get_slice("text_mask")[int(episode_indices.item())]
        if text_tokens.ndim != 2 or text_tokens.shape[-1] != 4096:
            raise ValueError(f"Invalid language token shape {tuple(text_tokens.shape)}")
        if text_mask.shape != text_tokens.shape[:1]:
            raise ValueError(f"Invalid language mask shape {tuple(text_mask.shape)}")
        return text_tokens, text_mask.bool()

    def _sampler_index_cache_path(self) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {
                    "catalog_sha256": self.catalog_sha256,
                    "critic_feature_layers": self.critic_feature_layers,
                    "exclude_last_n_chunks": self.exclude_last_n_chunks,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        return self.catalog_path.with_name(f".federated_index_{key}.pt")

    def _index_cache_path(self) -> Path:
        return self._sampler_index_cache_path()

    def _reward_metadata_cache_path(self) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {
                    "reward_metadata_schema_version": (
                        FEDERATED_REWARD_METADATA_SCHEMA_VERSION
                    ),
                    "catalog_sha256": self.catalog_sha256,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        return self.catalog_path.with_name(
            f".federated_reward_metadata_{key}.pt"
        )

    def _sampler_index_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FEDERATED_INDEX_SCHEMA_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "critic_feature_layers": self.critic_feature_layers,
            "exclude_last_n_chunks": self.exclude_last_n_chunks,
            "rows": self.rows,
            "positions_by_episode": self.positions_by_episode,
            "transition_row_offset_by_path": self._transition_row_offset_by_path,
        }

    def _restore_sampler_index_payload(self, payload: dict[str, Any]) -> bool:
        if (
            payload.get("schema_version") != FEDERATED_INDEX_SCHEMA_VERSION
            or payload.get("catalog_sha256") != self.catalog_sha256
            or tuple(payload.get("critic_feature_layers", ()))
            != self.critic_feature_layers
            or payload.get("exclude_last_n_chunks") != self.exclude_last_n_chunks
        ):
            return False
        try:
            self.rows = payload["rows"]
            self.positions_by_episode = payload["positions_by_episode"]
            self._transition_row_offset_by_path = payload[
                "transition_row_offset_by_path"
            ]
        except KeyError:
            return False
        expected_count = sum(int(source.dataset_size) for source in self.sources)
        if len(self._transition_row_offset_by_path) != sum(
            len(source.episode_files) for source in self.sources
        ):
            return False
        if not self.rows or len(self.rows) > expected_count:
            return False
        if any(
            int(row["transition_idx"])
            != int(self._transition_row_offset_by_path[Path(row["path"])])
            + int(row["row"])
            for row in self.rows
        ):
            return False
        return True

    def _v2_artifact_paths(
        self, source: FederatedCanonicalCacheSource
    ) -> tuple[Path, Path] | None:
        if source.artifacts_manifest_path is None:
            return None
        manifest_path = source.artifacts_manifest_path
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing v2 source artifacts manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("cache_pipeline_version") != "canonical_precompute_v2":
            raise ValueError(f"Artifact manifest is not v2: {manifest_path}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError(f"v2 manifest has no artifacts section: {manifest_path}")
        try:
            index_path = Path(artifacts["sampler_index"])
            metadata_path = Path(artifacts["reward_metadata"])
        except KeyError as error:
            raise ValueError(f"v2 manifest is missing artifact path: {manifest_path}") from error
        root = manifest_path.parent
        if not index_path.is_absolute():
            index_path = root / index_path
        if not metadata_path.is_absolute():
            metadata_path = root / metadata_path
        return index_path.resolve(), metadata_path.resolve()

    def _load_v2_sampler_index(self) -> bool:
        artifact_paths = [self._v2_artifact_paths(source) for source in self.sources]
        if not any(path is not None for path in artifact_paths):
            return False
        if not all(path is not None for path in artifact_paths):
            raise ValueError("Either all or none of federated sources must declare v2 artifacts")
        self.rows = []
        self.positions_by_episode = {}
        self._transition_row_offset_by_path = {}
        for source, paths in zip(self.sources, artifact_paths):
            assert paths is not None
            index_path, _ = paths
            payload = torch.load(index_path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != 2 or payload.get("format") != "phase3_federated_sampler_index_v2":
                raise ValueError(f"Unsupported v2 sampler index: {index_path}")
            episode_ptr = payload["episode_ptr"].long()
            episode_paths = payload["episode_paths"]
            episode_keys = payload["episode_keys"]
            cache_rows = payload["cache_row"].long()
            rl_chunk_idx = payload["rl_chunk_idx"].long()
            episode_chunk_count = payload["episode_chunk_count"].long()
            dataset_ids = payload["dataset_id"].long()
            episode_indices = payload["episode_index"].long()
            if episode_ptr.ndim != 1 or int(episode_ptr[-1]) != int(cache_rows.shape[0]):
                raise ValueError(f"Invalid v2 episode_ptr in {index_path}")
            if len(episode_paths) != len(episode_keys) or len(episode_ptr) != len(episode_keys) + 1:
                raise ValueError(f"Invalid v2 episode table in {index_path}")
            source_row_offset = 0
            for episode_number, raw_key in enumerate(episode_keys):
                local_dataset_id, episode_index = (int(raw_key[0]), int(raw_key[1]))
                if local_dataset_id not in source.dataset_id_map:
                    raise KeyError(f"Unknown local dataset ID in v2 index: {local_dataset_id}")
                episode_key = (source.dataset_id_map[local_dataset_id], episode_index)
                path = (source.artifacts_root / episode_paths[episode_number]).resolve()
                if path not in source.episode_files:
                    raise ValueError(f"v2 index path is not in source cache: {path}")
                start = int(episode_ptr[episode_number])
                stop = int(episode_ptr[episode_number + 1])
                self._transition_row_offset_by_path[path] = source.transition_idx_offset + source_row_offset
                for row in range(start, stop):
                    if (
                        self.exclude_last_n_chunks > 0
                        and int(rl_chunk_idx[row])
                        >= max(0, int(episode_chunk_count[row]) - self.exclude_last_n_chunks)
                    ):
                        continue
                    position = len(self.rows)
                    self.rows.append(
                        {
                            "path": path,
                            "row": int(cache_rows[row]),
                            "transition_idx": source.transition_idx_offset + source_row_offset + int(cache_rows[row]),
                            "episode_key": episode_key,
                            "rl_chunk_idx": int(rl_chunk_idx[row]),
                            "source_id": source.source_id,
                        }
                    )
                    self.positions_by_episode.setdefault(episode_key, []).append(position)
                source_row_offset += stop - start
            if source_row_offset != source.dataset_size:
                raise ValueError(
                    f"v2 sampler row count mismatch for source {source.source_id}: "
                    f"index={source_row_offset} catalog={source.dataset_size}"
                )
        if not self.rows:
            raise ValueError("v2 sampler index has no eligible rows")
        return True

    def _load_or_build_sampler_index(self) -> None:
        """Restore the offset-based sampler index without an NCCL collective."""
        if self._load_v2_sampler_index():
            return
        index_path = self._sampler_index_cache_path()
        if index_path.is_file():
            try:
                payload = torch.load(index_path, map_location="cpu", weights_only=False)
                if self._restore_sampler_index_payload(payload):
                    return
            except (OSError, RuntimeError, ValueError):
                pass

        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        is_builder = not distributed or torch.distributed.get_rank() == 0
        if not is_builder:
            deadline = time.monotonic() + 1800.0
            while time.monotonic() < deadline:
                if index_path.is_file():
                    try:
                        payload = torch.load(
                            index_path, map_location="cpu", weights_only=False
                        )
                        if self._restore_sampler_index_payload(payload):
                            return
                    except (OSError, RuntimeError, ValueError):
                        pass
                time.sleep(0.25)
            raise TimeoutError(
                f"Timed out waiting for federated index: {index_path}"
            )

        self._build_sampler_index()
        payload = self._sampler_index_payload()
        temporary_path = index_path.with_name(
            f".{index_path.name}.{os.getpid()}.tmp"
        )
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, index_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _reward_metadata_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FEDERATED_REWARD_METADATA_SCHEMA_VERSION,
            "catalog_sha256": self.catalog_sha256,
            "reward_metadata": self._reward_metadata,
        }

    def _restore_reward_metadata_payload(self, payload: dict[str, Any]) -> bool:
        if (
            payload.get("schema_version")
            != FEDERATED_REWARD_METADATA_SCHEMA_VERSION
            or payload.get("catalog_sha256") != self.catalog_sha256
        ):
            return False
        reward_metadata = payload.get("reward_metadata")
        if not isinstance(reward_metadata, dict):
            return False
        if set(reward_metadata) != set(FEDERATED_REWARD_METADATA_KEYS):
            return False
        expected_count = sum(int(source.dataset_size) for source in self.sources)
        if expected_count <= 0 or any(
            not torch.is_tensor(value)
            or value.ndim != 1
            or int(value.shape[0]) != expected_count
            for value in reward_metadata.values()
        ):
            return False
        expected_transition_idx = torch.cat(
            [
                torch.arange(
                    source.transition_idx_offset,
                    source.transition_idx_offset + source.dataset_size,
                    dtype=torch.long,
                )
                for source in self.sources
            ]
        )
        if not torch.equal(
            reward_metadata["transition_idx"], expected_transition_idx
        ):
            return False
        self._reward_metadata = reward_metadata
        return True

    def _load_v2_reward_metadata(self) -> bool:
        artifact_paths = [self._v2_artifact_paths(source) for source in self.sources]
        if not any(path is not None for path in artifact_paths):
            return False
        if not all(path is not None for path in artifact_paths):
            raise ValueError("Either all or none of federated sources must declare v2 artifacts")
        parts = {key: [] for key in FEDERATED_REWARD_METADATA_KEYS}
        for source, paths in zip(self.sources, artifact_paths):
            assert paths is not None
            _, metadata_path = paths
            payload = torch.load(metadata_path, map_location="cpu", weights_only=False)
            if payload.get("schema_version") != 1 or payload.get("format") != "phase3_federated_reward_metadata_v1":
                raise ValueError(f"Unsupported v2 reward metadata: {metadata_path}")
            metadata = payload.get("reward_metadata")
            if not isinstance(metadata, dict) or set(metadata) != set(FEDERATED_REWARD_METADATA_KEYS):
                raise ValueError(f"Invalid v2 reward metadata keys: {metadata_path}")
            count = int(source.dataset_size)
            if any(not torch.is_tensor(value) or value.ndim != 1 or int(value.shape[0]) != count for value in metadata.values()):
                raise ValueError(f"v2 reward metadata row count mismatch: {metadata_path}")
            metadata = {key: value.cpu() for key, value in metadata.items()}
            metadata["transition_idx"] = torch.arange(
                source.transition_idx_offset,
                source.transition_idx_offset + count,
                dtype=torch.long,
            )
            metadata["source_dataset_id"] = metadata["dataset_id"].long()
            metadata["dataset_id"] = self._remap_dataset_ids(metadata["dataset_id"], source)
            for key in FEDERATED_REWARD_METADATA_KEYS:
                parts[key].append(metadata[key])
        self._reward_metadata = {
            key: torch.cat(values, dim=0) for key, values in parts.items()
        }
        return True

    def _load_or_build_reward_metadata(self) -> None:
        """Build reward-only metadata separately from the sampler index."""
        if self._load_v2_reward_metadata():
            self._apply_success_overlay_to_reward_metadata()
            return
        metadata_path = self._reward_metadata_cache_path()
        if metadata_path.is_file():
            try:
                payload = torch.load(
                    metadata_path, map_location="cpu", weights_only=False
                )
                if self._restore_reward_metadata_payload(payload):
                    self._apply_success_overlay_to_reward_metadata()
                    return
            except (OSError, RuntimeError, ValueError):
                pass

        distributed = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        is_builder = not distributed or torch.distributed.get_rank() == 0
        if not is_builder:
            deadline = time.monotonic() + 1800.0
            while time.monotonic() < deadline:
                if metadata_path.is_file():
                    try:
                        payload = torch.load(
                            metadata_path, map_location="cpu", weights_only=False
                        )
                        if self._restore_reward_metadata_payload(payload):
                            return
                    except (OSError, RuntimeError, ValueError):
                        pass
                time.sleep(0.25)
            raise TimeoutError(
                f"Timed out waiting for federated reward metadata: {metadata_path}"
            )

        self._build_reward_metadata()
        payload = self._reward_metadata_payload()
        temporary_path = metadata_path.with_name(
            f".{metadata_path.name}.{os.getpid()}.tmp"
        )
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, metadata_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_artifact_manifest_overrides(self) -> dict[int, Path]:
        """Resolve optional explicit v2 artifact manifests from the config."""

        if self.artifacts_manifest_path is None:
            return {}
        path = self.artifacts_manifest_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing Phase 3 v2 artifacts manifest: {path}")
        payload = json.loads(path.read_text())
        if payload.get("cache_pipeline_version") == "canonical_precompute_v2":
            source_id = payload.get("source_id")
            if source_id is None:
                source_ids = [int(item["source_id"]) for item in self.catalog["sources"]]
                if len(source_ids) != 1:
                    raise ValueError(
                        "A single-source v2 artifacts manifest cannot be used "
                        "for a multi-source federated catalog"
                    )
                source_id = source_ids[0]
            return {int(source_id): path}
        if payload.get("format") != "phase3_federated_artifacts_manifest_v2":
            raise ValueError(
                "Unsupported Phase 3 v2 artifacts manifest format: "
                f"{payload.get('format')}"
            )
        result: dict[int, Path] = {}
        for item in payload.get("sources", []):
            source_id = int(item["source_id"])
            manifest_path = Path(item["manifest_path"])
            if not manifest_path.is_absolute():
                manifest_path = path.parent / manifest_path
            if source_id in result:
                raise ValueError(f"Duplicate v2 artifact source_id={source_id}")
            result[source_id] = manifest_path.resolve()
        if not result:
            raise ValueError("v2 artifacts manifest contains no sources")
        return result

    def _load_sources(self) -> tuple[FederatedCanonicalCacheSource, ...]:
        raw_sources = self.catalog.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("Federated cache catalog must contain sources")
        source_ids: set[int] = set()
        global_dataset_ids: set[int] = set()
        transition_ranges: list[tuple[int, int, int]] = []
        sources: list[FederatedCanonicalCacheSource] = []
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise TypeError("Federated cache sources must be objects")
            source_id = int(raw_source["source_id"])
            if source_id < 0 or source_id in source_ids:
                raise ValueError(f"Invalid or duplicate source_id: {source_id}")
            source_ids.add(source_id)
            task_name = str(raw_source.get("task_name", "")).strip()
            if not task_name:
                raise ValueError(f"source_id={source_id} has no task_name")
            cache_dir = Path(raw_source["cache_dir"])
            if not cache_dir.is_absolute():
                cache_dir = self.catalog_path.parent / cache_dir
            cache_dir = cache_dir.resolve()
            manifest_path = cache_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Missing source cache manifest: {manifest_path}"
                )
            manifest_bytes = manifest_path.read_bytes()
            expected_manifest_sha256 = raw_source.get("manifest_sha256")
            if expected_manifest_sha256 is not None:
                actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
                if actual_manifest_sha256 != str(expected_manifest_sha256):
                    raise ValueError(
                        f"Manifest hash mismatch for source_id={source_id}: "
                        f"{actual_manifest_sha256} != {expected_manifest_sha256}"
                    )
            manifest = json.loads(manifest_bytes)
            if manifest.get("schema_version") != CANONICAL_CACHE_SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported canonical schema for source_id={source_id}: "
                    f"{manifest.get('schema_version')}"
                )
            if manifest.get("cache_format") != CANONICAL_CACHE_FORMAT:
                raise ValueError(
                    f"Unsupported canonical format for source_id={source_id}: "
                    f"{manifest.get('cache_format')}"
                )
            cached_feature_layers = tuple(
                int(layer) for layer in manifest["cached_feature_layers"]
            )
            layer_offsets = _resolve_layer_offsets(
                cached_feature_layers,
                self.critic_feature_layers,
            )
            raw_datasets = raw_source.get("datasets")
            if not isinstance(raw_datasets, list) or not raw_datasets:
                raise ValueError(
                    f"source_id={source_id} must declare dataset mappings"
                )
            dataset_id_map: dict[int, int] = {}
            for raw_dataset in raw_datasets:
                if not isinstance(raw_dataset, dict):
                    raise TypeError("Federated dataset mappings must be objects")
                local_id = int(raw_dataset["local_dataset_id"])
                global_id = int(raw_dataset["global_dataset_id"])
                if local_id in dataset_id_map:
                    raise ValueError(
                        f"Duplicate local dataset ID {local_id} in source {source_id}"
                    )
                if global_id < 0 or global_id in global_dataset_ids:
                    raise ValueError(
                        f"Invalid or duplicate global dataset ID: {global_id}"
                    )
                dataset_id_map[local_id] = global_id
                global_dataset_ids.add(global_id)
            transition_idx_offset = int(raw_source["transition_idx_offset"])
            manifest_dataset_size = int(manifest.get("dataset_size", 0))
            dataset_size = int(
                raw_source.get("transition_count", manifest_dataset_size)
            )
            if transition_idx_offset < 0 or dataset_size <= 0:
                raise ValueError(
                    f"Invalid transition range for source_id={source_id}: "
                    f"offset={transition_idx_offset} size={dataset_size}"
                )
            transition_ranges.append(
                (
                    transition_idx_offset,
                    transition_idx_offset + dataset_size,
                    source_id,
                )
            )
            episode_files = tuple(
                sorted(cache_dir.glob("dataset-*/episode_d*_ep*.safetensors"))
            )
            expected_episode_count = int(manifest.get("episode_count", 0))
            if not episode_files or len(episode_files) != expected_episode_count:
                raise ValueError(
                    f"Source cache episode count mismatch for source_id={source_id}: "
                    f"files={len(episode_files)} manifest={expected_episode_count}"
                )
            artifacts_manifest_path = self._artifact_manifest_by_source.get(source_id)
            raw_artifacts_manifest = raw_source.get("artifacts_manifest_path")
            if raw_artifacts_manifest is not None:
                artifacts_manifest_path = Path(str(raw_artifacts_manifest))
                if not artifacts_manifest_path.is_absolute():
                    artifacts_manifest_path = self.catalog_path.parent / artifacts_manifest_path
                artifacts_manifest_path = artifacts_manifest_path.resolve()
            elif (
                artifacts_manifest_path is None
                and manifest.get("cache_pipeline_version") == "canonical_precompute_v2"
            ):
                artifacts_manifest_path = manifest_path
            sources.append(
                FederatedCanonicalCacheSource(
                    source_id=source_id,
                    task_name=task_name,
                    cache_dir=cache_dir,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    cached_feature_layers=cached_feature_layers,
                    layer_offsets=layer_offsets,
                    dataset_id_map=dataset_id_map,
                    transition_idx_offset=transition_idx_offset,
                    dataset_size=dataset_size,
                    episode_files=episode_files,
                    artifacts_manifest_path=artifacts_manifest_path,
                    artifacts_root=(
                        artifacts_manifest_path.parent
                        if artifacts_manifest_path is not None
                        else None
                    ),
                )
            )
        ordered_ranges = sorted(transition_ranges)
        for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
            if previous[1] > current[0]:
                raise ValueError(
                    "Federated transition ranges overlap: "
                    f"source {previous[2]} ends at {previous[1]}, "
                    f"source {current[2]} starts at {current[0]}"
                )
        return tuple(sources)

    def _episode_files(self) -> list[Path]:
        return [path for source in self.sources for path in source.episode_files]

    @property
    def federated_reward_metadata(self) -> dict[str, torch.Tensor]:
        if set(self._reward_metadata) != set(FEDERATED_REWARD_METADATA_KEYS):
            raise RuntimeError("Federated reward metadata index is unavailable")
        return self._reward_metadata

    def _source_for_path(self, path: Path) -> FederatedCanonicalCacheSource:
        source = self._source_by_path.get(Path(path))
        if source is None:
            raise KeyError(f"Episode path is not registered in the catalog: {path}")
        return source

    def episode_payload_bytes(self, episode_key: tuple[int, int]) -> int:
        """Estimate the resident payload, including the language sidecar."""
        cached = self._episode_byte_estimates.get(episode_key)
        if cached is not None:
            return cached
        path = self.episode_path(episode_key)
        source = self._source_for_path(path)
        total = _safetensors_payload_bytes(
            path,
            layer_offsets=source.layer_offsets,
        )
        if self.language_sidecar_catalog_path is not None:
            local_dataset_id = next(
                (
                    local_id
                    for local_id, global_id in source.dataset_id_map.items()
                    if int(global_id) == int(episode_key[0])
                ),
                None,
            )
            if local_dataset_id is None:
                raise KeyError(
                    "No federated local dataset mapping for "
                    f"episode_key={episode_key}"
                )
            sidecar_path = self._language_sidecar_by_dataset[
                (source.source_id, local_dataset_id)
            ]
            with safe_open(sidecar_path, framework="pt", device="cpu") as tensors:
                for key in ("text_tokens", "text_mask"):
                    view = tensors.get_slice(key)
                    shape = list(view.get_shape())
                    shape[0] = 1
                    dtype = view.get_dtype()
                    if isinstance(dtype, str):
                        dtype_bytes = {
                            "BOOL": 1,
                            "U8": 1,
                            "I8": 1,
                            "I16": 2,
                            "I32": 4,
                            "I64": 8,
                            "F16": 2,
                            "BF16": 2,
                            "F32": 4,
                            "F64": 8,
                        }
                        try:
                            element_size = dtype_bytes[dtype]
                        except KeyError as error:
                            raise ValueError(
                                f"Unsupported safetensors dtype {dtype!r}: "
                                f"{sidecar_path}"
                            ) from error
                    else:
                        element_size = torch.empty(
                            (), dtype=dtype
                        ).element_size()
                    total += math.prod(shape) * element_size
        self._episode_byte_estimates[episode_key] = int(total)
        return int(total)

    @staticmethod
    def _remap_dataset_ids(
        dataset_id: torch.Tensor,
        source: FederatedCanonicalCacheSource,
    ) -> torch.Tensor:
        remapped = torch.empty_like(dataset_id)
        for local_id in dataset_id.unique().tolist():
            local_id = int(local_id)
            if local_id not in source.dataset_id_map:
                raise KeyError(
                    f"Source {source.source_id} has no mapping for "
                    f"local dataset ID {local_id}"
                )
            remapped[dataset_id == local_id] = source.dataset_id_map[local_id]
        return remapped

    def _remap_metadata(
        self,
        payload: dict[str, torch.Tensor],
        source: FederatedCanonicalCacheSource,
        path: Path,
    ) -> dict[str, torch.Tensor]:
        if "transition_idx" in payload:
            source_transition_idx = payload["transition_idx"].long()
            payload["source_transition_idx"] = source_transition_idx
            row_offset = self._transition_row_offset_by_path[Path(path)]
            payload["transition_idx"] = torch.arange(
                source_transition_idx.shape[0],
                dtype=torch.long,
            ) + row_offset
        if "dataset_id" in payload:
            payload["source_dataset_id"] = payload["dataset_id"].long()
            payload["dataset_id"] = self._remap_dataset_ids(
                payload["dataset_id"], source
            )
        return payload

    def _build_sampler_index(self) -> None:
        seen_transition_indices: set[int] = set()
        episode_paths: dict[tuple[int, int], Path] = {}
        for source in self.sources:
            source_row_offset = 0
            for path in source.episode_files:
                self._transition_row_offset_by_path[path] = (
                    source.transition_idx_offset + source_row_offset
                )
                with safe_open(path, framework="pt", device="cpu") as tensors:
                    keys = set(tensors.keys())
                    missing = set(CANONICAL_FEATURE_KEYS).difference(keys)
                    if missing:
                        raise KeyError(
                            f"Canonical episode cache {path} is missing "
                            f"{sorted(missing)}"
                        )
                    transition_idx = tensors.get_tensor("transition_idx")
                    dataset_id = tensors.get_tensor("dataset_id")
                    episode_index = tensors.get_tensor("episode_index")
                    rl_chunk_idx = tensors.get_tensor("rl_chunk_idx")
                    episode_chunk_count = tensors.get_tensor(
                        "episode_chunk_count"
                    )
                    count = int(transition_idx.shape[0])
                    for row_id in range(count):
                        if (
                            self.exclude_last_n_chunks > 0
                            and int(rl_chunk_idx[row_id])
                            >= max(
                                0,
                                int(episode_chunk_count[row_id])
                                - self.exclude_last_n_chunks,
                            )
                        ):
                            continue
                        global_transition_idx = (
                            source.transition_idx_offset
                            + source_row_offset
                            + row_id
                        )
                        if global_transition_idx in seen_transition_indices:
                            raise ValueError(
                                "Duplicate federated transition index: "
                                f"{global_transition_idx}"
                            )
                        seen_transition_indices.add(global_transition_idx)
                        local_dataset_id = int(dataset_id[row_id])
                        if local_dataset_id not in source.dataset_id_map:
                            raise KeyError(
                                f"Source {source.source_id} has no mapping for "
                                f"local dataset ID {local_dataset_id}"
                            )
                        episode_key = (
                            source.dataset_id_map[local_dataset_id],
                            int(episode_index[row_id]),
                        )
                        registered_path = episode_paths.setdefault(episode_key, path)
                        if registered_path != path:
                            raise ValueError(
                                f"Federated episode key {episode_key} appears in "
                                f"both {registered_path} and {path}"
                            )
                        position = len(self.rows)
                        self.rows.append(
                            {
                                "path": path,
                                "row": row_id,
                                "transition_idx": global_transition_idx,
                                "episode_key": episode_key,
                                "rl_chunk_idx": int(rl_chunk_idx[row_id]),
                                "source_id": source.source_id,
                            }
                        )
                        self.positions_by_episode.setdefault(
                            episode_key, []
                        ).append(position)
                    source_row_offset += count
            if source_row_offset != source.dataset_size:
                raise ValueError(
                    f"Source {source.source_id} row count mismatch: "
                    f"files={source_row_offset} manifest={source.dataset_size}"
                )
        if not self.rows:
            raise ValueError(
                "Federated canonical cache has no trainable rows after applying "
                f"exclude_last_n_chunks={self.exclude_last_n_chunks}"
            )

    def _scan_source_reward_metadata(
        self,
        source: FederatedCanonicalCacheSource,
    ) -> dict[str, torch.Tensor]:
        indexed_global_offsets = tuple(
            int(self._transition_row_offset_by_path[Path(path)])
            for path in source.episode_files
        )
        return _scan_federated_source_reward_metadata(
            (source, indexed_global_offsets)
        )

    def _build_reward_metadata(self) -> None:
        configured_workers = int(
            os.environ.get("WAN_VA_FEDERATED_REWARD_INDEX_WORKERS", "16")
        )
        max_workers = max(1, min(configured_workers, len(self.sources)))
        jobs = [
            (
                source,
                tuple(
                    int(self._transition_row_offset_by_path[Path(path)])
                    for path in source.episode_files
                ),
            )
            for source in self.sources
        ]
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            source_metadata = list(
                executor.map(_scan_federated_source_reward_metadata, jobs)
            )
        self._reward_metadata = {
            key: torch.cat(
                [metadata[key] for metadata in source_metadata], dim=0
            )
            for key in FEDERATED_REWARD_METADATA_KEYS
        }
        self._apply_success_overlay_to_reward_metadata()

    def _apply_success_overlay_to_reward_metadata(self) -> None:
        if self.success_overlay is None:
            return
        values = self._reward_metadata["episode_success"].clone()
        transition_idx = self._reward_metadata["transition_idx"]
        for source in self.sources:
            start = int(source.transition_idx_offset)
            stop = start + int(source.dataset_size)
            source_mask = (transition_idx >= start) & (transition_idx < stop)
            local_ids = torch.empty_like(self._reward_metadata["dataset_id"][source_mask])
            global_to_local = {int(global_id): int(local_id) for local_id, global_id in source.dataset_id_map.items()}
            for global_id, local_id in global_to_local.items():
                local_ids[self._reward_metadata["dataset_id"][source_mask] == global_id] = local_id
            values[source_mask] = _apply_success_overlay(
                values[source_mask],
                source_id=source.source_id,
                dataset_id=local_ids,
                episode_index=self._reward_metadata["episode_index"][source_mask],
                overlay=self.success_overlay,
            )
        self._reward_metadata["episode_success"] = values

    def load_episode_payload(
        self,
        path: Path,
        *,
        materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        source = self._source_for_path(path)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            payload: dict[str, torch.Tensor] = {}
            for key in tensors.keys():
                value = tensors.get_tensor(key)
                if key in CANONICAL_FEATURE_KEYS:
                    value = _select_layers(value, source.layer_offsets)
                    if not value.is_contiguous():
                        value = value.contiguous()
                if materialize:
                    value = value.clone()
                payload[key] = value
        if self.language_sidecar_catalog_path is not None:
            text_tokens, text_mask = self._load_episode_language(source, payload)
            payload["text_tokens"] = text_tokens.clone() if materialize else text_tokens
            payload["text_mask"] = text_mask.clone() if materialize else text_mask
        if self.success_overlay is not None:
            payload["episode_success"] = _apply_success_overlay(
                payload.get("episode_success", torch.zeros_like(payload["done"], dtype=torch.bool)),
                source_id=source.source_id,
                dataset_id=payload["dataset_id"],
                episode_index=payload["episode_index"],
                overlay=self.success_overlay,
            )
        return self._remap_metadata(payload, source, path)

    def load_episode_range_payload(
        self,
        path: Path,
        start: int,
        stop: int,
        *,
        materialize: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Load a federated episode prefix/range without full-file materialization."""

        if start < 0 or stop <= start:
            raise ValueError(f"invalid episode range [{start}, {stop})")
        source = self._source_for_path(path)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            payload: dict[str, torch.Tensor] = {}
            for key in tensors.keys():
                value = _read_tensor_range(tensors, key, start, stop)
                if key in CANONICAL_FEATURE_KEYS:
                    value = _select_layers(value, source.layer_offsets)
                    if not value.is_contiguous():
                        value = value.contiguous()
                if materialize:
                    value = value.clone()
                payload[key] = value
        if self.language_sidecar_catalog_path is not None:
            text_tokens, text_mask = self._load_episode_language(source, payload)
            payload["text_tokens"] = text_tokens.clone() if materialize else text_tokens
            payload["text_mask"] = text_mask.clone() if materialize else text_mask
        if self.success_overlay is not None:
            payload["episode_success"] = _apply_success_overlay(
                payload.get(
                    "episode_success",
                    torch.zeros_like(payload["done"], dtype=torch.bool),
                ),
                source_id=source.source_id,
                dataset_id=payload["dataset_id"],
                episode_index=payload["episode_index"],
                overlay=self.success_overlay,
            )
        return self._remap_metadata(payload, source, path)

    def load_episode_metadata(self, path: Path) -> dict[str, torch.Tensor]:
        source = self._source_for_path(path)
        with safe_open(path, framework="pt", device="cpu") as tensors:
            payload = {
                key: tensors.get_tensor(key)
                for key in tensors.keys()
                if key not in CANONICAL_FEATURE_KEYS
            }
        if self.success_overlay is not None:
            payload["episode_success"] = _apply_success_overlay(
                payload.get("episode_success", torch.zeros_like(payload["done"], dtype=torch.bool)),
                source_id=source.source_id,
                dataset_id=payload["dataset_id"],
                episode_index=payload["episode_index"],
                overlay=self.success_overlay,
            )
        return self._remap_metadata(payload, source, path)


def phase3_canonical_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("samples must be non-empty")
    features: dict[str, torch.Tensor] = {}
    for key in samples[0]["features"]:
        values = [sample["features"][key] for sample in samples]
        if key == "text_tokens":
            features[key] = _pad_and_stack_text_tokens(values)
        elif key == "text_mask":
            features[key] = _pad_and_stack_text_masks(values)
        else:
            features[key] = _pad_and_stack_feature(values)
    batch: dict[str, Any] = {
        "_phase3_canonical_cache": True,
        "features": features,
    }
    for key in (
        "transition_idx",
        "discount",
        "state_valid",
        "jepa_distance",
        "goal_jepa_distance",
        "episode_goal_jepa_distance_max",
        "cached_reward",
        "done",
        "dataset_id",
        "episode_index",
        "rl_chunk_idx",
        "episode_success",
        "episode_chunk_count",
        "previous_latents_mask",
        "latents_mask",
        "previous_actions_mask",
        "actions_mask",
    ):
        batch[key] = torch.stack([sample[key] for sample in samples], dim=0)
    return batch


class EpisodeLocalBatchSampler(Sampler[list[int]]):
    """Shuffle episodes each epoch and sample local random chunks per batch."""

    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        batch_size: int,
        *,
        episodes_per_batch: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = False,
        context_length_aware: bool = False,
        preserve_episode_group_order: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.episodes_per_batch = int(episodes_per_batch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.context_length_aware = bool(context_length_aware)
        self.preserve_episode_group_order = bool(
            preserve_episode_group_order
        )
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        episodes = sorted(self.dataset.positions_by_episode)
        rng.shuffle(episodes)
        episodes = episodes[self.rank :: max(self.world_size, 1)]
        batches: list[list[int]] = []
        for start in range(0, len(episodes), self.episodes_per_batch):
            group = episodes[start : start + self.episodes_per_batch]
            rng.shuffle(group)
            positions: list[int] = []
            for episode in group:
                episode_positions = list(
                    self.dataset.positions_by_episode[episode]
                )
                if self.context_length_aware:
                    episode_positions.sort(
                        key=lambda position: self.dataset.rows[position][
                            "rl_chunk_idx"
                        ]
                    )
                else:
                    rng.shuffle(episode_positions)
                positions.extend(episode_positions)
            group_batches: list[list[int]] = []
            for batch_start in range(0, len(positions), self.batch_size):
                batch = positions[batch_start : batch_start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    group_batches.append(batch)
            if self.preserve_episode_group_order:
                # Keep batches for the same primary episode adjacent, while
                # shuffling pure batches within that episode. A boundary
                # batch stays last so its next episode remains resident for
                # the following cluster. The batch multiset is unchanged.
                block_rng = random.Random(
                    self.seed
                    + self.epoch * 1_000_003
                    + self.rank * 10_007
                    + start
                )
                episode_clusters: dict[
                    tuple[int, int],
                    tuple[list[list[int]], list[list[int]]],
                ] = {}
                episode_order: list[tuple[int, int]] = []
                for batch in group_batches:
                    primary_episode = self.dataset.rows[batch[0]]["episode_key"]
                    if primary_episode not in episode_clusters:
                        episode_clusters[primary_episode] = ([], [])
                        episode_order.append(primary_episode)
                    pure_batches, boundary_batches = episode_clusters[
                        primary_episode
                    ]
                    batch_episodes = {
                        self.dataset.rows[position]["episode_key"]
                        for position in batch
                    }
                    target = (
                        pure_batches
                        if len(batch_episodes) == 1
                        else boundary_batches
                    )
                    target.append(batch)
                clustered_batches: list[list[int]] = []
                for episode in episode_order:
                    pure_batches, boundary_batches = episode_clusters[episode]
                    block_rng.shuffle(pure_batches)
                    block_rng.shuffle(boundary_batches)
                    clustered_batches.extend(pure_batches)
                    clustered_batches.extend(boundary_batches)
                group_batches = clustered_batches
            batches.extend(group_batches)
        if not self.preserve_episode_group_order:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        rng = random.Random(self.seed + self.epoch)
        episodes = sorted(self.dataset.positions_by_episode)
        rng.shuffle(episodes)
        episodes = episodes[self.rank :: max(self.world_size, 1)]
        batch_count = 0
        for start in range(0, len(episodes), self.episodes_per_batch):
            group = episodes[start : start + self.episodes_per_batch]
            total = sum(
                len(self.dataset.positions_by_episode[episode])
                for episode in group
            )
            if self.drop_last:
                batch_count += total // self.batch_size
            else:
                batch_count += math.ceil(total / self.batch_size)
        return batch_count


def _sampler_seed(seed: int, epoch: int, stream: str) -> int:
    payload = f"{int(seed)}\0{int(epoch)}\0{stream}".encode("utf-8")
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=16).digest(),
        byteorder="big",
    )


class GlobalBalancedEpisodeBatchSampler(Sampler[list[int]]):
    """Build equal, non-overlapping rank batches without episode-tail bias."""

    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        batch_size: int,
        *,
        episodes_per_batch: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        drop_last: bool = True,
        context_length_aware: bool = False,
        preserve_episode_group_order: bool = False,
        source_balanced: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if episodes_per_batch <= 0:
            raise ValueError("episodes_per_batch must be positive")
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        if rank < 0 or rank >= world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if not drop_last:
            raise ValueError("global balanced batching requires drop_last=true")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.episodes_per_batch = int(episodes_per_batch)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.drop_last = True
        self.context_length_aware = bool(context_length_aware)
        self.preserve_episode_group_order = bool(
            preserve_episode_group_order
        )
        self.source_balanced = bool(source_balanced)
        self.source_ids = tuple(
            sorted({int(row.get("source_id", -1)) for row in dataset.rows})
        )
        if self.source_balanced:
            if self.preserve_episode_group_order:
                raise ValueError(
                    "source-balanced sampling is incompatible with "
                    "preserve_episode_group_order=true"
                )
            if len(self.source_ids) < 2 or -1 in self.source_ids:
                raise ValueError(
                    "source-balanced sampling requires federated rows from at "
                    "least two sources"
                )
            if self.world_size % len(self.source_ids):
                raise ValueError(
                    "source-balanced sampling requires world_size divisible by "
                    "the number of sources"
                )
        self.epoch = 0
        self.start_batch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_batch(self, start_batch: int) -> None:
        start_batch = int(start_batch)
        batches_per_epoch = len(self)
        if start_batch < 0 or start_batch > batches_per_epoch:
            raise ValueError(
                "start_batch must satisfy 0 <= start_batch <= "
                f"{batches_per_epoch}, got {start_batch}"
            )
        self.start_batch = start_batch

    def _candidate_cost(self, batch: list[int]) -> tuple[int, int]:
        chunk_indices = [
            int(self.dataset.rows[position].get("rl_chunk_idx", 0))
            for position in batch
        ]
        return max(chunk_indices, default=0), sum(chunk_indices)

    def _global_steps(self) -> list[list[list[int]]]:
        global_batch_size = self.batch_size * self.world_size
        omitted_count = len(self.dataset) % global_batch_size
        omission_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "omissions")
        )
        omitted = set(omission_rng.sample(range(len(self.dataset)), omitted_count))

        episode_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "episodes")
        )
        episodes = sorted(self.dataset.positions_by_episode)
        episode_rng.shuffle(episodes)

        candidates: list[list[int]] = []
        carry: list[int] = []
        for start in range(0, len(episodes), self.episodes_per_batch):
            group = episodes[start : start + self.episodes_per_batch]
            group_positions: list[int] = []
            for episode in group:
                episode_positions = [
                    position
                    for position in self.dataset.positions_by_episode[episode]
                    if position not in omitted
                ]
                if self.context_length_aware:
                    episode_positions.sort(
                        key=lambda position: self.dataset.rows[position].get(
                            "rl_chunk_idx", 0
                        )
                    )
                else:
                    position_rng = random.Random(
                        _sampler_seed(
                            self.seed,
                            self.epoch,
                            f"positions:{episode[0]}:{episode[1]}",
                        )
                    )
                    position_rng.shuffle(episode_positions)
                group_positions.extend(episode_positions)
            carry.extend(group_positions)
            full_count = len(carry) // self.batch_size
            for batch_index in range(full_count):
                offset = batch_index * self.batch_size
                candidates.append(carry[offset : offset + self.batch_size])
            carry = carry[full_count * self.batch_size :]

        if carry:
            raise RuntimeError(
                "global balanced sampler left a partial local batch: "
                f"size={len(carry)}"
            )
        expected_candidates = len(self) * self.world_size
        if len(candidates) != expected_candidates:
            raise RuntimeError(
                "global balanced sampler candidate count mismatch: "
                f"expected={expected_candidates} actual={len(candidates)}"
            )

        tie_rng = random.Random(_sampler_seed(self.seed, self.epoch, "cost-ties"))
        decorated = [
            (self._candidate_cost(batch), tie_rng.random(), batch)
            for batch in candidates
        ]
        decorated.sort(key=lambda item: (item[0], item[1]))
        global_steps = [
            [item[2] for item in decorated[start : start + self.world_size]]
            for start in range(0, len(decorated), self.world_size)
        ]

        step_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "global-steps")
        )
        step_rng.shuffle(global_steps)
        rank_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "rank-assignments")
        )
        for step in global_steps:
            rank_rng.shuffle(step)
        return global_steps

    def _rank_local_batches(self) -> list[list[int]]:
        """Return an equal, disjoint, group-local stream for this rank.

        The global-balanced locality path must preserve the same episode-group
        contract as ``BoundedGroupPrefetchLoader``. Earlier code flattened all
        shuffled episodes into one stream and cut that stream into rank shards;
        as a result, ``episodes_per_batch`` was effectively ignored here and
        ranks could advance through different episode groups at the same
        optimizer step. We now form global batches group-by-group, so every
        rank consumes one local batch from the same ordered episode group at a
        time. A single global batch may straddle a group boundary when the
        preceding group does not end on a global-batch boundary; this preserves
        complete coverage without duplicating or dropping transitions.
        """
        global_batch_size = self.batch_size * self.world_size
        omitted_count = len(self.dataset) % global_batch_size
        omission_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "locality-omissions")
        )
        omitted = set(
            omission_rng.sample(range(len(self.dataset)), omitted_count)
        )

        episode_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "locality-episodes")
        )
        episodes = sorted(self.dataset.positions_by_episode)
        episode_rng.shuffle(episodes)

        global_steps: list[list[list[int]]] = []
        carry: list[int] = []
        for group_start in range(0, len(episodes), self.episodes_per_batch):
            group = episodes[
                group_start : group_start + self.episodes_per_batch
            ]
            group_positions: list[int] = []
            for episode in group:
                episode_positions = [
                    position
                    for position in self.dataset.positions_by_episode[episode]
                    if position not in omitted
                ]
                if self.context_length_aware:
                    episode_positions.sort(
                        key=lambda position: self.dataset.rows[position].get(
                            "rl_chunk_idx", 0
                        )
                    )
                else:
                    position_rng = random.Random(
                        _sampler_seed(
                            self.seed,
                            self.epoch,
                            f"locality-positions:{episode[0]}:{episode[1]}",
                        )
                    )
                    position_rng.shuffle(episode_positions)
                group_positions.extend(episode_positions)

            carry.extend(group_positions)
            while len(carry) >= global_batch_size:
                global_batch = carry[:global_batch_size]
                del carry[:global_batch_size]
                global_steps.append(
                    [
                        global_batch[start : start + self.batch_size]
                        for start in range(0, global_batch_size, self.batch_size)
                    ]
                )

        if carry:
            raise RuntimeError(
                "group-local sampler left a partial global batch: "
                f"size={len(carry)}"
            )
        if len(global_steps) != len(self):
            raise RuntimeError(
                "group-local sampler step count mismatch: "
                f"expected={len(self)} actual={len(global_steps)}"
            )

        rank_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "locality-rank-assignments")
        )
        batches: list[list[int]] = []
        for global_step in global_steps:
            rank_rng.shuffle(global_step)
            batches.append(global_step[self.rank])
        if any(len(batch) != self.batch_size for batch in batches):
            raise RuntimeError(
                "group-local sampler violated fixed-size rank batches"
            )
        return batches

    def _source_candidate_batches(self, source_id: int) -> list[list[int]]:
        episode_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, f"source-episodes:{source_id}")
        )
        episodes = [
            episode
            for episode in sorted(self.dataset.positions_by_episode)
            if int(
                self.dataset.rows[
                    self.dataset.positions_by_episode[episode][0]
                ]["source_id"]
            )
            == source_id
        ]
        episode_rng.shuffle(episodes)

        candidates: list[list[int]] = []
        carry: list[int] = []
        for episode in episodes:
            positions = list(self.dataset.positions_by_episode[episode])
            if self.context_length_aware:
                positions.sort(
                    key=lambda position: self.dataset.rows[position].get(
                        "rl_chunk_idx", 0
                    )
                )
            else:
                position_rng = random.Random(
                    _sampler_seed(
                        self.seed,
                        self.epoch,
                        f"source-positions:{source_id}:{episode[0]}:{episode[1]}",
                    )
                )
                position_rng.shuffle(positions)
            carry.extend(positions)
            full_count = len(carry) // self.batch_size
            for batch_index in range(full_count):
                offset = batch_index * self.batch_size
                candidates.append(carry[offset : offset + self.batch_size])
            carry = carry[full_count * self.batch_size :]
        if not candidates:
            raise ValueError(
                f"source {source_id} has no complete local batches"
            )
        return candidates

    def _iter_source_balanced(self) -> Iterator[list[int]]:
        candidates_by_source = {
            source_id: self._source_candidate_batches(source_id)
            for source_id in self.source_ids
        }
        target_count = max(
            len(candidates) for candidates in candidates_by_source.values()
        )
        slots_per_source = self.world_size // len(self.source_ids)
        target_count -= target_count % slots_per_source
        if target_count <= 0:
            raise ValueError("source-balanced sampling has no global batches")

        def replayed_batches(
            source_id: int,
            candidates: list[list[int]],
        ) -> Iterator[list[int]]:
            source_rng = random.Random(
                _sampler_seed(self.seed, self.epoch, f"source-replay:{source_id}")
            )
            emitted = 0
            while emitted < target_count:
                cycle = list(candidates)
                source_rng.shuffle(cycle)
                for batch in cycle:
                    if emitted >= target_count:
                        return
                    emitted += 1
                    yield batch

        source_iterators = {
            source_id: replayed_batches(source_id, candidates)
            for source_id, candidates in candidates_by_source.items()
        }
        rank_rng = random.Random(
            _sampler_seed(self.seed, self.epoch, "source-rank-assignments")
        )
        for start in range(0, target_count, slots_per_source):
            step = [
                next(source_iterators[source_id])
                for source_id in self.source_ids
                for _ in range(slots_per_source)
            ]
            rank_rng.shuffle(step)
            yield step[self.rank]

    def __iter__(self) -> Iterator[list[int]]:
        if self.source_balanced:
            for batch_index, batch in enumerate(self._iter_source_balanced()):
                if batch_index >= self.start_batch:
                    yield batch
            return
        if self.preserve_episode_group_order:
            yield from self._rank_local_batches()[self.start_batch :]
            return
        for step in self._global_steps()[self.start_batch :]:
            yield step[self.rank]

    def __len__(self) -> int:
        if self.source_balanced:
            slots_per_source = self.world_size // len(self.source_ids)
            candidate_counts = [
                sum(
                    len(positions)
                    for positions in self.dataset.positions_by_episode.values()
                    if int(self.dataset.rows[positions[0]]["source_id"])
                    == source_id
                )
                // self.batch_size
                for source_id in self.source_ids
            ]
            return max(candidate_counts) // slots_per_source
        return len(self.dataset) // (self.batch_size * self.world_size)


@dataclass(frozen=True)
class _EpisodeLoadResult:
    payload: dict[str, torch.Tensor]
    byte_count: int
    load_time: float


class _EpisodeStore:
    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        *,
        max_bytes: int,
        load_workers: int,
    ) -> None:
        self.dataset = dataset
        self.max_bytes = max(1, int(max_bytes))
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, int(load_workers)),
            thread_name_prefix="phase3-episode-load",
        )
        self.lock = threading.Lock()
        self.cache: OrderedDict[
            tuple[int, int], tuple[dict[str, torch.Tensor], int]
        ] = OrderedDict()
        self.pending: dict[tuple[int, int], Future[_EpisodeLoadResult]] = {}
        self.protected: set[tuple[int, int]] = set()
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.load_count = 0
        self.load_seconds = 0.0
        self.cache_hits = 0
        self.pending_hits = 0
        self.wait_seconds = 0.0
        self.eviction_count = 0
        self.evicted_bytes = 0
        self.warm_count = 0
        self.warm_seconds = 0.0
        self.warmed_paths: set[Path] = set()

    @staticmethod
    def _warm_path(path: Path) -> None:
        """Touch file-backed pages to warm the shared OS page cache."""
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            if size <= 0:
                return
            with mmap.mmap(fd, length=0, access=mmap.ACCESS_READ) as mapped:
                page = max(1, int(getattr(mmap, "PAGESIZE", 4096)))
                for offset in range(0, size, page):
                    mapped[offset]
                mapped[size - 1]
        finally:
            os.close(fd)

    def warm_many(self, episode_keys: list[tuple[int, int]]) -> None:
        paths = []
        with self.lock:
            for key in dict.fromkeys(episode_keys):
                path = self.dataset.episode_path(key)
                if path not in self.warmed_paths:
                    self.warmed_paths.add(path)
                    paths.append(path)
        if not paths:
            return
        start = time.perf_counter()
        for path in paths:
            self._warm_path(path)
        with self.lock:
            self.warm_count += len(paths)
            self.warm_seconds += time.perf_counter() - start

    def _load(self, episode_key: tuple[int, int]) -> _EpisodeLoadResult:
        start = time.perf_counter()
        payload = self.dataset.load_episode_payload(
            self.dataset.episode_path(episode_key),
            materialize=True,
        )
        return _EpisodeLoadResult(
            payload=payload,
            byte_count=_tensor_bytes(payload),
            load_time=time.perf_counter() - start,
        )

    def _evict_locked(self) -> None:
        while self.resident_bytes > self.max_bytes and self.cache:
            victim = next(
                (key for key in self.cache if key not in self.protected),
                None,
            )
            if victim is None:
                break
            _, byte_count = self.cache.pop(victim)
            self.resident_bytes -= byte_count
            self.eviction_count += 1
            self.evicted_bytes += byte_count

    def _complete(
        self,
        episode_key: tuple[int, int],
        future: Future[_EpisodeLoadResult],
    ) -> None:
        if future.cancelled() or future.exception() is not None:
            with self.lock:
                self.pending.pop(episode_key, None)
            return
        result = future.result()
        with self.lock:
            self.pending.pop(episode_key, None)
            previous = self.cache.pop(episode_key, None)
            if previous is not None:
                self.resident_bytes -= previous[1]
            self.cache[episode_key] = (result.payload, result.byte_count)
            self.resident_bytes += result.byte_count
            self.peak_resident_bytes = max(
                self.peak_resident_bytes,
                self.resident_bytes,
            )
            self.load_count += 1
            self.load_seconds += result.load_time
            self._evict_locked()

    def prefetch(
        self,
        episode_key: tuple[int, int],
    ) -> Future[_EpisodeLoadResult]:
        register_callback = False
        with self.lock:
            cached = self.cache.get(episode_key)
            if cached is not None:
                self.cache.move_to_end(episode_key)
                self.cache_hits += 1
                future: Future[_EpisodeLoadResult] = Future()
                future.set_result(
                    _EpisodeLoadResult(cached[0], cached[1], 0.0)
                )
                return future
            pending = self.pending.get(episode_key)
            if pending is not None:
                self.pending_hits += 1
                return pending
            future = self.executor.submit(self._load, episode_key)
            self.pending[episode_key] = future
            register_callback = True
        if register_callback:
            future.add_done_callback(
                lambda completed, key=episode_key: self._complete(key, completed)
            )
        return future

    def prefetch_many(self, episode_keys: list[tuple[int, int]]) -> None:
        for episode_key in dict.fromkeys(episode_keys):
            self.prefetch(episode_key)

    def get_many(
        self,
        episode_keys: list[tuple[int, int]],
    ) -> tuple[dict[tuple[int, int], dict[str, torch.Tensor]], float]:
        unique_keys = list(dict.fromkeys(episode_keys))
        futures = {key: self.prefetch(key) for key in unique_keys}
        wait_start = time.perf_counter()
        payloads = {key: future.result().payload for key, future in futures.items()}
        wait_time = time.perf_counter() - wait_start
        with self.lock:
            self.wait_seconds += wait_time
            for key in unique_keys:
                if key in self.cache:
                    self.cache.move_to_end(key)
        return payloads, wait_time

    def update_protected(self, episode_keys: list[tuple[int, int]]) -> None:
        with self.lock:
            self.protected = set(episode_keys)
            self._evict_locked()

    def metrics(self) -> dict[str, float]:
        with self.lock:
            return {
                "loader/episode_load_count_total": float(self.load_count),
                "loader/episode_load_seconds_total": self.load_seconds,
                "loader/episode_cache_hits_total": float(self.cache_hits),
                "loader/episode_pending_hits_total": float(self.pending_hits),
                "loader/episode_wait_seconds_total": self.wait_seconds,
                "loader/episode_cache_resident_gib": (
                    self.resident_bytes / 1024**3
                ),
                "loader/episode_cache_peak_gib": (
                    self.peak_resident_bytes / 1024**3
                ),
                "loader/episode_cache_evictions_total": float(
                    self.eviction_count
                ),
                "loader/episode_cache_evicted_gib": (
                    self.evicted_bytes / 1024**3
                ),
                "loader/episode_cache_warm_count": float(self.warm_count),
                "loader/episode_cache_warm_seconds": self.warm_seconds,
            }

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


class EpisodeAwarePrefetchLoader:
    """Build ordered batches from asynchronously preloaded episode payloads."""

    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        sampler: Sampler[list[int]],
        *,
        episode_cache_max_bytes: int,
        episode_load_workers: int,
        episode_lookahead_batches: int,
        batch_prefetch_depth: int,
        batch_workers: int,
        pin_memory: bool,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.episode_cache_max_bytes = max(1, int(episode_cache_max_bytes))
        self.episode_load_workers = max(1, int(episode_load_workers))
        self.episode_lookahead_batches = max(1, int(episode_lookahead_batches))
        self.batch_prefetch_depth = max(1, int(batch_prefetch_depth))
        self.batch_workers = max(1, int(batch_workers))
        self.pin_memory = bool(pin_memory)

    def _episode_keys(self, batch: list[int]) -> list[tuple[int, int]]:
        return [self.dataset.episode_key_for_position(position) for position in batch]

    def _build_batch(
        self,
        batch_indices: list[int],
        store: _EpisodeStore,
    ) -> dict[str, Any]:
        build_start = time.perf_counter()
        episode_keys = self._episode_keys(batch_indices)
        payloads, episode_wait = store.get_many(episode_keys)
        samples = [
            self.dataset.sample_from_payload(payloads[episode_key], position)
            for position, episode_key in zip(batch_indices, episode_keys)
        ]
        batch = phase3_canonical_collate(samples)
        if self.pin_memory:
            batch = _pin_memory(batch)
        batch["_phase3_loader_metrics"] = {
            "loader/episode_wait_time": episode_wait,
            "loader/batch_build_time": time.perf_counter() - build_start,
            "loader/batch_episode_count": float(len(set(episode_keys))),
            **store.metrics(),
        }
        return batch

    def __iter__(self):
        batches = list(iter(self.sampler))
        store = _EpisodeStore(
            self.dataset,
            max_bytes=self.episode_cache_max_bytes,
            load_workers=self.episode_load_workers,
        )
        batch_executor = ThreadPoolExecutor(
            max_workers=self.batch_workers,
            thread_name_prefix="phase3-batch-build",
        )
        pending: OrderedDict[int, Future[dict[str, Any]]] = OrderedDict()
        next_submit = 0

        def maintain_window(anchor: int) -> None:
            end = min(
                len(batches),
                anchor + self.episode_lookahead_batches,
            )
            episode_keys = [
                episode_key
                for batch in batches[anchor:end]
                for episode_key in self._episode_keys(batch)
            ]
            store.update_protected(episode_keys)
            store.prefetch_many(episode_keys)

        def fill() -> None:
            nonlocal next_submit
            while (
                next_submit < len(batches)
                and len(pending) < self.batch_prefetch_depth
            ):
                batch_index = next_submit
                pending[batch_index] = batch_executor.submit(
                    self._build_batch,
                    batches[batch_index],
                    store,
                )
                next_submit += 1

        try:
            maintain_window(0)
            fill()
            while pending:
                batch_index, future = pending.popitem(last=False)
                ready_wait_start = time.perf_counter()
                batch = future.result()
                ready_wait = time.perf_counter() - ready_wait_start
                maintain_window(batch_index + 1)
                fill()
                batch["_phase3_loader_metrics"][
                    "loader/batch_ready_wait_time"
                ] = ready_wait
                yield batch
        finally:
            batch_executor.shutdown(wait=True, cancel_futures=True)
            store.close()

    def __len__(self) -> int:
        return len(self.sampler)


class BoundedGroupPrefetchLoader:
    """Keep episode groups resident while preserving sampler batch order.

    This loader is opt-in.  It groups the already-materialized sampler stream
    by consecutive episode keys, admits the active group and then admits
    future groups in order until ``max_resident_bytes`` or
    ``group_prefetch_depth`` is reached.  A group is protected until all of
    its batches have been yielded; only then can its tensors be evicted.

    The implementation deliberately reuses ``_EpisodeStore`` and the v2
    episode-sharded safetensors layout.  It therefore remains compatible with
    existing caches and retains the legacy path when the new policy is absent.
    """

    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        sampler: Sampler[list[int]],
        *,
        group_size: int,
        group_prefetch_depth: int,
        max_resident_bytes: int,
        load_workers: int,
        batch_prefetch_depth: int,
        batch_workers: int,
        pin_memory: bool,
        full: bool = False,
        warmup: bool = False,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.group_size = max(1, int(group_size))
        self.group_prefetch_depth = max(1, int(group_prefetch_depth))
        self.max_resident_bytes = max(1, int(max_resident_bytes))
        self.load_workers = max(1, int(load_workers))
        self.batch_prefetch_depth = max(1, int(batch_prefetch_depth))
        self.batch_workers = max(1, int(batch_workers))
        self.pin_memory = bool(pin_memory)
        self.full = bool(full)
        self.warmup = bool(warmup)

    @staticmethod
    def _episode_keys(dataset, batch: list[int]) -> list[tuple[int, int]]:
        return [dataset.episode_key_for_position(position) for position in batch]

    def _groups(self, batches: list[list[int]]) -> list[list[tuple[int, int]]]:
        """Make deterministic episode groups from the sampler's stream.

        The sampler already determines the episode order.  We only read the
        first occurrence of each episode key; transition order within that
        stream is otherwise untouched.
        """
        groups: list[list[tuple[int, int]]] = []
        current: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for batch in batches:
            for key in dict.fromkeys(self._episode_keys(self.dataset, batch)):
                if key in seen:
                    continue
                seen.add(key)
                current.append(key)
                if len(current) >= self.group_size:
                    groups.append(current)
                    current = []
        if current:
            groups.append(current)
        return groups

    def _group_bytes(self, group: list[tuple[int, int]]) -> int:
        return sum(
            int(self.dataset.episode_payload_bytes(key))
            for key in group
        )

    def _build_batch(
        self,
        batch_indices: list[int],
        store: _EpisodeStore,
    ) -> dict[str, Any]:
        build_start = time.perf_counter()
        episode_keys = list(dict.fromkeys(self._episode_keys(self.dataset, batch_indices)))
        payloads, episode_wait = store.get_many(episode_keys)
        samples = [
            self.dataset.sample_from_payload(
                payloads[self.dataset.episode_key_for_position(position)],
                position,
            )
            for position in batch_indices
        ]
        batch = phase3_canonical_collate(samples)
        if self.pin_memory:
            batch = _pin_memory(batch)
        batch["_phase3_loader_metrics"] = {
            "loader/group_episode_wait_time": episode_wait,
            "loader/batch_build_time": time.perf_counter() - build_start,
            "loader/batch_episode_count": float(len(episode_keys)),
            **store.metrics(),
        }
        return batch

    def __iter__(self):
        batches = list(iter(self.sampler))
        groups = self._groups(batches)
        if self.full:
            admitted_bytes = sum(self._group_bytes(group) for group in groups)
            if admitted_bytes > self.max_resident_bytes:
                raise MemoryError(
                    "full cache residency exceeds phase3_cache_max_resident_gib: "
                    f"estimated={admitted_bytes / 1024**3:.2f} GiB, "
                    f"budget={self.max_resident_bytes / 1024**3:.2f} GiB"
                )
        store = _EpisodeStore(
            self.dataset,
            max_bytes=self.max_resident_bytes,
            load_workers=self.load_workers,
        )
        batch_executor = ThreadPoolExecutor(
            max_workers=self.batch_workers,
            thread_name_prefix="phase3-group-batch-build",
        )
        pending: OrderedDict[int, Future[dict[str, Any]]] = OrderedDict()
        next_submit = 0
        admitted: set[tuple[int, int]] = set()

        def admit_group_window(group_index: int) -> None:
            end_group = (
                len(groups)
                if self.full
                else min(len(groups), group_index + self.group_prefetch_depth + 1)
            )
            selected: list[tuple[int, int]] = []
            selected_bytes = 0
            for group in groups[group_index:end_group]:
                group_bytes = self._group_bytes(group)
                if (
                    not selected
                    and group_bytes > self.max_resident_bytes
                ):
                    raise MemoryError(
                        "episode group exceeds phase3_cache_max_resident_gib: "
                        f"group={group_bytes / 1024**3:.2f} GiB, "
                        f"budget={self.max_resident_bytes / 1024**3:.2f} GiB"
                    )
                if selected and selected_bytes + group_bytes > self.max_resident_bytes:
                    break
                selected.extend(group)
                selected_bytes += group_bytes
            protected = list(dict.fromkeys(selected))
            store.update_protected(protected)
            if self.warmup:
                store.warm_many(protected)
            for key in protected:
                if key not in admitted:
                    admitted.add(key)
                    store.prefetch(key)

        def fill() -> None:
            nonlocal next_submit
            while next_submit < len(batches) and len(pending) < self.batch_prefetch_depth:
                index = next_submit
                pending[index] = batch_executor.submit(
                    self._build_batch, batches[index], store
                )
                next_submit += 1

        try:
            if self.full:
                admit_group_window(0)
                # Full mode is an explicit preload contract: do not yield a
                # batch until every episode has completed loading.
                store.get_many(list(admitted))
            elif self.warmup:
                admit_group_window(0)
                store.get_many(list(admitted))
            group_for_batch: list[int] = []
            for index, batch in enumerate(batches):
                key_set = set(self._episode_keys(self.dataset, batch))
                while len(group_for_batch) <= index:
                    group_for_batch.append(0)
                for group_index, group in enumerate(groups):
                    if key_set.intersection(group):
                        group_for_batch[index] = group_index
                        break
            active_group = -1
            while next_submit < len(batches) or pending:
                if pending:
                    next_index = next(iter(pending))
                    requested_group = group_for_batch[next_index]
                else:
                    requested_group = group_for_batch[next_submit]
                if requested_group != active_group:
                    active_group = requested_group
                    admit_group_window(active_group)
                fill()
                if not pending:
                    continue
                batch_index, future = pending.popitem(last=False)
                batch = future.result()
                batch["_phase3_loader_metrics"][
                    "loader/group_index"
                ] = float(group_for_batch[batch_index])
                yield batch
        finally:
            batch_executor.shutdown(wait=True, cancel_futures=True)
            store.close()

    def __len__(self) -> int:
        return len(self.sampler)


@dataclass(frozen=True)
class _TransitionRangeLoadResult:
    payload: dict[str, torch.Tensor]
    range_start: int
    range_stop: int
    byte_count: int
    load_time: float


class _TransitionRangeStore:
    """Asynchronously retain bounded episode prefixes for QTran-L context."""

    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        *,
        max_bytes: int,
        load_workers: int,
    ) -> None:
        self.dataset = dataset
        self.max_bytes = max(1, int(max_bytes))
        self.executor = ThreadPoolExecutor(
            max_workers=max(1, int(load_workers)),
            thread_name_prefix="phase3-transition-range-load",
        )
        self.lock = threading.Lock()
        self.cache: OrderedDict[
            tuple[int, int], tuple[dict[str, torch.Tensor], int, int]
        ] = OrderedDict()
        self.pending: dict[
            tuple[tuple[int, int], int], Future[_TransitionRangeLoadResult]
        ] = {}
        self.protected: set[tuple[int, int]] = set()
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.load_count = 0
        self.load_seconds = 0.0
        self.cache_hits = 0
        self.pending_hits = 0
        self.wait_seconds = 0.0

    def _load(
        self,
        episode_key: tuple[int, int],
        range_stop: int,
    ) -> _TransitionRangeLoadResult:
        start = time.perf_counter()
        payload = self.dataset.load_episode_range_payload(
            self.dataset.episode_path(episode_key),
            0,
            int(range_stop),
            materialize=True,
        )
        actual_stop = int(payload["done"].shape[0])
        if actual_stop != int(range_stop):
            raise RuntimeError(
                f"range read for {episode_key} returned {actual_stop} rows, "
                f"expected {range_stop}"
            )
        return _TransitionRangeLoadResult(
            payload=payload,
            range_start=0,
            range_stop=actual_stop,
            byte_count=_tensor_bytes(payload),
            load_time=time.perf_counter() - start,
        )

    def _evict_locked(self) -> None:
        while self.resident_bytes > self.max_bytes and self.cache:
            victim = next(
                (key for key in self.cache if key not in self.protected),
                None,
            )
            if victim is None:
                break
            _, _, byte_count = self.cache.pop(victim)
            self.resident_bytes -= byte_count

    def _complete(
        self,
        pending_key: tuple[tuple[int, int], int],
        future: Future[_TransitionRangeLoadResult],
    ) -> None:
        if future.cancelled() or future.exception() is not None:
            with self.lock:
                self.pending.pop(pending_key, None)
            return
        result = future.result()
        episode_key, requested_stop = pending_key
        with self.lock:
            self.pending.pop(pending_key, None)
            previous = self.cache.pop(episode_key, None)
            if previous is not None:
                self.resident_bytes -= previous[2]
                if previous[1] > result.range_stop:
                    self.cache[episode_key] = previous
                    self.resident_bytes += previous[2]
                    self.cache.move_to_end(episode_key)
                else:
                    self.cache[episode_key] = (
                        result.payload,
                        result.range_stop,
                        result.byte_count,
                    )
                    self.resident_bytes += result.byte_count
            else:
                self.cache[episode_key] = (
                    result.payload,
                    result.range_stop,
                    result.byte_count,
                )
                self.resident_bytes += result.byte_count
            self.peak_resident_bytes = max(
                self.peak_resident_bytes,
                self.resident_bytes,
            )
            self.load_count += 1
            self.load_seconds += result.load_time
            self._evict_locked()

    def prefetch(
        self,
        episode_key: tuple[int, int],
        range_stop: int,
    ) -> Future[_TransitionRangeLoadResult]:
        range_stop = int(range_stop)
        if range_stop <= 0:
            raise ValueError("range_stop must be positive")
        pending_key = (episode_key, range_stop)
        with self.lock:
            cached = self.cache.get(episode_key)
            if cached is not None and cached[1] >= range_stop:
                self.cache.move_to_end(episode_key)
                self.cache_hits += 1
                future: Future[_TransitionRangeLoadResult] = Future()
                future.set_result(
                    _TransitionRangeLoadResult(
                        cached[0], 0, cached[1], cached[2], 0.0
                    )
                )
                return future
            pending = self.pending.get(pending_key)
            if pending is not None:
                self.pending_hits += 1
                return pending
            future = self.executor.submit(self._load, episode_key, range_stop)
            self.pending[pending_key] = future
        future.add_done_callback(
            lambda completed, key=pending_key: self._complete(key, completed)
        )
        return future

    def get_many(
        self,
        requests: dict[tuple[int, int], int],
    ) -> tuple[dict[tuple[int, int], dict[str, torch.Tensor]], float]:
        futures = {
            key: self.prefetch(key, range_stop)
            for key, range_stop in requests.items()
        }
        wait_start = time.perf_counter()
        results = {key: future.result() for key, future in futures.items()}
        wait_time = time.perf_counter() - wait_start
        with self.lock:
            self.wait_seconds += wait_time
            for key in requests:
                if key in self.cache:
                    self.cache.move_to_end(key)
        return {key: result.payload for key, result in results.items()}, wait_time

    def update_protected(self, episode_keys: list[tuple[int, int]]) -> None:
        with self.lock:
            self.protected = set(episode_keys)
            self._evict_locked()

    def metrics(self) -> dict[str, float]:
        with self.lock:
            return {
                "loader/transition_range_load_count_total": float(self.load_count),
                "loader/transition_range_load_seconds_total": self.load_seconds,
                "loader/transition_range_cache_hits_total": float(self.cache_hits),
                "loader/transition_range_pending_hits_total": float(self.pending_hits),
                "loader/transition_range_wait_seconds_total": self.wait_seconds,
                "loader/transition_range_cache_resident_gib": (
                    self.resident_bytes / 1024**3
                ),
                "loader/transition_range_cache_peak_gib": (
                    self.peak_resident_bytes / 1024**3
                ),
            }

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


class TransitionRangePrefetchLoader:
    """Fetch selected rows as coalesced episode prefixes/ranges.

    The sampler is deliberately consumed exactly as before.  For the
    layer-context QTran-L critics, a requested row loads the prefix ending at
    that row because its value/query context includes preceding video chunks.
    MLP critics use the same prefix implementation initially; this keeps one
    parity-tested path and can still avoid materializing the remainder of a
    large episode.
    """

    def __init__(
        self,
        dataset: Phase3CanonicalEpisodeCacheDataset,
        sampler: Sampler[list[int]],
        *,
        transition_prefetch_max_bytes: int,
        transition_prefetch_workers: int,
        transition_prefetch_lookahead: int,
        transition_batch_workers: int,
        batch_prefetch_depth: int,
        max_active_episodes: int,
        pin_memory: bool,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.max_bytes = max(1, int(transition_prefetch_max_bytes))
        self.load_workers = max(1, int(transition_prefetch_workers))
        self.lookahead = max(1, int(transition_prefetch_lookahead))
        self.batch_workers = max(1, int(transition_batch_workers))
        self.batch_prefetch_depth = max(1, int(batch_prefetch_depth))
        self.max_active_episodes = max(1, int(max_active_episodes))
        self.pin_memory = bool(pin_memory)

    def _episode_keys(self, batch: list[int]) -> list[tuple[int, int]]:
        return [self.dataset.episode_key_for_position(position) for position in batch]

    def _requests_for_batches(
        self,
        batches: list[list[int]],
    ) -> dict[tuple[int, int], int]:
        requests: dict[tuple[int, int], int] = {}
        for batch in batches:
            for position in batch:
                row = self.dataset.rows[int(position)]
                key = row["episode_key"]
                stop = int(row["row"]) + 1
                requests[key] = max(requests.get(key, 0), stop)
        return requests

    def _build_batch(
        self,
        batch_indices: list[int],
        store: _TransitionRangeStore,
    ) -> dict[str, Any]:
        build_start = time.perf_counter()
        requests = self._requests_for_batches([batch_indices])
        payloads, range_wait = store.get_many(requests)
        samples = []
        for position in batch_indices:
            key = self.dataset.episode_key_for_position(position)
            samples.append(
                self.dataset.sample_from_payload_range(
                    payloads[key], position, range_start=0
                )
            )
        batch = phase3_canonical_collate(samples)
        if self.pin_memory:
            batch = _pin_memory(batch)
        batch["_phase3_loader_metrics"] = {
            "loader/transition_range_wait_time": range_wait,
            "loader/batch_build_time": time.perf_counter() - build_start,
            "loader/batch_episode_count": float(len(requests)),
            **store.metrics(),
        }
        return batch

    def __iter__(self):
        batches = list(iter(self.sampler))
        store = _TransitionRangeStore(
            self.dataset,
            max_bytes=self.max_bytes,
            load_workers=self.load_workers,
        )
        batch_executor = ThreadPoolExecutor(
            max_workers=self.batch_workers,
            thread_name_prefix="phase3-transition-range-batch",
        )
        pending: OrderedDict[int, Future[dict[str, Any]]] = OrderedDict()
        next_submit = 0

        def maintain_window(anchor: int) -> None:
            end = min(len(batches), anchor + self.lookahead)
            window = batches[anchor:end]
            requests = self._requests_for_batches(window)
            ordered_keys = list(requests)
            protected = ordered_keys[: self.max_active_episodes]
            store.update_protected(protected)
            for key in ordered_keys[: self.max_active_episodes]:
                store.prefetch(key, requests[key])

        def fill() -> None:
            nonlocal next_submit
            while (
                next_submit < len(batches)
                and len(pending) < self.batch_prefetch_depth
            ):
                batch_index = next_submit
                pending[batch_index] = batch_executor.submit(
                    self._build_batch,
                    batches[batch_index],
                    store,
                )
                next_submit += 1

        try:
            maintain_window(0)
            fill()
            while pending:
                batch_index, future = pending.popitem(last=False)
                ready_wait_start = time.perf_counter()
                batch = future.result()
                ready_wait = time.perf_counter() - ready_wait_start
                maintain_window(batch_index + 1)
                fill()
                batch["_phase3_loader_metrics"][
                    "loader/batch_ready_wait_time"
                ] = ready_wait
                yield batch
        finally:
            batch_executor.shutdown(wait=True, cancel_futures=True)
            store.close()

    def __len__(self) -> int:
        return len(self.sampler)


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


class AsyncCudaPrefetcher:
    """Produce FIFO CUDA batches without blocking an already-ready batch."""

    _END = object()

    def __init__(
        self,
        loader,
        device: torch.device,
        *,
        depth: int = 2,
    ) -> None:
        self.loader = loader
        self.device = device
        self.depth = max(1, int(depth))

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        ready: queue.Queue[Any] = queue.Queue(maxsize=self.depth)
        stop = threading.Event()

        def put(value: Any) -> bool:
            while not stop.is_set():
                try:
                    ready.put(value, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce() -> None:
            torch.cuda.set_device(self.device)
            stream = torch.cuda.Stream(device=self.device)
            try:
                for batch in self.loader:
                    if stop.is_set():
                        break
                    with torch.cuda.stream(stream):
                        cuda_batch = _move_to_device(
                            batch,
                            self.device,
                            non_blocking=True,
                        )
                        event = torch.cuda.Event()
                        event.record(stream)
                    # Retain the source tensors until the consumer finishes
                    # using this batch; pinned H2D copies may still be live.
                    if not put((cuda_batch, event, batch, None)):
                        return
            except BaseException as exc:
                put((None, None, None, exc))
            finally:
                put(self._END)

        producer = threading.Thread(
            target=produce,
            name="phase3-cuda-prefetch",
            daemon=False,
        )
        producer.start()
        try:
            while True:
                item = ready.get()
                if item is self._END:
                    break
                batch, event, source_batch, error = item
                if error is not None:
                    raise error
                current = torch.cuda.current_stream(self.device)
                current.wait_event(event)
                _record_stream(batch, current)
                yield batch
                del source_batch
        finally:
            stop.set()
            # The producer may still be unwinding the episode/batch executors.
            # Do not let interpreter teardown race a live CUDA-owning thread.
            producer.join()

    def __len__(self) -> int:
        return len(self.loader)
