from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ChunkRecord:
    dataset_idx: int
    episode_key: tuple[int, int]
    rl_chunk_idx: int
    frame_start: int
    valid_frames: int
    reward: float
    done: bool
    truncated: bool
    discount: float
    previous_record_idx: int | None = None
    next_record_idx: int | None = None
    mc_return: float = 0.0
    episode_success: bool = False
    episode_chunk_count: int = 0
    # Absolute position in the logical episode prefix, measured in valid
    # latent frames. This is separate from frame_start, which is local to
    # the storage segment and is used to slice that segment.
    episode_frame_start: int = 0


class ChunkTransitionDataset(Dataset):
    """View latent storage segments as inference-sized offline RL transitions."""

    _FRAME_DIMS = {
        "latents": 1,
        "actions": 1,
        "actions_mask": 1,
        "latents_mask": 0,
    }
    _OPTIONAL_FRAME_DIMS = {
        "jepa_target": 0,
    }
    _OPTIONAL_KEYS = {
        "jepa_available",
    }

    def __init__(
        self,
        base_dataset: Dataset,
        infer_latent_chunk_size: int,
        action_per_frame: int,
        gamma: float = 0.99,
        reward_source: str = "sparse_success",
        include_sparse_success_reward: bool = False,
        jepa_reward_weight: float = 1.0,
        success_reward_weight: float = 1.0,
        require_outcomes: bool = True,
        assume_missing_success_is_success: bool = False,
        include_previous: bool = True,
        include_next: bool = False,
        exclude_last_n_chunks: int = 0,
    ) -> None:
        if infer_latent_chunk_size <= 0:
            raise ValueError("infer_latent_chunk_size must be positive")
        if action_per_frame <= 0:
            raise ValueError("action_per_frame must be positive")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if exclude_last_n_chunks < 0:
            raise ValueError("exclude_last_n_chunks must be non-negative")
        if reward_source not in {
            "sparse_success",
            "jepa_delta_distance",
            "negative_predicted_actual_jepa_distance",
        }:
            raise ValueError(
                "reward_source must be 'sparse_success', "
                "'jepa_delta_distance', or "
                "'negative_predicted_actual_jepa_distance'"
            )
        if not hasattr(base_dataset, "get_rl_segment_metadata"):
            raise TypeError(
                "base_dataset must implement get_rl_segment_metadata(idx)"
            )

        self.base_dataset = base_dataset
        self.chunk_size = infer_latent_chunk_size
        self.action_per_frame = action_per_frame
        self.gamma = gamma
        self.reward_source = reward_source
        self.include_sparse_success_reward = include_sparse_success_reward
        self.jepa_reward_weight = float(jepa_reward_weight)
        self.success_reward_weight = float(success_reward_weight)
        self.assume_missing_success_is_success = bool(
            assume_missing_success_is_success
        )
        self.include_previous = include_previous
        self.include_next = include_next
        self.exclude_last_n_chunks = int(exclude_last_n_chunks)
        self.records = self._build_records(require_outcomes=require_outcomes)
        self.sample_indices = self._build_sample_indices()
        if not self.sample_indices:
            raise ValueError(
                "ChunkTransitionDataset has no trainable rows after applying "
                f"exclude_last_n_chunks={self.exclude_last_n_chunks}"
            )

    def _build_sample_indices(self) -> list[int]:
        if self.exclude_last_n_chunks == 0:
            return list(range(len(self.records)))
        return [
            index
            for index, record in enumerate(self.records)
            if record.rl_chunk_idx
            < max(0, record.episode_chunk_count - self.exclude_last_n_chunks)
        ]

    def _build_records(self, require_outcomes: bool) -> list[ChunkRecord]:
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for dataset_idx in range(len(self.base_dataset)):
            metadata = dict(
                self.base_dataset.get_rl_segment_metadata(dataset_idx)
            )
            metadata.setdefault("dataset_idx", dataset_idx)
            dataset_id = int(metadata.get("dataset_id", 0))
            episode_index = int(metadata["episode_index"])
            if require_outcomes and "success" not in metadata:
                if self.assume_missing_success_is_success:
                    metadata["success"] = True
                else:
                    raise ValueError(
                        f"Episode {(dataset_id, episode_index)} has no success label"
                    )
            grouped.setdefault((dataset_id, episode_index), []).append(metadata)

        records: list[ChunkRecord] = []
        episode_ranges: list[tuple[int, int]] = []
        for episode_key in sorted(grouped):
            segments = sorted(
                grouped[episode_key],
                key=lambda item: (
                    int(item["start_frame"]),
                    int(item["end_frame"]),
                ),
            )
            self._validate_segments(episode_key, segments)
            episode_start = len(records)
            chunk_index = 0
            episode_frame_cursor = 0
            success = bool(segments[0].get("success", False))
            truncated = bool(segments[0].get("truncated", False))

            for segment_index, segment in enumerate(segments):
                frame_count = int(segment["latent_frame_count"])
                for frame_start in range(0, frame_count, self.chunk_size):
                    valid_frames = min(
                        self.chunk_size, frame_count - frame_start
                    )
                    reward = self._base_reward(
                        segment,
                        frame_start=frame_start,
                        valid_frames=valid_frames,
                    )
                    records.append(
                        ChunkRecord(
                            dataset_idx=int(segment["dataset_idx"]),
                            episode_key=episode_key,
                            rl_chunk_idx=chunk_index,
                            frame_start=frame_start,
                            valid_frames=valid_frames,
                            reward=reward,
                            done=False,
                            truncated=False,
                            discount=self.gamma,
                            episode_frame_start=(
                                episode_frame_cursor + frame_start
                            ),
                        )
                    )
                    chunk_index += 1
                # Keep an odd storage-segment remainder as a masked short
                # chunk. Advance by valid frames rather than padded storage
                # frames so the next segment starts immediately after real
                # data.
                episode_frame_cursor += frame_count

            if len(records) == episode_start:
                continue
            episode_end = len(records)
            episode_ranges.append((episode_start, episode_end))
            for index in range(episode_start, episode_end):
                is_terminal = index == episode_end - 1
                record = records[index]
                records[index] = ChunkRecord(
                    **{
                        **record.__dict__,
                        "reward": self._final_reward(
                            record.reward,
                            success=success,
                            is_terminal=is_terminal,
                        ),
                        "done": is_terminal,
                        "truncated": bool(truncated and is_terminal),
                        "discount": 0.0 if is_terminal else record.discount,
                        "episode_success": success,
                        "episode_chunk_count": episode_end - episode_start,
                        "previous_record_idx": (
                            None if index == episode_start else index - 1
                        ),
                        "next_record_idx": None if is_terminal else index + 1,
                    }
                )

        for episode_start, episode_end in episode_ranges:
            running_return = 0.0
            for index in range(episode_end - 1, episode_start - 1, -1):
                record = records[index]
                running_return = (
                    record.reward + record.discount * running_return
                )
                records[index] = ChunkRecord(
                    **{**record.__dict__, "mc_return": running_return}
                )
        return records

    def _base_reward(
        self,
        segment: dict[str, Any],
        frame_start: int,
        valid_frames: int,
    ) -> float:
        if self.reward_source == "sparse_success":
            return 0.0
        if self.reward_source == "negative_predicted_actual_jepa_distance":
            return 0.0
        reward_config = segment.get("reward_config")
        if not isinstance(reward_config, dict):
            raise ValueError(
                "Missing reward_config for reward_source='jepa_delta_distance'"
            )
        if reward_config.get("reward_source") != "jepa_delta_distance":
            raise ValueError(
                "reward_config.reward_source must be 'jepa_delta_distance'"
            )
        rewards = reward_config.get("latent_rewards")
        if not isinstance(rewards, list):
            raise ValueError("reward_config.latent_rewards must be a list")
        reward_end = frame_start + valid_frames
        if reward_end > len(rewards):
            raise ValueError(
                "reward_config.latent_rewards is shorter than the segment's "
                "latent frame count"
            )
        return self.jepa_reward_weight * float(
            sum(float(value) for value in rewards[frame_start:reward_end])
        )

    def _final_reward(
        self,
        base_reward: float,
        success: bool,
        is_terminal: bool,
    ) -> float:
        sparse_reward = float(success and is_terminal)
        if self.reward_source == "sparse_success":
            return self.success_reward_weight * sparse_reward
        if self.reward_source == "negative_predicted_actual_jepa_distance":
            if self.include_sparse_success_reward:
                return base_reward + self.success_reward_weight * sparse_reward
            return base_reward
        if self.include_sparse_success_reward:
            return base_reward + self.success_reward_weight * sparse_reward
        return base_reward

    @staticmethod
    def _validate_segments(
        episode_key: tuple[int, int],
        segments: list[dict[str, Any]],
    ) -> None:
        outcome_fields = ("success", "truncated", "termination_reason")
        first = segments[0]
        for previous, current in zip(segments, segments[1:]):
            if int(current["start_frame"]) < int(previous["start_frame"]):
                raise ValueError(f"Unordered segments in episode {episode_key}")
            if (
                int(current["start_frame"]) == int(previous["start_frame"])
                and int(current["end_frame"]) == int(previous["end_frame"])
            ):
                raise ValueError(f"Duplicate segment in episode {episode_key}")
        termination_frame = first.get("termination_frame")
        if (
            termination_frame is not None
            and int(segments[-1]["end_frame"]) < int(termination_frame)
        ):
            raise ValueError(
                f"Episode {episode_key} ends before termination_frame"
            )
        for segment in segments[1:]:
            for field in outcome_fields:
                if (
                    field in first
                    and field in segment
                    and segment[field] != first[field]
                ):
                    raise ValueError(
                        f"Inconsistent {field} in episode {episode_key}"
                    )

    def __len__(self) -> int:
        return len(self.sample_indices)

    @staticmethod
    def _slice_and_pad(
        tensor: torch.Tensor,
        frame_start: int,
        valid_frames: int,
        chunk_size: int,
        frame_dim: int,
    ) -> torch.Tensor:
        sliced = tensor.narrow(frame_dim, frame_start, valid_frames)
        if valid_frames == chunk_size:
            return sliced
        padding = [0, 0] * tensor.ndim
        pair_index = 2 * (tensor.ndim - frame_dim - 1)
        padding[pair_index + 1] = chunk_size - valid_frames
        return F.pad(sliced, tuple(padding))

    def _load_record(self, record: ChunkRecord) -> dict[str, torch.Tensor]:
        segment = self.base_dataset[record.dataset_idx]
        chunk: dict[str, torch.Tensor] = {}
        for key, value in segment.items():
            if key in self._FRAME_DIMS:
                chunk[key] = self._slice_and_pad(
                    value,
                    record.frame_start,
                    record.valid_frames,
                    self.chunk_size,
                    self._FRAME_DIMS[key],
                )
            elif key in self._OPTIONAL_FRAME_DIMS:
                chunk[key] = self._slice_and_pad(
                    value,
                    record.frame_start,
                    record.valid_frames,
                    self.chunk_size,
                    self._OPTIONAL_FRAME_DIMS[key],
                )
            elif key in self._OPTIONAL_KEYS:
                chunk[key] = value
            elif key == "text_emb":
                chunk[key] = value

        required = set(self._FRAME_DIMS) | {"text_emb"}
        missing = required.difference(chunk)
        if missing:
            raise KeyError(f"Base sample is missing RL fields: {sorted(missing)}")
        return chunk

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx = int(self.sample_indices[int(idx)])
        record = self.records[record_idx]
        current = self._load_record(record)
        result: dict[str, Any] = {
            **current,
            "transition_idx": torch.tensor(record_idx, dtype=torch.long),
            "dataset_idx": torch.tensor(record.dataset_idx, dtype=torch.long),
            "episode_index": torch.tensor(
                record.episode_key[1], dtype=torch.long
            ),
            "rl_chunk_idx": torch.tensor(
                record.rl_chunk_idx, dtype=torch.long
            ),
            "reward": torch.tensor(record.reward, dtype=torch.float32),
            "done": torch.tensor(record.done, dtype=torch.bool),
            "truncated": torch.tensor(record.truncated, dtype=torch.bool),
            "discount": torch.tensor(record.discount, dtype=torch.float32),
            "mc_return": torch.tensor(record.mc_return, dtype=torch.float32),
            "episode_success": torch.tensor(
                record.episode_success, dtype=torch.bool
            ),
            "episode_chunk_count": torch.tensor(
                record.episode_chunk_count, dtype=torch.long
            ),
            "state_valid": torch.tensor(
                record.previous_record_idx is not None, dtype=torch.bool
            ),
            "next_transition_idx": torch.tensor(
                -1 if record.next_record_idx is None else record.next_record_idx,
                dtype=torch.long,
            ),
        }
        if "jepa_target" in current:
            goal_record = record
            while goal_record.next_record_idx is not None:
                goal_record = self.records[goal_record.next_record_idx]
            goal_chunk = self._load_record(goal_record)
            result["goal_jepa_target"] = goal_chunk["jepa_target"]
            result["goal_latents_mask"] = goal_chunk["latents_mask"]
            if "jepa_available" in goal_chunk:
                result["goal_jepa_available"] = goal_chunk["jepa_available"]

        if self.include_previous:
            if record.previous_record_idx is None:
                previous_chunk = {
                    key: torch.zeros_like(value)
                    for key, value in current.items()
                }
            else:
                previous_chunk = self._load_record(
                    self.records[record.previous_record_idx]
                )
            result.update(
                {
                    f"previous_{key}": value
                    for key, value in previous_chunk.items()
                }
            )
        if self.include_next:
            if record.next_record_idx is None:
                next_chunk = {
                    key: torch.zeros_like(value)
                    for key, value in current.items()
                }
            else:
                next_chunk = self._load_record(
                    self.records[record.next_record_idx]
                )
            result.update(
                {
                    f"next_{key}": value
                    for key, value in next_chunk.items()
                }
            )
            result["next_state_valid"] = torch.tensor(
                record.next_record_idx is not None, dtype=torch.bool
            )
        return result
