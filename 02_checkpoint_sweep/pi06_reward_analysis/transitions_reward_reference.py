from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# π0.6 time-cost scale used by the RoboCOIN critic: a nonterminal transition
# costs 0.001, successful termination costs 0, and failed termination costs 1.
TERMINAL_OUTCOME_COST_STEP_COST = 1e-3
TERMINAL_OUTCOME_COST_FAIL_PENALTY = 1.0


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
            "terminal_outcome_cost",
        }:
            raise ValueError(
                "reward_source must be 'sparse_success', "
                "'jepa_delta_distance', "
                "'negative_predicted_actual_jepa_distance', or "
                "'terminal_outcome_cost'"
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
            success = bool(segments[0].get("success", False))
            truncated = bool(segments[0].get("truncated", False))

            for segment_index, segment in enumerate(segments):
                frame_count = int(segment["latent_frame_count"])
                is_final_segment = segment_index == len(segments) - 1
                if (
                    not is_final_segment
                    and frame_count % self.chunk_size != 0
                ):
                    raise ValueError(
                        "Non-final storage segment in episode "
                        f"{episode_key} has {frame_count} latent frames, "
                        f"not divisible by chunk size {self.chunk_size}"
                    )
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
                        )
                    )
                    chunk_index += 1

            if len(records) == episode_start:
                continue
            episode_end = len(records)
            episode_ranges.append((episode_start, episode_end))
            for index in range(episode_start, episode_end):
                is_storage_end = index == episode_end - 1
                # A cached VA window is not automatically an episode. The
                # default preserves legacy datasets; canonical pretraining
                # loaders set source_is_terminal only for the real endpoint.
                is_terminal = bool(
                    is_storage_end
                    and segments[-1].get("source_is_terminal", True)
                )
                has_next_storage_record = index < episode_end - 1
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
                        # A nonterminal window tail bootstraps from its current
                        # state; it must never point into the next storage window.
                        "next_record_idx": (
                            index + 1 if has_next_storage_record else None
                        ),
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
        if self.reward_source == "terminal_outcome_cost":
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
        if self.reward_source == "terminal_outcome_cost":
            if not is_terminal:
                return -TERMINAL_OUTCOME_COST_STEP_COST
            return 0.0 if success else -TERMINAL_OUTCOME_COST_FAIL_PENALTY
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

class PairedWindowTransitionDataset(Dataset):
    """Pair every VA storage window with one chronological TD transition."""

    def __init__(
        self,
        base_dataset: Dataset,
        transition_dataset: ChunkTransitionDataset,
        selection_seed: int,
    ) -> None:
        if len(base_dataset) == 0:
            raise ValueError("paired VA dataset is empty")
        self.base_dataset = base_dataset
        self.transition_dataset = transition_dataset
        self.selection_seed = int(selection_seed)
        sample_for_record = {
            int(record_idx): sample_idx
            for sample_idx, record_idx in enumerate(transition_dataset.sample_indices)
        }
        valid_by_window: dict[int, list[int]] = defaultdict(list)
        all_by_window: dict[int, list[int]] = defaultdict(list)
        for record_idx, record in enumerate(transition_dataset.records):
            if record_idx not in sample_for_record:
                continue
            all_by_window[int(record.dataset_idx)].append(record_idx)
            if record.previous_record_idx is not None:
                valid_by_window[int(record.dataset_idx)].append(record_idx)

        self._critic_sample_index: list[int] = []
        true_terminal_selected = deterministic_selected = invalid = 0
        window_end_nonterminal_selected = 0
        true_terminal_records = sum(
            int(record.done) for record in transition_dataset.records
        )
        by_source = defaultdict(
            lambda: {"terminal": 0, "deterministic": 0, "invalid": 0}
        )
        for dataset_idx in range(len(base_dataset)):
            valid = valid_by_window.get(dataset_idx, [])
            terminal = [
                record_idx for record_idx in valid
                if transition_dataset.records[record_idx].done
            ]
            available = terminal or valid or all_by_window.get(dataset_idx, [])
            if not available:
                raise ValueError(f"VA window {dataset_idx} has no TD records")
            source = int(transition_dataset.records[available[0]].episode_key[0])
            if terminal:
                selected = terminal[-1]
                true_terminal_selected += 1
                by_source[source]["terminal"] += 1
            elif valid:
                # Stable across Python processes/ranks; do not use hash().
                position = (
                    self.selection_seed * 1000003 + dataset_idx * 97531
                ) % len(valid)
                selected = valid[position]
                deterministic_selected += 1
                by_source[source]["deterministic"] += 1
            else:
                # The paired transition stays in this window; Lc is zeroed.
                selected = available[0]
                invalid += 1
                by_source[source]["invalid"] += 1
            self._critic_sample_index.append(sample_for_record[selected])
            selected_record = transition_dataset.records[selected]
            if (
                selected_record.next_record_idx is None
                and not selected_record.done
            ):
                window_end_nonterminal_selected += 1

        self.selection_summary = {
            "windows": len(self._critic_sample_index),
            "true_terminal_selected": true_terminal_selected,
            "terminal_selected": true_terminal_selected,
            "window_end_nonterminal_selected": window_end_nonterminal_selected,
            "deterministic_selected": deterministic_selected,
            "critic_invalid": invalid,
            "critic_valid_fraction": (
                (len(self._critic_sample_index) - invalid)
                / len(self._critic_sample_index)
            ),
            "true_terminal_record_count": true_terminal_records,
            "by_source_id": {
                str(key): value for key, value in sorted(by_source.items())
            },
            "selection_seed": self.selection_seed,
        }

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        critic = self.transition_dataset[self._critic_sample_index[int(index)]]
        return {
            "va": self.base_dataset[int(index)],
            "critic": critic,
            "critic_valid": torch.tensor(
                bool(critic["state_valid"].item()), dtype=torch.bool
            ),
        }
