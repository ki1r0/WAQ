#!/usr/bin/env python3
"""Post-hoc TD-error curves for frozen RobotWin SFT rollouts.

Usage
-----
Prepare a replay or task-level goal once, then score one checkpoint:

    python scripts/plot_robotwin_sft_td_checkpoint_sweep.py score \
        --task place_can_basket \
        --checkpoint /path/to/checkpoints/checkpoint_00030000 \
        --sft-replay /path/to/sft_replay \
        --clean-root /path/to/clean50/place_can_basket \
        --clean-cache-dir /path/to/clean_cache/place_can_basket \
        --clean-episode 0 --clean-episode 1 \
        --reward-metadata /path/to/reward_metadata.pt \
        --expected-catalog-sha256 <sha256> \
        --output /path/to/output/place_can_basket/checkpoint_00030000

Sweep an inclusive checkpoint range with ``(start, end, step)``:

    python scripts/plot_robotwin_sft_td_checkpoint_sweep.py sweep \
        --task place_can_basket \
        --checkpoint-root /path/to/checkpoints \
        --start-step 30000 --end-step 50000 --step 2500 \
        --sft-replay /path/to/sft_replay \
        --clean-root /path/to/clean50/place_can_basket \
        --clean-cache-dir /path/to/clean_cache/place_can_basket \
        --clean-episode 0 --clean-episode 1 \
        --reward-metadata /path/to/reward_metadata.pt \
        --expected-catalog-sha256 <sha256> \
        --output-root /path/to/output

The sweep writes one directory per checkpoint plus ``sweep_summary.json``.

This is intentionally a two-stage diagnostic:

``prepare-sft`` deterministically replays the *base* model from archived SFT
observations and stores the resulting predicted latent chunks alongside real
latent chunks and recorded model-space actions.  The original stochastic
predicted chunks were not archived, so this is not a bitwise replay of SFT.

``score`` applies a requested QTran-L checkpoint to that immutable replay and
to the exact frozen Clean-50 Phase-3 feature-cache rows.  It uses the reward
configuration and gamma in the checkpoint.  The sole post-hoc override is the
goal-JEPA progress target: the mean terminal representation of the matching
Clean-50 task is used for every SFT and Clean trajectory.

``sweep`` repeats ``score`` for an inclusive checkpoint step range under one
checkpoint root.  Completed output directories are skipped, while partial
outputs and missing requested checkpoints fail closed.

The script never writes to a checkpoint, rollout, or training cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import cv2
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


REPO_ROOT = Path(os.environ.get("LINGBOT_ROOT", "/path/to/home/wy/lingbot-va-original-rl"))
VJEPA_SOURCE = Path(os.environ.get("VJEPA_SOURCE", "/path/to/home/wy/vjepa2_source_restore"))
for _path in (REPO_ROOT, VJEPA_SOURCE):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from wan_va.rl.guidance import _cat_context, _flatten_action_context, _flatten_video_context, build_clean_feature_input, load_q_guidance_artifact
from wan_va.rl.jepa_encoder import build_jepa_encoder
from wan_va.wan_va_server import VA_CONFIGS, VA_Server
from wan_va.wan_va_server_q_guiding import QGuidedVA_Server


CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
TERMINAL_OUTCOME_RAW = {"nonterminal": -1.0, "success": 0.0, "failure": -1000.0}
SUCCESS_LANGUAGE_SIDECAR = Path(os.environ.get(
    "ROBOTWIN_SUCCESS_LANGUAGE_SIDECAR",
    "/path/to/mnt2/robotwin_language_sidecar_rw50_exact_20260802/manifest.json",
))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def checkpoint_step(path: Path) -> int:
    suffix = path.name.removeprefix("checkpoint_")
    if not suffix.isdigit():
        raise ValueError(f"invalid checkpoint directory name: {path}")
    return int(suffix)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def clean_language_tokens(clean_root: Path, task: str, episodes: list[int]) -> tuple[dict[int, tuple[torch.Tensor, torch.Tensor]], dict[str, str]]:
    """Load the exact Clean-50 text sidecar rows used by QTran-L training."""
    if not SUCCESS_LANGUAGE_SIDECAR.is_file():
        raise FileNotFoundError(
            "the original Clean-50 language sidecar is unavailable; refusing "
            f"to substitute task-generic prompts: {SUCCESS_LANGUAGE_SIDECAR}"
        )
    manifest = json.loads(SUCCESS_LANGUAGE_SIDECAR.read_text(encoding="utf-8"))
    matches = []
    for source in manifest.get("sources", []):
        if source.get("task_name") != task:
            continue
        for dataset in source.get("datasets", []):
            if Path(str(dataset.get("source_dataset", ""))).resolve() == clean_root.resolve():
                matches.append(dataset)
    if len(matches) != 1:
        raise RuntimeError(f"expected one exact Clean-50 language sidecar entry for {task}, got {matches}")
    dataset = matches[0]
    episode_metadata = clean_root / "meta" / "episodes.jsonl"
    observed_sha = sha256(episode_metadata)
    if observed_sha != dataset.get("source_episodes_sha256"):
        raise RuntimeError(
            f"Clean-50 language metadata mismatch for {task}: "
            f"{observed_sha} != {dataset.get('source_episodes_sha256')}"
        )
    path = SUCCESS_LANGUAGE_SIDECAR.parent / str(dataset["file"])
    if not path.is_file() or sha256(path) != dataset.get("sha256"):
        raise RuntimeError(f"Clean-50 language sidecar integrity check failed: {path}")
    values: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        token_rows, mask_rows = source.get_slice("text_tokens"), source.get_slice("text_mask")
        for episode in episodes:
            tokens = token_rows[episode].clone().unsqueeze(0)
            mask = mask_rows[episode].clone().bool().unsqueeze(0)
            if not bool(mask.any()):
                raise RuntimeError(f"empty Clean-50 instruction embedding for {task} episode {episode}")
            values[episode] = (tokens, mask)
    return values, {
        "manifest": str(SUCCESS_LANGUAGE_SIDECAR),
        "manifest_sha256": sha256(SUCCESS_LANGUAGE_SIDECAR),
        "sidecar": str(path),
        "sidecar_sha256": sha256(path),
        "episodes_metadata_sha256": observed_sha,
    }


def terminal_goal(clean_root: Path) -> torch.Tensor:
    paths = sorted((clean_root / "jepa").glob("chunk-*/*.pt"))
    if len(paths) != 50:
        raise RuntimeError(f"expected 50 Jepa files under {clean_root}, got {len(paths)}")
    goals: list[torch.Tensor] = []
    for index, path in enumerate(paths, start=1):
        # The goal needs only the final two feature frames.  Memory mapping
        # avoids eagerly materialising all 50 complete sequences (tens of GB
        # of transient CPU RAM) before taking those terminal slices.
        payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        views = []
        for key in CAMERA_KEYS:
            value = payload[key]
            if value.ndim != 4 or value.shape[0] < 1:
                raise ValueError(f"invalid Jepa payload {path}: {key} {tuple(value.shape)}")
            # Slice before dtype conversion.  These mmap-backed tensors contain
            # an entire episode; calling ``.float()`` first materialises it all.
            terminal = value[0] if value.shape[0] == 1 else 0.5 * (value[-2] + value[-1])
            views.append(terminal.float())
        combined = torch.cat([torch.cat(views[1:], dim=1), views[0]], dim=0)
        goals.append(F.avg_pool2d(combined.permute(2, 0, 1).unsqueeze(0), 2, 2).squeeze(0).permute(1, 2, 0))
        # ``torch.load`` objects can retain mmap metadata until collection;
        # release each full-episode payload before advancing to the next.
        del payload, views, combined
        if index % 5 == 0 or index == len(paths):
            gc.collect()
            print(json.dumps({"event": "clean_goal_terminal_read", "completed": index, "total": len(paths)}), flush=True)
    goal = torch.stack(goals).mean(0).contiguous()
    if tuple(goal.shape) != (12, 10, 1664):
        raise ValueError(f"unexpected global goal shape {tuple(goal.shape)}")
    return goal


def load_global_goal(path: Path) -> torch.Tensor:
    """Load the immutable task-level Clean-50 terminal mean."""
    if not path.is_file():
        raise FileNotFoundError(f"global goal artifact is absent: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    goal = payload.get("global_goal") if isinstance(payload, dict) else payload
    if not isinstance(goal, torch.Tensor):
        raise ValueError(f"global goal artifact has no tensor under 'global_goal': {path}")
    goal = goal.float().contiguous()
    if tuple(goal.shape) != (12, 10, 1664):
        raise ValueError(f"unexpected global goal shape {tuple(goal.shape)} from {path}")
    return goal


def prepare_goal(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(args.output)
    goal = terminal_goal(args.clean_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "global_goal": goal,
            "provenance": {
                "clean_root": str(args.clean_root),
                "episode_count": 50,
                "definition": "mean of each Clean-50 episode's final two JEPA feature frames, assembled in RobotWin camera layout and 2x2 spatially pooled",
            },
        },
        args.output,
    )
    print(json.dumps({"event": "complete", "stage": "prepare-goal", "output": str(args.output), "shape": list(goal.shape), "sha256": sha256(args.output)}), flush=True)


def aligned_clean_jepa(clean_root: Path, episode_index: int, frames: int) -> torch.Tensor:
    path = clean_root / "jepa" / "chunk-000" / f"episode_{episode_index:06d}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    aligned: list[torch.Tensor] = []
    for key in CAMERA_KEYS:
        features = payload[key]
        current = torch.empty((frames, *features.shape[1:]), dtype=torch.float32)
        current[0] = features[0].float()
        if frames > 1:
            pairs = min((features.shape[0] - 1) // 2, frames - 1)
            if pairs:
                # Read precisely the needed feature pairs from the mmap-backed
                # episode rather than expanding the whole sequence into RAM.
                odd = features[1 : 1 + 2 * pairs : 2]
                even = features[2 : 2 + 2 * pairs : 2]
                current[1 : 1 + pairs] = 0.5 * (odd.float() + even.float())
            if pairs < frames - 1:
                current[1 + pairs :] = current[pairs] if pairs else current[0]
        aligned.append(current)
    combined = torch.cat([torch.cat(aligned[1:], dim=2), aligned[0]], dim=1)
    return F.avg_pool2d(combined.permute(0, 3, 1, 2), 2, 2).permute(0, 2, 3, 1).contiguous()


def cosine_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left, right = F.normalize(left.float(), dim=-1), F.normalize(right.float(), dim=-1)
    return (1.0 - (left * right).sum(-1)).mean(dim=tuple(range(1, left.ndim - 1)))


def split_robotwin_latents(server: QGuidedVA_Server, latents: torch.Tensor) -> list[torch.Tensor]:
    height, width = int(server.job_config.height) // 16, int(server.job_config.width) // 16
    wrist_h, wrist_w = height // 2, width // 2
    wrist, high = latents[:, :, :, :wrist_h], latents[:, :, :, wrist_h : wrist_h + height, :width]
    return [high, wrist[:, :, :, :, :wrist_w], wrist[:, :, :, :, wrist_w : wrist_w + wrist_w]]


def decode_latent_camera(server: QGuidedVA_Server, latents: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(server.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
    std = (1.0 / torch.tensor(server.vae.config.latents_std, device=latents.device, dtype=latents.dtype)).view(1, -1, 1, 1, 1)
    return (server.vae.decode(latents / std + mean, return_dict=False)[0].float() * 0.5 + 0.5).clamp(0, 1)


def jepa_from_latents(server: QGuidedVA_Server, encoder: torch.nn.Module, latents: torch.Tensor) -> torch.Tensor:
    converted: list[torch.Tensor] = []
    for camera in split_robotwin_latents(server, latents):
        video = decode_latent_camera(server, camera)
        video = torch.cat([video[:, :, :1], video], dim=2)
        mean = torch.tensor([.485, .456, .406], device=video.device, dtype=video.dtype).view(1, 3, 1, 1, 1)
        std = torch.tensor([.229, .224, .225], device=video.device, dtype=video.dtype).view(1, 3, 1, 1, 1)
        features = encoder((video - mean) / std)
        batch, _, frames, height, width = video.shape
        features = features.view(batch, frames // 2, height // 16, width // 16, features.shape[-1])
        target = torch.empty((batch, latents.shape[2], *features.shape[2:]), device=features.device, dtype=torch.float32)
        target[:, 0] = features[:, 0].float()
        if latents.shape[2] > 1:
            odd, even = features[:, 1::2].float(), features[:, 2::2].float()
            pairs = min(odd.shape[1], even.shape[1], latents.shape[2] - 1)
            if pairs:
                target[:, 1 : 1 + pairs] = 0.5 * (odd[:, :pairs] + even[:, :pairs])
            if pairs < latents.shape[2] - 1:
                target[:, 1 + pairs :] = target[:, pairs : pairs + 1] if pairs else target[:, :1]
        converted.append(target)
    combined = torch.cat([torch.cat(converted[1:], dim=3), converted[0]], dim=2)
    batch, frames, height, width, dim = combined.shape
    pooled = F.avg_pool2d(combined.permute(0, 1, 4, 2, 3).flatten(0, 1), 2, 2)
    return pooled.view(batch, frames, dim, height // 2, width // 2).permute(0, 1, 3, 4, 2).contiguous()


def move_observation_vaes(server: QGuidedVA_Server, device: str | torch.device) -> None:
    wrappers = [server.streaming_vae]
    if server.streaming_vae_half is not None:
        wrappers.append(server.streaming_vae_half)
    for wrapper in wrappers:
        wrapper.vae.to(device)
        wrapper.feat_cache = [None if value is None else value.to(device) for value in wrapper.feat_cache]


def encode_saved_observations(server: QGuidedVA_Server, observations: list[dict[str, np.ndarray]]) -> torch.Tensor:
    """Use the causal VAE in bounded groups, matching the evaluator cadence."""
    if not observations:
        raise ValueError("no saved observations")
    outputs = []
    for start in range(0, len(observations), 4):
        group = observations[start : start + 4]
        if len(group) != 4:
            raise RuntimeError(f"incomplete four-observation VAE group: {len(group)}")
        outputs.append(server._encode_obs({"obs": group}))
    return torch.cat(outputs, dim=2)


def make_replay_server(scratch: Path) -> VA_Server:
    config = deepcopy(VA_CONFIGS["robotwin_place_can_basket_official_dataset"])
    config.rank = config.local_rank = 0
    config.world_size = 1
    config.save_root = str(scratch)
    config.attn_mode = "torch"
    # The sampler plus two duplicate RobotWin VAE weight copies leave too
    # little room for a four-frame cache refresh on an 80 GB GPU.  Construct
    # VAE copies on CPU, retain separate streaming wrappers (and therefore
    # separate causal caches), then share one immutable parameter copy below.
    # This is diagnostic-only and does not affect any checkpoint, rollout, or
    # training process.
    config.enable_offload = True
    # The rollouts being replayed were generated without Q guidance.  Use the
    # base server rather than constructing an inactive critic/guidance artifact
    # that consumes VRAM but cannot change a zero-guidance prediction.
    server = VA_Server(config, robotwin_eval=True)
    if server.streaming_vae_half is not None:
        duplicate_vae = server.streaming_vae_half.vae
        server.streaming_vae_half.vae = server.vae
        server.streaming_vae_half.encoder = server.vae.encoder
        server.streaming_vae_half.quant_conv = server.vae.quant_conv
        del duplicate_vae
    # The base sampler remains on GPU 0.  The VAE and its causal feature
    # caches live on the idle GPU 1; _encode_obs transfers only the resulting
    # latents back to the sampler device.  That preserves the normal encoder
    # computation while removing the long-rollout GPU-0 memory conflict.
    move_observation_vaes(server, "cuda:1")
    # Keep prompt encoding GPU-resident.  The prompt cache avoids repeating it
    # across the multiple chunks of this isolated trajectory replay.
    server.text_encoder.to(server.device)
    return server


def make_server(checkpoint: Path, scratch: Path) -> QGuidedVA_Server:
    """Construct the QTran-L scorer without putting unused VAEs on the GPU."""
    config = deepcopy(VA_CONFIGS["robotwin_place_can_basket_official_dataset"])
    config.rank = config.local_rank = 0
    config.world_size = 1
    config.save_root = str(scratch)
    config.attn_mode = "torch"
    config.enable_offload = True
    server = QGuidedVA_Server(config, q_checkpoint=str(checkpoint), q_guidance_scale=0.0, q_history_chunks=10, robotwin_eval=True)
    server.text_encoder.to(server.device)
    return server


def reset_episode(server: QGuidedVA_Server, prompt: str) -> None:
    # With bounded VAE/text offload, encoding the same immutable instruction
    # on every replay reset dominates runtime.  Preserve the exact first
    # embedding and reuse it; the server state and all causal caches are still
    # reset normally for every trajectory.
    cache = getattr(server, "_td_prompt_cache", None)
    if cache is None:
        cache = {}
        server._td_prompt_cache = cache
    if prompt in cache:
        server._reset(prompt=None)
        prompt_embeds, negative_prompt_embeds = cache[prompt]
        server.prompt_embeds = prompt_embeds.to(device=server.device, dtype=server.dtype)
        server.negative_prompt_embeds = None if negative_prompt_embeds is None else negative_prompt_embeds.to(device=server.device, dtype=server.dtype)
    else:
        server._reset(prompt=prompt)
        cache[prompt] = (
            server.prompt_embeds.detach().cpu(),
            None if server.negative_prompt_embeds is None else server.negative_prompt_embeds.detach().cpu(),
        )
    for key in ("_q_previous_latents", "_q_previous_actions", "_q_previous_latent_mask", "_q_previous_action_mask"):
        setattr(server, key, None)


def h5_obs(handle: h5py.File, index: int) -> dict[str, np.ndarray]:
    paths = ("head_camera", "left_camera", "right_camera")
    values = []
    for camera in paths:
        encoded = np.frombuffer(handle[f"observation/{camera}/rgb"][index], dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"failed to decode {camera} frame {index}")
        values.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return dict(zip(CAMERA_KEYS, values))


def sft_sources(roots: Iterable[Path], task: str) -> list[dict[str, Any]]:
    by_identity: dict[tuple[int, str], dict[str, Any]] = {}
    for root in roots:
        for path in sorted(root.glob(f"rollouts/**/{task}/demo_clean/data/*.hdf5")):
            with h5py.File(path, "r") as handle:
                if str(handle.attrs.get("task_name")) != task:
                    continue
                seed = int(handle.attrs["seed"])
                prompt = str(handle.attrs["instruction"])
                item = {"path": str(path), "root": str(root), "seed": seed, "prompt": prompt, "success": bool(handle.attrs["success"]), "frames": int(handle["policy_action/raw"].shape[0])}
            key = (seed, prompt)
            previous = by_identity.get(key)
            if previous is None or item["path"] < previous["path"]:
                by_identity[key] = item
    return sorted(by_identity.values(), key=lambda row: (not row["success"], row["seed"], row["path"]))


def selected_sft_sources(roots: list[Path], task: str, count: int) -> list[dict[str, Any]]:
    rows = sft_sources(roots, task)
    successes = [row for row in rows if row["success"]][:count]
    failures = [row for row in rows if not row["success"]][:count]
    if not successes or not failures:
        raise RuntimeError(f"{task}: source has success={len(successes)}, failure={len(failures)} after de-duplication")
    for group, values in (("success", successes), ("failure", failures)):
        for item in values:
            item["outcome_group"] = group
    return successes + failures


def build_raw_action_chunks(raw: np.ndarray, valid: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    if raw.ndim != 2 or raw.shape[1] != 16 or raw.shape[0] < 17:
        raise ValueError(f"unexpected rollout raw action shape {raw.shape}")
    if bool(valid[0]) or np.isfinite(raw[0]).all():
        raise ValueError("SFT row zero must be the invalid action placeholder")
    chunks: list[tuple[np.ndarray, list[int]]] = []
    first = np.zeros((16, 2, 16), dtype=np.float32)
    first[:, 1] = raw[1:17].T
    if not valid[1:17].all() or not np.isfinite(first).all():
        raise ValueError("first SFT action chunk has invalid actions")
    chunks.append((first, [4, 8, 12, 16]))
    cursor = 17
    while cursor + 32 <= raw.shape[0]:
        action = raw[cursor : cursor + 32].reshape(2, 16, 16).transpose(2, 0, 1).astype(np.float32, copy=False)
        if not valid[cursor : cursor + 32].all() or not np.isfinite(action).all():
            break
        chunks.append((action, list(range(cursor + 3, cursor + 32, 4))))
        cursor += 32
    if not chunks:
        raise RuntimeError("no complete SFT chunks")
    return chunks


def deterministic_predictions(server: QGuidedVA_Server, prompt: str, chunks: list[tuple[np.ndarray, list[int]]], handle: h5py.File, seed: int) -> list[torch.Tensor]:
    reset_episode(server, prompt)
    predictions: list[torch.Tensor] = []
    for index, (_, observation_indices) in enumerate(chunks):
        torch.manual_seed(seed + index)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + index)
        if index == 0:
            _, predicted = server._infer({"obs": h5_obs(handle, 0)}, frame_st_id=0)
        else:
            _, predicted = server._infer({"obs": {}}, frame_st_id=server.frame_st_id)
        predictions.append(predicted.detach().cpu())
        if index + 1 < len(chunks):
            observed = [h5_obs(handle, observation_index) for observation_index in observation_indices]
            # Use fresh-base actions for its cache; score-time Q actions remain
            # the archived model-space policy_action/raw values.
            server._compute_kv_cache({"obs": observed})
    return predictions


def actual_latents(server: QGuidedVA_Server, prompt: str, chunks: list[tuple[np.ndarray, list[int]]], handle: h5py.File, predictions: list[torch.Tensor]) -> list[torch.Tensor]:
    reset_episode(server, prompt)
    # Prime the causal streaming VAE exactly as the evaluator's first
    # inference does.  A later four-observation cache refresh is valid only
    # after this one-frame causal context has populated its feature cache.
    server._encode_obs({"obs": [h5_obs(handle, 0)]})
    actual: list[torch.Tensor] = []
    for index, (_, observation_indices) in enumerate(chunks):
        observed = [h5_obs(handle, observation_index) for observation_index in observation_indices]
        encoded = encode_saved_observations(server, observed).detach().cpu()
        predicted = predictions[index]
        if index == 0 and encoded.shape[2] == predicted.shape[2] - 1:
            encoded = torch.cat([predicted[:, :, :1].to(encoded), encoded], dim=2)
        if encoded.shape[2] != predicted.shape[2]:
            raise RuntimeError(f"actual/predicted latent mismatch at chunk {index}: {encoded.shape} vs {predicted.shape}")
        actual.append(encoded)
    return actual


def prepare_sft(args: argparse.Namespace) -> None:
    if args.output.is_file() or args.output.exists():
        raise FileExistsError(args.output)
    selected = selected_sft_sources(args.sft_root, args.task, args.samples_per_outcome)
    # A task-level Clean-50 terminal mean is immutable.  When provided, load
    # it once rather than rebuilding it from the same 50 feature files.
    clean_goal = load_global_goal(args.global_goal) if args.global_goal else terminal_goal(args.clean_root)
    records: list[dict[str, Any]] = []
    for position, source in enumerate(selected, start=1):
        server = make_replay_server(args.output / f"server_scratch_{position:02d}")
        with h5py.File(source["path"], "r") as handle:
            raw = np.asarray(handle["policy_action/raw"], dtype=np.float32)
            valid = np.asarray(handle["policy_action/valid"], dtype=bool)
            chunks = build_raw_action_chunks(raw, valid)
            replay_seed = 20260812 + int(source["seed"])
            predicted = deterministic_predictions(server, source["prompt"], chunks, handle, replay_seed)
            actual = actual_latents(server, source["prompt"], chunks, handle, predicted)
        records.append({**source, "replay_seed": replay_seed, "chunks": [{"predicted": p, "actual": a, "raw_action": torch.from_numpy(action)} for (action, _), p, a in zip(chunks, predicted, actual)]})
        print(json.dumps({"event": "sft_replay", "task": args.task, "position": position, "total": len(selected), "outcome": source["outcome_group"], "seed": source["seed"], "chunks": len(chunks)}), flush=True)
        if position < len(selected):
            # The record is wholly CPU-backed at this point.  Discard the
            # heavyweight sampler before the next trajectory so transient
            # attention/VAE allocations cannot accumulate across records.
            del server
            gc.collect()
            torch.cuda.empty_cache()

    # Keep only the final server's VAE for predicted/actual latent decoding.
    # Its sampler, critic, and text encoder are no longer needed for rewards.
    del server.transformer
    del server.text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    # JEPA conversion consumes the saved latents on the sampler device.  Its
    # causal VAE state is no longer needed, so discard that state and bring the
    # single shared VAE back from GPU 1 only after the sampler is gone.
    server.streaming_vae.clear_cache()
    if server.streaming_vae_half is not None:
        server.streaming_vae_half.clear_cache()
    server.vae.to(server.device)
    encoder = build_jepa_encoder(args.jepa_checkpoint, server.device).eval().requires_grad_(False)
    goal = clean_goal.to(server.device)
    for record in records:
        distances, goals = [], []
        for chunk in record["chunks"]:
            predicted = chunk["predicted"].to(server.device)
            actual = chunk["actual"].to(server.device)
            pred_jepa, actual_jepa = jepa_from_latents(server, encoder, predicted), jepa_from_latents(server, encoder, actual)
            distances.append(float(cosine_distance(pred_jepa, actual_jepa).item()))
            expanded_goal = goal.to(actual_jepa).unsqueeze(0).unsqueeze(0).expand_as(actual_jepa)
            goals.append(float(cosine_distance(actual_jepa, expanded_goal).item()))
        record["prediction_jepa_distance"] = distances
        record["global_goal_jepa_distance"] = goals
    server.vae.to("cpu")
    del encoder
    del server
    torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": 1, "task": args.task, "global_goal": clean_goal, "records": records, "provenance": {"base_checkpoint": str(args.base_checkpoint), "base_checkpoint_sha256": sha256(args.base_checkpoint / "training_state.pt"), "jepa_checkpoint": str(args.jepa_checkpoint), "global_goal": str(args.global_goal) if args.global_goal else f"rebuilt from {args.clean_root}", "global_goal_sha256": sha256(args.global_goal) if args.global_goal else None, "sft_prediction_replay": "deterministic fresh base-model replay from archived observed frames; original SFT stochastic predicted latents/actions were not preserved", "q_action_source": "recorded policy_action/raw (model-space); executed_ee is intentionally not used"}}, args.output / "sft_replay.pt")
    json_dump(args.output / "selection.json", {"task": args.task, "records": [{key: value for key, value in record.items() if key not in {"chunks", "prediction_jepa_distance", "global_goal_jepa_distance"}} for record in records], "clean_goal_shape": list(clean_goal.shape)})
    print(json.dumps({"event": "complete", "stage": "prepare-sft", "output": str(args.output), "records": len(records)}), flush=True)


def reward_calibration(path: Path, expected_catalog_sha: str) -> dict[str, float | str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("catalog_sha256") != expected_catalog_sha:
        raise RuntimeError(f"frozen reward metadata catalog mismatch: {payload.get('catalog_sha256')}")
    metadata = payload.get("reward_metadata", payload)
    distances = metadata.get("jepa_distance")
    if not isinstance(distances, torch.Tensor) or not distances.numel():
        raise RuntimeError("frozen reward metadata lacks jepa_distance")
    raw = -distances.float()
    return {"catalog_sha256": expected_catalog_sha, "prediction_min": float(raw.min()), "prediction_max": float(raw.max()), "metadata_sha256": sha256(path)}


def scale(value: float, minimum: float, maximum: float) -> float:
    return 0.0 if maximum - minimum < 1e-6 else 2.0 * (value - minimum) / (maximum - minimum) - 1.0


def reward_rows(config: dict[str, Any], calibration: dict[str, float | str], prediction_distances: list[float], goal_distances: list[float], success: bool) -> list[dict[str, float]]:
    terms = config.get("reward_terms")
    if not isinstance(terms, list) or not terms:
        raise ValueError("checkpoint config has no reward_terms")
    count, maximum = len(prediction_distances), max(goal_distances)
    rows: list[dict[str, float]] = []
    for index, (prediction_distance, goal_distance) in enumerate(zip(prediction_distances, goal_distances)):
        done = index == count - 1
        progress = 1.0 if count == 1 else index / (count - 1)
        total, detail = 0.0, {}
        for term in terms:
            name, weight, normalization = str(term["name"]), float(term.get("weight", 1.0)), term.get("normalize", False)
            if name == "negative_predicted_actual_jepa_distance":
                value = -prediction_distance
                if normalization == "scale_neg1_pos1":
                    value = scale(value, float(calibration["prediction_min"]), float(calibration["prediction_max"]))
            elif name == "goal_jepa_progress":
                value = 1.0 if done else 1.0 - goal_distance / max(maximum, 1e-6) * math.exp(-float(config["goal_jepa_progress_lambda"]) * progress) * (1.0 - progress) ** float(config["goal_jepa_progress_beta"])
            elif name == "terminal_outcome_cost":
                value = TERMINAL_OUTCOME_RAW["success" if done and success else "failure" if done else "nonterminal"]
            elif name == "terminal_success":
                value = float(done and success)
            else:
                raise ValueError(f"unsupported checkpoint reward term for this diagnostic: {name}")
            if normalization not in (False, None, "scale_neg1_pos1"):
                raise ValueError(f"unsupported reward normalization {normalization!r} for {name}")
            detail[name] = value * weight
            total += detail[name]
        if config.get("final_reward_normalization", "none") != "none":
            raise ValueError("this diagnostic only supports checkpoints with final_reward_normalization=none")
        rows.append({"reward": total, "progress": progress, "done": float(done), "goal_distance_max": maximum, **{f"term_{key}": value for key, value in detail.items()}})
    return rows


def action_model_input(server: QGuidedVA_Server, raw: torch.Tensor) -> torch.Tensor:
    # Raw SFT actions are already the 16 used RobotWin model channels.
    return server.preprocess_action(raw.numpy()).to(server.device, dtype=server.dtype)


def q_values(server: QGuidedVA_Server, predicted: torch.Tensor, actions: torch.Tensor) -> tuple[float, float]:
    mask = server._valid_action_mask(actions, clamp_first_frame=False)
    with torch.no_grad(), server._q_feature_extraction_context(), server._raw_flex_attention_context():
        features = server._extract_q_features(predicted, actions, mask)
        if server.q_artifact.config.critic_type == "twin_language_query_transformer_v1":
            text_tokens = server.prompt_embeds.to(
                device=server.device,
                dtype=server.dtype,
            ).clone()
            text_mask = text_tokens.float().abs().sum(dim=-1).gt(0)
            expected_dim = int(server.q_artifact.config.critic_text_embed_dim)
            if text_tokens.ndim != 3 or text_tokens.shape[-1] != expected_dim:
                raise ValueError(
                    "QTran-L text tokens must have shape [B,T,D] with "
                    f"D={expected_dim}, got {tuple(text_tokens.shape)}"
                )
            if text_mask.shape != text_tokens.shape[:2]:
                raise ValueError(
                    "QTran-L text mask shape mismatch: "
                    f"{tuple(text_mask.shape)} vs {tuple(text_tokens.shape[:2])}"
                )
            if text_tokens.shape[0] != features["q_context_tokens"].shape[0]:
                raise ValueError("QTran-L text/context batch size mismatch")
            if not bool(text_mask.any(dim=1).all()):
                raise ValueError("QTran-L instruction embedding has no valid tokens")
            features = dict(features)
            features["text_tokens"] = text_tokens
            features["text_mask"] = text_mask
        q1, q2 = server.q_artifact.adapter.q_values(features, mask)
    return float(q1.item()), float(q2.item())


def target_value(server: QGuidedVA_Server, latents: torch.Tensor, actions: torch.Tensor, latent_mask: torch.Tensor) -> float:
    config = server.q_artifact.config
    feature_input = build_clean_feature_input(latents=latents.to(server.dtype), actions=actions.to(server.dtype), text_emb=server.prompt_embeds.to(server.dtype).clone(), patch_size=tuple(server.job_config.patch_size), chunk_size=config.infer_latent_chunk_size, window_size=config.window_size, latent_timestep=server._feature_timestep(config.value_video_feature_timestep_range), action_timestep=server._feature_timestep(config.value_action_feature_timestep_range))
    with torch.no_grad(), server._q_feature_extraction_context(), server._raw_flex_attention_context():
        outputs = server.transformer(feature_input, train_mode=True, return_features=True, critic_feature_layers=config.critic_feature_layers, stop_after_critic_feature_layer=True, activation_checkpointing=False)
        tokens = outputs[3]["video_tokens"]
        mask = latent_mask[:, None, :, None].expand(-1, tokens.shape[1], -1, tokens.shape[3]).flatten(2, 3)
        bundle = server.q_artifact.adapter.bundle
        if getattr(config, "critic_type", "") == "twin_language_query_transformer_v1":
            text_tokens = server.prompt_embeds.to(server.dtype).clone()
            text_mask = text_tokens.float().abs().sum(dim=-1).gt(0)
            return float(bundle.target_value(tokens.flatten(2, 3), mask, text_tokens, text_mask).item())
        return float(bundle.target_value(tokens.flatten(2, 3), mask).item())


def score_sft(server: QGuidedVA_Server, replay: dict[str, Any], config: dict[str, Any], calibration: dict[str, float | str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in replay["records"]:
        reset_episode(server, record["prompt"])
        chunk_rows: list[dict[str, Any]] = []
        for index, chunk in enumerate(record["chunks"]):
            predicted = chunk["predicted"].to(server.device)
            actual = chunk["actual"].to(server.device)
            actions = action_model_input(server, chunk["raw_action"])
            previous_latents, previous_actions, previous_mask = server._q_previous_latents, server._q_previous_actions, server._q_previous_latent_mask
            if previous_latents is None:
                prefix_latents, prefix_actions = actual, actions
                prefix_mask = torch.ones(1, actual.shape[2], dtype=torch.bool, device=server.device)
            else:
                prefix_latents, prefix_actions = torch.cat([previous_latents.to(actual), actual], 2), torch.cat([previous_actions.to(actions), actions], 2)
                prefix_mask = torch.cat([previous_mask.to(server.device), torch.ones(1, actual.shape[2], dtype=torch.bool, device=server.device)], 1)
            q1, q2 = q_values(server, predicted, actions)
            chunk_rows.append({"q1": q1, "q2": q2, "q_min": min(q1, q2), "target_value": target_value(server, prefix_latents, prefix_actions, prefix_mask)})
            server._store_previous_context(actual, actions)
        rewards = reward_rows(config, calibration, record["prediction_jepa_distance"], record["global_goal_jepa_distance"], bool(record["success"]))
        for index, (row, reward) in enumerate(zip(chunk_rows, rewards)):
            target = reward["reward"] if reward["done"] else reward["reward"] + float(config["gamma"]) * row["target_value"]
            residual = row["q_min"] - target
            rows.append({"category": f"sft_{record['outcome_group']}", "identity": f"seed-{record['seed']}", "source_path": record["path"], "success": bool(record["success"]), "chunk_index": index, "chunk_count": len(chunk_rows), "trajectory_progress": reward.pop("progress"), "td_target": target, "td_residual": residual, "absolute_td_error": abs(residual), "predicted_actual_jepa_distance": record["prediction_jepa_distance"][index], "global_goal_jepa_distance": record["global_goal_jepa_distance"][index], **row, **reward})
    return rows


def cache_context(episode: dict[str, torch.Tensor], row: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    current_mask = episode["actions_mask"][row].bool()
    state_valid = bool(episode["state_valid"][row])
    real_video = episode["real_video_tokens"]
    if state_valid:
        prefix = real_video[:row]
        prefix_mask = episode["latents_mask"][:row].bool()
    else:
        prefix = torch.zeros_like(real_video[:1])
        prefix_mask = torch.zeros_like(episode["latents_mask"][:1]).bool()
    target = real_video[: row + 1]
    target_mask = episode["latents_mask"][: row + 1].bool()
    # [rows,layers,frames,tokens,dim] -> [1,layers,total_frames,tokens,dim]
    def flatten_rows(value: torch.Tensor) -> torch.Tensor:
        return value.permute(1, 0, 2, 3, 4).reshape(1, value.shape[1], value.shape[0] * value.shape[2], value.shape[3], value.shape[4])
    value_tokens, value_mask = _flatten_video_context(flatten_rows(prefix), prefix_mask.reshape(1, -1))
    target_tokens, target_mask_out = _flatten_video_context(flatten_rows(target), target_mask.reshape(1, -1))
    action = episode["pred_action_tokens"][row : row + 1]
    action_tokens, action_mask = _flatten_action_context(action, current_mask.unsqueeze(0))
    q_tokens, q_mask = _cat_context((value_tokens, value_mask), (action_tokens, action_mask))
    return {"q_context_tokens": q_tokens, "q_context_mask": q_mask, "target_value_context_tokens": target_tokens, "target_value_context_mask": target_mask_out}, current_mask.unsqueeze(0)


def score_clean(artifact: Any, clean_cache_dir: Path, clean_root: Path, global_goal: torch.Tensor, calibration: dict[str, float | str], config: dict[str, Any], episodes: list[int], language_tokens: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None) -> list[dict[str, Any]]:
    # The immutable replay stage computed this exact task-level goal once.
    # Reuse it across the checkpoint sweep instead of rescanning all 50
    # Clean-50 terminal feature files for every checkpoint.
    goal = global_goal.float().cpu()
    output: list[dict[str, Any]] = []
    for episode_index in episodes:
        matches = sorted(clean_cache_dir.glob(f"dataset-*/episode_d*_ep{episode_index:09d}.safetensors"))
        if len(matches) != 1:
            raise FileNotFoundError(f"expected one staged cache for clean episode {episode_index}: {matches}")
        with safe_open(matches[0], framework="pt", device="cpu") as source:
            episode = {key: source.get_tensor(key) for key in source.keys()}
        count = int(episode["rl_chunk_idx"].numel())
        if not torch.equal(episode["rl_chunk_idx"], torch.arange(count)):
            raise ValueError(f"Clean episode {episode_index} has non-contiguous RL chunks")
        masks = episode["latents_mask"].bool()
        targets = aligned_clean_jepa(clean_root, episode_index, int(masks.sum()))
        distances = []
        cursor = 0
        for index, mask in enumerate(masks):
            # The Phase-3 mask is dense per RL chunk, while the aligned JEPA
            # target sequence contains only valid latent frames.  Consume that
            # compact sequence cumulatively rather than indexing it with the
            # chunk-local mask (which is wrong whenever a final chunk is short).
            valid = int(mask.sum())
            if valid <= 0:
                raise RuntimeError(f"Clean episode {episode_index} has an empty RL chunk at {index}")
            current = targets[cursor : cursor + valid]
            if current.shape[0] != valid:
                raise RuntimeError(f"Clean episode {episode_index} target/mask alignment ended at chunk {index}")
            cursor += valid
            distances.append(float(cosine_distance(current.unsqueeze(0), goal.unsqueeze(0)).item()))
        if cursor != targets.shape[0]:
            raise RuntimeError(f"Clean episode {episode_index} left {targets.shape[0] - cursor} unconsumed JEPA frames")
        rewards = reward_rows(config, calibration, [float(x) for x in episode["jepa_distance"]], distances, bool(episode["episode_success"][0]))
        for index, reward in enumerate(rewards):
            contexts, mask = cache_context(episode, index)
            device = next(artifact.adapter.parameters()).device
            contexts = {key: value.to(device) for key, value in contexts.items()}
            if language_tokens is not None:
                try:
                    text_tokens, text_mask = language_tokens[episode_index]
                except KeyError as error:
                    raise RuntimeError(f"missing language sidecar row for Clean-50 episode {episode_index}") from error
                contexts["text_tokens"] = text_tokens.to(device)
                contexts["text_mask"] = text_mask.to(device)
            with torch.no_grad():
                q1, q2 = artifact.adapter.q_values(contexts, mask)
                if language_tokens is not None:
                    value = artifact.adapter.bundle.target_value(contexts["target_value_context_tokens"], contexts["target_value_context_mask"], contexts["text_tokens"], contexts["text_mask"])
                else:
                    value = artifact.adapter.bundle.target_value(contexts["target_value_context_tokens"], contexts["target_value_context_mask"])
            q_min, next_value = min(float(q1.item()), float(q2.item())), float(value.item())
            target = reward["reward"] if reward["done"] else reward["reward"] + float(config["gamma"]) * next_value
            residual = q_min - target
            output.append({"category": "clean50", "identity": f"episode-{episode_index:06d}", "episode_index": episode_index, "chunk_index": index, "chunk_count": count, "trajectory_progress": reward.pop("progress"), "q1": float(q1.item()), "q2": float(q2.item()), "q_min": q_min, "target_value": next_value, "td_target": target, "td_residual": residual, "absolute_td_error": abs(residual), "predicted_actual_jepa_distance": float(episode["jepa_distance"][index]), "global_goal_jepa_distance": distances[index], **reward})
    return output


def interpolate(rows: list[dict[str, Any]], grid: np.ndarray, field: str) -> np.ndarray:
    ordered = sorted(rows, key=lambda row: row["chunk_index"])
    return np.interp(grid, [row["trajectory_progress"] for row in ordered], [row[field] for row in ordered])


def plot(rows: list[dict[str, Any]], task: str, checkpoint: int, output: Path, points: int) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, points)
    categories = (("sft_success", "Successful SFT"), ("sft_failure", "Failed SFT"), ("clean50", "Clean-50"))
    figure, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, constrained_layout=True)
    summary: dict[str, Any] = {}
    for row_index, (category, label) in enumerate(categories):
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["category"] == category:
                grouped[str(row["identity"])].append(row)
        if not grouped:
            raise RuntimeError(f"no rows for {category}")
        signed = np.stack([interpolate(values, grid, "td_residual") for values in grouped.values()])
        absolute = np.stack([interpolate(values, grid, "absolute_td_error") for values in grouped.values()])
        for axis, values, title, colour in ((axes[row_index, 0], signed, "Signed TD residual", "#296b9a"), (axes[row_index, 1], absolute, "Absolute TD error", "#b45520")):
            for curve in values:
                axis.plot(grid, curve, color=colour, alpha=.28, linewidth=1)
            axis.plot(grid, values.mean(0), color=colour, linewidth=2.4, label=f"mean (n={len(values)})")
            if len(values) > 1:
                axis.fill_between(grid, values.mean(0) - values.std(0, ddof=1), values.mean(0) + values.std(0, ddof=1), color=colour, alpha=.16)
            if title.startswith("Signed"):
                axis.axhline(0, color=".35", linestyle="--", linewidth=.8)
            axis.set_title(f"{label}: {title}")
            axis.set_ylabel("Q units")
            axis.grid(alpha=.2)
            axis.legend(frameon=False)
        summary[category] = {"trajectory_count": len(values), "signed_mean": signed.mean(0).tolist(), "signed_std": (signed.std(0, ddof=1) if len(signed) > 1 else np.zeros(points)).tolist(), "absolute_mean": absolute.mean(0).tolist(), "absolute_std": (absolute.std(0, ddof=1) if len(absolute) > 1 else np.zeros(points)).tolist()}
    for axis in axes[-1]:
        axis.set_xlabel("Normalized trajectory progress")
    figure.suptitle(f"{task}: TD-error curves, QTran-L checkpoint {checkpoint:,}")
    figure.text(.5, .005, "SFT rows use deterministic fresh base-model prediction replay; Clean-50 rows use frozen Phase-3 prediction features. Goal is mean Clean-50 terminal JEPA representation.", ha="center", fontsize=8.4)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {"progress_grid": grid.tolist(), "categories": summary}


def score(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    config = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    calibration = reward_calibration(args.reward_metadata, args.expected_catalog_sha256)
    replay = torch.load(args.sft_replay / "sft_replay.pt", map_location="cpu", weights_only=False)
    if replay.get("task") != args.task:
        raise ValueError(f"replay task {replay.get('task')} != requested {args.task}")
    language_tokens, language_provenance = clean_language_tokens(args.clean_root, args.task, args.clean_episode)
    server = make_server(args.checkpoint, args.output / "server_scratch")
    sft_rows = score_sft(server, replay, config, calibration)
    server.vae.to("cpu")
    move_observation_vaes(server, "cpu")
    del server
    torch.cuda.empty_cache()
    artifact = load_q_guidance_artifact(args.checkpoint, torch.device("cuda:0"))
    clean_rows = score_clean(artifact, args.clean_cache_dir, args.clean_root, replay["global_goal"], calibration, config, args.clean_episode, language_tokens)
    all_rows = sft_rows + clean_rows
    curves = plot(all_rows, args.task, checkpoint_step(args.checkpoint), args.output / "td_error_curves.png", args.points)
    payload = {"schema_version": 1, "diagnostic": "robotwin_sft_clean50_td_checkpoint_sweep", "task": args.task, "checkpoint": str(args.checkpoint), "checkpoint_step": checkpoint_step(args.checkpoint), "formula": "min(Q1,Q2) - [reward(config with global Clean-50 goal override) + gamma * (1-done) * target_V]", "reward_contract": {"checkpoint_config": str(args.checkpoint / "config.json"), "gamma": config["gamma"], "terms": config["reward_terms"], "final_reward_normalization": config.get("final_reward_normalization"), "frozen_prediction_calibration": calibration, "global_goal_override": "mean terminal Clean-50 JEPA representation for this task", "terminal_outcome_raw": TERMINAL_OUTCOME_RAW}, "provenance": replay["provenance"], "source_paths": {"sft_replay": str(args.sft_replay / "sft_replay.pt"), "clean_cache_dir": str(args.clean_cache_dir), "clean_root": str(args.clean_root), "clean_language_sidecar": language_provenance}, "curves": curves, "rows": all_rows}
    json_dump(args.output / "td_error_curves.json", payload)
    result = {
        "event": "complete",
        "stage": "score",
        "task": args.task,
        "checkpoint": checkpoint_step(args.checkpoint),
        "output": str(args.output),
        "rows": len(all_rows),
    }
    print(json.dumps(result), flush=True)
    return result


def sweep(args: argparse.Namespace) -> None:
    """Score an inclusive checkpoint range under one checkpoint directory."""
    if args.start_step < 0 or args.end_step < 0:
        raise ValueError("checkpoint steps must be non-negative")
    if args.start_step > args.end_step:
        raise ValueError("start-step must be <= end-step")
    if args.step <= 0:
        raise ValueError("step must be positive")

    checkpoint_root = args.checkpoint_root.resolve()
    output_root = args.output_root.resolve()
    requested_steps = list(range(args.start_step, args.end_step + 1, args.step))
    checkpoints: list[tuple[int, Path]] = []
    for step in requested_steps:
        checkpoint = checkpoint_root / f"checkpoint_{step:08d}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"requested checkpoint is absent: {checkpoint}")
        if not (checkpoint / "config.json").is_file():
            raise FileNotFoundError(f"checkpoint config is absent: {checkpoint / 'config.json'}")
        checkpoints.append((step, checkpoint))

    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for step, checkpoint in checkpoints:
        output = output_root / args.task / checkpoint.name
        curve_json = output / "td_error_curves.json"
        curve_png = output / "td_error_curves.png"
        if curve_json.is_file() and curve_png.is_file():
            skipped.append({"checkpoint": step, "output": str(output), "status": "complete-existing"})
            print(json.dumps({"event": "skip", "stage": "score", "task": args.task, "checkpoint": step, "output": str(output), "reason": "complete output exists"}), flush=True)
            continue
        if output.exists():
            raise FileExistsError(
                f"partial sweep output exists; refusing to overwrite: {output}"
            )
        score_args = argparse.Namespace(**vars(args))
        score_args.checkpoint = checkpoint
        score_args.output = output
        results.append(score(score_args))

    summary = {
        "schema_version": 1,
        "diagnostic": "robotwin_sft_clean50_td_checkpoint_sweep",
        "stage": "sweep",
        "task": args.task,
        "checkpoint_root": str(checkpoint_root),
        "output_root": str(output_root),
        "start_step": args.start_step,
        "end_step": args.end_step,
        "step": args.step,
        "requested_checkpoints": requested_steps,
        "completed": results,
        "skipped": skipped,
    }
    json_dump(output_root / args.task / "sweep_summary.json", summary)
    print(json.dumps({"event": "complete", "stage": "sweep", "task": args.task, "completed": len(results), "skipped": len(skipped), "summary": str(output_root / args.task / "sweep_summary.json")}), flush=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-sft")
    prepare.add_argument("--task", required=True, choices=("place_can_basket", "hanging_mug"))
    prepare.add_argument("--sft-root", type=Path, required=True, action="append")
    prepare.add_argument("--clean-root", type=Path, required=True)
    prepare.add_argument("--global-goal", type=Path, help="Existing mean Clean-50 terminal JEPA embedding for this task.")
    prepare.add_argument("--base-checkpoint", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--samples-per-outcome", type=int, default=10)
    prepare.add_argument("--jepa-checkpoint", type=Path, default=Path("/path/to/home/wy/giga/weight/vjepa2.1/vjepa2_1_vitG_384.pt"))
    goal = sub.add_parser("prepare-goal")
    goal.add_argument("--clean-root", type=Path, required=True)
    goal.add_argument("--output", type=Path, required=True)
    scored = sub.add_parser("score")
    scored.add_argument("--task", required=True, choices=("place_can_basket", "hanging_mug"))
    scored.add_argument("--checkpoint", type=Path, required=True)
    scored.add_argument("--sft-replay", type=Path, required=True)
    scored.add_argument("--clean-root", type=Path, required=True)
    scored.add_argument("--clean-cache-dir", type=Path, required=True)
    scored.add_argument("--clean-episode", type=int, action="append", required=True)
    scored.add_argument("--reward-metadata", type=Path, required=True)
    scored.add_argument("--expected-catalog-sha256", required=True)
    scored.add_argument("--output", type=Path, required=True)
    scored.add_argument("--points", type=int, default=41)
    ranged = sub.add_parser("sweep", help="score an inclusive checkpoint step range")
    ranged.add_argument("--task", required=True, choices=("place_can_basket", "hanging_mug"))
    ranged.add_argument("--checkpoint-root", type=Path, required=True)
    ranged.add_argument("--start-step", type=int, required=True)
    ranged.add_argument("--end-step", type=int, required=True)
    ranged.add_argument("--step", type=int, required=True)
    ranged.add_argument("--sft-replay", type=Path, required=True)
    ranged.add_argument("--clean-root", type=Path, required=True)
    ranged.add_argument("--clean-cache-dir", type=Path, required=True)
    ranged.add_argument("--clean-episode", type=int, action="append", required=True)
    ranged.add_argument("--reward-metadata", type=Path, required=True)
    ranged.add_argument("--expected-catalog-sha256", required=True)
    ranged.add_argument("--output-root", type=Path, required=True)
    ranged.add_argument("--points", type=int, default=41)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.command == "prepare-goal":
        prepare_goal(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    if args.command == "prepare-sft":
        prepare_sft(args)
    elif args.command == "score":
        score(args)
    else:
        sweep(args)


if __name__ == "__main__":
    main()
