#!/usr/bin/env python3
"""Convert the Cosmos LIBERO JPEG-HDF5 release to a LingBot-VA latent dataset.

The converter deliberately keeps the source trajectories read-only.  It writes
the small LeRobot v2.1 metadata/parquet contract and then materialises the
LingBot VAE/text/JEPA sidecars directly from the JPEG frames (no intermediate
MP4 is required).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


SOURCE_DEFAULT = Path("/mnt/zengqixin/datasets/LIBERO-Cosmos-Policy")
OUTPUT_DEFAULT = Path("/mnt/zengqixin/datasets/LIBERO-Cosmos-Policy-lingbotva-20260905")
FPS = 20.0
TARGET_FPS = 20.0
IMG_SIZE = (128, 128)
CAMERA_KEYS = ("observation.images.front", "observation.images.wrist")


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _decode(value: Any, size: tuple[int, int] | None = None) -> np.ndarray:
    raw = bytes(value) if not isinstance(value, bytes) else value
    with Image.open(io.BytesIO(raw)) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size, Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.uint8)


def _group_key(name: str) -> tuple[int, str]:
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)) if m else 10**9, name)


def records(
    source: Path,
    limit: int = 0,
    smoke: bool = False,
    include_success_only: bool = False,
) -> list[dict[str, Any]]:
    """Return deterministic episode descriptors without loading image payloads."""
    out: list[dict[str, Any]] = []
    # all_episodes are independent files; inspect them first so a bounded
    # smoke does not need to open one of the very large success-only archives.
    for p in sorted((source / "all_episodes").glob("*.hdf5")):
        try:
            with h5py.File(p, "r") as f:
                task = str(f.attrs.get("task_description", ""))
                success = bool(f.attrs.get("success", False))
                if all(k in f for k in ("actions", "proprio", "primary_images_jpeg", "wrist_images_jpeg")):
                    out.append({
                        "source": "all_episodes",
                        "path": str(p),
                        "group": None,
                        "success": success,
                        "task": task,
                    })
                    if limit and len(out) >= limit:
                        return out
                    if smoke:
                        break
        except (OSError, RuntimeError, KeyError) as exc:
            print(f"[skip] cannot inspect {p}: {exc}", flush=True)
    if smoke:
        # Keep one all-episode item and append one success-only demo.
        all_item = out[:1]
        out.clear()
        success_root = source / "success_only"
        for p in sorted(success_root.glob("**/*.hdf5")):
            try:
                with h5py.File(p, "r") as f:
                    data = f.get("data")
                    if data is None:
                        continue
                    for gname in sorted(data.keys(), key=_group_key):
                        g = data[gname]
                        if "actions" not in g or "obs/agentview_rgb_jpeg" not in g:
                            continue
                        stem = p.stem.removesuffix("_demo")
                        match = re.search(r"_SCENE\d+_(.*)$", stem)
                        task = (match.group(1) if match else stem).replace("_", " ")
                        out.append({"source": "success_only", "path": str(p), "group": gname, "success": True, "task": task})
                        return out + all_item
            except (OSError, RuntimeError, KeyError) as exc:
                print(f"[skip] cannot inspect {p}: {exc}", flush=True)
        return all_item
    # `success_only` in the Cosmos release is a filtered subset of
    # `all_episodes` (the per-task counts match the successful all-episode
    # counts).  Do not append it to the canonical full set by default: that
    # would duplicate every successful trajectory.  It remains available for
    # an explicit smoke or a deliberately separate success-only conversion.
    if not include_success_only:
        return out

    success_root = source / "success_only"
    for p in sorted(success_root.glob("**/*.hdf5")):
        try:
            with h5py.File(p, "r") as f:
                data = f.get("data")
                if data is None:
                    continue
                for gname in sorted(data.keys(), key=_group_key):
                    g = data[gname]
                    if "actions" not in g or "obs/agentview_rgb_jpeg" not in g:
                        continue
                    stem = p.stem.removesuffix("_demo")
                    match = re.search(r"_SCENE\d+_(.*)$", stem)
                    task = (match.group(1) if match else stem).replace("_", " ")
                    out.append({"source": "success_only", "path": str(p), "group": gname, "success": True, "task": task})
                    if limit and len(out) >= limit:
                        return out
        except (OSError, RuntimeError, KeyError) as exc:
            print(f"[skip] cannot inspect {p}: {exc}", flush=True)
    return out


def _open_episode(rec: dict[str, Any], load_images: bool = True) -> tuple[np.ndarray, np.ndarray, list[Any], list[Any], str]:
    with h5py.File(rec["path"], "r") as f:
        if rec["source"] == "success_only":
            g = f["data"][rec["group"]]
            actions = np.asarray(g["actions"], dtype=np.float32)
            state = np.asarray(g["robot_states"], dtype=np.float32)
            front = list(g["obs/agentview_rgb_jpeg"]) if load_images else []
            wrist = list(g["obs/eye_in_hand_rgb_jpeg"]) if load_images else []
            task = rec["task"]
        else:
            actions = np.asarray(f["actions"], dtype=np.float32)
            state = np.asarray(f["proprio"], dtype=np.float32)
            front = list(f["primary_images_jpeg"]) if load_images else []
            wrist = list(f["wrist_images_jpeg"]) if load_images else []
            task = str(f.attrs.get("task_description", rec["task"]))
    n = min(len(actions), len(state))
    if load_images:
        n = min(n, len(front), len(wrist))
    if n < 5:
        raise ValueError(f"episode has too few aligned rows: {n}")
    return actions[:n], state[:n], front[:n], wrist[:n], task


def _sample_indices(n: int) -> list[int]:
    # Frame4 uses the source 20 Hz timeline. Keep contiguous frame ids and
    # trim only the trailing remainder so each action segment is 4k+1 frames.
    ids = list(range(0, n, 1))
    count = ((len(ids) - 1) // 4) * 4 + 1
    return ids[:count]


def _fixed_list(values: np.ndarray, width: int, dtype: pa.DataType) -> pa.Array:
    flat = pa.array(values.reshape(-1).tolist(), type=dtype)
    return pa.FixedSizeListArray.from_arrays(flat, width)


def _stats(values: np.ndarray) -> dict[str, Any]:
    x = values.astype(np.float64)
    return {
        "min": np.nanmin(x, axis=0).tolist(),
        "max": np.nanmax(x, axis=0).tolist(),
        "mean": np.nanmean(x, axis=0).tolist(),
        "std": np.nanstd(x, axis=0).tolist(),
        "count": [int(x.shape[0])],
    }


def write_metadata(source: Path, out: Path, selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    task_to_idx: dict[str, int] = {}
    episode_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    total_frames = 0
    all_actions: list[np.ndarray] = []
    all_states: list[np.ndarray] = []
    for ep, rec in enumerate(selected):
        actions, state, _front, _wrist, task = _open_episode(rec, load_images=False)
        task = task.strip()
        if task not in task_to_idx:
            task_to_idx[task] = len(task_to_idx)
        task_idx = task_to_idx[task]
        n_raw = int(actions.shape[0])
        usable = _sample_indices(n_raw)
        actions = actions[usable]
        state = state[usable]
        n = int(len(usable))
        end_raw = int(usable[-1] + 1)
        chunk = ep // 1000
        data_dir = out / "data" / f"chunk-{chunk:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        idx = np.arange(total_frames, total_frames + n, dtype=np.int64)
        table = pa.table({
            "observation.state": _fixed_list(state, state.shape[1], pa.float32()),
            "action": _fixed_list(actions, actions.shape[1], pa.float32()),
            "timestamp": pa.array(np.arange(n, dtype=np.float32) / FPS),
            "frame_index": pa.array(np.arange(n, dtype=np.int64)),
            "episode_index": pa.array(np.full(n, ep, dtype=np.int64)),
            "index": pa.array(idx),
            "task_index": pa.array(np.full(n, task_idx, dtype=np.int64)),
        })
        hf_features = {
            "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": int(state.shape[1]), "_type": "Sequence"},
            "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": int(actions.shape[1]), "_type": "Sequence"},
            "timestamp": {"dtype": "float32", "_type": "Value"},
            "frame_index": {"dtype": "int64", "_type": "Value"},
            "episode_index": {"dtype": "int64", "_type": "Value"},
            "index": {"dtype": "int64", "_type": "Value"},
            "task_index": {"dtype": "int64", "_type": "Value"},
        }
        table = table.replace_schema_metadata({b"huggingface": json.dumps({"info": {"features": hf_features}}).encode()})
        pq.write_table(table, data_dir / f"episode_{ep:06d}.parquet", compression="zstd")
        episode_rows.append({
            "episode_index": ep,
            "tasks": [task],
            "length": n,
            "success": bool(rec["success"]),
            # The latent/action segment intentionally ends at the last
            # frame retained by _sample_indices.  The critic transition view
            # treats this aligned endpoint as the logical terminal frame;
            # using the raw HDF5 endpoint here would make its segment
            # validation reject every episode whose length is not 4k+1.
            "termination_frame": end_raw - 1,
            "termination_reason": "success" if rec["success"] else "failure",
            "truncated": False,
            "source": rec["source"],
            "source_path": rec["path"],
            "source_group": rec["group"],
            "action_config": [{"start_frame": 0, "end_frame": end_raw, "action_text": task, "skill": ""}],
        })
        stats_rows.append({
            "episode_index": ep,
            "stats": {"observation.state": _stats(state), "action": _stats(actions),
                       "timestamp": _stats((np.arange(n, dtype=np.float32) / FPS)[:, None]),
                       "frame_index": _stats(np.arange(n, dtype=np.float32)[:, None]),
                       "episode_index": _stats(np.full((n, 1), ep, dtype=np.float32)),
                       "index": _stats(idx.astype(np.float32)[:, None]),
                       "task_index": _stats(np.full((n, 1), task_idx, dtype=np.float32))}
        })
        total_frames += n
        all_actions.append(actions)
        all_states.append(state)

    with (out / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for row in episode_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / "episodes_stats.jsonl").open("w", encoding="utf-8") as f:
        for row in stats_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (out / "meta" / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task, task_idx in sorted(task_to_idx.items(), key=lambda kv: kv[1]):
            f.write(json.dumps({"task_index": task_idx, "task": task}, ensure_ascii=False) + "\n")

    features = {
        "observation.images.front": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "info": {"video.height": 256, "video.width": 256, "video.codec": "jpeg", "video.pix_fmt": "rgb24", "video.fps": 20, "video.channels": 3, "has_audio": False}},
        "observation.images.wrist": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channel"], "info": {"video.height": 256, "video.width": 256, "video.codec": "jpeg", "video.pix_fmt": "rgb24", "video.fps": 20, "video.channels": 3, "has_audio": False}},
        "observation.state": {"dtype": "float32", "shape": [int(all_states[0].shape[1])], "names": [f"state_{i}" for i in range(int(all_states[0].shape[1]))]},
        "action": {"dtype": "float32", "shape": [int(all_actions[0].shape[1])], "names": [f"action_{i}" for i in range(int(all_actions[0].shape[1]))]},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None}, "frame_index": {"dtype": "int64", "shape": [1], "names": None}, "episode_index": {"dtype": "int64", "shape": [1], "names": None}, "index": {"dtype": "int64", "shape": [1], "names": None}, "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    info = {"codebase_version": "v2.1", "robot_type": "panda", "total_episodes": len(episode_rows), "total_frames": total_frames, "total_tasks": len(task_to_idx), "total_videos": 0, "total_chunks": math.ceil(len(episode_rows) / 1000), "chunks_size": 1000, "fps": 20, "splits": {"train": f"0:{len(episode_rows)}"}, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet", "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4", "features": features, "cosmos_source": str(source), "cosmos_success_only_episodes": sum(r["source"] == "success_only" for r in episode_rows), "cosmos_all_episodes": sum(r["source"] == "all_episodes" for r in episode_rows), "lingbotva_latent_fps": TARGET_FPS, "lingbotva_action_downsample_factor": 1, "lingbotva_video_materialized": False}
    _json_dump(out / "meta" / "info.json", info)
    _json_dump(out / "meta" / "conversion_manifest.json", {"source": str(source), "episodes": episode_rows, "camera_keys": list(CAMERA_KEYS), "source_fps": FPS, "target_fps": TARGET_FPS, "target_image_size": IMG_SIZE, "action_source_dim": 7, "action_target_dim": 30, "action_downsample_factor": 1, "action_downsample_policy": "contiguous"})
    return episode_rows


def encode_sidecars(
    out: Path,
    episode_rows: list[dict[str, Any]],
    rank: int,
    world: int,
    do_jepa: bool,
    jepa_checkpoint: str | None = None,
) -> None:
    # Importing the repository modules is intentionally delayed: metadata-only
    # conversion remains CPU-only and does not need the model environment.
    repo = Path("/home/zengqixin/wy/lingbot-va-original-rl")
    sys.path.insert(0, str(repo))
    from preprocessing.extract_latent_vae import WanVAEEncoder, extract_text_embeddings, load_text_encoder, load_tokenizer
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    model_root = Path("/home/zengqixin/wy/giga/weight/lingbot-va-posttrain-robotwin")
    vae = WanVAEEncoder(str(model_root / "vae"), device)
    tokenizer = load_tokenizer(str(model_root / "tokenizer"))
    text_encoder = load_text_encoder(str(model_root / "text_encoder"), torch_dtype=torch.bfloat16, torch_device=device)
    jepa_encoder = None
    jepa_fns = None
    if do_jepa:
        vroot = repo.parent / "vjepa2"
        sys.path.insert(0, str(vroot))
        from preprocessing.extract_jepa_features import build_jepa_encoder, encode_batch, frames_to_tensor
        jepa_ckpt = Path(
            jepa_checkpoint
            or os.environ.get("VJEPA_CHECKPOINT", "")
            or "/luhongchao/shared/weights/vjepa2.1/vjepa2_1_vitG_384.pt"
        )
        # The shared weight is mounted at different paths on different
        # workers; keep the historical path as the default but fall back to
        # the current shared mount when it exists.
        if not jepa_ckpt.exists():
            fallback = Path("/mnt/zengqixin/weight/vjepa2.1/vjepa2_1_vitG_384.pt")
            if fallback.exists():
                jepa_ckpt = fallback
        if not jepa_ckpt.exists():
            raise FileNotFoundError(jepa_ckpt)
        jepa_encoder = build_jepa_encoder(str(jepa_ckpt), device)
        jepa_fns = (encode_batch, frames_to_tensor)

    for ep in range(rank, len(episode_rows), world):
        row = episode_rows[ep]
        actions, state, front, wrist, task = _open_episode({"source": row["source"], "path": row["source_path"], "group": row["source_group"], "task": row["tasks"][0], "success": row["success"]})
        ids = _sample_indices(len(actions))
        imgs = {CAMERA_KEYS[0]: np.stack([_decode(front[i], IMG_SIZE) for i in ids]), CAMERA_KEYS[1]: np.stack([_decode(wrist[i], IMG_SIZE) for i in ids])}
        with torch.inference_mode():
            text_emb = extract_text_embeddings(task, tokenizer, text_encoder, device).squeeze(0).cpu() if task else None
        chunk = ep // 1000
        for cam_key, frames in imgs.items():
            tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
            with torch.inference_mode():
                latent, c, f_lat, h_lat, w_lat = vae.encode_video(tensor)
            payload = {"latent": latent, "latent_num_frames": int(f_lat), "latent_height": int(h_lat), "latent_width": int(w_lat), "video_num_frames": len(frames), "video_height": IMG_SIZE[1], "video_width": IMG_SIZE[0], "text_emb": text_emb, "text": task, "frame_ids": ids, "start_frame": 0, "end_frame": int(ids[-1] + 1), "fps": TARGET_FPS, "ori_fps": FPS}
            p = out / "latents" / f"chunk-{chunk:03d}" / cam_key / f"episode_{ep:06d}_0_{ids[-1] + 1}.pth"
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, p)
        if do_jepa:
            enc_batch, to_tensor = jepa_fns
            jdict: dict[str, Any] = {"frame_ids": np.asarray(ids, dtype=np.int64)}
            for cam_key, frames in imgs.items():
                # Prepend one duplicate so each JEPA token covers a VAE 4-frame window.
                x = np.concatenate([frames[0:1], frames], axis=0)
                feat = enc_batch(jepa_encoder, [to_tensor(x)], device)[0]
                jdict[cam_key] = feat
            jp = out / "jepa" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.pt"
            jp.parent.mkdir(parents=True, exist_ok=True)
            torch.save(jdict, jp)
        if ep % 10 == 0 or ep == len(episode_rows) - 1:
            print(f"[rank {rank}] encoded {ep + 1}/{len(episode_rows)}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, default=SOURCE_DEFAULT)
    p.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--encode-existing", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    p.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    p.add_argument("--jepa", action="store_true")
    p.add_argument("--jepa-checkpoint", type=str, default=None)
    p.add_argument("--smoke", action="store_true", help="one success_only and one all_episodes record")
    p.add_argument(
        "--include-success-only",
        action="store_true",
        help="append success_only records (off by default because they duplicate successful all_episodes)",
    )
    args = p.parse_args()
    if args.encode_existing:
        rows = [json.loads(x) for x in (args.output / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        encode_sidecars(
            args.output,
            rows,
            args.rank,
            args.world_size,
            args.jepa,
            args.jepa_checkpoint,
        )
        return
    recs = records(
        args.source,
        limit=args.limit,
        smoke=args.smoke,
        include_success_only=args.include_success_only,
    )
    if args.limit and not args.smoke:
        recs = recs[: args.limit]
    rows = write_metadata(args.source, args.output, recs)
    print(json.dumps({"episodes": len(rows), "success": sum(bool(x["success"]) for x in rows), "failure": sum(not bool(x["success"]) for x in rows)}, ensure_ascii=False), flush=True)
    if not args.metadata_only:
        encode_sidecars(
            args.output,
            rows,
            args.rank,
            args.world_size,
            args.jepa,
            args.jepa_checkpoint,
        )


if __name__ == "__main__":
    import torch
    main()
