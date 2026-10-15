#!/usr/bin/env python3
"""Exact-frame V-JEPA extraction for the open500 frame4 dataset.

This is a thin runner around the repository JEPA implementation.  Its only
local policy is to use the same contiguous source-frame decimation and
``(n-1) % 4 == 0`` trimming as ``extract_latent_vae.py``; the repository JEPA
runner historically used ``linspace`` for the trim and could therefore write
frame_ids that did not match the VAE sidecars.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path("/home/zengqixin/wy/lingbot-va-original-rl")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from preprocessing import extract_jepa_features as ej


def _read_frames_exact(video_path: str, target_fps: float, height: int, width: int):
    """Decode and sample exactly the frame ids used by the VAE extractor."""
    cap = cv2.VideoCapture(video_path)
    orig_fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    decoded = None
    if orig_fps <= 0 or total_frames <= 0:
        cap.release()
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            orig_fps = float(stream.average_rate)
            decoded = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
        total_frames = len(decoded)

    if target_fps >= orig_fps:
        step = 1
        actual_fps = orig_fps
    else:
        step = round(orig_fps / target_fps)
        actual_fps = orig_fps / step
    if step < 1 or not np.isclose(actual_fps, target_fps):
        raise ValueError(f"Cannot make exact {target_fps} Hz stream from {orig_fps} Hz")

    sample_indices = list(range(0, total_frames, step))
    target_count = len(sample_indices)
    adjusted = target_count
    if (adjusted - 1) % 4 != 0:
        adjusted = ((adjusted - 1) // 4) * 4 + 1
    if adjusted > target_count:
        adjusted -= 4
    adjusted = max(adjusted, 1)
    sample_indices = sample_indices[:adjusted]

    if decoded is None:
        frames = []
        ids = []
        wanted = set(sample_indices)
        cur = 0
        while cur < total_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if cur in wanted:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC))
                ids.append(cur)
            cur += 1
        cap.release()
        if len(frames) != len(sample_indices):
            # AV1 often reports metadata correctly but OpenCV cannot decode;
            # retry in software while retaining the same contiguous indices.
            import av
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                pyav_fps = float(stream.average_rate)
                decoded = [f.to_ndarray(format="rgb24") for f in container.decode(stream)]
            if not np.isclose(pyav_fps, orig_fps):
                raise RuntimeError(f"decoder FPS mismatch: {orig_fps} vs {pyav_fps}")
            frames = [cv2.resize(decoded[i], (width, height), interpolation=cv2.INTER_CUBIC)
                      for i in sample_indices]
            ids = sample_indices
    else:
        frames = [cv2.resize(decoded[i], (width, height), interpolation=cv2.INTER_CUBIC)
                  for i in sample_indices]
        ids = sample_indices

    if len(frames) != len(sample_indices):
        raise RuntimeError(f"decoded {len(frames)}/{len(sample_indices)} frames: {video_path}")
    return np.stack(frames, axis=0), np.asarray(ids, dtype=np.int64)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--camera-keys", nargs="+", required=True)
    p.add_argument("--target-fps", type=float, default=20.0)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()

    # process_dataset resolves this global when reading each video.
    ej.read_frames = _read_frames_exact
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    import torch
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    root = Path(args.dataset_root)
    datasets = ej.find_dataset_roots(root)
    if not datasets:
        raise RuntimeError(f"no dataset root containing videos/: {root}")
    if rank == 0:
        print(f"[exact-jepa] world={world} datasets={datasets} fps={args.target_fps}", flush=True)

    encoder = ej.build_jepa_encoder(args.checkpoint, device)
    for i, ds in enumerate(datasets, 1):
        processed, skipped, errors = ej.process_dataset(
            dataset_root=ds,
            encoder=encoder,
            device=device,
            rank=rank,
            local_rank=local_rank,
            world_size=world,
            cam_resolutions={},
            default_height=args.height,
            default_width=args.width,
            target_fps=args.target_fps,
            batch_size=args.batch_size,
            chunk_ids=None,
            camera_keys=args.camera_keys,
            skip_existing=args.skip_existing,
            temporal_window_frames=0,
            ds_index=i,
            ds_total=len(datasets),
        )
        print(f"[rank {rank}] processed={processed} skipped={skipped} errors={errors}", flush=True)


if __name__ == "__main__":
    main()
