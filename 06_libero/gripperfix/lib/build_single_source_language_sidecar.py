#!/usr/bin/env python3
"""Build a language text sidecar for a single-source federated LIBERO catalog.

Adapted from build_libero_long_language_sidecar.py (20260805) for the pi0.6
tree: emits the `robotwin_federated_episode_text_v1` manifest format that
wan_va/rl/canonical_cache.py in lingbot-va-original-rl-pi0.6 expects, and
handles exactly one catalog source instead of the fixed [0, 1] pair.

Text embeddings are not recomputed: each episode latent payload already stores
`text_emb` (512, 4096) produced by the policy's T5 text encoder. This script
verifies the stored text against episodes.jsonl, verifies the token mask
against a tokenizer pass, and repacks the embeddings into one safetensors
sidecar indexed by raw episode index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import T5TokenizerFast

EPISODE_RE = re.compile(r"^episode_(\d+)_")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_episode_indices(cache_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in sorted(cache_dir.glob("dataset-*/episode_*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            values = tensors.get_tensor("episode_index").unique()
            datasets = tensors.get_tensor("dataset_id").unique()
        if values.numel() != 1 or datasets.numel() != 1:
            raise RuntimeError(f"mixed identity in canonical episode {path}")
        index = int(values.item())
        if int(datasets.item()) != 0:
            raise RuntimeError(f"unexpected local dataset ID in {path}")
        if index in result:
            raise RuntimeError(f"duplicate cached episode index {index}")
        result[index] = path
    if not result:
        raise RuntimeError(f"no canonical episode files under {cache_dir}")
    return result


def latent_paths(root: Path, camera_key: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    chunk_dirs = sorted((root / "latents").glob("chunk-*"))
    if not chunk_dirs:
        raise RuntimeError(f"no latent chunk dirs under {root / 'latents'}")
    for chunk_dir in chunk_dirs:
        base = chunk_dir / camera_key
        if not base.is_dir():
            continue
        for path in sorted(base.glob("episode_*.pth")):
            match = EPISODE_RE.match(path.name)
            if not match:
                raise RuntimeError(f"unexpected latent filename {path}")
            index = int(match.group(1))
            if index in result:
                raise RuntimeError(f"duplicate latent episode {index} under {base}")
            result[index] = path
    if not result:
        raise RuntimeError(f"no latents for {camera_key} under {root / 'latents'}")
    return result


def read_episodes(root: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if [int(row["episode_index"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"non-contiguous episode metadata in {root}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--camera-key", type=str, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    catalog_bytes = args.catalog.read_bytes()
    catalog = json.loads(catalog_bytes)
    if len(catalog["sources"]) != 1:
        raise RuntimeError("expected a single-source catalog")
    source = catalog["sources"][0]
    source_id = int(source["source_id"])
    if len(source["datasets"]) != 1 or int(source["datasets"][0]["local_dataset_id"]) != 0:
        raise RuntimeError("expected exactly one dataset with local_dataset_id 0")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_root / "tokenizer")
    root = args.source_root
    cache_dir = Path(source["cache_dir"])
    episodes = read_episodes(root)
    selected = cache_episode_indices(cache_dir)
    paths = latent_paths(root, args.camera_key)
    missing = sorted(set(selected) - set(paths))
    if missing:
        raise RuntimeError(f"missing latent paths: {missing[:8]}")
    if max(selected) >= len(episodes):
        raise RuntimeError("cache index exceeds metadata")

    lengths: dict[int, int] = {}
    for index in selected:
        text = prompt_clean(str(episodes[index]["tasks"][0]))
        lengths[index] = len(
            tokenizer(text, add_special_tokens=True, truncation=True, max_length=512)[
                "input_ids"
            ]
        )
    max_tokens = max(lengths.values())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    indices = sorted(selected)
    relative = Path(f"source-{source_id:02d}") / "dataset-000000.safetensors"
    output = args.output_dir / relative
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (max(indices) + 1, max_tokens, 4096)
    tokens = torch.zeros(shape, dtype=torch.bfloat16)
    mask = torch.zeros(shape[:2], dtype=torch.bool)
    for position, index in enumerate(indices):
        payload = torch.load(paths[index], map_location="cpu", weights_only=False)
        expected_text = str(episodes[index]["tasks"][0])
        payload_text = payload.get("text")
        if isinstance(payload_text, list) and len(payload_text) == 1:
            payload_text = payload_text[0]
        if payload_text != expected_text:
            raise RuntimeError(f"text mismatch for episode {index}")
        text = payload.get("text_emb")
        if not isinstance(text, torch.Tensor) or tuple(text.shape) != (512, 4096):
            raise RuntimeError(f"invalid text embedding for episode {index}")
        valid = lengths[index]
        nonzero = text.float().abs().sum(dim=-1).gt(0)
        if int(nonzero.sum()) != valid or not bool(nonzero[:valid].all()) or bool(nonzero[valid:].any()):
            raise RuntimeError(f"text token mask mismatch for episode {index}")
        tokens[index, :valid] = text[:valid].to(torch.bfloat16)
        mask[index, :valid] = True
        if (position + 1) % 200 == 0:
            print(f"packed {position + 1}/{len(indices)} episodes", flush=True)

    temporary = output.with_suffix(".safetensors.tmp")
    temporary.unlink(missing_ok=True)
    episodes_path = root / "meta" / "episodes.jsonl"
    selected_bytes = json.dumps(indices, separators=(",", ":")).encode()
    save_file(
        {"text_tokens": tokens, "text_mask": mask},
        temporary,
        metadata={
            "source_id": str(source_id),
            "local_dataset_id": "0",
            "task_name": str(source["task_name"]),
            "source_dataset": str(root),
            "source_episodes_sha256": sha256(episodes_path),
            "selected_episode_indices_sha256": hashlib.sha256(selected_bytes).hexdigest(),
            "embedding_source": "stored_episode_text_emb",
        },
    )
    os.replace(temporary, output)
    print(f"saved source={source_id} selected={len(indices)} rows={shape[0]}", flush=True)

    manifest = {
        "schema_version": 1,
        "format": "robotwin_federated_episode_text_v1",
        "federated_catalog": str(args.catalog),
        "federated_catalog_sha256": hashlib.sha256(catalog_bytes).hexdigest(),
        "embedding_source": "stored_episode_text_emb",
        "text_encoder_provenance": str(args.model_root / "text_encoder"),
        "tokenizer": str(args.model_root / "tokenizer"),
        "text_embed_dim": 4096,
        "max_tokens": max_tokens,
        "sources": [
            {
                "source_id": source_id,
                "task_name": source["task_name"],
                "datasets": [
                    {
                        "local_dataset_id": 0,
                        "episode_count": len(indices),
                        "sidecar_row_count": shape[0],
                        "selected_episode_indices_sha256": hashlib.sha256(selected_bytes).hexdigest(),
                        "source_dataset": str(root),
                        "source_episodes_sha256": sha256(episodes_path),
                        "file": str(relative),
                        "sha256": sha256(output),
                    }
                ],
            }
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "COMPLETE").write_text("complete\n")
    print(f"manifest={args.output_dir / 'manifest.json'} max_tokens={max_tokens}")


if __name__ == "__main__":
    main()
