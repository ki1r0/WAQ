#!/usr/bin/env python3
"""Single-source catalog + overlay builder (parameterized; no source editing).

Generalization of va2_rebuild_20260907/build_va2_step3_catalogs.py: the base
config name, task/source names and episode filter are CLI arguments so the
same file serves both the VA2 (track A) and original-VA (track B) rebuilds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(args: argparse.Namespace) -> None:
    cache, source = args.cache_dir, args.source_root
    manifest_path = cache / "manifest.json"
    episodes_path = source / "meta" / "episodes.jsonl"
    if not manifest_path.is_file() or not episodes_path.is_file():
        raise SystemExit(f"missing cache/source metadata: {manifest_path} {episodes_path}")
    rows = read_jsonl(episodes_path)
    if args.max_episodes is not None:
        rows = [r for r in rows if int(r["episode_index"]) < args.max_episodes]
    rows.sort(key=lambda r: int(r["episode_index"]))
    expected = list(range(len(rows)))
    if [int(r["episode_index"]) for r in rows] != expected:
        raise SystemExit("non-contiguous source episode indices")
    cache_files = sorted(cache.glob("dataset-*/episode_*.safetensors"))
    if len(cache_files) != len(rows):
        raise SystemExit(f"cache episode count mismatch: files={len(cache_files)} rows={len(rows)}")
    transitions, seen = 0, set()
    for path in cache_files:
        with safe_open(str(path), framework="pt", device="cpu") as tensors:
            ep = tensors.get_tensor("episode_index").unique().tolist()
            ds = tensors.get_tensor("dataset_id").unique().tolist()
            n = int(tensors.get_tensor("transition_idx").shape[0])
        if len(ep) != 1 or len(ds) != 1 or int(ds[0]) != 0:
            raise SystemExit(f"invalid cache identity: {path}")
        if int(ep[0]) in seen:
            raise SystemExit(f"duplicate cache episode {int(ep[0])}")
        seen.add(int(ep[0]))
        transitions += n
    if seen != set(expected):
        raise SystemExit(f"cache episode index mismatch: missing={sorted(set(expected) - seen)[:5]}")
    manifest_sha = sha256(manifest_path)
    catalog = {
        "schema_version": 1,
        "cache_format": "phase3_federated_canonical_episode",
        "run_contract": {"name": args.contract_name, "sampling": "single_source_uniform",
                         "note": args.note},
        "task_count": 10,
        "episode_count": len(rows),
        "transition_count": transitions,
        "sources": [{
            "source_id": 0,
            "task_name": args.task_name,
            "base_config_name": args.base_config,
            "cache_dir": str(cache),
            "manifest_sha256": manifest_sha,
            "transition_idx_offset": 0,
            "transition_count": transitions,
            "datasets": [{
                "local_dataset_id": 0,
                "global_dataset_id": 0,
                "source": args.source_name,
                "benchmark": "libero_10",
                "episode_count": len(rows),
            }],
        }],
    }
    args.catalog_out.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_out.write_text(json.dumps(catalog, indent=2) + "\n")
    cat_sha = sha256(args.catalog_out)
    overrides, successes = [], 0
    for row in rows:
        ok = bool(row.get("success", True))
        successes += int(ok)
        if not ok:
            overrides.append({"dataset_id": 0, "episode_index": int(row["episode_index"]),
                              "episode_success": False})
    overlay = {
        "format": "phase3_episode_success_overlay_v1",
        "schema_version": 1,
        "mode": "replace",
        "federated_catalog_sha256": cat_sha,
        "source_manifest_sha256_by_source": {"0": manifest_sha},
        "source_defaults": {"0": True},
        "episode_overrides": overrides,
    }
    args.overlay_out.parent.mkdir(parents=True, exist_ok=True)
    args.overlay_out.write_text(json.dumps(overlay, indent=2) + "\n")
    print(json.dumps({"episodes": len(rows), "transitions": transitions,
                      "successes": successes, "failures": len(overrides),
                      "catalog": str(args.catalog_out)}, indent=2))


def validate(args: argparse.Namespace) -> None:
    cat = json.loads(args.catalog_out.read_text())
    overlay = json.loads(args.overlay_out.read_text())
    if cat["cache_format"] != "phase3_federated_canonical_episode" or len(cat["sources"]) != 1:
        raise SystemExit("invalid catalog")
    if overlay["federated_catalog_sha256"] != sha256(args.catalog_out):
        raise SystemExit("overlay/catalog sha mismatch")
    if args.sidecar_dir is not None:
        side = json.loads((args.sidecar_dir / "manifest.json").read_text())
        if side["federated_catalog_sha256"] != sha256(args.catalog_out):
            raise SystemExit("sidecar/catalog sha mismatch")
        if not (args.sidecar_dir / "COMPLETE").is_file():
            raise SystemExit("sidecar COMPLETE missing")
    print(f"VALIDATION_PASS catalog={args.catalog_out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--base-config")
    ap.add_argument("--task-name")
    ap.add_argument("--source-name")
    ap.add_argument("--contract-name")
    ap.add_argument("--note", default="")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--catalog-out", type=Path, required=True)
    ap.add_argument("--overlay-out", type=Path, required=True)
    ap.add_argument("--sidecar-dir", type=Path)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    if args.validate_only:
        validate(args)
        return
    for req in ("cache_dir", "source_root", "base_config", "task_name",
                "source_name", "contract_name"):
        if getattr(args, req) is None:
            ap.error(f"build mode requires --{req.replace('_', '-')}")
    build(args)


if __name__ == "__main__":
    main()
