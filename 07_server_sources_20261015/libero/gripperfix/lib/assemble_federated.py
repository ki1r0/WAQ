#!/usr/bin/env python3
"""Merge two single-source (catalog, overlay, sidecar) triples into the
federated triple the critic trainer consumes.

Reproduces the layout of va2_rebuild_20260907/catalog_va2_combined_1000_*.json
exactly: source_id 0/1, cumulative transition_idx_offset, global_dataset_id =
source index, per-source manifest shas, and a sidecar dir whose source-NN
entries symlink the single-source sidecars.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, nargs=2, required=True,
                    help="single-source catalogs, in source_id order")
    ap.add_argument("--overlay", type=Path, nargs=2, required=True)
    ap.add_argument("--sidecar", type=Path, nargs=2, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cats = [json.loads(p.read_text()) for p in args.catalog]
    ovls = [json.loads(p.read_text()) for p in args.overlay]
    sides = [json.loads((p / "manifest.json").read_text()) for p in args.sidecar]
    for i, (c, o, s) in enumerate(zip(cats, ovls, sides)):
        if len(c["sources"]) != 1:
            raise SystemExit(f"input {i} is not single-source")
        if o["federated_catalog_sha256"] != sha256(args.catalog[i]):
            raise SystemExit(f"input {i}: overlay/catalog sha mismatch")
        if s["federated_catalog_sha256"] != sha256(args.catalog[i]):
            raise SystemExit(f"input {i}: sidecar/catalog sha mismatch")
        if not (args.sidecar[i] / "COMPLETE").is_file():
            raise SystemExit(f"input {i}: sidecar COMPLETE missing")

    sources, offset = [], 0
    for i, c in enumerate(cats):
        src = json.loads(json.dumps(c["sources"][0]))
        src["source_id"] = i
        src["transition_idx_offset"] = offset
        for d in src["datasets"]:
            d["global_dataset_id"] = i
        sources.append(src)
        offset += int(src["transition_count"])
    catalog = {
        "schema_version": 1,
        "cache_format": "phase3_federated_canonical_episode",
        "run_contract": {"name": args.name, "sampling": "global_balanced_batching",
                         "note": args.note},
        "task_count": 10,
        "episode_count": sum(int(c["episode_count"]) for c in cats),
        "transition_count": offset,
        "sources": sources,
    }
    cat_path = out / "catalog.json"
    cat_path.write_text(json.dumps(catalog, indent=2) + "\n")
    cat_sha = sha256(cat_path)

    overrides = []
    for i, o in enumerate(ovls):
        for entry in o["episode_overrides"]:
            e = dict(entry)
            e["source_id"] = i
            overrides.append(e)
    overlay = {
        "format": "phase3_episode_success_overlay_v1",
        "schema_version": 1,
        "mode": "replace",
        "federated_catalog_sha256": cat_sha,
        "source_manifest_sha256_by_source": {str(i): s["manifest_sha256"]
                                             for i, s in enumerate(sources)},
        "source_defaults": {str(i): True for i in range(len(sources))},
        "episode_overrides": overrides,
    }
    ovl_path = out / "success_overlay.json"
    ovl_path.write_text(json.dumps(overlay, indent=2) + "\n")

    side_dir = out / "sidecar"
    side_dir.mkdir(exist_ok=True)
    for key in ("embedding_source", "format", "max_tokens", "schema_version",
                "text_embed_dim", "text_encoder_provenance", "tokenizer"):
        if sides[0][key] != sides[1][key]:
            raise SystemExit(f"sidecar mismatch on {key}: {sides[0][key]} vs {sides[1][key]}")
    side_sources = []
    for i, s in enumerate(sides):
        link = side_dir / f"source-{i:02d}"
        target = (args.sidecar[i] / "source-00").resolve()
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(target, link)
        entry = json.loads(json.dumps(s["sources"][0]))
        entry["source_id"] = i
        for d in entry["datasets"]:
            d["file"] = f"source-{i:02d}/" + d["file"].split("/", 1)[1]
        side_sources.append(entry)
    manifest = {
        "embedding_source": sides[0]["embedding_source"],
        "federated_catalog": str(cat_path),
        "federated_catalog_sha256": cat_sha,
        "format": sides[0]["format"],
        "max_tokens": sides[0]["max_tokens"],
        "schema_version": sides[0]["schema_version"],
        "sources": side_sources,
        "text_embed_dim": sides[0]["text_embed_dim"],
        "text_encoder_provenance": sides[0]["text_encoder_provenance"],
        "tokenizer": sides[0]["tokenizer"],
    }
    (side_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (side_dir / "COMPLETE").write_text("ok\n")
    for i, s in enumerate(side_sources):
        f = side_dir / s["datasets"][0]["file"]
        if not f.is_file():
            raise SystemExit(f"sidecar payload missing after link: {f}")
    print(json.dumps({"catalog": str(cat_path), "episodes": catalog["episode_count"],
                      "transitions": catalog["transition_count"],
                      "failures_marked": len(overrides),
                      "sidecar": str(side_dir)}, indent=2))
    print("ASSEMBLE_OK")


if __name__ == "__main__":
    main()
