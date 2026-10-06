#!/usr/bin/env python3
"""Repair the missing LeRobot v2.1 episode statistics for a read-only mix view.

The mix view keeps Cosmos parquet/latent files as symlinks and compacts the
successful episode indices.  Its per-episode stats must therefore be selected
from the immutable source stats by ``source_episode_index`` and re-keyed to the
view's compact episode index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix-root", type=Path, required=True)
    ap.add_argument("--source-root", type=Path, required=True)
    args = ap.parse_args()

    mix_meta = args.mix_root / "meta"
    source_meta = args.source_root / "meta"
    episodes_path = mix_meta / "episodes.jsonl"
    source_stats_path = source_meta / "episodes_stats.jsonl"
    output_path = mix_meta / "episodes_stats.jsonl"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing file: {output_path}")
    if not episodes_path.is_file() or not source_stats_path.is_file():
        raise FileNotFoundError("mix episodes.jsonl or source episodes_stats.jsonl is missing")

    mix_rows = read_jsonl(episodes_path)
    source_rows = read_jsonl(source_stats_path)
    source_by_index = {int(row["episode_index"]): row for row in source_rows}
    if len(source_by_index) != len(source_rows):
        raise RuntimeError("source episodes_stats has duplicate episode indices")

    out_rows: list[dict] = []
    for new_index, row in enumerate(mix_rows):
        if int(row["episode_index"]) != new_index:
            raise RuntimeError(f"mix episode index is not compact at row {new_index}")
        old_index = int(row["source_episode_index"])
        source_row = source_by_index.get(old_index)
        if source_row is None:
            raise RuntimeError(f"missing source stats for source_episode_index={old_index}")
        out_row = {"episode_index": new_index, "stats": source_row["stats"]}
        out_rows.append(out_row)

    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in out_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "pass",
        "mix_episodes": len(mix_rows),
        "source_stats": len(source_rows),
        "written": len(out_rows),
        "output": str(output_path),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
