#!/usr/bin/env python3
"""Small, bounded state helper for the LIBERO node-level eval queue."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


TASK_RE = re.compile(r"task_(\d{5})")
TRIAL_RE = re.compile(r"trial_(\d{3})\.json$")


def collect(root: Path):
    records = defaultdict(list)
    record_roots = []
    old_client_root = root / "client"
    if old_client_root.is_dir():
        record_roots.extend(
            worker / "task_records" / "libero_10"
            for worker in old_client_root.glob("worker-*")
            if worker.is_dir()
        )
    # The node queue has task/server/client siblings.  Only descend into the
    # explicitly named client directories; never scan the whole result tree
    # (which can include large Ceph-backed logs and caches).
    queue_root = root / "node_queue_20260910"
    if queue_root.is_dir():
        for node in queue_root.iterdir():
            if not node.is_dir() or node.name in {"claims", "logs", "assignments"}:
                continue
            for task_root in node.glob("task_*"):
                if not task_root.is_dir():
                    continue
                record_roots.extend(
                    client / "task_records" / "libero_10"
                    for client in task_root.glob("client-gpu*")
                    if client.is_dir()
                )
    for record_root in record_roots:
        if not record_root.is_dir():
            continue
        for task_dir in record_root.glob("task_*"):
            task_match = TASK_RE.search(task_dir.name)
            if not task_match:
                continue
            task = int(task_match.group(1))
            if not 0 <= task < 10:
                continue
            for path in task_dir.glob("trial_*.json"):
                trial_match = TRIAL_RE.search(path.name)
                if not trial_match:
                    continue
                trial = int(trial_match.group(1))
                if not 0 <= trial < 50:
                    continue
                try:
                    data = json.loads(path.read_text())
                except Exception:
                    continue
                if data.get("status") != "completed":
                    continue
                records[(task, trial)].append(
                    (path, bool(data.get("success", False)))
                )
    return records


def task_stats(records, task: int):
    completed = sorted(trial for (t, trial) in records if t == task)
    success = sum(1 for (t, _), values in records.items()
                  if t == task and any(ok for _, ok in values))
    duplicates = sum(max(0, len(values) - 1)
                     for (t, _), values in records.items() if t == task)
    return {
        "task": task,
        "completed": len(completed),
        "success": success,
        "remaining": 50 - len(completed),
        "duplicates": duplicates,
        "trials": completed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--task", type=int)
    parser.add_argument("--gpu-count", type=int)
    parser.add_argument("--assign-csv", action="store_true")
    parser.add_argument("--all-summary", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    records = collect(root)

    if args.all_summary:
        for task in range(10):
            print(json.dumps(task_stats(records, task), sort_keys=True))
        return

    if args.assign_csv:
        if args.task is None or not args.gpu_count or args.gpu_count < 1:
            raise SystemExit("--task and positive --gpu-count required")
        done = {trial for (task, trial) in records if task == args.task}
        pending = [trial for trial in range(50) if trial not in done]
        base, extra = divmod(len(pending), args.gpu_count)
        offset = 0
        for gpu in range(args.gpu_count):
            count = base + (1 if gpu < extra else 0)
            chunk = pending[offset:offset + count]
            offset += count
            print(f"{len(chunk)}|{','.join(str(x) for x in chunk)}")
        return

    if args.task is None:
        raise SystemExit("--task required")
    print(json.dumps(task_stats(records, args.task), sort_keys=True))


if __name__ == "__main__":
    main()
