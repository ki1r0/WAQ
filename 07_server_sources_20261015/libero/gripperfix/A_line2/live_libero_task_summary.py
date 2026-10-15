#!/usr/bin/env python3
"""Live, read-only aggregation for one LIBERO task evaluation.

The evaluator owns the worker files.  This process only reads completed trial
records and atomically replaces ``metrics/summary.json`` until the supplied
runner PID exits or all expected trials are accounted for.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDENTITY_FIELDS = ("policy_checkpoint", "policy_id")
IN_PROGRESS_STATUSES = {"running", "in_progress", "pending", "started", "active", "queued"}


def runner_alive(pid: int) -> bool:
    """Return whether *pid* still exists, without inspecting or changing it."""

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _record_identity(record: dict[str, Any]) -> tuple[Any, ...] | None:
    values = tuple(record.get(field) for field in IDENTITY_FIELDS)
    return values if any(value is not None for value in values) else None


def _matches_expected(
    record: dict[str, Any],
    expected: dict[str, str | None],
) -> bool:
    return all(
        value is None or record.get(field) == value
        for field, value in expected.items()
    )


def collect(
    output_root: Path,
    *,
    task: int = 4,
    expected: int = 50,
    expected_identity: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    """Read this run's task-4 records and return one progress snapshot."""

    record_paths = sorted(
        output_root.glob(
            "client/worker-*/task_records/libero_10/task_00004/trial_*.json"
        )
    )
    parse_retries = 0
    invalid_unkeyed = 0
    invalid_trials: set[tuple[int, int]] = set()
    candidates: dict[tuple[int, int], tuple[bool, tuple[Any, ...] | None]] = {}
    conflict_trials: set[tuple[int, int]] = set()
    duplicate_records = 0
    checkpoint_mismatches = 0

    for path in record_paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            # A worker may be replacing or still writing a JSON file.  Leave
            # it out of the denominator and try it again on the next pass.
            parse_retries += 1
            continue

        if not isinstance(record, dict):
            invalid_unkeyed += 1
            continue

        trial = record.get("trial_index")
        record_task = record.get("task_index", task)
        if (
            not _is_int(record_task)
            or record_task != task
            or not _is_int(trial)
            or not 0 <= trial < expected
        ):
            invalid_unkeyed += 1
            continue

        key = (record_task, trial)
        status = record.get("status")
        if status in IN_PROGRESS_STATUSES:
            # Valid in-progress records are neither success nor failure.
            continue
        if status != "completed":
            invalid_trials.add(key)
            continue
        success = record.get("success")
        if not isinstance(success, bool):
            invalid_trials.add(key)
            continue

        if expected_identity and not _matches_expected(record, expected_identity):
            invalid_trials.add(key)
            checkpoint_mismatches += 1
            continue

        identity = _record_identity(record)
        prior = candidates.get(key)
        if prior is None and key not in conflict_trials:
            candidates[key] = (success, identity)
            continue

        duplicate_records += 1
        if prior is not None and (prior[0] != success or prior[1] != identity):
            # Never choose one side of a duplicate disagreement.
            conflict_trials.add(key)
            candidates.pop(key, None)

    for key in invalid_trials:
        candidates.pop(key, None)

    success_count = sum(value[0] for value in candidates.values())
    failure_count = len(candidates) - success_count
    completed = success_count + failure_count
    invalid = invalid_unkeyed + len(invalid_trials | conflict_trials)
    remaining = max(0, expected - completed)
    rate = success_count / completed if completed else None

    now = datetime.now(timezone.utc).isoformat()
    return {
        "time": now,
        "timestamp": now,
        "task": task,
        "expected": expected,
        "completed": completed,
        "success": success_count,
        "failure": failure_count,
        "invalid": invalid,
        "remaining": remaining,
        "rate": rate,
        "duplicate_records": duplicate_records,
        "duplicate_conflicts": len(conflict_trials),
        "parse_retries": parse_retries,
        "invalid_unkeyed_records": invalid_unkeyed,
        "checkpoint_mismatches": checkpoint_mismatches,
        "record_files": len(record_paths),
    }


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace *path* while leaving worker records untouched."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _step_from_root(output_root: Path) -> str | None:
    for part in reversed(output_root.parts):
        match = re.search(r"step(\d+)", part)
        if match:
            return match.group(1)
    return None


def run(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root)
    summary_path = output_root / "metrics" / "summary.json"
    expected_identity = {
        "policy_checkpoint": args.policy_checkpoint,
        "policy_id": args.policy_id,
        "run_id": args.run_id,
    }
    if not any(expected_identity.values()):
        expected_identity = None  # type: ignore[assignment]

    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_handlers = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    started = time.monotonic()
    next_refresh = 0.0
    saw_output_root = False

    try:
        while True:
            now = time.monotonic()
            if output_root.is_dir():
                saw_output_root = True
            if now >= next_refresh and saw_output_root:
                snapshot = collect(
                    output_root,
                    task=args.task,
                    expected=args.expected,
                    expected_identity=expected_identity,
                )
                snapshot.update(
                    {
                        "output_root": str(output_root),
                        "runner_pid": args.runner_pid,
                        "runner_alive": runner_alive(args.runner_pid),
                        "checkpoint_step": _step_from_root(output_root),
                    }
                )
                write_atomic(summary_path, snapshot)
                next_refresh = now + args.interval
                if snapshot["completed"] >= args.expected:
                    break

            alive = runner_alive(args.runner_pid)
            startup_expired = (
                not saw_output_root
                and now - started >= args.startup_timeout
            )
            if stop_requested or not alive or startup_expired:
                break

            wait_for = max(0.1, min(1.0, next_refresh - time.monotonic()))
            time.sleep(wait_for)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if output_root.is_dir():
            snapshot = collect(
                output_root,
                task=args.task,
                expected=args.expected,
                expected_identity=expected_identity,
            )
            snapshot.update(
                {
                    "output_root": str(output_root),
                    "runner_pid": args.runner_pid,
                    "runner_alive": runner_alive(args.runner_pid),
                    "checkpoint_step": _step_from_root(output_root),
                    "final": True,
                }
            )
            write_atomic(summary_path, snapshot)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runner-pid", type=int, required=True)
    parser.add_argument("--task", type=int, default=4)
    parser.add_argument("--expected", type=int, default=50)
    parser.add_argument("--interval", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--policy-id")
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
