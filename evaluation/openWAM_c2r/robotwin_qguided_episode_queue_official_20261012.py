#!/usr/bin/env python3
"""Episode-level dynamic queue for OpenWAM's official RoboTwin evaluator.

Each GPU claims one randomized demo episode at a time. Evaluation is performed
by OpenWAM's ``eval_policy_wrapper.py`` and the repository's
bundled ``openwam2robotwin_interface`` policy. A fresh run starts from zero; importing
completed episodes from a prior task-sharded run is optional.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = (
    Path(os.environ["OPENWAM_C2R_IMPORT_SOURCE_ROOT"]).expanduser()
    if os.environ.get("OPENWAM_C2R_IMPORT_SOURCE_ROOT")
    else None
)
RUN_ROOT = Path(os.environ.get("OPENWAM_C2R_RUN_ROOT", "./openwam_c2r_run")).expanduser()
ROBOTWIN_ROOT = Path(os.environ.get("ROBOTWIN_ROOT", ".")).expanduser()
ROBOTWIN_OVERLAY = Path(
    os.environ.get("ROBOTWIN_SOURCE_OVERLAY", str(PACKAGE_ROOT / "robotwin_source_overlay"))
).expanduser()
REPO = Path(os.environ.get("OPENWAM_REPO", ".")).expanduser()
ROBOTWIN_PYTHON = Path(os.environ.get("ROBOTWIN_PYTHON", "python3")).expanduser()
OPENWAM_PYTHON = Path(os.environ.get("OPENWAM_PYTHON", "python3")).expanduser()
SFT_DIR = Path(os.environ.get("OPENWAM_SFT_DIR", ".")).expanduser()
SFT_NAME = os.environ.get("OPENWAM_SFT_NAME", "checkpoint_step_5000.safetensors")
CRITIC = Path(os.environ.get("OPENWAM_Q_CHECKPOINT", ".")).expanduser()
TASK_LIST_FILE = Path(
    os.environ.get("ROBOTWIN_TASK_LIST", str(PACKAGE_ROOT / "tasks.txt"))
).expanduser()
ADAPTER = Path(
    os.environ.get("OPENWAM_Q_ADAPTER", str(PACKAGE_ROOT / "openwam_qguided_deploy.py"))
).expanduser()
EVAL_WRAPPER = PACKAGE_ROOT / "eval_policy_wrapper.py"
POLICY_TEMPLATE = PACKAGE_ROOT / "policy_config.yml"
CKPT_SETTING = os.environ.get(
    "OPENWAM_C2R_CKPT_SETTING", "openwam_sft5000_rlcritic7266_qguided"
)
PORT_BASE = int(os.environ.get("OPENWAM_Q_PORT_BASE", "18848"))
GPUS_PER_NODE = int(os.environ.get("OPENWAM_C2R_GPUS_PER_NODE", "8"))
NODE_COUNT = int(os.environ.get("OPENWAM_C2R_NODE_COUNT", "6"))
TOTAL_WORKERS = GPUS_PER_NODE * NODE_COUNT
TARGET_EPISODES = int(os.environ.get("OPENWAM_C2R_EPISODES_PER_TASK", "100"))
MAX_RETRIES = int(os.environ.get("OPENWAM_C2R_MAX_RETRIES", "2"))
EPISODE_TIMEOUT = int(os.environ.get("OPENWAM_C2R_EPISODE_TIMEOUT", "1800"))
NEW_SEED_BASE = int(os.environ.get("OPENWAM_C2R_SEED_BASE", "2000000"))

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SHARD_SUMMARY_RE = re.compile(r"Success rate:\s*(\d+)/(\d+)\b")
EPISODE_RESULT_RE = re.compile(
    r"Success rate:\s*(\d+)/1\b.*?current seed:\s*(\d+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def task_names() -> list[str]:
    if not TASK_LIST_FILE.is_file():
        raise FileNotFoundError(f"RobotWin task list not found: {TASK_LIST_FILE}")
    tasks = [
        line.strip()
        for line in TASK_LIST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(tasks) != 50 or len(set(tasks)) != 50:
        raise RuntimeError(f"Expected 50 unique RobotWin tasks in {TASK_LIST_FILE}, found {len(tasks)}")
    return tasks


def read_attempt_counts(attempt_dir: Path, task: str, attempt: int) -> dict[str, Any]:
    """Read completed shard totals from the official evaluator's summaries."""
    shards: dict[int, dict[str, int]] = {}
    for result_file in attempt_dir.rglob("_result.txt"):
        gpu_match = next(
            (
                match
                for parent in result_file.parents
                if (match := re.fullmatch(r"gpu-(\d+)", parent.name))
            ),
            None,
        )
        if gpu_match is None:
            continue
        gpu = int(gpu_match.group(1))
        if gpu in shards:
            raise RuntimeError(f"Multiple result files for {task} attempt {attempt}, GPU {gpu}")
        try:
            ratio = float(result_file.read_text(encoding="utf-8").splitlines()[-1])
        except (OSError, ValueError, IndexError):
            continue
        log_paths = list((SOURCE_ROOT / "logs").glob(f"{task}.attempt{attempt}.*.gpu{gpu}.log"))
        if len(log_paths) != 1:
            continue
        text = ANSI_RE.sub("", log_paths[0].read_text(encoding="utf-8", errors="replace"))
        summaries = SHARD_SUMMARY_RE.findall(text)
        if not summaries:
            continue
        successes, episodes = map(int, summaries[-1])
        if not episodes or not 0 <= successes <= episodes:
            continue
        # RoboTwin's outer writer divides by its fixed 100, while the wrapper
        # reports the actual shard size. Compare counts, not shard percentages.
        if abs(ratio * TARGET_EPISODES - successes) > 1e-5:
            continue
        shards[gpu] = {"episodes": episodes, "successes": successes}

    return {
        "attempt": attempt,
        "episodes": sum(shard["episodes"] for shard in shards.values()),
        "successes": sum(shard["successes"] for shard in shards.values()),
        "shards": shards,
        "source_attempt_dir": str(attempt_dir),
    }


def import_previous_results() -> None:
    """Preserve old task totals; old sharded logs do not record per-episode labels."""
    if SOURCE_ROOT is None:
        raise RuntimeError(
            "Set OPENWAM_C2R_IMPORT_SOURCE_ROOT to import results from an earlier run"
        )
    state_root = RUN_ROOT / "queue"
    marker = state_root / "import_complete.json"
    workers_root = RUN_ROOT / "workers"
    if workers_root.exists() and any(workers_root.glob("*.json")):
        raise RuntimeError("Cannot import previous totals after new workers have started")
    state_root.mkdir(parents=True, exist_ok=True)
    lock = state_root / ".import.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"Import is already running: {lock}") from exc

    try:
        reports: dict[str, Any] = {}
        prior_counts: dict[str, dict[str, Any]] = {}
        done_root = SOURCE_ROOT / "queue/done"
        for task in task_names():
            attempt_root = SOURCE_ROOT / "runtime" / task
            attempts = sorted(
                (path for path in attempt_root.glob("attempt-*") if path.is_dir()),
                key=lambda path: int(path.name.split("-")[1]),
            )
            attempt_summaries: list[dict[str, Any]] = []
            for attempt_dir in attempts:
                attempt = int(attempt_dir.name.split("-")[1])
                summary = read_attempt_counts(attempt_dir, task, attempt)
                if summary["episodes"]:
                    attempt_summaries.append(summary)

            done_path = done_root / f"{task}.json"
            if done_path.is_file():
                done = json.loads(done_path.read_text(encoding="utf-8"))
                attempt = int(done["attempt"])
                matching = [item for item in attempt_summaries if item["attempt"] == attempt]
                if not matching:
                    raise RuntimeError(f"Done marker lacks verified shard totals: {task}")
                selected = matching[0]
                if selected["episodes"] != int(done["episodes"]):
                    raise RuntimeError(f"Done marker episode count disagrees with shards: {task}")
                if selected["successes"] != int(done["successes"]):
                    raise RuntimeError(f"Done marker success count disagrees with shards: {task}")
            else:
                selected = max(
                    attempt_summaries,
                    key=lambda item: (item["episodes"], item["attempt"]),
                    default={"attempt": None, "episodes": 0, "successes": 0},
                )
            if selected["episodes"] > TARGET_EPISODES:
                raise RuntimeError(f"Imported episode count exceeds target for {task}")
            prior_counts[task] = {
                "episodes": int(selected["episodes"]),
                "successes": int(selected["successes"]),
                "source_attempt": selected["attempt"],
                "source": "done_marker" if done_path.is_file() else "best_completed_shards",
            }
            if selected["episodes"] == TARGET_EPISODES:
                atomic_json(
                    RUN_ROOT / "summaries" / f"{task}.json",
                    {
                        "task": task,
                        "episodes": TARGET_EPISODES,
                        "successes": int(selected["successes"]),
                        "success_rate": int(selected["successes"]) / TARGET_EPISODES,
                        "instruction_type": "unseen",
                        "task_config": "demo_randomized",
                        "source": "imported_current_run",
                        "time_utc": utc_now(),
                    },
                )
            reports[task] = {
                **prior_counts[task],
                "episodes_available": int(selected["episodes"]),
            }
        atomic_json(state_root / "prior_counts.json", prior_counts)
        atomic_json(marker, {"source_root": str(SOURCE_ROOT), "tasks": reports, "time_utc": utc_now()})
        print(json.dumps(reports, indent=2, sort_keys=True))
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


class EpisodeQueue:
    def __init__(self, node_index: int):
        if not 0 <= node_index < NODE_COUNT:
            raise ValueError(f"node index must be in [0, {NODE_COUNT - 1}]")
        self.node_index = node_index
        self.host = os.uname().nodename
        self.worker_root = RUN_ROOT / "workers"
        self.results_root = RUN_ROOT / "results"
        self.claims_root = RUN_ROOT / "queue/claims"
        self.failed_root = RUN_ROOT / "queue/failed"
        self.runtime_root = RUN_ROOT / "runtime"
        self.logs_root = RUN_ROOT / "logs"
        self.config_root = RUN_ROOT / "configs"
        self.tasks = task_names()
        prior_counts_path = RUN_ROOT / "queue/prior_counts.json"
        if prior_counts_path.is_file():
            self.prior_counts = json.loads(prior_counts_path.read_text(encoding="utf-8"))
        else:
            self.prior_counts = {
                task: {"episodes": 0, "successes": 0, "source": "fresh_run"}
                for task in self.tasks
            }
        self.cursors: dict[tuple[str, int], int] = {}
        self.server_processes: dict[int, subprocess.Popen] = {}
        self.server_logs: dict[int, Any] = {}
        self.log_lock = threading.Lock()
        for path in (
            self.worker_root,
            self.results_root,
            self.claims_root,
            self.failed_root,
            self.runtime_root,
            self.logs_root,
            self.config_root,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        line = f"{utc_now()} host={self.host} {message}"
        with self.log_lock:
            print(line, flush=True)
            with (self.logs_root / f"queue_{self.host}.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")

    def server_matches(self, gpu: int) -> bool:
        expected_port = PORT_BASE + gpu
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", "replace"
                )
                environment = (entry / "environ").read_bytes().decode("utf-8", "replace")
            except (OSError, PermissionError):
                continue
            if (
                str(ADAPTER) in command
                and str(SFT_DIR) in command
                and SFT_NAME in command
                and f"--port {expected_port}" in command
                and str(CRITIC) in environment
            ):
                return True
        return False

    def ensure_server(self, gpu: int) -> None:
        port = PORT_BASE + gpu
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                if not self.server_matches(gpu):
                    raise RuntimeError(f"port {port} is open but not owned by the expected OpenWAM server")
                return
        except OSError:
            if self.server_matches(gpu):
                raise RuntimeError(f"expected OpenWAM server exists but port {port} is not ready")

        server_log = (self.logs_root / f"{self.host}_server_gpu{gpu}.log").open(
            "a", encoding="utf-8", buffering=1
        )
        self.server_logs[gpu] = server_log
        env = os.environ.copy()
        env.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "PYTHONPATH": os.pathsep.join((str(REPO), env.get("PYTHONPATH", ""))),
                "OPENWAM_Q_CHECKPOINT": str(CRITIC),
                "OPENWAM_Q_CHUNK_SIZE": "16",
                "OPENWAM_Q_FEATURE_LAYER": "0",
                "OPENWAM_Q_GUIDANCE_INTERVAL": "5",
                "OPENWAM_Q_GUIDANCE_BUDGET": "0.08",
                "OPENWAM_Q_GUIDANCE_SIGMA_PEAK": "0.25",
                "OPENWAM_Q_GUIDANCE_SIGMA_CONCENTRATION": "8.0",
            }
        )
        command = [
            str(OPENWAM_PYTHON),
            str(ADAPTER),
            "--ckpt-dir",
            str(SFT_DIR),
            "--ckpt-name",
            SFT_NAME,
            "--device",
            "cuda:0",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--denoise-steps",
            "10",
            "--denoise-mode",
            "sync",
            "--inference-mode",
            "sync",
            "--compile-enabled",
            "true",
            "optimization.dit_cache.enabled=true",
            "optimization.dit_cache.cosine_threshold=0.99",
            "optimization.dit_cache.max_skips=3",
            "optimization.compile.enabled=true",
            "inference.cfg_scale=1.0",
        ]
        server = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.server_processes[gpu] = server
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"OpenWAM GPU {gpu} server exited with {server.returncode}")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    if self.server_matches(gpu):
                        self.log(f"server_ready gpu={gpu} port={port} pid={server.pid}")
                        return
            except OSError:
                pass
            time.sleep(2)
        raise TimeoutError(f"OpenWAM server did not become ready on GPU {gpu}, port {port}")

    def gpu_config(self, gpu: int) -> Path:
        path = self.config_root / f"{self.host}.gpu{gpu}.yml"
        text = POLICY_TEMPLATE.read_text(encoding="utf-8")
        text, host_count = re.subn(r"(?m)^host:.*$", 'host: "127.0.0.1"', text, count=1)
        text, port_count = re.subn(r"(?m)^port:.*$", f"port: {PORT_BASE + gpu}", text, count=1)
        if host_count != 1 or port_count != 1:
            raise RuntimeError(f"Could not set OpenWAM server address in {POLICY_TEMPLATE}")
        path.write_text(text, encoding="utf-8")
        return path

    def result_files(self, task: str) -> list[Path]:
        return list((self.results_root / task).glob("seed-*.json"))

    def active_claims(self, task: str) -> list[Path]:
        directory = self.claims_root / task
        return (
            [path for path in directory.iterdir() if path.is_dir() and not path.name.startswith(".")]
            if directory.is_dir()
            else []
        )

    def done_count(self, task: str) -> int:
        prior = int(self.prior_counts[task]["episodes"])
        return prior + len(self.result_files(task))

    def success_count(self, task: str) -> int:
        prior = int(self.prior_counts[task]["successes"])
        recent = sum(
            bool(json.loads(path.read_text(encoding="utf-8"))["success"])
            for path in self.result_files(task)
        )
        return prior + recent

    def reserve_episode(self, gpu: int) -> tuple[str, int, Path] | None:
        worker_id = self.node_index * GPUS_PER_NODE + gpu
        stride = TOTAL_WORKERS
        for task in self.tasks:
            claim_parent = self.claims_root / task
            claim_parent.mkdir(parents=True, exist_ok=True)
            reservation_lock = claim_parent / ".reserve"
            deadline = time.monotonic() + 120
            while True:
                try:
                    reservation_lock.mkdir()
                    break
                except FileExistsError:
                    try:
                        stale = time.time() - reservation_lock.stat().st_mtime > 120
                    except FileNotFoundError:
                        continue
                    if stale:
                        stale_lock = claim_parent / f".reserve-stale-{int(time.time())}-{worker_id}"
                        try:
                            os.replace(reservation_lock, stale_lock)
                        except OSError:
                            pass
                        continue
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out reserving episode for {task}")
                    time.sleep(0.05)
            try:
                if self.done_count(task) >= TARGET_EPISODES:
                    continue
                if self.done_count(task) + len(self.active_claims(task)) >= TARGET_EPISODES:
                    continue
                key = (task, worker_id)
                candidate = self.cursors.setdefault(key, NEW_SEED_BASE + worker_id)
                for _ in range(10000):
                    self.cursors[key] = candidate + stride
                    result = self.results_root / task / f"seed-{candidate}.json"
                    failure = self.failed_root / task / f"seed-{candidate}.json"
                    claim = claim_parent / str(candidate)
                    if result.exists() or failure.exists():
                        candidate += stride
                        continue
                    try:
                        claim.mkdir()
                    except FileExistsError:
                        candidate += stride
                        continue
                    atomic_json(
                        claim / "owner.json",
                        {
                            "candidate_seed": candidate,
                            "gpu": gpu,
                            "host": self.host,
                            "pid": os.getpid(),
                            "worker_id": worker_id,
                            "time_utc": utc_now(),
                        },
                    )
                    return task, candidate, claim
            finally:
                try:
                    reservation_lock.rmdir()
                except OSError:
                    pass
        return None

    def run_episode(self, task: str, candidate_seed: int, gpu: int) -> dict[str, Any] | None:
        worker_id = self.node_index * GPUS_PER_NODE + gpu
        stride = TOTAL_WORKERS
        config = self.gpu_config(gpu)
        for attempt in range(MAX_RETRIES + 1):
            episode_root = (
                self.runtime_root
                / task
                / f"candidate-{candidate_seed}"
                / f"attempt-{attempt}"
                / f"{self.host}-gpu{gpu}"
            )
            episode_root.mkdir(parents=True, exist_ok=False)
            task_config = episode_root / "task_config"
            task_config.mkdir()
            config_text = (ROBOTWIN_ROOT / "task_config/demo_randomized.yml").read_text(
                encoding="utf-8"
            )
            config_text, changed = re.subn(
                r"(?m)^eval_video_log:\s*.*$", "eval_video_log: false", config_text, count=1
            )
            if changed != 1:
                raise RuntimeError("Could not disable RoboTwin episode video logging")
            (task_config / "demo_randomized.yml").write_text(config_text, encoding="utf-8")

            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "ROBOTWIN_PATH": str(ROBOTWIN_OVERLAY),
                    "ROBOTWIN_RUNTIME_ROOT": str(episode_root),
                    "ROBOTWIN_TEST_NUM": "1",
                    "ROBOTWIN_SEED_START": str(candidate_seed),
                    "ROBOTWIN_SEED_STRIDE": str(stride),
                    "ROBOTWIN_SKIP_VERSION_CHECK": "1",
                    "PYTHONPATH": os.pathsep.join(
                        (
                            str(ROBOTWIN_OVERLAY),
                            str(ROBOTWIN_ROOT),
                            str(PACKAGE_ROOT / "benchmarks/robotwin"),
                            str(PACKAGE_ROOT),
                            str(REPO),
                            env.get("PYTHONPATH", ""),
                        )
                    ),
                    "MPLCONFIGDIR": str(RUN_ROOT / "tmp/matplotlib" / self.host / f"gpu{gpu}"),
                }
            )
            Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
            egl_json = (
                ROBOTWIN_PYTHON.parent.parent
                / "lib/python3.10/site-packages/sapien/vulkan_library/10_nvidia.json"
            )
            if egl_json.is_file() and not env.get("__EGL_VENDOR_LIBRARY_FILENAMES"):
                env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(egl_json)

            command = [
                str(ROBOTWIN_PYTHON),
                str(EVAL_WRAPPER),
                "--config",
                str(config),
                "--overrides",
                "--task_name",
                task,
                "--task_config",
                "demo_randomized",
                "--ckpt_setting",
                CKPT_SETTING,
                "--seed",
                "0",
                "--policy_name",
                "openwam2robotwin_interface",
            ]
            log_path = (
                self.logs_root
                / f"episode.{task}.seed{candidate_seed}.{self.host}.gpu{gpu}.attempt{attempt}.log"
            )
            with log_path.open("w", encoding="utf-8", buffering=1) as log:
                log.write("COMMAND " + subprocess.list2cmdline(command) + "\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    cwd=ROBOTWIN_ROOT,
                    env={**env, "PYTHONUNBUFFERED": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = process.wait(timeout=EPISODE_TIMEOUT)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    code = 124
            if code == 0:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                plain = ANSI_RE.sub("", text)
                matches = EPISODE_RESULT_RE.findall(plain)
                result_files = list(episode_root.glob("eval_result/**/_result.txt"))
                if matches and len(result_files) == 1:
                    successes, actual_seed = matches[-1]
                    return {
                        "task": task,
                        "task_config": "demo_randomized",
                        "instruction_type": "unseen",
                        "candidate_seed": candidate_seed,
                        "seed": int(actual_seed),
                        "success": int(successes) == 1,
                        "attempt": attempt,
                        "gpu": gpu,
                        "worker_id": worker_id,
                        "source": "openwam_official_robotwin_wrapper",
                        "eval_log": str(log_path),
                        "result_file": str(result_files[0]),
                        "completed_utc": utc_now(),
                    }
            self.log(
                f"episode_retry task={task} candidate_seed={candidate_seed} gpu={gpu} "
                f"attempt={attempt} exit={code}"
            )
            time.sleep(2)
        return None

    def finish_episode(self, task: str, candidate: int, claim: Path, item: dict[str, Any] | None) -> None:
        try:
            if item is None:
                atomic_json(
                    self.failed_root / task / f"seed-{candidate}.json",
                    {
                        "task": task,
                        "candidate_seed": candidate,
                        "host": self.host,
                        "time_utc": utc_now(),
                        "reason": "episode failed after retry limit",
                    },
                )
                self.log(f"episode_skipped task={task} candidate_seed={candidate}")
                return
            result = self.results_root / task / f"seed-{int(item['seed'])}.json"
            if result.exists():
                self.log(
                    f"episode_duplicate_discarded task={task} candidate={candidate} "
                    f"actual_seed={item['seed']}"
                )
                return
            atomic_json(result, item)
            self.log(
                f"episode_done task={task} seed={item['seed']} success={int(bool(item['success']))} "
                f"gpu={item['gpu']} candidate={candidate}"
            )
            if self.done_count(task) >= TARGET_EPISODES:
                episodes = self.done_count(task)
                successes = self.success_count(task)
                atomic_json(
                    RUN_ROOT / "summaries" / f"{task}.json",
                    {
                        "task": task,
                        "episodes": episodes,
                        "successes": successes,
                        "success_rate": successes / episodes,
                        "prior_episodes": int(self.prior_counts[task]["episodes"]),
                        "prior_successes": int(self.prior_counts[task]["successes"]),
                        "instruction_type": "unseen",
                        "task_config": "demo_randomized",
                        "time_utc": utc_now(),
                    },
                )
        finally:
            try:
                owner = claim / "owner.json"
                if owner.exists():
                    owner.unlink()
                claim.rmdir()
            except OSError:
                pass

    def work(self, gpu: int) -> None:
        worker_id = self.node_index * GPUS_PER_NODE + gpu
        while True:
            assignment = self.reserve_episode(gpu)
            if assignment is None:
                if all(
                    self.done_count(task) >= TARGET_EPISODES
                    for task in self.tasks
                ):
                    return
                time.sleep(2)
                continue
            task, candidate, claim = assignment
            try:
                item = self.run_episode(task, candidate, gpu)
                self.finish_episode(task, candidate, claim, item)
                if item is not None:
                    key = (task, worker_id)
                    self.cursors[key] = int(item["seed"]) + TOTAL_WORKERS
                else:
                    key = (task, worker_id)
                    self.cursors[key] = candidate + TOTAL_WORKERS
            except Exception as exc:
                self.log(f"episode_exception task={task} candidate_seed={candidate} error={exc!r}")
                self.finish_episode(task, candidate, claim, None)

    def run(self) -> None:
        if not os.environ.get("OPENWAM_C2R_RUN_ROOT"):
            raise RuntimeError(
                "Set OPENWAM_C2R_RUN_ROOT to a shared directory visible from every worker node"
            )
        required_paths = (
            REPO,
            REPO / "openwam/deploy/server.py",
            REPO / "openwam/deploy/engine.py",
            REPO / "openwam/model/architectures/base.py",
            ROBOTWIN_ROOT,
            ROBOTWIN_ROOT / "task_config/demo_randomized.yml",
            ROBOTWIN_OVERLAY / "script/eval_policy.py",
            PACKAGE_ROOT / "benchmarks/robotwin/openwam2robotwin_interface.py",
            PACKAGE_ROOT / "benchmarks/robotwin/step_limits.yml",
            PACKAGE_ROOT / "benchmarks/utils/__init__.py",
            EVAL_WRAPPER,
            POLICY_TEMPLATE,
            ADAPTER,
            SFT_DIR / SFT_NAME,
            CRITIC,
        )
        missing = [str(path) for path in required_paths if not path.exists()]
        if missing:
            raise RuntimeError("Missing evaluation dependencies:\n  " + "\n  ".join(missing))
        guidance_hooks = (
            (REPO / "openwam/deploy/server.py", "server_builder"),
            (REPO / "openwam/deploy/engine.py", "action_velocity_guidance"),
            (REPO / "openwam/model/architectures/base.py", "action_velocity_guidance"),
        )
        unpatched = [
            str(path)
            for path, marker in guidance_hooks
            if marker not in path.read_text(encoding="utf-8")
        ]
        if unpatched:
            raise RuntimeError(
                "OpenWAM Q-guidance patch is not active in:\n  " + "\n  ".join(unpatched)
            )
        for gpu in range(GPUS_PER_NODE):
            self.ensure_server(gpu)
        atomic_json(
            self.worker_root / f"{self.host}.json",
            {"node_index": self.node_index, "pid": os.getpid(), "time_utc": utc_now()},
        )
        self.log(f"worker_start node_index={self.node_index} gpus=0-7 total_workers={TOTAL_WORKERS}")
        threads = [
            threading.Thread(target=self.work, args=(gpu,), name=f"gpu-{gpu}", daemon=False)
            for gpu in range(GPUS_PER_NODE)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.log("worker_exit all_tasks_complete")
        for gpu, server in self.server_processes.items():
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait(timeout=10)
            log = self.server_logs.get(gpu)
            if log is not None:
                log.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-previous", action="store_true")
    parser.add_argument("--node-index", type=int, choices=range(NODE_COUNT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENWAM_C2R_RUN_ROOT"):
        raise SystemExit("Set OPENWAM_C2R_RUN_ROOT to a shared directory visible from every worker node")
    if not RUN_ROOT.is_absolute():
        raise SystemExit("OPENWAM_C2R_RUN_ROOT must be an absolute path on shared storage")
    if args.import_previous:
        if SOURCE_ROOT is None:
            raise SystemExit("--import-previous requires OPENWAM_C2R_IMPORT_SOURCE_ROOT")
        import_previous_results()
        return 0
    if args.node_index is None:
        raise SystemExit("--node-index is required unless --import-previous is used")
    EpisodeQueue(args.node_index).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
