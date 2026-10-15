import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
import hashlib
import json
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
import os
import imageio
import cv2


def write_json(data, path):
    """Write the small progress record without importing the training dataset stack."""
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_task_classification():
    classification_path = Path(benchmark.__file__).with_name(
        "task_classification.json")
    if not classification_path.exists():
        return {}
    with classification_path.open() as handle:
        classifications = json.load(handle)
    return {
        suite: {entry["name"]: entry for entry in entries}
        for suite, entries in classifications.items()
    }


TASK_CLASSIFICATION = _load_task_classification()


def _resolve_init_state_path(task):
    """Resolve the same LIBERO-plus init-state file selected by the benchmark."""
    from libero.libero import get_libero_path

    init_root = Path(get_libero_path("init_states"))
    name = task.init_states_file
    candidates = [init_root / task.problem_folder / name]
    if "_language_" in name:
        candidates.append(
            init_root / task.problem_folder /
            (name.split("_language_")[0] + "." + name.split(".")[-1]))
    if "_view_" in name:
        candidates.append(
            init_root / task.problem_folder /
            (name.split("_view_")[0] + "." + name.split(".")[-1]))
    if "_table_" in name:
        candidates.append(
            init_root / task.problem_folder /
            (name.split("_table_")[0] + "." + name.split(".")[-1]))
    if "_tb_" in name:
        candidates.append(
            init_root / task.problem_folder /
            (name.split("_tb_")[0] + "." + name.split(".")[-1]))
    if "_light_" in name:
        candidates.append(
            init_root / task.problem_folder /
            (name.split("_light_")[0] + "." + name.split(".")[-1]))
    if "_add_" in name or "_level" in name:
        candidates.append(init_root / "libero_newobj" / task.problem_folder / name)
    return next((path for path in candidates if path.exists()), candidates[0])


def _resolve_bddl_path(task, path):
    """Resolve legacy generated BDDL names to the native asset filename."""
    if path.exists():
        return path
    stem = path.stem
    for marker in ("_view_", "_initstate_"):
        if marker in stem:
            stem = stem.split(marker, 1)[0]
    candidates = [path.parent / f"{stem}.bddl", path.parent / f"{task.name}.bddl"]
    candidates.extend(sorted(path.parent.glob(f"{stem}*.bddl")))
    return next((candidate for candidate in candidates if candidate.exists()), path)


def _task_identity(libero_benchmark, benchmark_instance, task_idx, episode_idx,
                   init_states, policy_id, policy_checkpoint, inference_config,
                   run_id):
    task = benchmark_instance.get_task(task_idx)
    classification = TASK_CLASSIFICATION.get(libero_benchmark, {}).get(task.name, {})
    bddl_path = _resolve_bddl_path(
        task, Path(benchmark_instance.get_task_bddl_file_path(task_idx)))
    init_path = _resolve_init_state_path(task)
    init_state_index = episode_idx % init_states.shape[0]
    return {
        "identity_version": 1,
        "run_id": run_id,
        "policy_id": policy_id,
        "policy_checkpoint": policy_checkpoint,
        "inference_config": inference_config,
        "benchmark": libero_benchmark,
        "task_index": task_idx,
        "task_id": classification.get("id"),
        "task_name": task.name,
        "prompt": task.language,
        "perturbation_category": classification.get("category"),
        "difficulty_level": classification.get("difficulty_level"),
        "bddl_path": str(bddl_path),
        "bddl_sha256": _sha256_file(bddl_path),
        "init_states_path": str(init_path),
        "init_states_file": task.init_states_file,
        "init_state_index": init_state_index,
        "init_state_count": int(init_states.shape[0]),
        "trial_index": episode_idx,
    }


def _write_evaluation_record(out_dir, identity, success, env_steps, video_path,
                             server_metadata):
    record = {
        **identity,
        "status": "completed",
        "success": bool(success),
        "env_steps": int(env_steps),
        "video_path": None if video_path is None else str(video_path),
        "server_metadata": server_metadata,
    }
    record_path = (Path(out_dir) / "task_records" / identity["benchmark"] /
                   f"task_{identity['task_index']:05d}" /
                   f"trial_{identity['trial_index']:03d}.json")
    record_path.parent.mkdir(exist_ok=True, parents=True)
    temporary_path = record_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(record_path)
    return record_path


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    # The SFT server is configured for the training camera names; retain the
    # legacy aliases for video output and older evaluation consumers.
    return {
        "observation.images.front": agentview,
        "observation.images.wrist": eye_in_hand,
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs), np.asarray(obs["robot0_joint_pos"], dtype=np.float32)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, policy_id,
            policy_checkpoint, inference_config, run_id, server_metadata):
    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    prompt = benchmark_instance.get_task(task_idx).language
    task = benchmark_instance.get_task(task_idx)
    bddl_path = _resolve_bddl_path(
        task, Path(benchmark_instance.get_task_bddl_file_path(task_idx)))
    env_args = {
                "bddl_file_name": str(bddl_path),
                "camera_heights": 128,
                "camera_widths": 128,
    }
    init_states = benchmark_instance.get_task_init_states(task_idx)
    identity = _task_identity(
        libero_benchmark, benchmark_instance, task_idx, episode_idx, init_states,
        policy_id, policy_checkpoint, inference_config, run_id)

    cur_env = construct_single_env(env_args)
    first_obs, initial_state = init_single_env(
        cur_env, init_states[episode_idx % init_states.shape[0]])

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    while cur_env.env.timestep < 800:
        ret = model.infer(
            dict(obs=first_obs, prompt=prompt, state=initial_state))
        action = np.asarray(ret['action'])

        key_frame_list = []
        if action.ndim == 2:
            if server_metadata.get("inference_protocol") == "original_temporal_action_kv":
                raise RuntimeError(
                    "Original LIBERO protocol must return a temporal action "
                    f"tensor, got flattened shape {tuple(action.shape)}"
                )
            action_per_frame = int(server_metadata.get("action_per_frame", 4))
            if action.shape[1] != 7 or action.shape[0] % action_per_frame:
                raise ValueError(
                    "Expected LIBERO action sequence with shape "
                    f"(T, 7), got {action.shape}")
            for step, ee_action in enumerate(action):
                observes, done = env_one_step(cur_env, ee_action)
                if done:
                    break
                if (step + 1) % action_per_frame == 0:
                    pass  # NO_VIDEO: retain RGB only for inference, do not buffer video frames
                    key_frame_list.append(observes)
            frame_chunk_size = int(server_metadata.get("frame_chunk_size", 4))
            if frame_chunk_size == 2:
                # SFT frame-chunk-2 uses one action state frame per observed
                # latent frame, with four actions on each frame.
                action_state = action.T.reshape(
                    action.shape[1], frame_chunk_size, action_per_frame)
            else:
                # The Long frame-chunk-4 contract folds four RGB observations
                # into one latent frame and keeps its full action horizon.
                action_state = action.T.reshape(
                    action.shape[1], 1, action.shape[0])
        else:
            expected_frame_chunk = int(server_metadata.get("frame_chunk_size", 4))
            expected_action_per_frame = int(server_metadata.get("action_per_frame", 4))
            expected_shape = (7, expected_frame_chunk, expected_action_per_frame)
            if tuple(action.shape) != expected_shape:
                raise RuntimeError(
                    "Original LIBERO protocol returned an unexpected action "
                    f"shape: expected {expected_shape}, got {tuple(action.shape)}"
                )
            action_per_frame = action.shape[2] // 4
            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    ee_action = action[:, i, j]
                    observes, done = env_one_step(cur_env, ee_action)
                    if done:
                        break
                    if (j+1) % action_per_frame == 0:
                        pass  # NO_VIDEO: retain RGB only for inference, do not buffer video frames
                        key_frame_list.append(observes)

                if done:
                    break
            action_state = action

        first = False

        if done:
            break
        else:
            model.infer(dict(
                obs=key_frame_list,
                compute_kv_cache=True,
                imagine=False,
                state=action_state,
            ))

    out_file = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}" / f"{episode_idx}_{done}.mp4"
    out_file.parent.mkdir(exist_ok=True, parents=True)

    # NO_VIDEO: OffScreenRenderEnv RGB remains enabled for policy input; skip MP4 encoding/writing.
    out_file = None

    env_steps = cur_env.env.timestep
    cur_env.close()
    record_path = _write_evaluation_record(
        out_dir, identity, done, env_steps, out_file, server_metadata)
    return done, record_path


def run(libero_benchmark, port, out_dir, test_num, trial_offset=0, task_range=None,
        task_indices=None,
        policy_id="unspecified", policy_checkpoint=None, inference_config=None,
        run_id=None, resume_map=None):
    '''
        task_range: [start, end) for splitting tasks
        task_indices: explicit task ids for non-contiguous native manifests
    '''
    if task_indices is not None:
        task_ids = list(task_indices)
        num_tasks = len(task_ids)
        progress_bar = tqdm(task_ids, total=num_tasks)
    elif task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)
    server_metadata = model.get_server_metadata()
    protocol = server_metadata.get("inference_protocol")
    if protocol != "original_temporal_action_kv":
        raise RuntimeError(
            "LIBERO evaluation requires the original temporal action/KV "
            f"protocol, got {protocol!r}. Refusing to produce a score."
        )

    video_save_root_dict = None
    resume_episodes = None
    if resume_map is not None:
        with open(resume_map, "r", encoding="utf-8") as f:
            raw_resume = json.load(f)
        resume_episodes = {int(k): [int(x) for x in v] for k, v in raw_resume.items()}

    if trial_offset < 0:
        raise ValueError("trial_offset must be non-negative")
    default_episode_list = range(trial_offset, trial_offset + test_num)
    for task_idx in progress_bar:
        if resume_episodes is not None:
            episode_list = resume_episodes.get(int(task_idx), [])
        else:
            episode_list = default_episode_list
        completed_before = 0
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            completed_before = len(video_save_list)
            episode_list = range(
                trial_offset + len(video_save_list), trial_offset + test_num)
        else:
            succ_num = 0.
            if resume_episodes is not None:
                # Remaining-only maps prevent re-running immutable records.  Use
                # the records already on disk as the cumulative summary prefix,
                # so the per-task summary still describes all 50 trials.
                valid_indices = set(
                    range(trial_offset, trial_offset + test_num))
                record_root = (
                    Path(out_dir) / "task_records" / libero_benchmark /
                    f"task_{int(task_idx):05d}")
                existing_records = []
                for record_file in sorted(record_root.glob("trial_*.json")):
                    try:
                        record = json.loads(record_file.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue
                    record_index = record.get("trial_index")
                    if (record.get("status") == "completed" and
                            record_index in valid_indices):
                        existing_records.append(record)
                completed_before = len(existing_records)
                succ_num = float(sum(
                    bool(record.get("success"))
                    for record in existing_records))

        if resume_episodes is not None:
            print(
                f"Resume task={task_idx}: skip_existing={completed_before}, "
                f"remaining={len(episode_list)}, success_prefix={succ_num}")

        for completed, episode_idx in enumerate(
            tqdm(episode_list, total=len(episode_list)), start=completed_before + 1
        ):
            res_i, record_path = run_one(
                model, libero_benchmark, task_idx, out_dir, episode_idx,
                policy_id, policy_checkpoint, inference_config, run_id,
                server_metadata)
            succ_num += res_i
            succ_rate = succ_num / completed
            print(
                f"Success rate: {succ_rate}, success num: {succ_num}, "
                f"total num: {completed}, record: {record_path}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": completed,
                "succ_rate": succ_rate,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--task-indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit task ids for non-contiguous native manifests",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--trial-offset",
        type=int,
        default=0,
        help="First native trial index; use disjoint offsets for concurrent clients.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="unspecified",
        help="Immutable policy label recorded with every completed rollout.",
    )
    parser.add_argument(
        "--policy-checkpoint",
        type=str,
        default=None,
        help="Checkpoint path or content hash recorded with every rollout.",
    )
    parser.add_argument(
        "--inference-config",
        type=str,
        default=None,
        help="Inference config name or content hash recorded with every rollout.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Caller-supplied immutable evaluation-run identifier.",
    )
    parser.add_argument(
        "--resume-map",
        type=str,
        default=None,
        help="JSON mapping task index to exact episode indices to run.",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
