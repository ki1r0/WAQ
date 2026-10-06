import sys
import os
import subprocess
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import cv2
from pathlib import Path

robowin_root = Path(os.environ.get("ROBOTWIN_ROOT", "/path/to/robotwin"))
if str(robowin_root) not in sys.path:
    sys.path.insert(0, str(robowin_root))


import os
os.chdir(robowin_root)

from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

# R6 no-video evaluation variant. Camera observations remain enabled because
# the policy needs them, but visualization generation, frame accumulation, and
# MP4 encoding are disabled.

import yaml
from datetime import datetime
import importlib
import argparse
import pdb
from evaluation.robotwin.geometry import euler2quat
import numpy as np

from description.utils.generate_episode_instructions import *
import traceback

import imageio
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path

from evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from evaluation.robotwin.test_render import Sapien_TEST

def write_json(data: dict, fpath: Path) -> None:
    """Write data to a JSON file.

    Creates parent directories if they don't exist.

    Args:
        data (dict): The dictionary to write.
        fpath (Path): The path to the output JSON file.
    """
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def append_jsonl(data: dict, fpath: Path) -> None:
    fpath.parent.mkdir(exist_ok=True, parents=True)
    with open(fpath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, sort_keys=True) + "\n")


def load_jsonl(fpath: Path) -> list[dict]:
    rows = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def claim_dynamic_replay_entry(queue_dir: Path, worker_rank: int) -> dict | None:
    """Atomically claim one replay row, waiting until work or completion.

    ``mkdir`` is the only scheduling primitive, so two clients cannot own the
    same seed.  The full replay row (instruction and episode_info included) is
    preserved; this changes scheduling only, not episode semantics.
    """
    jobs_dir = queue_dir / "jobs"
    claims_dir = queue_dir / "claims"
    done_dir = queue_dir / "done"
    events_path = queue_dir / f"events-worker-{worker_rank}.jsonl"
    jobs = sorted(jobs_dir.glob("job-*.jsonl"))
    if not jobs:
        raise RuntimeError(f"dynamic replay queue has no jobs: {jobs_dir}")
    while True:
        if (queue_dir / "ABORT").exists():
            raise RuntimeError(f"dynamic replay queue aborted: {queue_dir}")
        for job_path in jobs:
            job_name = job_path.stem
            if (done_dir / f"{job_name}.json").is_file():
                continue
            claim_dir = claims_dir / job_name
            try:
                claim_dir.mkdir()
            except FileExistsError:
                continue
            rows = load_jsonl(job_path)
            if len(rows) != 1:
                raise RuntimeError(f"expected one replay row in {job_path}, got {len(rows)}")
            replay_entry = rows[0]
            claim = {
                "event": "claim",
                "job": job_name,
                "worker_rank": int(worker_rank),
                "seed": int(replay_entry["seed"]),
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            write_json(claim, claim_dir / "claim.json")
            append_jsonl(claim, events_path)
            replay_entry["_dynamic_job"] = job_name
            return replay_entry
        if len(list(done_dir.glob("job-*.json"))) == len(jobs):
            return None
        time.sleep(1.0)


def complete_dynamic_replay_entry(
    queue_dir: Path,
    worker_rank: int,
    replay_entry: dict,
    success: bool,
    worker_episode: int,
) -> None:
    job_name = str(replay_entry["_dynamic_job"])
    done_path = queue_dir / "done" / f"{job_name}.json"
    if done_path.exists():
        raise RuntimeError(f"duplicate completion for {job_name}")
    record = {
        "event": "done",
        "job": job_name,
        "worker_rank": int(worker_rank),
        "worker_episode": int(worker_episode),
        "seed": int(replay_entry["seed"]),
        "success": bool(success),
        "timestamp": datetime.now().astimezone().isoformat(),
    }
    tmp_path = done_path.with_suffix(f".tmp.{os.getpid()}")
    write_json(record, tmp_path)
    os.replace(tmp_path, done_path)
    append_jsonl(record, queue_dir / f"events-worker-{worker_rank}.jsonl")


def to_uint8_rgb(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        if img.size and float(np.nanmax(img)) <= 1.0001:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


def normalize_predicted_video(video):
    arr = np.asarray(video)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 1, -1)
    if arr.ndim != 4:
        raise ValueError(f"Expected predicted video with 4 or 5 dims, got {arr.shape}")
    return [to_uint8_rgb(frame) for frame in arr]


def real_obs_prediction_layout(obs, pred_shape):
    pred_h, pred_w = int(pred_shape[0]), int(pred_shape[1])
    high = to_uint8_rgb(obs["observation.images.cam_high"])
    left = to_uint8_rgb(obs["observation.images.cam_left_wrist"])
    right = to_uint8_rgb(obs["observation.images.cam_right_wrist"])

    # RobotWin latents are encoded in the LingBot T-shape layout:
    # top row = left/right wrist cameras, bottom row = high camera.
    top_h = max(1, pred_h // 3)
    bottom_h = max(1, pred_h - top_h)
    left_w = max(1, pred_w // 2)
    right_w = max(1, pred_w - left_w)
    top = np.hstack(
        [
            cv2.resize(left, (left_w, top_h), interpolation=cv2.INTER_AREA),
            cv2.resize(right, (right_w, top_h), interpolation=cv2.INTER_AREA),
        ]
    )
    bottom = cv2.resize(high, (pred_w, bottom_h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(np.vstack([top, bottom]))


def prediction_reward_record(predicted_video, real_obs_list):
    if predicted_video is None or not real_obs_list:
        return None
    predicted_frames = normalize_predicted_video(predicted_video)
    frame_count = min(len(predicted_frames), len(real_obs_list))
    if frame_count <= 0:
        return None
    predicted_frames = predicted_frames[-frame_count:]
    real_obs_list = real_obs_list[-frame_count:]

    frame_mse = []
    frame_mae = []
    for predicted, obs in zip(predicted_frames, real_obs_list):
        real = real_obs_prediction_layout(obs, predicted.shape)
        predicted_f = predicted.astype(np.float32) / 255.0
        real_f = real.astype(np.float32) / 255.0
        diff = predicted_f - real_f
        frame_mse.append(float(np.mean(diff * diff)))
        frame_mae.append(float(np.mean(np.abs(diff))))

    pixel_mse = float(np.mean(frame_mse))
    pixel_mae = float(np.mean(frame_mae))
    return {
        "pixel_frames": int(frame_count),
        "pixel_mse": pixel_mse,
        "pixel_mae": pixel_mae,
        "pixel_reward": -pixel_mse,
        "pixel_frame_mse": frame_mse,
        "pixel_frame_mae": frame_mae,
    }

def add_title_bar(img, text, font_scale=0.8, thickness=2):
    """Add a black title bar with text above the image"""
    h, w, _ = img.shape
    bar_height = 40
    
    # Create black background bar
    title_bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    
    # Calculate text position to center it
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = (w - text_w) // 2
    text_y = (bar_height + text_h) // 2 - 5
    
    cv2.putText(title_bar, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return np.vstack([title_bar, img])

def quaternion_to_euler(quat):
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    quat: [rx, ry, rz, rw] format
    Return: [roll, pitch, yaw] (radians)
    """
    # scipy uses [x, y, z, w] format
    rotation = R.from_quat(quat)
    euler = rotation.as_euler('xyz', degrees=False)  # returns [roll, pitch, yaw]
    return euler

def visualize_action_step(action_history, step_idx, window=50):
    """
    Plot dual-arm action curves:
    Subplot 1: Left arm XYZ Position + Gripper
    Subplot 2: Left arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    Subplot 3: Right arm XYZ Position + Gripper
    Subplot 4: Right arm Euler angles (Roll, Pitch, Yaw) - converted from quaternion
    
    Input data format: [left_x, left_y, left_z, left_rx, left_ry, left_rz, left_rw, left_gripper,
                   right_x, right_y, right_z, right_rx, right_ry, right_rz, right_rw, right_gripper]
    Total 16 dimensions
    """
    # Create four subplots, sharing the X-axis
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 8), dpi=100, sharex=True)
    
    # 1. Determine slice range
    start = max(0, step_idx - window)
    end = step_idx + 1
    
    # 2. Get data subset
    history_subset = np.array(action_history)[start:end]
    
    # 3. Generate X-axis based on actual data length
    actual_len = len(history_subset)
    x_axis = range(start, start + actual_len)
    
    if actual_len > 0 and history_subset.shape[1] >= 16:
        # Convert quaternions to Euler angles
        left_euler = []
        right_euler = []
        
        for action in history_subset:
            # Left arm quaternion to Euler angles
            left_quat = action[3:7]  # [rx, ry, rz, rw]
            left_rpy = quaternion_to_euler(left_quat)
            left_euler.append(left_rpy)
            
            # Right arm quaternion to Euler angles
            right_quat = action[11:15]  # [rx, ry, rz, rw]
            right_rpy = quaternion_to_euler(right_quat)
            right_euler.append(right_rpy)
        
        left_euler = np.array(left_euler)
        right_euler = np.array(right_euler)
        
        # --- Left Arm ---
        # Subplot 1: Left Arm Translation (XYZ) + Gripper
        ax1.plot(x_axis, history_subset[:, 0], label='left_x', color='r', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 1], label='left_y', color='g', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 2], label='left_z', color='b', linewidth=1.5)
        ax1.plot(x_axis, history_subset[:, 7], label='left_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Position (m)')
        ax1.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f"Step {step_idx}: Left Arm Position & Gripper")

        # Subplot 2: Left Arm Euler Angles (Roll, Pitch, Yaw)
        ax2.plot(x_axis, left_euler[:, 0], label='left_roll', color='c', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 1], label='left_pitch', color='m', linewidth=1.5)
        ax2.plot(x_axis, left_euler[:, 2], label='left_yaw', color='y', linewidth=1.5)
        ax2.set_ylabel('Rotation (rad)')
        ax2.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Left Arm Rotation (RPY from Quaternion)")

        # --- Right Arm ---
        # Subplot 3: Right Arm Translation (XYZ) + Gripper
        ax3.plot(x_axis, history_subset[:, 8], label='right_x', color='r', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 9], label='right_y', color='g', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 10], label='right_z', color='b', linewidth=1.5, linestyle='--')
        ax3.plot(x_axis, history_subset[:, 15], label='right_grip', color='orange', 
                 linestyle=':', linewidth=2, alpha=0.8)
        ax3.set_ylabel('Position (m)')
        ax3.legend(loc='upper right', fontsize='x-small', ncol=4)
        ax3.grid(True, alpha=0.3)
        ax3.set_title("Right Arm Position & Gripper")

        # Subplot 4: Right Arm Euler Angles (Roll, Pitch, Yaw)
        ax4.plot(x_axis, right_euler[:, 0], label='right_roll', color='c', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 1], label='right_pitch', color='m', linewidth=1.5, linestyle='--')
        ax4.plot(x_axis, right_euler[:, 2], label='right_yaw', color='y', linewidth=1.5, linestyle='--')
        ax4.set_ylabel('Rotation (rad)')
        ax4.legend(loc='upper right', fontsize='x-small', ncol=3)
        ax4.grid(True, alpha=0.3)
        ax4.set_title("Right Arm Rotation (RPY from Quaternion)")

    # Set X-axis display range to maintain sliding window effect
    ax1.set_xlim(max(0, step_idx - window), max(window, step_idx))
    ax3.set_xlabel('Step')
    ax4.set_xlabel('Step')
    
    plt.tight_layout()
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba())
    img = img[:, :, :3]
    
    # Convert to uint8
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
        
    plt.close(fig)
    return img


def save_comparison_video(real_obs_list, imagined_video, action_history, save_path, fps=15):
    if not real_obs_list:
        return

    n_real = len(real_obs_list)
    if imagined_video is not None:
        imagined_video = np.concatenate(imagined_video, 0)
        n_imagined = len(imagined_video) 
    else:
        n_imagined = 0
    n_frames = n_real # Based on real observation frames
    
    print(f"Saving video: Real {n_real} frames, Imagined {n_imagined} frames...")

    final_frames = []

    for i in range(n_frames):
        obs = real_obs_list[i]
        cam_high = obs["observation.images.cam_high"]
        cam_left = obs["observation.images.cam_left_wrist"]
        cam_right = obs["observation.images.cam_right_wrist"]

        base_h = cam_high.shape[0]
        
        def resize_h(img, h):
            if img.shape[0] != h:
                w = int(img.shape[1] * h / img.shape[0])
                img = cv2.resize(img, (w, h))
            img = np.ascontiguousarray(img)
            if img.dtype != np.uint8:
                img = (img * 255).astype(np.uint8)
            return img

        row_real = np.hstack([
            resize_h(cam_high, base_h), 
            resize_h(cam_left, base_h), 
            resize_h(cam_right, base_h)
        ])
        
        row_real = np.ascontiguousarray(row_real)

        row_real = add_title_bar(row_real, "Real Observation (High / Left / Right)")

        target_width = row_real.shape[1]

        if imagined_video is not None and i < n_imagined:
            img_frame = imagined_video[i]
            if img_frame.dtype != np.uint8 and img_frame.max() <= 1.0001:
                img_frame = (img_frame * 255).astype(np.uint8)
            elif img_frame.dtype != np.uint8:
                img_frame = img_frame.astype(np.uint8)

            h = int(img_frame.shape[0] * target_width / img_frame.shape[1])
            row_imagined = cv2.resize(img_frame, (target_width, h))
        else:
            row_imagined = np.zeros((300, target_width, 3), dtype=np.uint8)
            cv2.putText(row_imagined, "Coming soon", (target_width//2 - 100, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)

        row_imagined = np.ascontiguousarray(row_imagined)
        row_imagined = add_title_bar(row_imagined, "Imagined Video Stream")
        full_frame = np.vstack([row_real, row_imagined])
        full_frame = np.ascontiguousarray(full_frame)
        final_frames.append(full_frame)

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"Combined video saved to: {save_path}")


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(robowin_root, "task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def main(usr_args):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    save_root = usr_args["save_root"]
    policy_name = usr_args["policy_name"]
    video_guidance_scale = usr_args["video_guidance_scale"]
    action_guidance_scale = usr_args["action_guidance_scale"]
    instruction_type = 'seen'
    save_dir = None
    video_save_dir = None
    video_size = None

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    # Enforce no-video behavior independently of YAML/CLI boolean parsing.
    args["eval_video_log"] = False

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting
    args["save_root"] = save_root
    args["save_rollouts"] = bool(usr_args.get("save_rollouts", False))
    args["rollout_save_mode"] = usr_args.get("rollout_save_mode", "failures")
    args["rollout_save_root"] = usr_args.get("rollout_save_root")
    args["record_prediction_reward"] = bool(
        usr_args.get("record_prediction_reward", False)
    )
    args["prediction_reward_log_path"] = usr_args.get("prediction_reward_log_path")
    args["timing_log_path"] = usr_args.get("timing_log_path")
    args["eval_variant"] = usr_args.get("eval_variant", "default")
    args["replay_manifest_write_path"] = usr_args.get("replay_manifest_write_path")
    args["replay_manifest_read_path"] = usr_args.get("replay_manifest_read_path")
    args["dynamic_replay_queue_dir"] = usr_args.get("dynamic_replay_queue_dir")
    args["dynamic_worker_rank"] = usr_args.get("dynamic_worker_rank", -1)

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{current_time}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        camera_config = get_camera_config(args["camera"]["head_camera_type"])
        video_size = str(camera_config["w"]) + "x" + str(camera_config["h"])
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 10000 * (1 + seed)
    suc_nums = []
    test_num = usr_args["test_num"]

    
    model = WebsocketClientPolicy(host=usr_args.get("server_host", "HOST_ADDRESS"), port=usr_args['port'])

    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   save_visualization=False,
                                   video_guidance_scale=video_guidance_scale,
                                   action_guidance_scale=action_guidance_scale)
    suc_nums.append(suc_num)

    file_path = os.path.join(save_dir, f"_result.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {current_time}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")

def format_obs(observation, prompt):
    return {
                "observation.images.cam_high": observation["observation"]["head_camera"]["rgb"], # H,W,3
                "observation.images.cam_left_wrist": observation["observation"]["left_camera"]["rgb"],
                "observation.images.cam_right_wrist": observation["observation"]["right_camera"]["rgb"],
                "observation.state": observation["joint_action"]["vector"],
                "task": prompt,
            }

def rollout_action_data(raw_action, executed_ee_action, valid=True):
    return {
        "policy_action": {
            "raw": np.asarray(raw_action, dtype=np.float32),
            "executed_ee": np.asarray(executed_ee_action, dtype=np.float32),
            "valid": np.bool_(valid),
        }
    }


def write_rollout_sidecars(rollout_dir, episode_index, prompt, episode_info):
    instruction_path = rollout_dir / "instructions" / f"episode{episode_index}.json"
    write_json({"seen": [prompt], "unseen": []}, instruction_path)

    scene_info_path = rollout_dir / "scene_info.json"
    if scene_info_path.exists():
        with scene_info_path.open("r", encoding="utf-8") as f:
            scene_info = json.load(f)
    else:
        scene_info = {}
    scene_info[f"episode_{episode_index}"] = {"info": episode_info}
    write_json(scene_info, scene_info_path)

def add_eef_pose(new_pose, init_pose):
    new_pose_R = R.from_quat(new_pose[3:7][None])
    init_pose_R = R.from_quat(init_pose[3:7][None])
    out_rot = (init_pose_R * new_pose_R).as_quat().reshape(-1)
    out_trans = new_pose[:3] + init_pose[:3]
    return np.concatenate([out_trans, out_rot, new_pose[7:8]])

def add_init_pose(new_pose, init_pose):
    left_pose = add_eef_pose(new_pose[:8], init_pose[:8])
    right_pose = add_eef_pose(new_pose[8:], init_pose[8:])
    return np.concatenate([left_pose, right_pose])

def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                save_visualization=False,
                video_guidance_scale=5.0,
                action_guidance_scale=5.0):
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []


    now_seed = st_seed
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    save_rollouts = bool(args.get("save_rollouts", False))
    rollout_save_mode = args.get("rollout_save_mode", "failures")
    rollout_root = Path(args.get("rollout_save_root") or Path(args["save_root"]) / "rollouts")
    rollout_dir = rollout_root / task_name / args["task_config"]
    record_prediction_reward = bool(args.get("record_prediction_reward", False))
    prediction_reward_log_path = Path(
        args.get("prediction_reward_log_path")
        or Path(args["save_root"]) / "metrics" / "prediction_rewards.jsonl"
    )
    timing_log_path = Path(
        args.get("timing_log_path")
        or Path(args["save_root"]) / "metrics" / "rollout_timing.jsonl"
    )
    eval_variant = str(args.get("eval_variant", "default"))
    replay_manifest_write_path = (
        Path(args["replay_manifest_write_path"])
        if args.get("replay_manifest_write_path")
        else None
    )
    replay_manifest_read_path = (
        Path(args["replay_manifest_read_path"])
        if args.get("replay_manifest_read_path")
        else None
    )
    replay_entries = None
    dynamic_replay_queue_dir = (
        Path(args["dynamic_replay_queue_dir"])
        if args.get("dynamic_replay_queue_dir")
        else None
    )
    dynamic_worker_rank = int(args.get("dynamic_worker_rank", -1))
    if dynamic_replay_queue_dir is not None and replay_manifest_read_path is not None:
        raise ValueError("dynamic replay queue and fixed replay manifest are mutually exclusive")
    if replay_manifest_read_path is not None:
        replay_entries = load_jsonl(replay_manifest_read_path)
        print(
            f"Loaded replay manifest: {replay_manifest_read_path} "
            f"({len(replay_entries)} episodes)"
        )
    rollout_episode_num = 0
    if save_rollouts:
        while (rollout_dir / "data" / f"episode{rollout_episode_num}.hdf5").exists():
            rollout_episode_num += 1
        print(f"Rollout dataset: {rollout_dir} (next episode {rollout_episode_num})")

    while succ_seed < test_num:
        render_freq = args["render_freq"]
        args["render_freq"] = 0
        replay_entry = None
        if dynamic_replay_queue_dir is not None:
            replay_entry = claim_dynamic_replay_entry(
                dynamic_replay_queue_dir, dynamic_worker_rank
            )
            if replay_entry is None:
                break
            now_seed = int(replay_entry["seed"])
            episode_info = {"info": replay_entry.get("episode_info", {})}
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        elif replay_entries is not None:
            if succ_seed >= len(replay_entries):
                break
            replay_entry = replay_entries[succ_seed]
            now_seed = int(replay_entry["seed"])
            episode_info = {"info": replay_entry.get("episode_info", {})}
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        elif expert_check:
            args["save_data"] = False
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                TASK_ENV.close_env()
                now_seed += 1
                args["render_freq"] = render_freq
                print(f"error occurs ! {e}")
                traceback.print_exc()
                continue

        if replay_entry is not None:
            pass
        elif (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        if save_rollouts:
            args["save_data"] = True
            args["save_path"] = str(rollout_dir)
            rollout_candidate_index = rollout_episode_num
        else:
            args["save_data"] = False
            rollout_candidate_index = now_id
        TASK_ENV.setup_demo(now_ep_num=rollout_candidate_index, seed=now_seed, is_test=True, **args)
        episode_info_list = [episode_info["info"]]
        if replay_entry is not None:
            instruction = str(replay_entry["instruction"])
        else:
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        if replay_manifest_write_path is not None:
            append_jsonl(
                {
                    "event": "replay_episode",
                    "variant": eval_variant,
                    "task": task_name,
                    "task_config": args["task_config"],
                    "worker_seed": int(st_seed),
                    "worker_episode": int(TASK_ENV.test_num),
                    "seed": int(now_seed),
                    "instruction": str(instruction),
                    "episode_info": episode_info["info"],
                    "rollout_episode_index": int(rollout_candidate_index),
                },
                replay_manifest_write_path,
            )
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        if TASK_ENV.eval_video_path is not None:
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    f"{TASK_ENV.eval_video_path}/episode{TASK_ENV.test_num}.mp4",
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)

        succ = False

        episode_wall_start = time.perf_counter()
        timing = {
            "reset_inference_seconds": 0.0,
            "policy_inference_seconds": 0.0,
            "kv_inference_seconds": 0.0,
            "simulation_step_seconds": 0.0,
            "policy_inference_calls": 0,
            "kv_inference_calls": 0,
            "simulation_steps": 0,
        }

        prompt = TASK_ENV.get_instruction()
        timed_start = time.perf_counter()
        ret = model.infer(dict(reset = True, prompt=prompt, save_visualization=save_visualization))
        timing["reset_inference_seconds"] += time.perf_counter() - timed_start
        
        first = True
        full_obs_list = []
        gen_video_list = []
        full_action_history = []
        rollout_started = False
        policy_action_space = None

        initial_obs = TASK_ENV.get_obs() 
        inint_eef_pose = initial_obs['endpose']['left_endpose'] + \
        [initial_obs['endpose']['left_gripper']] + \
        initial_obs['endpose']['right_endpose'] + \
        [initial_obs['endpose']['right_gripper']]
        inint_eef_pose = np.array(inint_eef_pose, dtype=np.float64)
        if inint_eef_pose.shape != (16,):
            raise ValueError(
                f"Expected 16-D dual-arm EE state, got {inint_eef_pose.shape}"
            )
        initial_formatted_obs = format_obs(initial_obs, prompt)
        if save_visualization:
            full_obs_list.append(initial_formatted_obs)
        first_obs = None
        episode_chunk_index = 0
        while TASK_ENV.take_action_cnt<TASK_ENV.step_lim:
            if first:
                first_obs = format_obs(TASK_ENV.get_obs(), prompt)

            timed_start = time.perf_counter()
            ret = model.infer(dict(
                obs=first_obs,
                prompt=prompt,
                save_visualization=save_visualization,
                video_guidance_scale=video_guidance_scale,
                action_guidance_scale=action_guidance_scale,
                return_predicted_video=record_prediction_reward,
            )) #(TASK_ENV, model, observation)
            timing["policy_inference_seconds"] += time.perf_counter() - timed_start
            timing["policy_inference_calls"] += 1
            action = np.asarray(ret['action'])
            if action.ndim != 3 or action.shape[0] not in (14, 16):
                raise ValueError(
                    f"Expected policy action shape (14|16, F, H), got {action.shape}"
                )
            if action.shape[0] == 14:
                policy_action_space = "dual_ee_pose_rpy"
            else:
                policy_action_space = "dual_ee_delta_pose_quaternion"
            if save_rollouts and not rollout_started:
                TASK_ENV._take_picture(
                    extra_data=rollout_action_data(
                        np.full(action.shape[0], np.nan, dtype=np.float32),
                        np.full(16, np.nan, dtype=np.float32),
                        valid=False,
                    )
                )
                rollout_started = True
            if save_visualization and 'video' in ret:
                imagined_video = ret['video']
                gen_video_list.append(imagined_video)
            key_frame_list = []

            assert action.shape[2] % 4 == 0
            action_per_frame = action.shape[2] // 4

            start_idx = 1 if first else 0
            for i in range(start_idx, action.shape[1]):
                for j in range(action.shape[2]):
                    raw_action_step = action[:, i, j].flatten() 
                    if save_visualization:
                        full_action_history.append(raw_action_step)

                    ee_action = action[:, i, j]
                    if action.shape[0] == 14:
                        ee_action = np.concatenate([
                            ee_action[:3],
                            euler2quat(ee_action[3], ee_action[4], ee_action[5]),
                            ee_action[6:10],
                            euler2quat(ee_action[10], ee_action[11], ee_action[12]),
                            ee_action[13:14]
                        ])
                    elif action.shape[0] == 16:
                        ee_action =  add_init_pose(ee_action, inint_eef_pose)
                        ee_action = np.concatenate([
                            ee_action[:3],
                            ee_action[3:7] / np.linalg.norm(ee_action[3:7]),
                            ee_action[7:11],
                            ee_action[11:15] / np.linalg.norm(ee_action[11:15]),
                            ee_action[15:16]
                        ])
                    else:
                        raise NotImplementedError
                    timed_start = time.perf_counter()
                    TASK_ENV.take_action(ee_action, action_type='ee')
                    timing["simulation_step_seconds"] += time.perf_counter() - timed_start
                    timing["simulation_steps"] += 1
                    if save_rollouts:
                        TASK_ENV._take_picture(
                            extra_data=rollout_action_data(raw_action_step, ee_action)
                        )
                   
                    if (j+1) % action_per_frame == 0:
                        obs = format_obs(TASK_ENV.get_obs(), prompt)
                        if save_visualization:
                            full_obs_list.append(obs)
                        key_frame_list.append(obs)

            expected_observations = (action.shape[1] - start_idx) * 4
            if len(key_frame_list) != expected_observations:
                raise RuntimeError(
                    "KV-cache observation cadence mismatch: expected "
                    f"{expected_observations} frames for {action.shape[1]} "
                    f"latent frames, got {len(key_frame_list)}."
                )

            first = False

            reward_record = None
            if record_prediction_reward:
                try:
                    reward_record = prediction_reward_record(
                        ret.get("video"),
                        key_frame_list,
                    )
                except Exception as e:
                    reward_record = {"prediction_reward_error": repr(e)}

            timed_start = time.perf_counter()
            kv_ret = model.infer(dict(
                obs=key_frame_list,
                compute_kv_cache=True,
                imagine=False,
                save_visualization=save_visualization,
                state=action,
            ))
            timing["kv_inference_seconds"] += time.perf_counter() - timed_start
            timing["kv_inference_calls"] += 1
            if record_prediction_reward:
                record = {
                    "event": "prediction_reward_chunk",
                    "variant": eval_variant,
                    "task": task_name,
                    "task_config": args["task_config"],
                    "episode": int(TASK_ENV.test_num),
                    "worker_seed": int(st_seed),
                    "seed": int(now_seed),
                    "chunk_index": int(episode_chunk_index),
                    "take_action_cnt": int(TASK_ENV.take_action_cnt),
                    "real_frames": int(len(key_frame_list)),
                }
                if reward_record is not None:
                    record.update(reward_record)
                if isinstance(kv_ret, dict) and isinstance(
                    kv_ret.get("prediction_metrics"), dict
                ):
                    record.update(kv_ret["prediction_metrics"])
                append_jsonl(record, prediction_reward_log_path)
                episode_chunk_index += 1
  
            if TASK_ENV.eval_success:
                succ = True
                break
      

        if save_visualization:
            vis_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'visualization' / task_name
            vis_dir.mkdir(parents=True, exist_ok=True)
            video_name = f"{TASK_ENV.test_num}_{prompt.replace(' ', '_')}_{succ}.mp4"
            out_img_file = vis_dir / video_name
            save_comparison_video(
                real_obs_list=full_obs_list,
                imagined_video=None, #gen_video_list,
                action_history=full_action_history,
                save_path=str(out_img_file),
                fps=15 # Suggest adjusting fps based on simulation step
            )
        if TASK_ENV.eval_video_path is not None:
            TASK_ENV._del_eval_video_ffmpeg()

        if save_rollouts:
            keep_rollout = rollout_save_mode == "all" or not succ
            if keep_rollout:
                timed_out = not succ and TASK_ENV.take_action_cnt >= TASK_ENV.step_lim
                termination_reason = "task_success" if succ else ("timeout" if timed_out else "task_failure")
                TASK_ENV.merge_pkl_to_hdf5_video_with_meta(
                    success=succ,
                    termination_reason=termination_reason,
                    truncated=timed_out,
                    extra_metadata={
                        "source": "policy_rollout",
                        "policy_name": args["policy_name"],
                        "seed": int(now_seed),
                        "instruction": str(prompt),
                        "policy_action_space": policy_action_space or "unknown",
                        "executed_action_space": "dual_ee_absolute_pose_quaternion",
                    },
                )
                write_rollout_sidecars(
                    rollout_dir,
                    rollout_candidate_index,
                    prompt,
                    episode_info["info"],
                )
                TASK_ENV.remove_data_cache()
                rollout_episode_num += 1
            else:
                TASK_ENV.remove_data_cache()

        control_loop_wall_seconds = time.perf_counter() - episode_wall_start
        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        timing_record = {
            "event": "rollout_timing",
            "variant": eval_variant,
            "task": task_name,
            "task_config": args["task_config"],
            "episode": int(TASK_ENV.test_num),
            "worker_seed": int(st_seed),
            "seed": int(now_seed),
            "success": bool(succ),
            "take_action_cnt": int(TASK_ENV.take_action_cnt),
            "chunks": int(episode_chunk_index),
            "control_loop_wall_seconds": float(control_loop_wall_seconds),
            **{
                key: float(value) if key.endswith("_seconds") else int(value)
                for key, value in timing.items()
            },
        }
        timing_record["mean_policy_inference_seconds"] = (
            timing_record["policy_inference_seconds"]
            / timing_record["policy_inference_calls"]
            if timing_record["policy_inference_calls"]
            else None
        )
        timing_record["mean_kv_inference_seconds"] = (
            timing_record["kv_inference_seconds"] / timing_record["kv_inference_calls"]
            if timing_record["kv_inference_calls"]
            else None
        )
        timing_record["mean_simulation_step_seconds"] = (
            timing_record["simulation_step_seconds"] / timing_record["simulation_steps"]
            if timing_record["simulation_steps"]
            else None
        )
        append_jsonl(timing_record, timing_log_path)

        if record_prediction_reward:
            append_jsonl(
                {
                    "event": "episode_result",
                    "variant": eval_variant,
                    "task": task_name,
                    "task_config": args["task_config"],
                    "episode": int(TASK_ENV.test_num),
                    "worker_seed": int(st_seed),
                    "seed": int(now_seed),
                    "success": bool(succ),
                    "take_action_cnt": int(TASK_ENV.take_action_cnt),
                    "chunks": int(episode_chunk_index),
                },
                prediction_reward_log_path,
            )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        save_dir = Path(args['save_root']) / f'stseed-{st_seed}' / 'metrics' / task_name
        save_dir.mkdir(parents=True, exist_ok=True)
        out_json_file = save_dir / 'res.json'
        write_json({
          "succ_num": float(TASK_ENV.suc),
          "total_num": float(TASK_ENV.test_num),
          "succ_rate": float(TASK_ENV.suc / TASK_ENV.test_num),
        }, out_json_file)
        
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        if dynamic_replay_queue_dir is not None:
            complete_dynamic_replay_entry(
                dynamic_replay_queue_dir,
                dynamic_worker_rank,
                replay_entry,
                succ,
                TASK_ENV.test_num - 1,
            )
        now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    parser.add_argument("--port", type=int, default=8000, help='remote policy socket port.')
    parser.add_argument("--save_root", type=str, default="results/default_vis_path")
    parser.add_argument("--video_guidance_scale", type=float, default=5.0)
    parser.add_argument("--action_guidance_scale", type=float, default=5.0)
    parser.add_argument("--test_num", type=int, default=100)
    parser.add_argument("--server_host", type=str, default="HOST_ADDRESS", help="remote policy server host/IP.")
    parser.add_argument(
        "--save_rollouts",
        action="store_true",
        default=None,
        help="Save policy rollouts in RoboTwin HDF5 format",
    )
    parser.add_argument(
        "--rollout_save_root",
        type=str,
        default=None,
        help="Rollout dataset root; defaults to <save_root>/rollouts",
    )
    parser.add_argument(
        "--rollout_save_mode",
        choices=("failures", "all"),
        default=None,
        help="Save only failed rollouts or every rollout",
    )
    parser.add_argument(
        "--record_prediction_reward",
        action="store_true",
        default=None,
        help="Record real-vs-predicted video distance rewards during rollout",
    )
    parser.add_argument(
        "--prediction_reward_log_path",
        type=str,
        default=None,
        help="JSONL path for per-chunk prediction reward records",
    )
    parser.add_argument(
        "--timing_log_path",
        type=str,
        default=None,
        help="JSONL path for per-episode rollout timing records",
    )
    parser.add_argument(
        "--eval_variant",
        type=str,
        default=None,
        help="Variant label to store in metrics and reward logs",
    )
    parser.add_argument(
        "--replay_manifest_write_path",
        type=str,
        default=None,
        help="Write accepted episode seeds/instructions to this JSONL manifest",
    )
    parser.add_argument(
        "--replay_manifest_read_path",
        type=str,
        default=None,
        help="Replay episode seeds/instructions from this JSONL manifest",
    )
    parser.add_argument(
        "--dynamic_replay_queue_dir",
        type=str,
        default=None,
        help="Atomic shared replay queue; preserves a persistent client episode loop",
    )
    parser.add_argument(
        "--dynamic_worker_rank",
        type=int,
        default=None,
        help="Worker rank recorded in dynamic queue claims/completions",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    config['port'] = args.port
    config['save_root'] = args.save_root
    config['video_guidance_scale'] = args.video_guidance_scale
    config['action_guidance_scale'] = args.action_guidance_scale
    config['test_num'] = args.test_num
    config['server_host'] = args.server_host
    if args.save_rollouts is not None:
        config['save_rollouts'] = args.save_rollouts
    else:
        config.setdefault('save_rollouts', False)
    if args.rollout_save_root is not None:
        config['rollout_save_root'] = args.rollout_save_root
    else:
        config.setdefault('rollout_save_root', None)
    if args.rollout_save_mode is not None:
        config['rollout_save_mode'] = args.rollout_save_mode
    else:
        config.setdefault('rollout_save_mode', 'failures')
    if args.record_prediction_reward is not None:
        config['record_prediction_reward'] = args.record_prediction_reward
    else:
        config.setdefault('record_prediction_reward', False)
    if args.prediction_reward_log_path is not None:
        config['prediction_reward_log_path'] = args.prediction_reward_log_path
    else:
        config.setdefault('prediction_reward_log_path', None)
    if args.timing_log_path is not None:
        config['timing_log_path'] = args.timing_log_path
    else:
        config.setdefault('timing_log_path', None)
    if args.eval_variant is not None:
        config['eval_variant'] = args.eval_variant
    else:
        config.setdefault('eval_variant', 'default')
    if args.replay_manifest_write_path is not None:
        config['replay_manifest_write_path'] = args.replay_manifest_write_path
    else:
        config.setdefault('replay_manifest_write_path', None)
    if args.replay_manifest_read_path is not None:
        config['replay_manifest_read_path'] = args.replay_manifest_read_path
    else:
        config.setdefault('replay_manifest_read_path', None)
    if args.dynamic_replay_queue_dir is not None:
        config['dynamic_replay_queue_dir'] = args.dynamic_replay_queue_dir
    else:
        config.setdefault('dynamic_replay_queue_dir', None)
    if args.dynamic_worker_rank is not None:
        config['dynamic_worker_rank'] = args.dynamic_worker_rank
    else:
        config.setdefault('dynamic_worker_rank', -1)

    return config


if __name__ == "__main__":
    
    Sapien_TEST()
    usr_args = parse_args_and_config()
    main(usr_args)
