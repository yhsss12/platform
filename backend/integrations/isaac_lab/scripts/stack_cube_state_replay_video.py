#!/usr/bin/env python3
# Copyright (c) 2026 EAI Platform
"""State-based Stack Cube HDF5 replay: set sim state per frame and render MP4."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from preview_video_utils import build_preview_from_frames as _build_preview

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="State-based Stack Cube HDF5 replay video.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--select_episodes", type=int, nargs="*", default=[0])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--live_frame_dir", type=str, required=True)
parser.add_argument("--live_frame_every", type=int, default=1)
parser.add_argument("--live_status_out", type=str, default="")
parser.add_argument("--preview_video_out", type=str, default="")
parser.add_argument("--replay_mode", type=str, default="state", choices=["state", "action"])

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    print(f"ERROR: isaaclab unavailable: {exc}", file=sys.stderr)
    sys.exit(1)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.replay_mode != "state":
    print("ERROR: stack_cube_state_replay_video.py requires --replay_mode state", file=sys.stderr)
    sys.exit(2)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

STACK_CUBE_VIEWER_EYE = (1.35, 1.35, 1.05)
STACK_CUBE_VIEWER_LOOKAT = (0.5, 0.0, 0.35)
CUBE_NAMES = ("cube_1", "cube_2", "cube_3")


def configure_single_env_live_viewer(env_cfg, env_index: int = 0) -> None:
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = max(0, int(env_index))
    env_cfg.viewer.eye = STACK_CUBE_VIEWER_EYE
    env_cfg.viewer.lookat = STACK_CUBE_VIEWER_LOOKAT


def focus_live_viewport(env, env_index: int = 0) -> None:
    env_index = max(0, min(int(env_index), max(0, env.num_envs - 1)))
    vcc = getattr(env, "viewport_camera_controller", None)
    if vcc is not None:
        vcc.set_view_env_index(env_index)
        vcc.update_view_location(eye=STACK_CUBE_VIEWER_EYE, lookat=STACK_CUBE_VIEWER_LOOKAT)
        return
    if hasattr(env, "sim") and hasattr(env, "scene"):
        origin = env.scene.env_origins[env_index].detach().cpu().numpy()
        eye = origin + np.array(STACK_CUBE_VIEWER_EYE, dtype=float)
        target = origin + np.array(STACK_CUBE_VIEWER_LOOKAT, dtype=float)
        env.sim.set_camera_view(eye=eye, target=target)


def _rgb_is_valid(arr: np.ndarray, *, min_mean: float = 2.0, min_std: float = 2.0) -> bool:
    if arr.size == 0:
        return False
    return float(arr.mean()) > min_mean and float(arr.std()) > min_std


def capture_viewport_rgb(env, simulation_app, *, env_index: int = 0, max_attempts: int = 24) -> np.ndarray | None:
    if not hasattr(env, "render"):
        return None
    for _ in range(max_attempts):
        focus_live_viewport(env, env_index)
        if hasattr(env, "sim"):
            env.sim.render()
        if simulation_app is not None:
            simulation_app.update()
        try:
            rgb = env.render(recompute=False)
        except Exception:
            continue
        if rgb is None:
            continue
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3 and _rgb_is_valid(arr):
            return arr[:, :, :3].copy()
    return None


def _write_status(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _capture_frame(env, live_dir: Path, frames_dir: Path, frame_count: list[int], simulation_app) -> bool:
    arr = capture_viewport_rgb(env, simulation_app, max_attempts=12)
    if arr is None:
        return False
    try:
        import cv2
    except ImportError:
        return False
    bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
    frame_count[0] += 1
    cv2.imwrite(str(frames_dir / f"frame_{frame_count[0]:06d}.jpg"), bgr)
    cv2.imwrite(str(live_dir / "latest.jpg"), bgr)
    return True


def _load_episode_arrays(demo_group: h5py.Group) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    if "states" in demo_group and "articulation" in demo_group["states"]:
        robot = demo_group["states"]["articulation"]["robot"]
        out["robot_joint_pos"] = np.asarray(robot["joint_position"])
        out["robot_joint_vel"] = np.asarray(robot["joint_velocity"])
        if "rigid_object" in demo_group["states"]:
            ro = demo_group["states"]["rigid_object"]
            for cube in CUBE_NAMES:
                if cube in ro:
                    out[f"{cube}_root_pose"] = np.asarray(ro[cube]["root_pose"])
                    out[f"{cube}_root_velocity"] = np.asarray(ro[cube]["root_velocity"])
    elif "obs" in demo_group:
        obs = demo_group["obs"]
        out["robot_joint_pos"] = np.asarray(obs["joint_pos"])
        out["robot_joint_vel"] = np.asarray(obs.get("joint_vel", obs["joint_pos"]))
        positions = np.asarray(obs["cube_positions"]).reshape(-1, 3, 3)
        orientations = None
        if "cube_orientations" in obs:
            orientations = np.asarray(obs["cube_orientations"]).reshape(-1, 3, 4)
        for idx, cube in enumerate(CUBE_NAMES):
            pos = positions[:, idx, :]
            if orientations is not None:
                quat = orientations[:, idx, :]
            else:
                quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(pos), 1))
            out[f"{cube}_root_pose"] = np.concatenate([pos, quat], axis=1)
            out[f"{cube}_root_velocity"] = np.zeros((len(pos), 6), dtype=np.float32)
    else:
        raise KeyError("demo missing states and obs joint data")
    out["num_steps"] = int(out["robot_joint_pos"].shape[0])
    return out


def _apply_state_frame(env, frame_idx: int, arrays: dict[str, np.ndarray], device: torch.device) -> None:
    env_ids = torch.tensor([0], device=device, dtype=torch.long)
    robot = env.scene["robot"]
    joint_pos = torch.as_tensor(arrays["robot_joint_pos"][frame_idx], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel = torch.as_tensor(arrays["robot_joint_vel"][frame_idx], device=device, dtype=torch.float32).unsqueeze(0)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    scene_keys = set(getattr(env.scene, "keys", lambda: [])())
    for cube in CUBE_NAMES:
        pose_key = f"{cube}_root_pose"
        if pose_key not in arrays or cube not in scene_keys:
            continue
        asset = env.scene[cube]
        pose = torch.as_tensor(arrays[pose_key][frame_idx], device=device, dtype=torch.float32).unsqueeze(0)
        vel_key = f"{cube}_root_velocity"
        vel = torch.zeros((1, 6), device=device, dtype=torch.float32)
        if vel_key in arrays:
            vel = torch.as_tensor(arrays[vel_key][frame_idx], device=device, dtype=torch.float32).unsqueeze(0)
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        if hasattr(asset, "write_root_velocity_to_sim"):
            asset.write_root_velocity_to_sim(vel, env_ids=env_ids)

    env.scene.write_data_to_sim()
    env.sim.forward()


def main() -> int:
    if not args_cli.enable_cameras:
        print("enable_cameras=false; state replay skipped")
        return 0

    dataset_path = Path(args_cli.dataset_file)
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)

    live_dir = Path(args_cli.live_frame_dir)
    frames_dir = live_dir / "frames"
    live_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    status_path = Path(args_cli.live_status_out) if args_cli.live_status_out else None
    preview_out = Path(args_cli.preview_video_out) if args_cli.preview_video_out else None

    with h5py.File(dataset_path, "r") as handle:
        demo_names = sorted(k for k in handle["data"].keys() if k.startswith("demo_"))
        if not demo_names:
            print("No demos in dataset")
            return 1
        select = args_cli.select_episodes or [0]

    env_name = args_cli.task.split(":")[-1]
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    configure_single_env_live_viewer(env_cfg, 0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    env.reset()
    capture_viewport_rgb(env, simulation_app, env_index=0, max_attempts=60)

    frame_count = [0]
    step_count = [0]
    every = max(1, int(args_cli.live_frame_every or 1))

    _write_status(
        status_path,
        {
            "visualPhase": "state_replay",
            "replayMode": "state_based",
            "liveFrameAvailable": False,
            "status": "running",
        },
    )

    with h5py.File(dataset_path, "r") as handle, torch.inference_mode():
        for ep_idx in select:
            if ep_idx < 0 or ep_idx >= len(demo_names):
                continue
            demo = handle["data"][demo_names[ep_idx]]
            arrays = _load_episode_arrays(demo)
            num_steps = int(arrays["num_steps"])
            env.reset()
            for frame_i in range(num_steps):
                _apply_state_frame(env, frame_i, arrays, env.device)
                step_count[0] += 1
                if step_count[0] % every == 0 or frame_i == num_steps - 1:
                    _capture_frame(env, live_dir, frames_dir, frame_count, simulation_app)

    env.close()

    latest_ok = (live_dir / "latest.jpg").is_file()
    preview_ok = False
    if preview_out and frame_count[0] > 0:
        preview_ok = _build_preview(frames_dir, preview_out)

    _write_status(
        status_path,
        {
            "visualPhase": "state_replay",
            "replayMode": "state_based",
            "videoConsistency": "hdf5_state_replay",
            "liveFrameAvailable": latest_ok,
            "previewVideoAvailable": preview_ok,
            "frameCount": frame_count[0],
            "status": "completed" if preview_ok else "failed",
        },
    )
    print(
        f"state replay frames={frame_count[0]} steps={step_count[0]} preview={preview_ok} path={preview_out}"
    )
    return 0 if preview_ok else 2


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as exc:
        import traceback

        print(f"ERROR: state replay failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        exit_code = 2
    finally:
        simulation_app.close()
    raise SystemExit(exit_code)
