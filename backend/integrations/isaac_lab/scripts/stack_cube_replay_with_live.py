#!/usr/bin/env python3
# Copyright (c) 2026 EAI Platform
"""Replay Stack Cube HDF5 demo with live frame output and preview.mp4 synthesis."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from preview_video_utils import build_preview_from_frames as _build_preview

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Replay Stack Cube dataset with live preview frames.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--select_episodes", type=int, nargs="*", default=[0])
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--live_frame_dir", type=str, required=True)
parser.add_argument("--live_frame_every", type=int, default=3)
parser.add_argument("--live_status_out", type=str, default="")
parser.add_argument("--preview_video_out", type=str, default="")

try:
    from isaaclab.app import AppLauncher
except ImportError as exc:
    print(f"ERROR: isaaclab unavailable: {exc}", file=sys.stderr)
    sys.exit(1)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

STACK_CUBE_VIEWER_EYE = (1.35, 1.35, 1.05)
STACK_CUBE_VIEWER_LOOKAT = (0.5, 0.0, 0.35)


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


def capture_viewport_rgb(env, simulation_app, *, env_index: int = 0, max_attempts: int = 60) -> np.ndarray | None:
    if not hasattr(env, "render"):
        return None
    for attempt in range(max_attempts):
        focus_live_viewport(env, env_index)
        if hasattr(env, "sim"):
            env.sim.render()
        if simulation_app is not None:
            simulation_app.update()
        try:
            rgb = env.render(recompute=False)
        except Exception as exc:
            logger.debug("viewport capture attempt %s failed: %s", attempt, exc)
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


def main() -> int:
    if not args_cli.enable_cameras:
        print("enable_cameras=false; replay preview skipped")
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

    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    env_name = args_cli.task.split(":")[-1]
    episode_names = list(handler.get_episode_names())
    if not episode_names:
        print("No episodes in dataset")
        return 1

    select = args_cli.select_episodes or [0]
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.recorders = {}
    env_cfg.terminations = {}
    configure_single_env_live_viewer(env_cfg, 0)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array").unwrapped
    env.reset()
    capture_viewport_rgb(env, simulation_app, env_index=0, max_attempts=60)

    frame_count = [0]
    step_count = [0]
    every = max(1, int(args_cli.live_frame_every or 3))

    _write_status(
        status_path,
        {
            "visualPhase": "replay_preview",
            "liveFrameAvailable": False,
            "status": "running",
            "visualNumEnvs": 1,
            "parallelNumEnvs": 1,
            "visualMode": "single_env",
            "visualEnvIndex": 0,
        },
    )

    with torch.inference_mode():
        for ep_idx in select:
            if ep_idx < 0 or ep_idx >= len(episode_names):
                continue
            ep = handler.load_episode(episode_names[ep_idx], env.device)
            env.reset()
            actions = ep.data["actions"]
            for step_i in range(actions.shape[0]):
                env.step(actions[step_i : step_i + 1])
                step_count[0] += 1
                if step_count[0] % every == 0:
                    _capture_frame(env, live_dir, frames_dir, frame_count, simulation_app)

    env.close()
    handler.close()

    latest_ok = (live_dir / "latest.jpg").is_file()
    if latest_ok:
        try:
            from PIL import Image

            arr = np.array(Image.open(live_dir / "latest.jpg").convert("RGB"))
            latest_ok = _rgb_is_valid(arr)
        except OSError:
            latest_ok = False
    preview_ok = False
    if preview_out and latest_ok:
        preview_ok = _build_preview(frames_dir, preview_out)

    _write_status(
        status_path,
        {
            "visualPhase": "completed" if latest_ok else "replay_preview",
            "liveFrameAvailable": latest_ok,
            "previewVideoAvailable": preview_ok,
            "frameCount": frame_count[0],
            "status": "completed" if latest_ok else "failed",
        },
    )
    print(f"replay preview frames={frame_count[0]} latest={latest_ok} preview={preview_ok}")
    return 0 if latest_ok else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
