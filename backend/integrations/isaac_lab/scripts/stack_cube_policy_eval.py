#!/usr/bin/env python3
# Copyright (c) 2026 EAI Platform
"""Isaac Stack Cube Robomimic BC policy rollout evaluation with platform artifacts."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from preview_video_utils import build_preview_from_frames as _build_preview  # noqa: F401 — kept for replay parity

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Evaluate Stack Cube Robomimic BC policy with platform artifacts.")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task id.")
parser.add_argument("--checkpoint", type=str, required=True, help="Robomimic checkpoint path.")
parser.add_argument("--horizon", type=int, default=400, help="Rollout step horizon.")
parser.add_argument("--num_rollouts", type=int, default=1, help="Number of rollouts.")
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument("--output_dir", type=str, required=True, help="Platform eval job root directory.")
parser.add_argument("--live_frame_every", type=int, default=3, help="Capture viewport frame every N steps.")
parser.add_argument(
    "--norm_factor_min", type=float, default=None, help="Optional action normalization min."
)
parser.add_argument(
    "--norm_factor_max", type=float, default=None, help="Optional action normalization max."
)
parser.add_argument("--disable_fabric", action="store_true", default=False)

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
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

STACK_CUBE_VIEWER_EYE = (1.35, 1.35, 1.05)
STACK_CUBE_VIEWER_LOOKAT = (0.5, 0.0, 0.35)


def configure_single_env_live_viewer(env_cfg, env_index: int = 0) -> None:
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = max(0, int(env_index))
    env_cfg.viewer.eye = STACK_CUBE_VIEWER_EYE
    env_cfg.viewer.lookat = STACK_CUBE_VIEWER_LOOKAT


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def capture_viewport_rgb(
    env,
    *,
    simulation_app=None,
    env_index: int = 0,
    max_attempts: int = 60,
) -> np.ndarray | None:
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
        except Exception as exc:
            logger.debug("viewport capture failed: %s", exc)
            continue
        if rgb is None:
            continue
        arr = np.asarray(rgb)
        if arr.ndim == 3 and arr.shape[2] >= 3 and _rgb_is_valid(arr):
            return arr[:, :, :3].copy()
    return None


class EpisodeFrameCapture:
    def __init__(self, frames_dir: Path, every: int) -> None:
        self.frames_dir = frames_dir
        self.every = max(1, int(every))
        self.count = 0
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def maybe_capture(self, env, step_index: int) -> None:
        if step_index % self.every != 0:
            return
        arr = capture_viewport_rgb(env, simulation_app=simulation_app, max_attempts=12)
        if arr is None:
            return
        try:
            import cv2
        except ImportError:
            return
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        self.count += 1
        cv2.imwrite(str(self.frames_dir / f"frame_{self.count:06d}.jpg"), bgr)


def rollout_episode(
    policy,
    env,
    success_term,
    horizon: int,
    device,
    capture: EpisodeFrameCapture | None,
) -> tuple[bool, float, int, str | None]:
    policy.start_episode()
    obs_dict, _ = env.reset()
    total_reward = 0.0
    steps = 0

    for step in range(horizon):
        obs = copy.deepcopy(obs_dict["policy"])
        for ob in obs:
            obs[ob] = torch.squeeze(obs[ob])

        actions = policy(obs)
        if args_cli.norm_factor_min is not None and args_cli.norm_factor_max is not None:
            actions = (
                (actions + 1) * (args_cli.norm_factor_max - args_cli.norm_factor_min)
            ) / 2 + args_cli.norm_factor_min

        actions_t = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[1])
        obs_dict, reward, terminated, truncated, _ = env.step(actions_t)
        reward_val = float(reward.sum()) if hasattr(reward, "sum") else float(reward)
        total_reward += reward_val
        steps += 1

        if capture is not None:
            capture.maybe_capture(env, step)

        if bool(success_term.func(env, **success_term.params)[0]):
            return True, total_reward, steps, None
        if terminated:
            return False, total_reward, steps, "terminated"
        if truncated:
            return False, total_reward, steps, "truncated"

    return False, total_reward, steps, "horizon_reached"


def main() -> int:
    output_root = Path(args_cli.output_dir).resolve()
    results_dir = output_root / "results"
    videos_dir = output_root / "videos"
    results_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args_cli.checkpoint).expanduser()
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        print(f"ERROR: checkpoint missing: {checkpoint}", file=sys.stderr)
        return 1

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None

    capture_enabled = bool(getattr(args_cli, "enable_cameras", False))
    if capture_enabled:
        configure_single_env_live_viewer(env_cfg, 0)
    render_mode = "rgb_array" if capture_enabled else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode).unwrapped

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    random.seed(args_cli.seed)
    env.seed(args_cli.seed)
    env.reset()
    if capture_enabled:
        warmed = capture_viewport_rgb(env, simulation_app=simulation_app, max_attempts=60)
        print(f"[INFO] viewport warmup ok={warmed is not None}")

    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=device)

    per_episode: list[dict] = []
    successes = 0
    rewards: list[float] = []
    lengths: list[int] = []

    for trial in range(int(args_cli.num_rollouts)):
        print(f"[INFO] Starting rollout {trial}")
        frames_dir = output_root / "artifacts" / f"episode_{trial:02d}_frames"
        capture = EpisodeFrameCapture(frames_dir, args_cli.live_frame_every) if capture_enabled else None

        success, reward, length, failure_reason = rollout_episode(
            policy,
            env,
            success_term,
            int(args_cli.horizon),
            device,
            capture,
        )
        if success:
            successes += 1
        rewards.append(reward)
        lengths.append(length)

        video_rel = f"videos/episode_{trial:02d}.mp4"
        episode_row = {
            "episodeIndex": trial,
            "success": success,
            "reward": reward,
            "episodeLength": length,
            "failureReason": failure_reason,
            "videoPath": video_rel if frames_dir.is_dir() and any(frames_dir.glob("frame_*.jpg")) else None,
            "seed": int(args_cli.seed) + trial,
        }
        per_episode.append(episode_row)
        print(f"[INFO] Rollout {trial}: success={success} reward={reward:.4f} length={length}", flush=True)

    episode_count = len(per_episode)
    success_rate = (successes / episode_count) if episode_count else 0.0
    mean_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0
    mean_length = int(round(sum(lengths) / len(lengths))) if lengths else 0
    failure_count = episode_count - successes

    aggregate = {
        "taskEnv": args_cli.task,
        "evaluationMode": "trained_model_evaluation",
        "backendType": "isaac_robomimic_bc",
        "episodeCount": episode_count,
        "successRate": success_rate,
        "meanReward": mean_reward,
        "meanEpisodeLength": mean_length,
        "failureCount": failure_count,
        "horizon": int(args_cli.horizon),
        "seed": int(args_cli.seed),
        "checkpointPath": str(checkpoint),
    }

    _write_json(results_dir / "aggregate_result.json", aggregate)
    _write_json(results_dir / "per_episode_results.json", {"episodes": per_episode})

    print(f"\nSuccessful rollouts: {successes}, out of {episode_count} rollouts", flush=True)
    print(f"Success rate: {success_rate}", flush=True)
    print(f"Trial Results: {[row['success'] for row in per_episode]}\n", flush=True)

    env.close()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        simulation_app.close()
    sys.exit(code)
