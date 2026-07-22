#!/usr/bin/env python3
# Copyright (c) 2026 EAI Platform
"""Stack Cube Mimic generate_dataset 包装：在 env_loop 中写入 live/latest.jpg。"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import logging
import random
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from preview_video_utils import build_preview_from_frames

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Generate Stack Cube mimic dataset with live preview frames.")
parser.add_argument("--task", type=str, required=True, help="Mimic task id.")
parser.add_argument("--input_file", type=str, required=True, help="Annotated HDF5 input.")
parser.add_argument("--output_file", type=str, required=True, help="Output dataset HDF5.")
parser.add_argument("--generation_num_trials", type=int, default=1)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--live_frame_dir", type=str, default="", help="Job live/ directory.")
parser.add_argument("--live_frame_every", type=int, default=5, help="Capture every N sim steps.")
parser.add_argument("--visual_env_index", type=int, default=0, help="Env index for live preview camera.")
parser.add_argument("--live_status_out", type=str, default="", help="Optional live_status.json path.")
parser.add_argument("--preview_video_out", type=str, default="", help="Optional preview.mp4 output path.")
parser.add_argument(
    "--pause_subtask",
    action="store_true",
    help="Pause after every subtask during generation for debugging.",
)

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

from isaaclab.envs import ManagerBasedRLMimicEnv

import isaaclab_mimic.envs  # noqa: F401
import isaaclab_mimic.datagen.generation as gen_module
from isaaclab_mimic.datagen.generation import setup_async_generation, setup_env_config
from isaaclab_mimic.datagen.utils import get_env_name_from_dataset, setup_output_paths

import isaaclab_tasks  # noqa: F401

# Stack Cube 单环境预览相机（相对 env origin 的 eye / lookat）
STACK_CUBE_VIEWER_EYE = (1.35, 1.35, 1.05)
STACK_CUBE_VIEWER_LOOKAT = (0.5, 0.0, 0.35)


def configure_single_env_live_viewer(env_cfg, env_index: int = 0) -> None:
    """Live preview 聚焦单个 parallel env，避免 world 视角看到多环境网格。"""
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


def capture_viewport_rgb(
    env,
    simulation_app,
    *,
    env_index: int = 0,
    max_attempts: int = 60,
) -> np.ndarray | None:
    """Capture viewer rgb_array; Isaac returns zeros until sim.render() + warmup."""
    if not hasattr(env, "render"):
        return None
    for attempt in range(max_attempts):
        focus_live_viewport(env, env_index)
        if hasattr(env, "sim"):
            env.sim.render()
        if simulation_app is not None:
            simulation_app.update()
        try:
            # recompute=True skips sim.render() in Isaac Lab; always render explicitly first.
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


class LiveFrameWriter:
    def __init__(
        self,
        live_dir: Path,
        *,
        every: int = 5,
        status_path: Path | None = None,
        simulation_app=None,
        visual_env_index: int = 0,
        parallel_num_envs: int = 1,
    ):
        self.live_dir = live_dir
        self.frames_dir = live_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, every)
        self.status_path = status_path
        self.simulation_app = simulation_app
        self.visual_env_index = max(0, int(visual_env_index))
        self.parallel_num_envs = max(1, int(parallel_num_envs))
        self.step_count = 0
        self.frame_count = 0
        self.skipped_black = 0

    def _write_status(self, **extra: object) -> None:
        if not self.status_path:
            return
        latest = self.live_dir / "latest.jpg"
        visual_mode = "single_env" if self.parallel_num_envs <= 1 else "single_env"
        payload = {
            "liveFrameAvailable": False,
            "liveFrameBlack": False,
            "frameCount": self.frame_count,
            "stepCount": self.step_count,
            "skippedBlackFrames": self.skipped_black,
            "visualPhase": "live_generate",
            "visualNumEnvs": 1,
            "parallelNumEnvs": self.parallel_num_envs,
            "visualMode": visual_mode,
            "visualEnvIndex": self.visual_env_index,
            **extra,
        }
        if latest.is_file():
            try:
                from PIL import Image

                arr = np.array(Image.open(latest).convert("RGB"))
                valid = _rgb_is_valid(arr)
                payload["liveFrameAvailable"] = valid
                payload["liveFrameBlack"] = not valid
            except OSError:
                payload["liveFrameBlack"] = True
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_rgb(self, env, arr: np.ndarray) -> bool:
        if arr.ndim != 3 or arr.shape[2] < 3 or not _rgb_is_valid(arr):
            self.skipped_black += 1
            return False
        try:
            import cv2
        except ImportError:
            logger.warning("opencv-python unavailable; cannot write live frames")
            return False
        bgr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2BGR)
        self.frame_count += 1
        frame_path = self.frames_dir / f"frame_{self.frame_count:06d}.jpg"
        latest_path = self.live_dir / "latest.jpg"
        cv2.imwrite(str(frame_path), bgr)
        cv2.imwrite(str(latest_path), bgr)
        self._write_status()
        return True

    def maybe_capture(self, env: ManagerBasedRLMimicEnv) -> None:
        self.step_count += 1
        if self.step_count % self.every != 0:
            return
        arr = capture_viewport_rgb(
            env,
            self.simulation_app,
            env_index=self.visual_env_index,
            max_attempts=8,
        )
        if arr is None:
            self.skipped_black += 1
            return
        self.write_rgb(env, arr)

    def warmup(self, env: ManagerBasedRLMimicEnv) -> bool:
        arr = capture_viewport_rgb(
            env,
            self.simulation_app,
            env_index=self.visual_env_index,
            max_attempts=60,
        )
        if arr is None:
            logger.warning("live frame warmup failed: viewport still black after 60 attempts")
            return False
        return self.write_rgb(env, arr)


def env_loop_with_live(
    env: ManagerBasedRLMimicEnv,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    live_writer: LiveFrameWriter | None,
):
    """env_loop with optional live rgb capture after each step."""
    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0

    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while True:
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape)
            for _ in range(env.num_envs):
                env_id, action = asyncio_event_loop.run_until_complete(env_action_queue.get())
                actions[env_id] = action

            env.step(actions)
            if live_writer is not None:
                live_writer.maybe_capture(env)

            for _ in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != gen_module.num_attempts:
                prev_num_attempts = gen_module.num_attempts
                generated_success_rate = (
                    100 * gen_module.num_success / gen_module.num_attempts
                    if gen_module.num_attempts > 0
                    else 0.0
                )
                print("")
                print("*" * 50, "\033[K")
                print(
                    f"{gen_module.num_success}/{gen_module.num_attempts} ({generated_success_rate:.1f}%) successful demos generated by"
                    " mimic\033[K"
                )
                print("*" * 50, "\033[K")

                generation_guarantee = env.cfg.datagen_config.generation_guarantee
                generation_num_trials = env.cfg.datagen_config.generation_num_trials
                check_val = gen_module.num_success if generation_guarantee else gen_module.num_attempts
                if check_val >= generation_num_trials:
                    print(f"Reached {generation_num_trials} successes/attempts. Exiting.")
                    break

            if env.sim.is_stopped():
                break

    env.close()


def main() -> int:
    live_dir = Path(args_cli.live_frame_dir).expanduser() if args_cli.live_frame_dir else None
    status_path = Path(args_cli.live_status_out).expanduser() if args_cli.live_status_out else None
    preview_out = Path(args_cli.preview_video_out).expanduser() if args_cli.preview_video_out else None
    live_writer: LiveFrameWriter | None = None

    if live_dir and args_cli.enable_cameras:
        live_writer = LiveFrameWriter(
            live_dir,
            every=int(args_cli.live_frame_every or 5),
            status_path=status_path,
            simulation_app=simulation_app,
            visual_env_index=int(args_cli.visual_env_index or 0),
            parallel_num_envs=max(1, int(args_cli.num_envs)),
        )
        live_writer._write_status(status="starting")
    elif live_dir and not args_cli.enable_cameras:
        logger.info("enable_cameras=false; skipping live frame capture during generate")

    output_dir, output_file_name = setup_output_paths(args_cli.output_file)
    task_name = args_cli.task.split(":")[-1] if args_cli.task else None
    env_name = task_name or get_env_name_from_dataset(args_cli.input_file)

    env_cfg, success_term = setup_env_config(
        env_name=env_name,
        output_dir=output_dir,
        output_file_name=output_file_name,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        generation_num_trials=args_cli.generation_num_trials,
    )
    visual_env_index = max(0, min(int(args_cli.visual_env_index or 0), max(0, args_cli.num_envs - 1)))
    if live_writer is not None:
        configure_single_env_live_viewer(env_cfg, visual_env_index)
        live_writer.visual_env_index = visual_env_index
        live_writer.parallel_num_envs = max(1, int(args_cli.num_envs))
        print(
            f"live preview camera: env_index={visual_env_index} parallel_num_envs={args_cli.num_envs} "
            f"origin_type=env eye={STACK_CUBE_VIEWER_EYE}"
        )

    render_mode = "rgb_array" if args_cli.enable_cameras and live_writer is not None else None
    env = gym.make(env_name, cfg=env_cfg, render_mode=render_mode).unwrapped

    if not isinstance(env, ManagerBasedRLMimicEnv):
        raise ValueError("The environment should be derived from ManagerBasedRLMimicEnv")

    if "action_noise_dict" not in inspect.signature(env.target_eef_pose_to_action).parameters:
        logger.warning("Deprecated mimic API signature detected on %s", env_name)

    random.seed(env.cfg.datagen_config.seed)
    np.random.seed(env.cfg.datagen_config.seed)
    torch.manual_seed(env.cfg.datagen_config.seed)
    env.reset()
    if live_writer is not None:
        warmed = live_writer.warmup(env)
        print(f"live frame warmup ok={warmed} skipped_black={live_writer.skipped_black}")

    async_components = setup_async_generation(
        env=env,
        num_envs=args_cli.num_envs,
        input_file=args_cli.input_file,
        success_term=success_term,
        pause_subtask=args_cli.pause_subtask,
        motion_planners=None,
    )

    loop_fn = (
        lambda *a, **k: env_loop_with_live(*a, **k, live_writer=live_writer)
        if live_writer is not None
        else gen_module.env_loop
    )

    try:
        data_gen_tasks = asyncio.ensure_future(asyncio.gather(*async_components["tasks"]))
        loop_fn(
            env,
            async_components["reset_queue"],
            async_components["action_queue"],
            async_components["info_pool"],
            async_components["event_loop"],
        )
    except asyncio.CancelledError:
        print("Tasks were cancelled.")
    finally:
        data_gen_tasks.cancel()
        try:
            async_components["event_loop"].run_until_complete(data_gen_tasks)
        except asyncio.CancelledError:
            print("Remaining async tasks cancelled.")
        except Exception as exc:
            print(f"Error cancelling async tasks: {exc}")

    if live_writer is not None:
        live_writer._write_status(status="generate_complete")
    if preview_out and live_writer is not None and live_writer.frames_dir.is_dir():
        ok = build_preview_from_frames(live_writer.frames_dir, preview_out)
        print(f"preview.mp4 built={ok} path={preview_out}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
    finally:
        simulation_app.close()
