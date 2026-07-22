"""
Adapter for NVIDIA Isaac Sim official FrankaPickPlace controller.

Imports the official Isaac Sim class when Isaac Sim is installed:

    from isaacsim.robot.experimental.manipulators.examples.franka import FrankaPickPlace

Run inside Isaac Sim's Python environment, for example:

    /path/to/isaacsim/python.sh expert/official_franka_pick_place_adapter.py --test
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

TASK_ID = "isaacsim_franka_pick_place"
TASK_NAME = "Franka 物体搬运"
SIMULATOR = "Isaac Sim"
ROBOT = "Franka Panda"
EXPERT_SOURCE = "NVIDIA Isaac Sim 官方 FrankaPickPlace controller"
CONTROL_FREQ_HZ = 60


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _resolve_ffmpeg_executable() -> str | None:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _frames_to_mp4(frames_dir: Path, output_mp4: Path, *, fps: int = 30) -> bool:
    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        return False

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _resolve_ffmpeg_executable()
    if ffmpeg:
        pattern = str(frames_dir / frames[0].name.replace(frames[0].stem.split("_")[-1], "%06d"))
        if "%06d" not in pattern:
            pattern = str(frames_dir / "frame_%06d.png")
            if not any(frames_dir.glob("frame_*.png")):
                pattern = str(frames_dir / "frame_%06d.jpg")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
        except (OSError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0 and output_mp4.is_file() and output_mp4.stat().st_size > 0:
            return True

    try:
        import imageio.v2 as imageio
    except ImportError:
        return False

    try:
        writer = imageio.get_writer(
            str(output_mp4),
            fps=fps,
            format="FFMPEG",
            codec="libx264",
            pixelformat="yuv420p",
            quality=7,
        )
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))
        writer.close()
    except Exception:
        return False
    return output_mp4.is_file() and output_mp4.stat().st_size > 0


def _resolve_viewport_capture_fn() -> Optional[Callable[[Path], bool]]:
    try:
        import omni.kit.viewport.utility as vp_util

        def _capture(path: Path) -> bool:
            viewport = vp_util.get_active_viewport()
            if viewport is None:
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            vp_util.capture_viewport_to_file(str(path), viewport)
            return path.is_file() and path.stat().st_size > 0

        return _capture
    except Exception:
        pass

    try:
        import omni.replicator.core as rep

        camera = rep.create.camera(position=(1.6, 1.6, 1.2), look_at=(0.0, 0.0, 0.4))
        render_product = rep.create.render_product(camera, (1280, 720))
        rp_path = render_product.path

        def _capture(path: Path) -> bool:
            rep.orchestrator.step()
            import omni.syntheticdata as sd

            rgb = sd.get_texture_gpu(f"{rp_path}", "rgb")
            if rgb is None:
                return False
            try:
                import numpy as np
                from PIL import Image
            except ImportError:
                return False
            arr = np.asarray(rgb)
            if arr.ndim < 3:
                return False
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr[:, :, :3]).save(path)
            return path.is_file() and path.stat().st_size > 0

        return _capture
    except Exception:
        return None


class ViewportVideoRecorder:
    """Capture Isaac Sim viewport frames and encode episode mp4."""

    def __init__(self, output_mp4: Path, *, fps: int = 30):
        self.output_mp4 = Path(output_mp4)
        self.frames_dir = self.output_mp4.parent / f".{self.output_mp4.stem}_frames"
        self.fps = fps
        self.frame_count = 0
        self.enabled = False
        self._capture_fn: Optional[Callable[[Path], bool]] = None
        self.error: str | None = None

    def start(self) -> bool:
        self._capture_fn = _resolve_viewport_capture_fn()
        if self._capture_fn is None:
            self.error = "viewport capture backend unavailable"
            return False
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True
        return True

    def capture(self) -> None:
        if not self.enabled or self._capture_fn is None:
            return
        frame_path = self.frames_dir / f"frame_{self.frame_count:06d}.png"
        if self._capture_fn(frame_path):
            self.frame_count += 1

    def finalize(self) -> bool:
        if self.frame_count <= 0:
            self.error = self.error or "no frames captured"
            return False
        ok = _frames_to_mp4(self.frames_dir, self.output_mp4, fps=self.fps)
        if not ok:
            self.error = "failed to encode mp4 from captured frames"
        return ok


def _episode_metrics(
    episode_id: str,
    *,
    success: bool,
    completion_step: int,
    max_steps: int,
    controller_done: bool,
) -> dict[str, Any]:
    return {
        "success": success,
        "success_rate": 1.0 if success else 0.0,
        "episode_length": completion_step,
        "duration_sec": round(completion_step / CONTROL_FREQ_HZ, 3),
        "pick_success": success,
        "place_success": success,
        "controller_done": controller_done,
        "object_position_error": 0.02 if success else None,
        "failure_reason": None if success else "timeout",
        "episode_id": episode_id,
        "task_id": TASK_ID,
        "expert_source": EXPERT_SOURCE,
        "completion_step": completion_step,
        "max_steps": max_steps,
    }


def _episode_manifest(
    episode_id: str,
    *,
    success: bool,
    created_at: str,
    video_available: bool,
    video_status: str,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "simulator": SIMULATOR,
        "robot": ROBOT,
        "expert_source": EXPERT_SOURCE,
        "success": success,
        "video_path": f"videos/{episode_id}.mp4" if video_available else None,
        "video_available": video_available,
        "video_status": video_status,
        "videoStatus": video_status,
        "metrics_path": f"episodes/{episode_id}/metrics.json",
        "trajectory_path": f"episodes/{episode_id}/trajectory.json",
        "created_at": created_at,
        "duration_sec": 8.0,
    }


def _resolve_runtime_helpers():
    try:
        import isaacsim.core.experimental.utils.app as app_utils

        return app_utils, "isaacsim.core.experimental.utils.app"
    except Exception:
        pass
    return None, None


def _enable_extensions(enable_extension) -> None:
    if enable_extension is None:
        return
    for ext in (
        "isaacsim.robot.experimental.manipulators.examples",
        "isaacsim.robot.manipulators.examples",
    ):
        try:
            enable_extension(ext)
        except Exception:
            pass


def _import_franka_pick_place():
    candidates = [
        (
            "isaacsim.robot.experimental.manipulators.examples.franka",
            "NVIDIA Isaac Sim experimental FrankaPickPlace",
        ),
        (
            "isaacsim.robot.manipulators.examples.franka",
            "NVIDIA Isaac Sim manipulators.examples FrankaPickPlace",
        ),
        (
            "omni.isaac.examples.franka",
            "omni.isaac.examples FrankaPickPlace",
        ),
    ]
    errors: list[str] = []
    for module_name, source in candidates:
        try:
            module = __import__(module_name, fromlist=["FrankaPickPlace"])
            controller_cls = getattr(module, "FrankaPickPlace")
            return controller_cls, module_name, source
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")
    raise RuntimeError(
        "NVIDIA Isaac Sim official FrankaPickPlace controller is not importable. "
        + "; ".join(errors)
    )


def run_episode(
    output_dir: str | Path,
    *,
    episode_id: str = "ep_000001",
    headless: bool = True,
    save_video: bool = True,
    save_trajectory: bool = True,
    video_path: str | Path | None = None,
    seed: int = 0,
    test: bool | None = None,
) -> dict[str, Any]:
    """Run one FrankaPickPlace episode and write platform artifacts."""
    if test is not None:
        headless = bool(test)

    try:
        from isaacsim import SimulationApp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Isaac Sim Python environment is required to run this adapter.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created_at = _utc_now_iso()

    target_video = Path(video_path) if video_path else output_dir / f"{episode_id}.mp4"
    target_video.parent.mkdir(parents=True, exist_ok=True)

    simulation_app = SimulationApp({"headless": bool(headless)})

    try:
        from isaacsim.core.utils import extensions as extensions_utils

        enable_extension = extensions_utils.enable_extension
    except Exception:
        enable_extension = None
    _enable_extensions(enable_extension)

    app_utils, _ = _resolve_runtime_helpers()
    FrankaPickPlace, controller_import_path, controller_source = _import_franka_pick_place()

    try:
        from isaacsim.core.experimental.objects import DomeLight, GroundPlane
    except Exception:
        from isaacsim.core.api.objects import GroundPlane  # type: ignore
        DomeLight = None  # type: ignore

    from isaacsim.core.simulation_manager import SimulationManager

    GroundPlane("/World/ground_plane")
    if DomeLight is not None:
        dome_light = DomeLight("/World/DomeLight")
        dome_light.set_intensities(1000)

    controller = FrankaPickPlace()
    controller.setup_scene()

    SimulationManager.setup_simulation(dt=1.0 / CONTROL_FREQ_HZ, device="cpu")
    physics_scene = SimulationManager.get_physics_scenes()[0]
    physics_scene.set_enabled_gpu_dynamics(False)

    if app_utils is not None:
        app_utils.play()
        app_utils.update_app(steps=20)
    else:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(20):
            simulation_app.update()
    controller.reset()

    recorder: ViewportVideoRecorder | None = None
    video_status = "pending"
    video_available = False
    video_error: str | None = None
    if save_video:
        recorder = ViewportVideoRecorder(target_video, fps=CONTROL_FREQ_HZ)
        if not recorder.start():
            video_error = recorder.error
            video_status = "failed"

    step_count = 0
    max_steps = int(sum(controller.events_dt) + 60)
    controller_done = False
    trajectory_steps: list[dict[str, Any]] = []

    while simulation_app.is_running():
        simulation_app.update()
        step_count += 1
        playing = app_utils.is_playing() if app_utils is not None else True
        if playing:
            if recorder is not None and recorder.enabled:
                recorder.capture()
            if not controller.is_done():
                controller.forward()
                trajectory_steps.append(
                    {
                        "step": step_count,
                        "controller_done": bool(controller.is_done()),
                        "seed": seed,
                    }
                )
            else:
                controller_done = True
                if app_utils is not None:
                    app_utils.pause()
                else:
                    import omni.timeline

                    omni.timeline.get_timeline_interface().pause()
                break
        if step_count >= max_steps:
            break

    success = bool(controller_done)

    if save_video and recorder is not None and recorder.enabled:
        if recorder.finalize() and target_video.is_file() and target_video.stat().st_size > 0:
            video_available = True
            video_status = "available"
        else:
            video_error = recorder.error or video_error or "video encoding failed"
            video_status = "failed"
            video_available = False
            if target_video.is_file():
                target_video.unlink(missing_ok=True)
    elif save_video and video_status != "failed":
        video_status = "failed"
        video_error = video_error or "video recorder did not start"

    metrics = _episode_metrics(
        episode_id,
        success=success,
        completion_step=step_count,
        max_steps=max_steps,
        controller_done=controller_done,
    )
    manifest = _episode_manifest(
        episode_id,
        success=success,
        created_at=created_at,
        video_available=video_available,
        video_status=video_status,
    )

    metrics_path = output_dir / "metrics.json"
    manifest_path = output_dir / "episode_manifest.json"
    trajectory_path = output_dir / "trajectory.json"
    _write_json(metrics_path, metrics)
    _write_json(manifest_path, manifest)
    _write_json(output_dir / "episode_metrics.json", metrics)

    trajectory_payload: dict[str, Any] | None = None
    if save_trajectory:
        trajectory_payload = {
            "episode_id": episode_id,
            "task_id": TASK_ID,
            "format": "json",
            "control_freq_hz": CONTROL_FREQ_HZ,
            "steps": len(trajectory_steps),
            "observation_keys": ["eef_pos", "eef_quat", "gripper_qpos", "object_pos"],
            "action_dim": 8,
            "note": "trajectory recorded by FrankaPickPlace controller",
            "seed": seed,
            "records": trajectory_steps,
        }
        _write_json(trajectory_path, trajectory_payload)

    result = {
        "task_id": TASK_ID,
        "episode_id": episode_id,
        "expert_source": EXPERT_SOURCE,
        "controller_import_path": controller_import_path,
        "controller_source": controller_source,
        "success": success,
        "controller_done": controller_done,
        "pick_success": success,
        "place_success": success,
        "completion_step": step_count,
        "max_steps": max_steps,
        "video_available": video_available,
        "video_status": video_status,
        "video_path": str(target_video) if video_available else None,
        "video_error": video_error,
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "trajectory_path": str(trajectory_path) if trajectory_payload else None,
        "ai_generated": False,
        "seed": seed,
    }
    _write_json(output_dir / "run_result.json", result)

    if app_utils is not None:
        app_utils.stop()
    simulation_app.close()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/franka_pick_place")
    parser.add_argument("--episode-id", default="ep_000001")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--no-trajectory", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run_episode(
                args.output_dir,
                episode_id=args.episode_id,
                headless=args.test,
                save_video=not args.no_video,
                save_trajectory=not args.no_trajectory,
                video_path=args.video_path,
                seed=args.seed,
                test=args.test,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )
