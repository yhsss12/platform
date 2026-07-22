from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths
from app.services.dataset_naming import persist_manifest_display_fields
from app.services.task_config_metadata import build_job_resource_metadata
from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)
from app.services.evaluation.display_name import (
    build_evaluation_display_name,
    resolve_evaluation_type_label,
    resolve_task_display_name,
)
from app.services.cable_threading_eval_params import DEFAULT_CABLE_EVAL_DISPLAY_CAMERA

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
WORKING_DIR = PROJECT_ROOT / "integrations" / "CableThreadingMVP"
OUTPUT_ROOT = platform_paths.runs_root / "cable_threading"
PYTHON_BIN = Path("/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python")
RUN_PY = WORKING_DIR / "run.py"

ALLOWED_ROBOTS = frozenset({"Panda", "UR5e"})
ALLOWED_CABLE_MODELS = frozenset({"composite_cable", "composite_soft", "rmb", "flex"})
ALLOWED_DIFFICULTIES = frozenset({"easy", "medium", "hard"})
ALLOWED_POLICIES = frozenset({"scripted", "random", "robomimic", "diffusion_policy", "act", "pi0"})
ALLOWED_OUTPUT_FORMATS = frozenset({"npz", "hdf5", "lerobot"})
TRAINED_MODEL_POLICIES = frozenset({"robomimic", "diffusion_policy", "act", "pi0"})

TIMEOUT_EXPERT = 1200
TIMEOUT_EVAL_MIN = 600
TIMEOUT_EVAL_BASE = 120
TIMEOUT_EVAL_SECONDS_PER_STEP = 1.5
TIMEOUT_VIDEO = 300

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
DISPLAY_ASPECT_RATIO = "16:9"


def _resolve_eval_device(device: str = "", *, train_device: Optional[str] = None) -> str:
    """Resolve eval device; default to cuda (same as training_service._resolve_device)."""
    value = (train_device or device or "cuda").strip().lower()
    if value in {"cuda_if_available", "auto", "l20", ""}:
        return "cuda"
    return value if value in {"cpu", "cuda"} else "cuda"


def _compute_eval_timeout(*, episodes: int, horizon: int) -> int:
    """Dynamic eval timeout from episodes × horizon (steps), with a minimum floor."""
    steps = max(int(episodes), 1) * max(int(horizon), 1)
    return int(max(TIMEOUT_EVAL_MIN, TIMEOUT_EVAL_BASE + steps * TIMEOUT_EVAL_SECONDS_PER_STEP))


STDOUT_TAIL_LINES = 30

_CT_JOB_SUFFIX = r"(?:\d{8}_\d{6}_[a-f0-9]{4}|[a-z0-9_]+)"
JOB_ID_PATTERN = re.compile(rf"^ct_(gen|eval|vid)_{_CT_JOB_SUFFIX}$")
JOB_VIDEO_RELATIVE = Path("videos") / "demo.mp4"
JOB_GENERATE_VIDEO_RELATIVE = Path("videos") / "generate.mp4"
JOB_EVAL_VIDEO_RELATIVE = Path("videos") / "eval.mp4"
JOB_EVAL_BROWSER_VIDEO_RELATIVE = Path("videos") / "eval.browser.mp4"
JOB_LIVE_TIMELINE = Path("live") / "generate_timeline.json"
JOB_EVAL_TIMELINE = Path("live") / "eval_timeline.json"
JOB_LIVE_DIR = Path("live")
JOB_LIVE_FRAME = JOB_LIVE_DIR / "latest.jpg"
JOB_LIVE_STATUS = JOB_LIVE_DIR / "status.json"


@dataclass
class RunResult:
    returncode: int
    stdout: str
    command: list[str]
    log_path: Path


@dataclass
class AsyncJobRecord:
    job_id: str
    job_dir: Path
    command: list[str]
    started_at: str
    process: Optional[Any] = None


ASYNC_JOBS: dict[str, AsyncJobRecord] = {}


def _validate_robot(robot: str) -> str:
    if robot not in ALLOWED_ROBOTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"robot must be one of {sorted(ALLOWED_ROBOTS)}",
        )
    return robot


def _validate_cable_model(cable_model: str) -> str:
    if cable_model not in ALLOWED_CABLE_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"cableModel must be one of {sorted(ALLOWED_CABLE_MODELS)}",
        )
    return cable_model


def _validate_difficulty(difficulty: str) -> str:
    if difficulty not in ALLOWED_DIFFICULTIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"difficulty must be one of {sorted(ALLOWED_DIFFICULTIES)}",
        )
    return difficulty


def _validate_output_format(output_format: str) -> str:
    normalized = str(output_format or "hdf5").strip().lower()
    if normalized not in ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"outputFormat must be one of {sorted(ALLOWED_OUTPUT_FORMATS)}",
        )
    return normalized


def _lerobot_dataset_dir(job_root: Path) -> Path:
    return job_root / "datasets" / "lerobot_dataset"


def _lerobot_sidecar_path(job_root: Path, filename: str) -> Path:
    return _lerobot_dataset_dir(job_root) / filename


def _validate_policy(policy: str) -> str:
    if policy not in ALLOWED_POLICIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"policy must be one of {sorted(ALLOWED_POLICIES)}",
        )
    return policy


def make_job_id(prefix: str) -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{suffix}"


def _job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


def _assert_job_child(path: Path, job_root: Path, *, detail: str) -> Path:
    resolved = path.resolve()
    if not is_path_within(resolved, job_root):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=detail)
    return resolved


def validate_job_id(job_id: str) -> str:
    candidate = (job_id or "").strip()
    if ".." in candidate or "/" in candidate or "\\" in candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    if not JOB_ID_PATTERN.match(candidate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    return candidate


def resolve_job_video_path(job_id: str, episode: Optional[int] = None) -> Optional[Path]:
    """Resolve eval.mp4, generate.mp4, demo.mp4, or per-episode mp4 for a validated cable_threading job."""
    from app.services.imported_eval_bridge import resolve_imported_eval_video_path
    from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

    candidate = (job_id or "").strip()
    if is_imported_workspace_eval_job_id(candidate):
        return resolve_imported_eval_video_path(candidate, episode=episode)

    validated = validate_job_id(job_id)
    job_root = (_job_dir(validated)).resolve()
    if episode is not None:
        videos_dir = _assert_job_child(job_root / "videos", job_root, detail="Invalid video path")
        episode_candidates = [
            videos_dir / f"episode_{episode + 1:03d}.mp4",
            videos_dir / f"episode_{episode:03d}.mp4",
            videos_dir / f"episode_{episode}.mp4",
            videos_dir / f"episode-{episode}.mp4",
            videos_dir / f"eval_episode_{episode:03d}.mp4",
            videos_dir / f"eval_episode_{episode}.mp4",
        ]
        for candidate in episode_candidates:
            resolved = _assert_job_child(candidate, job_root, detail="Invalid video path")
            if resolved.is_file():
                browser = resolved.with_name(f"{resolved.stem}.browser{resolved.suffix}")
                if browser.is_file() and browser.stat().st_size > 0:
                    return browser
                return resolved
        return None
    eval_path = (job_root / JOB_EVAL_VIDEO_RELATIVE).resolve()
    eval_browser_path = (job_root / JOB_EVAL_BROWSER_VIDEO_RELATIVE).resolve()
    generate_path = (job_root / JOB_GENERATE_VIDEO_RELATIVE).resolve()
    demo_path = (job_root / JOB_VIDEO_RELATIVE).resolve()
    for candidate in (eval_browser_path, eval_path, generate_path, demo_path):
        _assert_job_child(candidate, job_root, detail="Invalid video path")
    if eval_browser_path.is_file():
        return eval_browser_path
    if eval_path.is_file():
        return eval_path
    if generate_path.is_file():
        return generate_path
    if demo_path.is_file():
        return demo_path
    return None


def _prepare_job_dirs(job_id: str, *, include_datasets: bool, include_videos: bool) -> Path:
    root = OUTPUT_ROOT / "jobs" / job_id
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    if include_datasets:
        (root / "datasets").mkdir(parents=True, exist_ok=True)
    if include_videos:
        (root / "videos").mkdir(parents=True, exist_ok=True)
    return root


def _build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MUJOCO_GL"] = "egl"
    return env


def _format_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def _run_subprocess(cmd: list[str], *, log_path: Path, timeout: int) -> RunResult:
    if not PYTHON_BIN.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Python interpreter not found: {PYTHON_BIN}",
        )
    if not RUN_PY.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"run.py not found: {RUN_PY}",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = _build_env()

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(WORKING_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + ("\n" if exc.stdout and exc.stderr else "") + (exc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Command timed out after {timeout}s",
        ) from exc

    combined_output = ""
    if proc.stdout:
        combined_output += proc.stdout
    if proc.stderr:
        if combined_output and not combined_output.endswith("\n"):
            combined_output += "\n"
        combined_output += proc.stderr
    log_path.write_text(combined_output, encoding="utf-8")

    return RunResult(
        returncode=proc.returncode,
        stdout=combined_output,
        command=cmd,
        log_path=log_path,
    )


def _parse_stdout_value(stdout: str, key: str) -> Optional[str]:
    prefix = f"{key}:"
    for line in stdout.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    return None


def _parse_stdout_float(stdout: str, key: str) -> Optional[float]:
    raw = _parse_stdout_value(stdout, key)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_stdout_int(stdout: str, key: str) -> Optional[int]:
    raw = _parse_stdout_value(stdout, key)
    if raw is None:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _stdout_tail(stdout: str, n: int = STDOUT_TAIL_LINES) -> list[str]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return lines[-n:]


def _path_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    size_bytes: Optional[int] = None
    if exists and path.is_file():
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
    return {
        "path": str(path),
        "exists": exists,
        "sizeBytes": size_bytes,
    }


def _load_eval_results_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read eval results json %s: %s", path, exc)
        return {}


def _artifact_paths_for_eval(job_root: Path) -> dict[str, Path]:
    return {
        "evalCsv": job_root / "results" / "eval.csv",
        "resultsJson": job_root / "results" / "eval.results.json",
        "failuresJson": job_root / "results" / "eval.failures.json",
        "log": job_root / "logs" / "run.log",
        "liveStatus": job_root / JOB_LIVE_STATUS,
        "liveFrame": job_root / JOB_LIVE_FRAME,
        "evalVideo": job_root / JOB_EVAL_VIDEO_RELATIVE,
        "evalBrowserVideo": job_root / JOB_EVAL_BROWSER_VIDEO_RELATIVE,
        "evalTimeline": job_root / JOB_EVAL_TIMELINE,
    }


def _build_eval_command(
    job_root: Path,
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
    policy: str,
    checkpoint: Optional[str] = None,
    device: str = "",
    grasp_mode: str = "attachment",
    attachment_mode: str = "policy",
    record_video: bool = True,
    eval_display_camera: str = DEFAULT_CABLE_EVAL_DISPLAY_CAMERA,
    allow_camera_fallback: bool = False,
    eval_executor: Optional[str] = None,
    controller_type: Optional[str] = None,
    action_mode: Optional[str] = None,
    train_config_path: Optional[str] = None,
    task_instruction: Optional[str] = None,
) -> list[str]:
    artifact_paths = _artifact_paths_for_eval(job_root)
    resolved_device = _resolve_eval_device(device)
    cmd = [
        str(PYTHON_BIN),
        str(RUN_PY),
        "eval",
        "--episodes",
        str(episodes),
        "--policy",
        policy,
        "--robot",
        robot,
        "--cable-model",
        cable_model,
        "--difficulty",
        difficulty,
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--out",
        str(artifact_paths["evalCsv"]),
        "--grasp-mode",
        grasp_mode,
        "--attachment-mode",
        attachment_mode,
    ]
    if policy in TRAINED_MODEL_POLICIES:
        if not checkpoint:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="checkpoint required for trained model policy evaluation",
            )
        cmd.extend(["--checkpoint", checkpoint, "--device", resolved_device])
        if policy in {"diffusion_policy", "pi0"}:
            if eval_executor:
                cmd.extend(["--eval-executor", eval_executor])
            if controller_type:
                cmd.extend(["--controller-type", controller_type])
            if action_mode:
                cmd.extend(["--action-mode", action_mode])
        if policy == "pi0":
            if train_config_path:
                cmd.extend(["--train-config", train_config_path])
            if task_instruction:
                cmd.extend(["--task-instruction", task_instruction])

    live_dir = job_root / JOB_LIVE_DIR
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "frames").mkdir(parents=True, exist_ok=True)
    timeline_path = job_root / JOB_EVAL_TIMELINE
    cmd.extend(
        [
            "--live-frame-dir",
            str(live_dir),
            "--live-frame-width",
            str(DISPLAY_WIDTH),
            "--live-frame-height",
            str(DISPLAY_HEIGHT),
            "--live-frame-every",
            "5",
            "--live-status-out",
            str(artifact_paths["liveStatus"]),
            "--live-display-camera",
            eval_display_camera,
            "--live-timeline-out",
            str(timeline_path),
        ]
    )
    if allow_camera_fallback:
        cmd.append("--allow-camera-fallback")
    if record_video:
        videos_dir = job_root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(
            [
                "--live-save-frames",
                "--episode-video-dir",
                str(videos_dir),
                "--live-video-fps",
                "20",
            ]
        )
    cmd.extend(["--job-id", job_root.name, "--record-step-metrics"])
    return cmd


def _parse_eval_episode_progress(log_content: str, total_episodes: int) -> int:
    completed = 0
    for line in log_content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                episode_index = int(parts[0])
            except ValueError:
                continue
            completed = max(completed, episode_index + 1)
    return min(completed, total_episodes) if total_episodes > 0 else completed


def _artifact_paths_for_generate(job_root: Path, *, include_hdf5: bool, include_lerobot: bool = False) -> dict[str, Path]:
    paths = {
        "npz": job_root / "datasets" / "dataset.npz",
        "hdf5": job_root / "datasets" / "dataset.hdf5",
        "manifest": job_root / "datasets" / "dataset.manifest.json",
        "collectCsv": job_root / "results" / "collect.csv",
        "failures": job_root / "results" / "failures.json",
        "log": job_root / "logs" / "run.log",
        "liveStatus": job_root / JOB_LIVE_STATUS,
        "liveFrame": job_root / JOB_LIVE_FRAME,
        "generateVideo": job_root / JOB_GENERATE_VIDEO_RELATIVE,
        "generateTimeline": job_root / JOB_LIVE_TIMELINE,
    }
    if include_lerobot:
        lerobot_dir = _lerobot_dataset_dir(job_root)
        paths["lerobot"] = lerobot_dir
        paths["lerobotMetadata"] = lerobot_dir / "metadata.json"
        paths["lerobotStats"] = lerobot_dir / "stats.json"
        paths["lerobotReport"] = lerobot_dir / "generation_report.json"
    if not include_hdf5:
        paths.pop("hdf5", None)
    return paths


def _build_generate_command(
    job_root: Path,
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
    save_hdf5: bool,
    output_format: str,
    include_live: bool,
    save_process_video: bool = True,
    lerobot_task_instruction: str = "thread the cable through the pole",
    lerobot_robot: str = "Panda",
    lerobot_fps: int = 20,
) -> list[str]:
    output_format = _validate_output_format(output_format)
    include_hdf5 = save_hdf5 or output_format == "hdf5"
    include_lerobot = output_format == "lerobot"
    artifact_paths = _artifact_paths_for_generate(
        job_root,
        include_hdf5=include_hdf5,
        include_lerobot=include_lerobot,
    )
    npz_out = artifact_paths["npz"]
    if include_lerobot and not include_hdf5 and output_format == "lerobot":
        npz_out = job_root / "datasets" / "debug" / "dataset.npz"

    cmd = [
        str(PYTHON_BIN),
        str(RUN_PY),
        "expert",
        "--episodes",
        str(episodes),
        "--robot",
        robot,
        "--cable-model",
        cable_model,
        "--difficulty",
        difficulty,
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--out",
        str(npz_out),
        "--results-out",
        str(artifact_paths["collectCsv"]),
        "--failures-out",
        str(artifact_paths["failures"]),
    ]
    if include_hdf5:
        cmd.extend(["--hdf5-out", str(artifact_paths["hdf5"])])
    if include_lerobot:
        cmd.extend(
            [
                "--lerobot-out",
                str(artifact_paths["lerobot"]),
                "--lerobot-task-instruction",
                lerobot_task_instruction,
                "--lerobot-robot",
                lerobot_robot,
                "--lerobot-fps",
                str(lerobot_fps),
                "--manifest-out",
                str(artifact_paths["manifest"]),
            ]
        )
    if include_live:
        live_dir = job_root / JOB_LIVE_DIR
        live_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(
            [
                "--live-frame-dir",
                str(live_dir),
                "--live-frame-width",
                str(DISPLAY_WIDTH),
                "--live-frame-height",
                str(DISPLAY_HEIGHT),
                "--live-frame-every",
                "5",
                "--live-status-out",
                str(artifact_paths["liveStatus"]),
                "--live-camera",
                "agentview",
            ]
        )
        if save_process_video:
            generate_video = job_root / JOB_GENERATE_VIDEO_RELATIVE
            generate_video.parent.mkdir(parents=True, exist_ok=True)
            timeline_path = job_root / JOB_LIVE_TIMELINE
            cmd.extend(
                [
                    "--live-save-frames",
                    "--live-video-out",
                    str(generate_video),
                    "--live-video-fps",
                    "20",
                    "--live-timeline-out",
                    str(timeline_path),
                ]
            )
    return cmd


def start_generate_async(
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
    save_hdf5: bool,
    output_format: str,
    save_process_video: bool = True,
    task_config_id: Optional[str] = None,
    lerobot_task_instruction: str = "thread the cable through the pole",
    lerobot_robot: str = "Panda",
    lerobot_fps: int = 20,
) -> dict[str, Any]:
    robot = _validate_robot(robot)
    cable_model = _validate_cable_model(cable_model)
    difficulty = _validate_difficulty(difficulty)
    output_format = _validate_output_format(output_format)
    if output_format == "lerobot":
        save_hdf5 = False
    elif output_format == "hdf5":
        save_hdf5 = True

    job_id = make_job_id("ct_gen")
    job_root = _prepare_job_dirs(
        job_id, include_datasets=True, include_videos=save_process_video
    )
    (job_root / JOB_LIVE_DIR).mkdir(parents=True, exist_ok=True)
    log_path = job_root / "logs" / "run.log"
    cmd = _build_generate_command(
        job_root,
        episodes=episodes,
        robot=robot,
        cable_model=cable_model,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        save_hdf5=save_hdf5,
        output_format=output_format,
        include_live=True,
        save_process_video=save_process_video,
        lerobot_task_instruction=lerobot_task_instruction,
        lerobot_robot=lerobot_robot or robot,
        lerobot_fps=lerobot_fps,
    )

    if not PYTHON_BIN.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Python interpreter not found: {PYTHON_BIN}",
        )
    if not RUN_PY.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"run.py not found: {RUN_PY}",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKING_DIR),
        env=_build_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ASYNC_JOBS[job_id] = AsyncJobRecord(
        job_id=job_id,
        job_dir=job_root,
        command=cmd,
        started_at=started_at,
        process=proc,
    )

    initial_status = {
        "status": "running",
        "jobType": "generate",
        "taskType": "cable_threading",
        "episode": 0,
        "episodes": episodes,
        "step": 0,
        "horizon": horizon,
        "frameCount": 0,
        "savedFrameCount": 0,
        "generateVideoStatus": "pending" if save_process_video else None,
        "generateVideo": None,
        "generateVideoExists": False,
        "generateVideoSizeBytes": 0,
        "latestFrame": "latest.jpg",
        "phase": "",
        "successfulEpisodes": 0,
        "seed": seed,
        "error": None,
    }
    status_path = job_root / JOB_LIVE_STATUS
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(initial_status, indent=2), encoding="utf-8")

    record_workspace_job_start(
        job_id=job_id,
        job_type="generate",
        task_type="cable_threading",
        runtime_path=str(job_root),
        runner="run.py",
        status="running",
        metadata=build_job_resource_metadata(
            task_type="cable_threading",
            task_config_id=task_config_id,
            extra={
                "episodes": episodes,
                "robot": robot,
                "cableModel": cable_model,
                "difficulty": difficulty,
                "horizon": horizon,
                "seed": seed,
            },
        ),
    )

    return {
        "jobId": job_id,
        "taskType": "cable_threading",
        "status": "running",
        "command": _format_command(cmd),
        "frameUrl": f"/api/workspace/cable-threading/jobs/{job_id}/frame",
        "statusUrl": f"/api/workspace/cable-threading/jobs/{job_id}/status",
    }


def _read_live_status_json(job_root: Path) -> dict[str, Any]:
    status_path = job_root / JOB_LIVE_STATUS
    if not status_path.is_file():
        return {}
    try:
        data = json.loads(status_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read live status %s: %s", status_path, exc)
        return {}


def _write_live_status_json(job_root: Path, live: dict[str, Any]) -> None:
    status_path = job_root / JOB_LIVE_STATUS
    try:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(json.dumps(live, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write live status %s: %s", status_path, exc)


def _derive_eval_failure_diagnosis(
    *,
    status_value: str,
    live: dict[str, Any],
    log_content: str,
    job_root: Path,
) -> dict[str, Any]:
    if status_value != "failed":
        return {}

    if str(live.get("failedStage") or "") == "obs_validation":
        stdout_path = job_root / "logs" / "stdout.log"
        stderr_path = job_root / "logs" / "stderr.log"
        run_log_path = job_root / "logs" / "run.log"
        log_paths: dict[str, str] = {
            "stdout": "logs/stdout.log" if stdout_path.is_file() else "logs/run.log",
            "stderr": "logs/stderr.log" if stderr_path.is_file() else "logs/run.log",
            "run": "logs/run.log" if run_log_path.is_file() else "logs/run.log",
        }
        return {
            "failedStage": "obs_validation",
            "failureReason": str(live.get("failureReason") or "obs_key_mismatch"),
            "errorMessage": str(
                live.get("errorMessage")
                or "策略评测环境与模型观测不匹配，未进入 rollout。"
            ),
            "logPaths": live.get("logPaths") or log_paths,
            "expectedObsKeys": live.get("expectedObsKeys"),
            "actualObsKeys": live.get("actualObsKeys"),
            "missingKeys": live.get("missingKeys"),
            "shapeMismatchKeys": live.get("shapeMismatchKeys"),
        }

    error_raw = str(live.get("error") or "").strip()
    combined = f"{error_raw}\n{log_content}".lower()

    failed_stage = str(live.get("failedStage") or "unknown")
    failure_reason = str(live.get("failureReason") or "unknown_error")
    error_message = str(
        live.get("errorMessage")
        or "评测任务执行失败，请查看评测日志获取详细错误信息。"
    )

    if failed_stage == "unknown" or failure_reason == "unknown_error":
        if "missing observation keys" in combined or "obs_key" in combined:
            failed_stage = "rollout"
            failure_reason = "obs_key_mismatch"
            error_message = (
                "策略评测环境与模型观测不匹配：评测环境未提供模型所需的观测项"
                "（例如 robot0_eye_in_hand_image）。"
            )
        elif any(token in combined for token in ("checkpoint", "model_final", "load_state_dict", "no such file")):
            if "checkpoint" in combined or "model" in combined:
                failed_stage = "model_loading"
                failure_reason = "model_load_failed"
                error_message = "模型加载失败，请确认 checkpoint 文件存在且与任务匹配。"
        elif any(token in combined for token in ("action dim", "action_dim", "shape mismatch")):
            failed_stage = "rollout"
            failure_reason = "action_dim_mismatch"
            error_message = "策略 action 维度与仿真环境不匹配。"
        elif any(token in combined for token in ("obs dim", "observation dim", "obs_dim")):
            failed_stage = "rollout"
            failure_reason = "obs_dim_mismatch"
            error_message = "策略观测维度与仿真环境不匹配。"
        elif "exited with code" in combined or "traceback" in combined:
            failed_stage = "rollout"
            failure_reason = "runner_exception"
            if error_raw:
                error_message = "评测 runner 执行异常，请查看日志中的 Traceback。"
        elif "evalvideo" in combined or "video" in combined:
            failed_stage = "video_generation"
            failure_reason = "video_generation_failed"
            error_message = "评测视频生成失败。"

    stdout_path = job_root / "logs" / "stdout.log"
    stderr_path = job_root / "logs" / "stderr.log"
    run_log_path = job_root / "logs" / "run.log"
    log_paths: dict[str, str] = {
        "stdout": "logs/stdout.log" if stdout_path.is_file() else "logs/run.log",
        "stderr": "logs/stderr.log" if stderr_path.is_file() else "logs/run.log",
        "run": "logs/run.log" if run_log_path.is_file() else "logs/run.log",
    }

    diagnosis = {
        "failedStage": failed_stage,
        "failureReason": failure_reason,
        "errorMessage": error_message,
        "logPaths": log_paths,
    }

    dirty = False
    for key, value in diagnosis.items():
        if live.get(key) != value:
            live[key] = value
            dirty = True
    if dirty:
        _write_live_status_json(job_root, live)

    return diagnosis


def _load_eval_aggregate_json(job_root: Path) -> dict[str, Any]:
    aggregate_path = job_root / "results" / "aggregate_result.json"
    if not aggregate_path.is_file():
        return {}
    try:
        data = json.loads(aggregate_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def get_job_status(job_id: str) -> dict[str, Any]:
    from app.services.imported_eval_bridge import get_imported_eval_cable_status
    from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

    candidate = (job_id or "").strip()
    if is_imported_workspace_eval_job_id(candidate):
        return get_imported_eval_cable_status(candidate)

    validated = validate_job_id(job_id)
    if validated.startswith("ct_eval_"):
        return get_eval_job_status(validated)
    return get_generate_job_status(validated)


def get_eval_job_status(job_id: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    sync_workspace_job_from_runtime(validated)
    job_root = _job_dir(validated)
    live = _read_live_status_json(job_root)
    artifact_paths = _artifact_paths_for_eval(job_root)
    paths = {key: _path_info(path) for key, path in artifact_paths.items()}

    record = ASYNC_JOBS.get(validated)
    if record and record.process and record.process.poll() is not None:
        return_code = record.process.returncode
        current_status = str(live.get("status") or "running")
        if current_status in {"queued", "running"}:
            live["status"] = "completed" if return_code == 0 else "failed"
            if return_code != 0 and not live.get("error"):
                live["error"] = f"evaluate process exited with code {return_code}"

    log_path = artifact_paths["log"]
    log_content = ""
    if log_path.is_file():
        try:
            log_content = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_content = ""

    total_episodes = int(live.get("episodes") or 0)
    if total_episodes <= 0:
        total_episodes = _parse_eval_episode_progress(log_content, 100)
    completed_episodes = _parse_eval_episode_progress(log_content, total_episodes)
    if completed_episodes > 0:
        live["completedEpisodes"] = completed_episodes
        if total_episodes > 0:
            live["progressPercent"] = min(
                100, round((completed_episodes / total_episodes) * 100)
            )

    results_data = _load_eval_results_json(artifact_paths["resultsJson"])
    aggregate_file = _load_eval_aggregate_json(job_root)
    metrics: dict[str, Any] = {}
    if results_data:
        metrics["successRate"] = results_data.get("success_rate")
        metrics["everSuccessRate"] = results_data.get("ever_success_rate")
        metrics["numEpisodes"] = results_data.get("num_episodes")
        metrics["aggregate"] = results_data.get("aggregate")
    if aggregate_file:
        metrics["aggregate"] = {**(metrics.get("aggregate") or {}), **aggregate_file}
        if metrics.get("successRate") is None:
            metrics["successRate"] = aggregate_file.get("final_success_rate") or aggregate_file.get(
                "success_rate"
            )
        if metrics.get("everSuccessRate") is None:
            metrics["everSuccessRate"] = aggregate_file.get("ever_success_rate")
        if metrics.get("numEpisodes") is None:
            metrics["numEpisodes"] = aggregate_file.get("total_episodes") or aggregate_file.get(
                "num_episodes"
            )
    if metrics.get("successRate") is None:
        metrics["successRate"] = _parse_stdout_float(log_content, "success_rate")

    status_value = str(live.get("status") or "running")
    if status_value in {"queued", "running"} and paths["resultsJson"]["exists"]:
        status_value = "completed"
        live["status"] = "completed"
    if status_value in {"queued", "running"} and paths["evalCsv"]["exists"] and not paths["resultsJson"]["exists"]:
        if record and record.process and record.process.poll() is not None:
            status_value = "failed" if record.process.returncode != 0 else "completed"
            live["status"] = status_value

    eval_video_path = artifact_paths["evalVideo"]
    eval_browser_video_path = artifact_paths["evalBrowserVideo"]
    eval_video_exists = bool(live.get("evalVideoExists"))
    eval_video_size = live.get("evalVideoSizeBytes")
    eval_browser_video_exists = bool(live.get("evalBrowserVideoExists"))
    eval_browser_video_size = live.get("evalBrowserVideoSizeBytes")
    video_resolution = live.get("videoResolution")
    browser_video_path_value = live.get("browserVideoPath")

    if eval_browser_video_path.is_file():
        eval_browser_video_exists = True
        if eval_browser_video_size is None:
            try:
                eval_browser_video_size = eval_browser_video_path.stat().st_size
            except OSError:
                eval_browser_video_size = 0
        if not live.get("evalBrowserVideo"):
            live["evalBrowserVideo"] = str(eval_browser_video_path)
        live["evalBrowserVideoExists"] = True
        if eval_browser_video_size is not None:
            live["evalBrowserVideoSizeBytes"] = eval_browser_video_size
        browser_video_path_value = str(eval_browser_video_path)

    if eval_video_path.is_file():
        eval_video_exists = True
        if eval_video_size is None:
            try:
                eval_video_size = eval_video_path.stat().st_size
            except OSError:
                eval_video_size = 0
        if not live.get("evalVideo"):
            live["evalVideo"] = str(eval_video_path)
        live["evalVideoExists"] = True
        if eval_video_size is not None:
            live["evalVideoSizeBytes"] = eval_video_size

    video_url = (
        f"/api/workspace/cable-threading/jobs/{validated}/video"
        if eval_video_exists or eval_browser_video_exists
        else None
    )

    timeline_path = artifact_paths["evalTimeline"]
    timeline_exists = timeline_path.is_file()
    timeline_url = (
        f"/api/workspace/cable-threading/jobs/{validated}/timeline"
        if timeline_exists
        else None
    )

    failure_diagnosis = _derive_eval_failure_diagnosis(
        status_value=status_value,
        live=live,
        log_content=log_content,
        job_root=job_root,
    )

    _apply_live_frame_path_info(paths, job_root, live)

    from app.services.evaluation.selected_evaluation_metrics import finalize_selected_evaluation_metrics
    from app.services.evaluation_replay_info import build_cable_threading_replay_info

    aggregate_for_metrics = metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else aggregate_file
    if isinstance(aggregate_for_metrics, dict) and aggregate_for_metrics:
        finalized = finalize_selected_evaluation_metrics(
            aggregate_for_metrics,
            job_root,
            None,
            task_type="cable_threading",
            persist=status_value in {"completed", "failed"},
            legacy_fallback=True,
        )
        metrics["aggregate"] = finalized["aggregate"]
        metrics["selectedMetricIds"] = finalized["selectedMetricIds"]
        metrics["metricResults"] = finalized["metricResults"]
        metrics["runMetrics"] = finalized.get("runMetrics") or {}

    replay_info = build_cable_threading_replay_info(
        validated,
        job_root,
        live=live,
        results_data=results_data,
        aggregate_file=metrics.get("aggregate") if isinstance(metrics.get("aggregate"), dict) else aggregate_file,
        status_value=status_value,
    )
    if replay_info.get("completedEpisodes") is not None:
        live["completedEpisodes"] = replay_info["completedEpisodes"]
    if replay_info.get("requestedEpisodes") and replay_info.get("completedEpisodes"):
        live["progressPercent"] = min(
            100,
            round(
                (int(replay_info["completedEpisodes"]) / int(replay_info["requestedEpisodes"])) * 100
            ),
        )

    from app.services.evaluation_workbench_basic_info import attach_workbench_basic_info

    status_payload = {
        "jobId": validated,
        "evalJobId": validated,
        "taskType": "cable_threading",
        "status": status_value,
        "live": live,
        **replay_info,
        "paths": paths,
        "metrics": metrics,
        "command": _format_command(record.command) if record else "",
        "startedAt": record.started_at if record else None,
        "evalVideoExists": eval_video_exists,
        "evalVideoSizeBytes": eval_video_size if eval_video_exists else 0,
        "evalVideoPath": str(eval_video_path) if eval_video_exists else None,
        "evalBrowserVideoExists": eval_browser_video_exists,
        "evalBrowserVideoSizeBytes": eval_browser_video_size if eval_browser_video_exists else 0,
        "evalBrowserVideoPath": (
            str(eval_browser_video_path) if eval_browser_video_exists else None
        ),
        "browserVideoPath": browser_video_path_value,
        "videoResolution": video_resolution,
        "evalVideoStatus": live.get("evalVideoStatus") or live.get("videoStatus"),
        "videoUrl": video_url,
        "timelineExists": timeline_exists,
        "timelinePath": str(timeline_path) if timeline_exists else None,
        "timelineUrl": timeline_url,
        **failure_diagnosis,
    }
    if metrics.get("selectedMetricIds"):
        status_payload["selectedMetricIds"] = metrics["selectedMetricIds"]
    if metrics.get("metricResults"):
        status_payload["metricResults"] = metrics["metricResults"]
    if metrics.get("runMetrics"):
        status_payload["runMetrics"] = metrics["runMetrics"]
    from app.services.replay_content_detection import detect_cable_threading_replay_content

    status_payload["replayContent"] = detect_cable_threading_replay_content(
        job_root,
        job_id=validated,
        metrics=metrics,
        live=live,
        is_eval_job=True,
    )
    return attach_workbench_basic_info(status_payload, eval_job_id=validated, job_root=job_root)


def _normalize_eval_episode_row(ep: dict[str, Any], index: int) -> dict[str, Any]:
    success = ep.get("success")
    if success is None:
        success = ep.get("final_success")
    return {
        "episode": ep.get("episode", index),
        "seed": ep.get("seed"),
        "success": bool(success) if success is not None else None,
        "final_success": bool(ep.get("final_success")) if ep.get("final_success") is not None else None,
        "thread_completion_max": ep.get("thread_completion_max") or ep.get("max_thread_completion"),
        "failure_reason": ep.get("failure_reason") or ep.get("fail_reason") or ep.get("error") or "",
        "video_path": ep.get("video_path") or ep.get("videoPath") or "",
        "videoUri": ep.get("videoUri") or ep.get("video_path") or ep.get("videoPath") or "",
        "episodeIndex": ep.get("episodeIndex") or (index + 1),
        "recordCamera": ep.get("recordCamera") or ep.get("record_camera") or "",
        "cameraFallbackUsed": ep.get("cameraFallbackUsed"),
    }


def build_cable_threading_eval_report_payload(
    job_id: str,
    job_root: Path,
    *,
    status_value: str,
    results_data: dict[str, Any],
    artifact_paths: dict[str, Path],
    eval_video_path: Optional[str] = None,
    browser_video_path: Optional[str] = None,
    video_resolution: Optional[str] = None,
) -> dict[str, Any]:
    aggregate_raw = results_data.get("aggregate") if isinstance(results_data.get("aggregate"), dict) else {}
    episodes_raw = results_data.get("episodes") if isinstance(results_data.get("episodes"), list) else []
    per_episode = [_normalize_eval_episode_row(ep, idx) for idx, ep in enumerate(episodes_raw) if isinstance(ep, dict)]

    success_episodes = sum(
        1
        for row in per_episode
        if row.get("success") is True or row.get("final_success") is True
    )
    total_episodes = int(results_data.get("num_episodes") or len(per_episode) or 0)

    aggregate: dict[str, Any] = {
        "task_name": "线缆穿杆",
        "total_episodes": total_episodes,
        "success_episodes": success_episodes,
        "final_success_rate": aggregate_raw.get("final_success_rate") or results_data.get("success_rate"),
        "ever_success_rate": aggregate_raw.get("ever_success_rate") or results_data.get("ever_success_rate"),
        "mean_thread_completion_max": aggregate_raw.get("mean_thread_completion_max"),
        "mean_endpoint_goal_error_final": aggregate_raw.get("mean_endpoint_goal_error_final"),
        "mean_straightness_error_final": aggregate_raw.get("mean_straightness_error_final"),
        "mean_anchor_error_final": aggregate_raw.get("mean_anchor_error_final"),
        "mean_tabletop_spread_final": aggregate_raw.get("mean_tabletop_spread_final"),
        "failure_reasons": aggregate_raw.get("failure_reasons") or {},
    }

    results_json = artifact_paths["resultsJson"]
    per_episode_path = job_root / "results" / "per_episode_results.json"
    aggregate_path = job_root / "results" / "aggregate_result.json"

    if aggregate_path.is_file():
        try:
            aggregate.update(json.loads(aggregate_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass

    from app.services.evaluation.selected_evaluation_metrics import finalize_selected_evaluation_metrics

    finalized = finalize_selected_evaluation_metrics(
        aggregate,
        job_root,
        None,
        task_type="cable_threading",
        persist=True,
        legacy_fallback=True,
    )
    aggregate = finalized["aggregate"]

    from app.services.evaluation_replay_info import build_cable_threading_replay_info

    replay_info = build_cable_threading_replay_info(
        job_id,
        job_root,
        live=_read_live_status_json(job_root),
        results_data=results_data,
        aggregate_file=aggregate,
        status_value=status_value,
    )

    return {
        "evalJobId": job_id,
        "taskType": "cable_threading",
        "status": status_value,
        "aggregate": aggregate,
        "selectedMetricIds": finalized["selectedMetricIds"],
        "metricResults": finalized["metricResults"],
        "runMetrics": finalized.get("runMetrics") or {},
        **replay_info,
        "perEpisode": per_episode,
        "episodes": episodes_raw,
        "successRate": aggregate.get("final_success_rate"),
        "everSuccessRate": aggregate.get("ever_success_rate"),
        "numEpisodes": total_episodes,
        "evalVideoPath": eval_video_path,
        "browserVideoPath": browser_video_path or aggregate.get("browserVideoPath"),
        "videoResolution": video_resolution or aggregate.get("videoResolution"),
        "videoStatus": aggregate.get("videoStatus"),
        "paths": {
            "resultsJson": _path_info(results_json),
            "evalCsv": _path_info(artifact_paths["evalCsv"]),
            "failuresJson": _path_info(artifact_paths["failuresJson"]),
            "log": _path_info(artifact_paths["log"]),
            "aggregateResult": _path_info(aggregate_path),
            "perEpisodeResults": _path_info(per_episode_path),
        },
        "artifacts": {
            "aggregateResult": str(aggregate_path) if aggregate_path.is_file() else str(results_json),
            "perEpisodeResults": str(per_episode_path) if per_episode_path.is_file() else None,
            "resultsJson": _path_info(results_json),
            "evalCsv": _path_info(artifact_paths["evalCsv"]),
            "log": _path_info(artifact_paths["log"]),
            "evalVideo": {"path": eval_video_path, "exists": bool(eval_video_path)},
        },
        "fileChecks": {
            "aggregateResult": aggregate_path.is_file() or results_json.is_file(),
            "perEpisodeResults": per_episode_path.is_file() or len(per_episode) > 0,
            "statusCompleted": status_value == "completed",
            "resultsDirectory": results_json.parent.is_dir(),
        },
    }


def _write_eval_report_artifacts(job_root: Path, payload: dict[str, Any]) -> None:
    results_dir = job_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    aggregate = payload.get("aggregate")
    per_episode = payload.get("perEpisode")
    if isinstance(aggregate, dict):
        aggregate_path = results_dir / "aggregate_result.json"
        aggregate_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    if isinstance(per_episode, list):
        per_path = results_dir / "per_episode_results.json"
        per_path.write_text(json.dumps(per_episode, indent=2, ensure_ascii=False), encoding="utf-8")


def get_eval_job_result(job_id: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    if not validated.startswith("ct_eval_"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="result endpoint only supports ct_eval_* jobs",
        )
    status_payload = get_eval_job_status(validated)
    job_root = _job_dir(validated)
    artifact_paths = _artifact_paths_for_eval(job_root)
    results_data = _load_eval_results_json(artifact_paths["resultsJson"])
    payload = build_cable_threading_eval_report_payload(
        validated,
        job_root,
        status_value=str(status_payload.get("status") or "unknown"),
        results_data=results_data,
        artifact_paths=artifact_paths,
        eval_video_path=status_payload.get("evalVideoPath"),
        browser_video_path=status_payload.get("browserVideoPath")
        or status_payload.get("evalBrowserVideoPath"),
        video_resolution=status_payload.get("videoResolution"),
    )
    _write_eval_report_artifacts(job_root, payload)
    sync_workspace_job_from_runtime(validated)
    return payload


def start_evaluate_async(
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
    policy: str,
    checkpoint: Optional[str] = None,
    device: str = "",
    task_config_id: Optional[str] = None,
    model_name: Optional[str] = None,
    record_video: bool = True,
    eval_display_camera: str = DEFAULT_CABLE_EVAL_DISPLAY_CAMERA,
    allow_camera_fallback: bool = False,
    eval_executor: Optional[str] = None,
    controller_type: Optional[str] = None,
    action_mode: Optional[str] = None,
    train_config_path: Optional[str] = None,
    task_instruction: Optional[str] = None,
    model_asset_id: Optional[str] = None,
    source_train_job_id: Optional[str] = None,
    state_dim: Optional[int] = None,
    action_dim: Optional[int] = None,
) -> dict[str, Any]:
    robot = _validate_robot(robot)
    cable_model = _validate_cable_model(cable_model)
    difficulty = _validate_difficulty(difficulty)
    policy = _validate_policy(policy)

    if policy in {"act", "diffusion_policy", "robomimic"}:
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "评测环境缺少 opencv (cv2) 依赖，无法运行图像策略评测。"
                    "请在训练/评测节点安装 opencv-python-headless。"
                ),
            ) from exc

    if policy in TRAINED_MODEL_POLICIES:
        checkpoint_path = (checkpoint or "").strip()
        if not checkpoint_path:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="所选模型 checkpoint 不可用，请先完成训练并确认模型资产已生成",
            )
        resolved_checkpoint = Path(checkpoint_path).resolve()
        if not resolved_checkpoint.is_file():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="所选模型 checkpoint 文件不存在或不可用，请重新选择已训练模型",
            )
        checkpoint = str(resolved_checkpoint)
    else:
        checkpoint = None

    job_id = make_job_id("ct_eval")
    job_root = _prepare_job_dirs(job_id, include_datasets=False, include_videos=True)
    live_dir = job_root / JOB_LIVE_DIR
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "frames").mkdir(parents=True, exist_ok=True)
    log_path = job_root / "logs" / "run.log"
    cmd = _build_eval_command(
        job_root,
        episodes=episodes,
        robot=robot,
        cable_model=cable_model,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        policy=policy,
        checkpoint=checkpoint,
        device=device,
        record_video=record_video,
        eval_display_camera=eval_display_camera,
        allow_camera_fallback=allow_camera_fallback,
        eval_executor=eval_executor,
        controller_type=controller_type,
        action_mode=action_mode,
        train_config_path=train_config_path,
        task_instruction=task_instruction,
    )

    if not PYTHON_BIN.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Python interpreter not found: {PYTHON_BIN}",
        )
    if not RUN_PY.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"run.py not found: {RUN_PY}",
        )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(WORKING_DIR),
        env=_build_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ASYNC_JOBS[job_id] = AsyncJobRecord(
        job_id=job_id,
        job_dir=job_root,
        command=cmd,
        started_at=started_at,
        process=proc,
    )

    initial_status = {
        "status": "running",
        "jobType": "evaluate",
        "taskType": "cable_threading",
        "policy": policy,
        "episodes": episodes,
        "requestedEpisodes": episodes,
        "completedEpisodes": 0,
        "successfulEpisodes": 0,
        "failedEpisodes": 0,
        "recordedVideoCount": 0,
        "horizon": horizon,
        "seed": seed,
        "recordVideo": record_video,
        "progressPercent": 0,
        "latestFrame": None,
        "error": None,
    }
    status_path = job_root / JOB_LIVE_STATUS
    status_path.write_text(json.dumps(initial_status, indent=2), encoding="utf-8")

    public_mode = "trained_model_evaluation" if policy in TRAINED_MODEL_POLICIES else "expert_policy_evaluation"
    user_task_name = (model_name or "").strip() or None
    generated_display_name = build_evaluation_display_name("cable_threading", public_mode)
    task_name = user_task_name or generated_display_name

    record_workspace_job_start(
        job_id=job_id,
        job_type="evaluation",
        task_type="cable_threading",
        runtime_path=str(job_root),
        runner="run.py",
        status="running",
        task_name=task_name,
        metadata=build_job_resource_metadata(
            task_type="cable_threading",
            task_config_id=task_config_id,
            extra={
                "displayName": generated_display_name,
                "templateDisplayName": generated_display_name,
                "taskDisplayName": resolve_task_display_name("cable_threading"),
                "evaluationTypeLabel": resolve_evaluation_type_label(public_mode),
                "evaluationMode": public_mode,
                "originalName": user_task_name,
                "episodes": episodes,
                "robot": robot,
                "cableModel": cable_model,
                "difficulty": difficulty,
                "policy": policy,
                "horizon": horizon,
                "seed": seed,
                **({"modelName": model_name.strip()} if model_name and model_name.strip() else {}),
                **({"modelAssetId": model_asset_id} if model_asset_id else {}),
                **({"sourceTrainJobId": source_train_job_id} if source_train_job_id else {}),
                **({"evalExecutor": eval_executor} if eval_executor else {}),
                **({"controllerType": controller_type} if controller_type else {}),
                **({"actionMode": action_mode} if action_mode else {}),
                **({"stateDim": state_dim} if state_dim is not None else {}),
                **({"actionDim": action_dim} if action_dim is not None else {}),
                **({"taskInstruction": task_instruction} if task_instruction else {}),
                **({"trainConfigPath": train_config_path} if train_config_path else {}),
            },
        ),
    )

    return {
        "evalJobId": job_id,
        "jobId": job_id,
        "taskType": "cable_threading",
        "status": "queued",
        "command": _format_command(cmd),
        "statusUrl": f"/api/workspace/cable-threading/jobs/{job_id}/status",
    }


def _build_cable_threading_failure_reason(summary: dict[str, Any]) -> str:
    import sys

    if str(WORKING_DIR) not in sys.path:
        sys.path.insert(0, str(WORKING_DIR))
    from examples.cable_threading.failure_reason import build_cable_threading_failure_reason

    return build_cable_threading_failure_reason(summary)


def _load_cable_failures_json(job_root: Path) -> list[dict[str, Any]]:
    failures_path = job_root / "results" / "failures.json"
    if not failures_path.is_file():
        return []
    try:
        data = json.loads(failures_path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read failures.json %s: %s", failures_path, exc)
    return []


def _append_cable_generation_failure_metrics(metrics: dict[str, Any], job_root: Path) -> None:
    failures = _load_cable_failures_json(job_root)
    if not failures:
        return
    summary_items: list[dict[str, Any]] = []
    for item in failures:
        episode_summary = item.get("summary") if isinstance(item.get("summary"), dict) else {}
        episode_index = item.get("episode", episode_summary.get("episode"))
        seed = item.get("seed", episode_summary.get("seed"))
        reason = (
            str(episode_summary.get("failure_reason") or item.get("failure_reason") or "").strip()
            or _build_cable_threading_failure_reason(episode_summary)
            or "未满足最终成功条件"
        )
        display_episode = int(episode_index) + 1 if episode_index is not None else None
        summary_items.append(
            {
                "episodeIndex": display_episode,
                "seed": seed,
                "success": False,
                "failureReason": reason,
            }
        )
    metrics["failedEpisodes"] = len(summary_items)
    metrics["failureSummary"] = summary_items
    metrics["episodeResults"] = summary_items


def _read_dataset_manifest_json(job_root: Path) -> dict[str, Any]:
    manifest_path = job_root / "datasets" / "dataset.manifest.json"
    if not manifest_path.is_file():
        return {}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _maybe_persist_cable_dataset_manifest(job_root: Path, job_id: str) -> dict[str, Any]:
    manifest_path = job_root / "datasets" / "dataset.manifest.json"
    if not manifest_path.is_file():
        return {}
    manifest = _read_dataset_manifest_json(job_root)
    lerobot_meta_path = _lerobot_sidecar_path(job_root, "metadata.json")
    lerobot_ready = lerobot_meta_path.is_file()
    hdf5_ready = (job_root / "datasets" / "dataset.hdf5").is_file()

    if lerobot_ready:
        fmt = "lerobot"
    elif hdf5_ready:
        fmt = "hdf5"
    else:
        fmt = "npz"

    enriched = persist_manifest_display_fields(
        manifest_path,
        task_type="cable_threading",
        source_job_id=job_id,
        simulator_backend="mujoco",
        dataset_format=fmt,
    )

    hdf5_path = job_root / "datasets" / "dataset.hdf5"
    manifest_path = job_root / "datasets" / "dataset.manifest.json"
    successful_episodes = int(
        enriched.get("successfulEpisodes")
        or enriched.get("num_successful")
        or enriched.get("numSuccessful")
        or 0
    )
    failed_episodes = int(enriched.get("num_failed") or enriched.get("failedEpisodes") or 0)
    dataset_id = str(enriched.get("datasetId") or f"ds_{job_id}")
    enriched.update(
        {
            "datasetId": dataset_id,
            "datasetName": enriched.get("displayName") or enriched.get("name") or dataset_id,
            "sourceJobId": job_id,
            "datasetFormat": fmt,
            "format": fmt,
            "primaryFormat": fmt,
            "datasetFormats": [fmt],
            "availableFormats": [fmt],
            "datasetFile": str(hdf5_path) if hdf5_ready else None,
            "hdf5Path": str(hdf5_path) if hdf5_ready else None,
            "successfulEpisodes": successful_episodes,
            "failedEpisodes": failed_episodes,
            "episodeCount": successful_episodes,
            "episodes": successful_episodes + failed_episodes,
            "trainable": bool(hdf5_ready and successful_episodes > 0),
            "directTrainable": bool(hdf5_ready and successful_episodes > 0),
            "jointActionAvailable": bool(enriched.get("joint_action_available")),
            "usage": "training",
            "artifacts": {
                **(enriched.get("artifacts") if isinstance(enriched.get("artifacts"), dict) else {}),
                **({"hdf5": str(hdf5_path)} if hdf5_ready else {}),
                "manifest": str(manifest_path),
            },
        }
    )

    if lerobot_ready:
        try:
            lerobot_meta = json.loads(lerobot_meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            lerobot_meta = {}
        lerobot_dir = _lerobot_dataset_dir(job_root)
        enriched["datasetFormats"] = ["lerobot"] + (["hdf5"] if hdf5_ready else [])
        enriched["primaryFormat"] = "lerobot"
        enriched["availableFormats"] = enriched["datasetFormats"]
        enriched["format"] = "lerobot"
        enriched["datasetFormat"] = "lerobot"
        enriched["lerobot"] = {
            "status": "ready",
            "path": str(lerobot_dir.relative_to(job_root)),
            "metadataPath": str((lerobot_dir / "metadata.json").relative_to(job_root)),
            "statsPath": str((lerobot_dir / "stats.json").relative_to(job_root)),
            "reportPath": str((lerobot_dir / "generation_report.json").relative_to(job_root)),
            "taskInstruction": lerobot_meta.get("task_instruction"),
            "robot": lerobot_meta.get("robot"),
            "stateDim": lerobot_meta.get("state_dim"),
            "actionDim": lerobot_meta.get("action_dim"),
            "pi0Ready": bool(lerobot_meta.get("pi0Ready")),
            "pi0ReadyReason": lerobot_meta.get("pi0ReadyReason") or "",
        }
    manifest_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return enriched


def _register_completed_cable_dataset(job_root: Path, job_id: str) -> dict[str, Any]:
    """Finalize manifest and database indexes once an async generation completes."""
    marker = job_root / "metadata" / "platform_dataset_registration.json"
    if marker.is_file():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if (
                isinstance(payload, dict)
                and payload.get("status") == "completed"
                and payload.get("version") == 2
            ):
                return payload
        except (OSError, json.JSONDecodeError):
            pass

    manifest = _maybe_persist_cable_dataset_manifest(job_root, job_id)
    hdf5_path = job_root / "datasets" / "dataset.hdf5"
    backfill: dict[str, Any] = {}
    if hdf5_path.is_file():
        from app.services.workspace_dataset_backfill_service import backfill_hdf5_dataset_records

        backfill = backfill_hdf5_dataset_records(
            dry_run=False,
            overwrite=True,
            hdf5_paths=[hdf5_path],
        )
        if backfill.get("errors"):
            raise RuntimeError("; ".join(str(item) for item in backfill["errors"]))

    from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

    invalidate_workspace_dataset_list_cache()
    sync_workspace_job_from_runtime(job_id, overwrite_artifacts=True)
    payload = {
        "version": 2,
        "status": "completed",
        "jobId": job_id,
        "datasetId": manifest.get("datasetId") or f"ds_{job_id}",
        "manifestPath": str(job_root / "datasets" / "dataset.manifest.json"),
        "hdf5Path": str(hdf5_path) if hdf5_path.is_file() else None,
        "backfill": backfill,
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def get_generate_job_status(job_id: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    sync_workspace_job_from_runtime(validated)
    job_root = _job_dir(validated)
    live = _read_live_status_json(job_root)

    record = ASYNC_JOBS.get(validated)
    if record and record.process and record.process.poll() is not None:
        return_code = record.process.returncode
        if live.get("status") == "running":
            live["status"] = "completed" if return_code == 0 else "failed"
            if return_code != 0 and not live.get("error"):
                live["error"] = f"expert process exited with code {return_code}"

    include_hdf5 = bool(live.get("savedHdf5")) or (job_root / "datasets" / "dataset.hdf5").exists()
    include_lerobot = bool(live.get("savedLerobot")) or _lerobot_sidecar_path(job_root, "metadata.json").is_file()
    artifact_paths = _artifact_paths_for_generate(
        job_root,
        include_hdf5=include_hdf5,
        include_lerobot=include_lerobot,
    )
    paths = {key: _path_info(path) for key, path in artifact_paths.items()}

    metrics: dict[str, Any] = {}
    final_rate = live.get("finalSuccessRate")
    if final_rate is not None:
        metrics["finalSuccessRate"] = final_rate
    if live.get("successfulEpisodes") is not None:
        metrics["successfulEpisodes"] = live.get("successfulEpisodes")
    if live.get("episodes") is not None:
        metrics["episodes"] = live.get("episodes")

    status_value = str(live.get("status") or "running")
    dataset_ready = (
        (include_lerobot and paths.get("lerobotMetadata", {}).get("exists"))
        or paths["npz"]["exists"]
    )
    if status_value == "running" and dataset_ready and paths["collectCsv"]["exists"]:
        status_value = "completed"
        live["status"] = "completed"

    if status_value == "completed":
        manifest = _maybe_persist_cable_dataset_manifest(job_root, validated)
        if manifest:
            if metrics.get("successfulEpisodes") is None and manifest.get("successfulEpisodes") is not None:
                metrics["successfulEpisodes"] = manifest.get("successfulEpisodes")
            if metrics.get("episodes") is None and manifest.get("totalEpisodes") is not None:
                metrics["episodes"] = manifest.get("totalEpisodes")
            if metrics.get("successfulEpisodes") is None and manifest.get("num_successful") is not None:
                metrics["successfulEpisodes"] = manifest.get("num_successful")
            if metrics.get("episodes") is None:
                success = manifest.get("num_successful")
                failed = manifest.get("num_failed")
                if isinstance(success, int) and isinstance(failed, int):
                    metrics["episodes"] = success + failed
            if manifest.get("seed") is not None:
                metrics["seed"] = manifest.get("seed")
                live.setdefault("seed", manifest.get("seed"))
        if live.get("seed") is not None and metrics.get("seed") is None:
            metrics["seed"] = live.get("seed")
        _append_cable_generation_failure_metrics(metrics, job_root)
        if metrics.get("failedEpisodes") is None and metrics.get("successfulEpisodes") is not None:
            total = metrics.get("episodes")
            if isinstance(total, int) and total > metrics["successfulEpisodes"]:
                metrics["failedEpisodes"] = total - metrics["successfulEpisodes"]
        try:
            registration = _register_completed_cable_dataset(job_root, validated)
            live["platformDatasetRegistered"] = True
            live["datasetId"] = registration.get("datasetId")
        except Exception as exc:
            logger.exception("cable dataset registration failed job=%s", validated)
            live["platformDatasetRegistered"] = False
            live["datasetRegistrationError"] = str(exc)

    generate_video_path = artifact_paths["generateVideo"]
    generate_video_exists = bool(live.get("generateVideoExists"))
    generate_video_size = live.get("generateVideoSizeBytes")
    if generate_video_path.exists():
        generate_video_exists = True
        if generate_video_size is None:
            try:
                generate_video_size = generate_video_path.stat().st_size
            except OSError:
                generate_video_size = 0
        if not live.get("generateVideo"):
            live["generateVideo"] = str(generate_video_path)
        live["generateVideoExists"] = True
        if generate_video_size is not None:
            live["generateVideoSizeBytes"] = generate_video_size

    video_url = (
        f"/api/workspace/cable-threading/jobs/{validated}/video"
        if generate_video_exists
        else None
    )

    timeline_path = artifact_paths["generateTimeline"]
    timeline_exists = timeline_path.is_file()
    timeline_url = (
        f"/api/workspace/cable-threading/jobs/{validated}/timeline"
        if timeline_exists
        else None
    )

    _apply_live_frame_path_info(paths, job_root, live)

    from app.services.replay_content_detection import detect_cable_threading_replay_content

    replay_content = detect_cable_threading_replay_content(
        job_root,
        job_id=validated,
        metrics=metrics,
        live=live,
        is_eval_job=False,
    )

    return {
        "jobId": validated,
        "taskType": "cable_threading",
        "status": status_value,
        "live": live,
        "paths": paths,
        "metrics": metrics,
        "replayContent": replay_content,
        "command": _format_command(record.command) if record else "",
        "startedAt": record.started_at if record else None,
        "generateVideoExists": generate_video_exists,
        "generateVideoSizeBytes": generate_video_size if generate_video_exists else 0,
        "generateVideoPath": str(generate_video_path) if generate_video_exists else None,
        "videoUrl": video_url,
        "timelineExists": timeline_exists,
        "timelinePath": str(timeline_path) if timeline_exists else None,
        "timelineUrl": timeline_url,
    }


def _live_frame_ready(job_root: Path, live: dict[str, Any]) -> bool:
    """True only when latest.jpg exists and rollout marked at least one valid frame."""
    frame_path = job_root / JOB_LIVE_FRAME
    if not frame_path.is_file():
        return False
    if live.get("hasValidFrame") is False:
        return False
    if live.get("hasValidFrame") is True:
        return True
    # Legacy jobs without hasValidFrame: require ready frame status.
    frame_status = str(live.get("frameStatus") or "")
    return frame_status == "ready" and int(live.get("frameCount") or 0) > 0


def _apply_live_frame_path_info(paths: dict[str, dict[str, Any]], job_root: Path, live: dict[str, Any]) -> None:
    info = paths.get("liveFrame")
    if not isinstance(info, dict):
        return
    ready = _live_frame_ready(job_root, live)
    info["exists"] = ready
    if ready:
        try:
            info["sizeBytes"] = (job_root / JOB_LIVE_FRAME).stat().st_size
        except OSError:
            info["sizeBytes"] = None
    else:
        info["sizeBytes"] = 0 if info.get("path") else None


def resolve_job_timeline_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated).resolve()
    candidates = (
        [job_root / JOB_EVAL_TIMELINE, job_root / JOB_LIVE_TIMELINE]
        if validated.startswith("ct_eval_")
        else [job_root / JOB_LIVE_TIMELINE]
    )
    for timeline_path in candidates:
        resolved = _assert_job_child(timeline_path, job_root, detail="Invalid timeline path")
        if resolved.is_file():
            return resolved
    return None


def resolve_job_frame_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated).resolve()
    frame_path = _assert_job_child(job_root / JOB_LIVE_FRAME, job_root, detail="Invalid frame path")
    if not frame_path.is_file():
        return None
    live = _read_live_status_json(job_root)
    if not _live_frame_ready(job_root, live):
        return None
    return frame_path


def resolve_job_log_path(job_id: str) -> Path:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated).resolve()
    return _assert_job_child(job_root / "logs" / "run.log", job_root, detail="Invalid log path")


def read_job_log_tail(job_id: str, lines: int = 40) -> str:
    from app.services.imported_eval_bridge import read_imported_eval_log_tail
    from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

    candidate = (job_id or "").strip()
    if is_imported_workspace_eval_job_id(candidate):
        return read_imported_eval_log_tail(candidate, lines=lines)

    log_path = resolve_job_log_path(job_id)
    if not log_path.is_file():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return ""


def run_generate(
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
    save_hdf5: bool,
    output_format: str,
    lerobot_task_instruction: str = "thread the cable through the pole",
    lerobot_robot: str = "Panda",
    lerobot_fps: int = 20,
) -> dict[str, Any]:
    robot = _validate_robot(robot)
    cable_model = _validate_cable_model(cable_model)
    difficulty = _validate_difficulty(difficulty)
    output_format = _validate_output_format(output_format)
    if output_format == "lerobot":
        save_hdf5 = False
    elif output_format == "hdf5":
        save_hdf5 = True

    job_id = make_job_id("ct_gen")
    job_root = _prepare_job_dirs(job_id, include_datasets=True, include_videos=False)

    npz_path = job_root / "datasets" / "dataset.npz"
    hdf5_path = job_root / "datasets" / "dataset.hdf5"
    manifest_path = job_root / "datasets" / "dataset.manifest.json"
    collect_csv = job_root / "results" / "collect.csv"
    failures_json = job_root / "results" / "failures.json"
    log_path = job_root / "logs" / "run.log"

    include_hdf5 = save_hdf5 or output_format == "hdf5"
    include_lerobot = output_format == "lerobot"

    cmd = _build_generate_command(
        job_root,
        episodes=episodes,
        robot=robot,
        cable_model=cable_model,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        save_hdf5=save_hdf5,
        output_format=output_format,
        include_live=False,
        lerobot_task_instruction=lerobot_task_instruction,
        lerobot_robot=lerobot_robot or robot,
        lerobot_fps=lerobot_fps,
    )

    result = _run_subprocess(cmd, log_path=log_path, timeout=TIMEOUT_EXPERT)
    status_value = "completed" if result.returncode == 0 else "failed"

    final_success_rate = _parse_stdout_float(result.stdout, "final_success_rate")
    successful_episodes = _parse_stdout_int(result.stdout, "successful_episodes")

    if status_value == "completed" and manifest_path.is_file():
        _maybe_persist_cable_dataset_manifest(job_root, job_id)
        from app.services.workspace_dataset_list_cache import invalidate_workspace_dataset_list_cache

        invalidate_workspace_dataset_list_cache()

    artifact_paths = _artifact_paths_for_generate(
        job_root,
        include_hdf5=include_hdf5,
        include_lerobot=include_lerobot,
    )
    paths = {key: _path_info(path) for key, path in artifact_paths.items()}

    return {
        "jobId": job_id,
        "taskType": "cable_threading",
        "status": status_value,
        "command": _format_command(cmd),
        "paths": paths,
        "metrics": {
            "finalSuccessRate": final_success_rate,
            "successfulEpisodes": successful_episodes,
            "episodes": episodes,
            "returnCode": result.returncode,
        },
        "stdoutTail": _stdout_tail(result.stdout),
    }


def run_evaluate(
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
    policy: str,
    checkpoint: Optional[str] = None,
    device: str = "",
) -> dict[str, Any]:
    robot = _validate_robot(robot)
    cable_model = _validate_cable_model(cable_model)
    difficulty = _validate_difficulty(difficulty)
    policy = _validate_policy(policy)

    job_id = make_job_id("ct_eval")
    job_root = _prepare_job_dirs(job_id, include_datasets=False, include_videos=False)

    artifact_paths = _artifact_paths_for_eval(job_root)
    eval_csv = artifact_paths["evalCsv"]
    results_json = artifact_paths["resultsJson"]
    failures_json = artifact_paths["failuresJson"]
    log_path = artifact_paths["log"]

    cmd = _build_eval_command(
        job_root,
        episodes=episodes,
        robot=robot,
        cable_model=cable_model,
        difficulty=difficulty,
        horizon=horizon,
        seed=seed,
        policy=policy,
        checkpoint=checkpoint,
        device=device,
    )

    result = _run_subprocess(
        cmd,
        log_path=log_path,
        timeout=_compute_eval_timeout(episodes=episodes, horizon=horizon),
    )
    status_value = "completed" if result.returncode == 0 else "failed"

    results_data = _load_eval_results_json(results_json)
    metrics: dict[str, Any] = {
        "successRate": results_data.get("success_rate"),
        "everSuccessRate": results_data.get("ever_success_rate"),
        "numEpisodes": results_data.get("num_episodes", episodes),
        "aggregate": results_data.get("aggregate"),
        "returnCode": result.returncode,
    }
    if metrics["successRate"] is None:
        metrics["successRate"] = _parse_stdout_float(result.stdout, "success_rate")
    if metrics["numEpisodes"] is None:
        metrics["numEpisodes"] = episodes

    paths = {
        "evalCsv": _path_info(eval_csv),
        "resultsJson": _path_info(results_json),
        "failuresJson": _path_info(failures_json),
        "log": _path_info(log_path),
    }

    return {
        "jobId": job_id,
        "taskType": "cable_threading",
        "status": status_value,
        "command": _format_command(cmd),
        "paths": paths,
        "metrics": metrics,
        "stdoutTail": _stdout_tail(result.stdout),
    }


def run_video(
    *,
    episodes: int,
    robot: str,
    cable_model: str,
    difficulty: str,
    horizon: int,
    seed: int,
) -> dict[str, Any]:
    robot = _validate_robot(robot)
    cable_model = _validate_cable_model(cable_model)
    difficulty = _validate_difficulty(difficulty)

    job_id = make_job_id("ct_vid")
    job_root = _prepare_job_dirs(job_id, include_datasets=False, include_videos=True)

    video_path = job_root / "videos" / "demo.mp4"
    log_path = job_root / "logs" / "run.log"

    cmd = [
        str(PYTHON_BIN),
        str(RUN_PY),
        "video",
        "--episodes",
        str(episodes),
        "--robot",
        robot,
        "--cable-model",
        cable_model,
        "--difficulty",
        difficulty,
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
        "--video-out",
        str(video_path),
    ]

    result = _run_subprocess(cmd, log_path=log_path, timeout=TIMEOUT_VIDEO)
    status_value = "completed" if result.returncode == 0 else "failed"

    video_info = _path_info(video_path)
    return {
        "jobId": job_id,
        "taskType": "cable_threading",
        "status": status_value,
        "command": _format_command(cmd),
        "paths": {
            "video": video_info,
            "log": _path_info(log_path),
        },
        "videoExists": bool(video_info["exists"]),
        "videoSizeBytes": video_info.get("sizeBytes"),
        "stdoutTail": _stdout_tail(result.stdout),
    }


def get_hdf5_trajectory_meta(job_id: str, demo_name: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated)
    from app.services.cable_threading_hdf5_trajectory import get_demo_trajectory_meta, resolve_job_hdf5_path

    hdf5_path = resolve_job_hdf5_path(job_root)
    if hdf5_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dataset.hdf5 not found")
    try:
        return get_demo_trajectory_meta(hdf5_path, demo_name)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def get_hdf5_trajectory_frame(
    job_id: str,
    demo_name: str,
    *,
    camera: str,
    frame_index: int,
    quality: int = 85,
) -> bytes:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated)
    from app.services.cable_threading_hdf5_trajectory import get_demo_frame_jpeg, resolve_job_hdf5_path

    hdf5_path = resolve_job_hdf5_path(job_root)
    if hdf5_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dataset.hdf5 not found")
    try:
        return get_demo_frame_jpeg(
            hdf5_path,
            demo_name,
            camera=camera,
            frame_index=frame_index,
            quality=quality,
        )
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def get_hdf5_trajectory_step(
    job_id: str,
    demo_name: str,
    *,
    step_index: int,
) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated)
    from app.services.cable_threading_hdf5_trajectory import get_demo_step_detail, resolve_job_hdf5_path

    hdf5_path = resolve_job_hdf5_path(job_root)
    if hdf5_path is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dataset.hdf5 not found")
    try:
        return get_demo_step_detail(hdf5_path, demo_name, step_index=step_index)
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
