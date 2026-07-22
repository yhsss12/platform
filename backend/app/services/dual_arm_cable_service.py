from __future__ import annotations

import json
import logging
import os
import re
import secrets
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.platform_paths import is_path_within, platform_paths
from app.services.task_config_metadata import build_job_resource_metadata
from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
INTEGRATION_DIR = PROJECT_ROOT / "integrations" / "DualArmCableManipulation"
OUTPUT_ROOT = platform_paths.runs_root / "dual_arm_cable"
PYTHON_BIN = Path("/home/ubuntu/miniconda3/envs/cable/bin/python")
PLATFORM_RUNNER = INTEGRATION_DIR / "platform_runner.py"
EVAL_POLICY_ROLLOUT = INTEGRATION_DIR / "examples" / "eval_policy_rollout.py"
SCENE_XML = INTEGRATION_DIR / "assets/scenes/dual_fr3_table_scene.xml"

JOB_ID_PATTERN = re.compile(r"^dac_(gen|eval)_\d{8}_\d{6}_[a-f0-9]{4}$")
JOB_STATUS = Path("status.json")
JOB_LOG = Path("logs") / "run.log"
JOB_FRAME = Path("live") / "latest.jpg"
JOB_VIDEO = Path("videos") / "generate.mp4"
JOB_RESULT = Path("results") / "episode_result.json"
JOB_MANIFEST = Path("results") / "episode_manifest.json"

ALLOWED_STRETCH_MODES = frozenset({"ema_jump", "fixed_distance", "fixed_force"})
ALLOWED_RELEASE_MODES = frozenset({"three_phase", "direct_open", "slow_open"})

LOG_TAIL_LINES = 40


@dataclass
class AsyncJobRecord:
    job_id: str
    job_dir: Path
    command: list[str]
    started_at: str
    process: Optional[Any] = None


ASYNC_JOBS: dict[str, AsyncJobRecord] = {}


def _on_generate_job_finished(job_id: str, return_code: int) -> None:
    sync_workspace_job_from_runtime(job_id)
    if return_code != 0:
        return
    try:
        from app.services import dual_arm_cable_dataset_service as dataset_svc

        result = dataset_svc.auto_build_il_dataset_after_generate(job_id)
        log_path = _job_dir(job_id) / JOB_LOG
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[service] auto_il_export={json.dumps(result, ensure_ascii=False)}\n")
    except Exception as exc:
        logger.exception("auto IL export failed for job=%s", job_id)
        log_path = _job_dir(job_id) / JOB_LOG
        try:
            with open(log_path, "a", encoding="utf-8") as log_file:
                log_file.write(f"[service] auto_il_export_error={exc}\n")
        except OSError:
            pass
    sync_workspace_job_from_runtime(job_id)


def _spawn_generate_completion_monitor(job_id: str, proc: subprocess.Popen) -> None:
    def _run() -> None:
        return_code = proc.wait()
        ASYNC_JOBS.pop(job_id, None)
        _on_generate_job_finished(job_id, return_code)

    thread = threading.Thread(
        target=_run,
        daemon=True,
        name=f"dac-gen-monitor-{job_id}",
    )
    thread.start()


def make_job_id(prefix: str = "dac_gen") -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{suffix}"


def _job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


def validate_job_id(job_id: str) -> str:
    candidate = (job_id or "").strip()
    if not JOB_ID_PATTERN.match(candidate):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID format",
        )
    return candidate


def _assert_job_root(job_root: Path) -> Path:
    resolved = job_root.resolve()
    if not is_path_within(resolved, OUTPUT_ROOT / "jobs"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid job path",
        )
    return resolved


def _build_child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["MUJOCO_GL"] = "egl"
    conda_prefix = os.environ.get("CABLE_CONDA_PREFIX", "/home/ubuntu/miniconda3/envs/cable")
    env["CABLE_CONDA_PREFIX"] = conda_prefix

    ld_parts = [
        f"{conda_prefix}/lib",
        f"{conda_prefix}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
    ]
    existing_ld = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(ld_parts + ([existing_ld] if existing_ld else []))
    env["PATH"] = f"{conda_prefix}/bin:{env.get('PATH', '')}"

    pythonpath_parts = [
        str(INTEGRATION_DIR),
        str(INTEGRATION_DIR / "detectron2"),
        str(INTEGRATION_DIR / "detectron2/build/lib.linux-x86_64-cpython-310"),
        str(INTEGRATION_DIR / "perception/Mask2Former"),
        str(INTEGRATION_DIR / "perception/Mask2Former/demo"),
        str(INTEGRATION_DIR / "perception/Mask2Former/mask2former/modeling/pixel_decoder/ops"),
    ]
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = ":".join(pythonpath_parts + ([existing_pp] if existing_pp else []))
    return env


def _read_status_json(job_root: Path) -> dict[str, Any]:
    status_path = job_root / JOB_STATUS
    if not status_path.is_file():
        return {"status": "queued", "message": "job queued"}
    try:
        return json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "running", "message": "status unreadable"}


def _load_episode_result(job_root: Path) -> dict[str, Any]:
    result_path = job_root / JOB_RESULT
    if not result_path.is_file():
        return {}
    try:
        return json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _extract_step_metrics(episode_result: dict[str, Any]) -> dict[str, Any]:
    steps = episode_result.get("steps") or []
    if not steps:
        return {}
    step0 = steps[0] if isinstance(steps[0], dict) else {}
    result = step0.get("result") if isinstance(step0.get("result"), dict) else {}
    return {
        "left_contact": result.get("left_contact"),
        "right_contact": result.get("right_contact"),
        "stretch_reached": result.get("stretch_reached"),
        "sag_m": result.get("sag_m"),
        "span_m": result.get("span_m"),
        "final_sag_m": result.get("final_sag_m"),
        "final_span_m": result.get("final_span_m"),
    }


def start_generate_async(
    *,
    max_cables: int,
    seed: int,
    record: bool,
    headless: bool,
    stretch_mode: str,
    release_mode: str,
    task_config_id: Optional[str] = None,
) -> dict[str, Any]:
    if stretch_mode not in ALLOWED_STRETCH_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"stretchMode must be one of {sorted(ALLOWED_STRETCH_MODES)}",
        )
    if release_mode not in ALLOWED_RELEASE_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"releaseMode must be one of {sorted(ALLOWED_RELEASE_MODES)}",
        )
    if not PYTHON_BIN.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Python interpreter not found: {PYTHON_BIN}",
        )
    if not PLATFORM_RUNNER.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"platform_runner not found: {PLATFORM_RUNNER}",
        )
    if not SCENE_XML.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"scene not found: {SCENE_XML}",
        )

    job_id = make_job_id("dac_gen")
    # New writes always target the external data root; `_job_dir` is reserved
    # for current-first, legacy-compatible reads.
    job_root = OUTPUT_ROOT / "jobs" / job_id
    for sub in ("logs", "live", "videos", "results", "episode"):
        (job_root / sub).mkdir(parents=True, exist_ok=True)

    initial_status = {
        "jobId": job_id,
        "taskType": "dual_arm_cable_manipulation",
        "status": "queued",
        "progress": 0.05,
        "phase": "queued",
        "maxCables": max_cables,
        "succeededCables": 0,
        "videoExists": False,
        "videoPath": None,
        "resultPath": None,
        "message": "任务已创建，等待启动",
    }
    (job_root / JOB_STATUS).write_text(
        json.dumps(initial_status, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cmd = [
        str(PYTHON_BIN),
        str(PLATFORM_RUNNER),
        "--job-id",
        job_id,
        "--output-dir",
        str(job_root),
        "--scene",
        str(SCENE_XML),
        "--max-cables",
        str(max_cables),
        "--seed",
        str(seed),
        "--stretch-mode",
        stretch_mode,
        "--release-mode",
        release_mode,
    ]
    if record:
        cmd.append("--record")
    if headless:
        cmd.append("--headless")

    log_path = job_root / JOB_LOG
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(f"[service] started_at={datetime.now().isoformat()}\n")
    log_file.write(f"[service] command={' '.join(cmd)}\n\n")
    log_file.flush()

    proc = subprocess.Popen(
        cmd,
        cwd=str(INTEGRATION_DIR),
        env=_build_child_env(),
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    _spawn_generate_completion_monitor(job_id, proc)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ASYNC_JOBS[job_id] = AsyncJobRecord(
        job_id=job_id,
        job_dir=job_root,
        command=cmd,
        started_at=started_at,
        process=proc,
    )

    record_workspace_job_start(
        job_id=job_id,
        job_type="generate",
        task_type="dual_arm_cable_manipulation",
        runtime_path=str(job_root),
        runner="platform_runner.py",
        status="running",
        metadata=build_job_resource_metadata(
            task_type="dual_arm_cable_manipulation",
            task_config_id=task_config_id,
            extra={
                "maxCables": max_cables,
                "seed": seed,
                "stretchMode": stretch_mode,
                "releaseMode": release_mode,
                "record": record,
                "headless": headless,
            },
        ),
    )

    api_prefix = f"/api/workspace/dual-arm-cable/jobs/{job_id}"
    return {
        "jobId": job_id,
        "taskType": "dual_arm_cable_manipulation",
        "status": "queued",
        "frameUrl": f"{api_prefix}/frame",
        "statusUrl": f"{api_prefix}/status",
    }


def get_job_status(job_id: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    sync_workspace_job_from_runtime(validated)
    job_root = _assert_job_root(_job_dir(validated))
    payload = _read_status_json(job_root)

    record = ASYNC_JOBS.get(validated)
    if record and record.process and record.process.poll() is not None:
        return_code = record.process.returncode
        current = str(payload.get("status") or "running")
        if current in {"queued", "running"} and not (job_root / JOB_RESULT).is_file():
            payload["status"] = "completed" if return_code == 0 else "failed"
            if return_code != 0 and not payload.get("message"):
                payload["message"] = f"episode process exited with code {return_code}"

    episode_result = _load_episode_result(job_root)
    step_metrics = _extract_step_metrics(episode_result)

    video_path = job_root / JOB_VIDEO
    episode_video_path = job_root / "episode" / "episode_video.mp4"
    result_path = job_root / JOB_RESULT
    frame_path = job_root / JOB_FRAME
    resolved_video = video_path if video_path.is_file() else (
        episode_video_path if episode_video_path.is_file() else None
    )
    video_exists = bool(payload.get("videoExists")) or resolved_video is not None
    live_frame_exists = bool(payload.get("liveFrameExists")) or frame_path.is_file()
    live_frame_seq = payload.get("liveFrameSeq")
    live_frame_updated_at = payload.get("liveFrameUpdatedAt")
    live_frame_source = payload.get("liveFrameSource")
    if not live_frame_updated_at and frame_path.is_file():
        try:
            from datetime import datetime, timezone

            live_frame_updated_at = datetime.fromtimestamp(
                frame_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            live_frame_updated_at = None
    if live_frame_seq is None and frame_path.is_file():
        live_frame_seq = 1
    episode_success = bool(
        payload.get("episodeSuccess")
        if payload.get("episodeSuccess") is not None
        else episode_result.get("episode_success")
    )
    succeeded = int(
        payload.get("succeededCables")
        if payload.get("succeededCables") is not None
        else episode_result.get("num_cables_succeeded") or 0
    )
    max_cables = int(
        payload.get("maxCables")
        if payload.get("maxCables") is not None
        else episode_result.get("max_cables") or 1
    )

    metrics = dict(payload.get("metrics") or {})
    metrics.update(
        {
            "episode_success": episode_success,
            "num_cables_succeeded": succeeded,
            "max_cables": max_cables,
            **{k: v for k, v in step_metrics.items() if v is not None},
        }
    )

    api_prefix = f"/api/workspace/dual-arm-cable/jobs/{validated}"
    return {
        "jobId": validated,
        "taskType": "dual_arm_cable_manipulation",
        "status": str(payload.get("status") or "running"),
        "progress": payload.get("progress"),
        "phase": payload.get("phase"),
        "maxCables": max_cables,
        "succeededCables": succeeded,
        "episodeSuccess": episode_success,
        "videoExists": video_exists,
        "liveFrameExists": live_frame_exists,
        "liveFrameSeq": live_frame_seq,
        "liveFrameUpdatedAt": live_frame_updated_at,
        "liveFrameSource": live_frame_source,
        "currentStep": payload.get("currentStep"),
        "episodeIndex": payload.get("episodeIndex"),
        "videoPath": str(resolved_video) if resolved_video is not None else None,
        "resultPath": str(result_path) if result_path.is_file() else payload.get("resultPath"),
        "runtimePath": str(job_root),
        "logPath": str(job_root / JOB_LOG),
        "manifestPath": str(job_root / "results" / "episode_manifest.json"),
        "message": str(payload.get("message") or ""),
        "metrics": metrics,
        "frameUrl": f"{api_prefix}/frame",
        "videoUrl": f"{api_prefix}/video" if video_exists else None,
        "logUrl": f"{api_prefix}/log",
        "resultUrl": f"{api_prefix}/result" if result_path.is_file() else None,
    }


def resolve_job_frame_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    frame = (job_root / JOB_FRAME).resolve()
    if not str(frame).startswith(str(job_root.resolve())):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid frame path")
    return frame if frame.is_file() else None


def resolve_job_video_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    for rel in (JOB_VIDEO, Path("episode") / "episode_video.mp4"):
        video = (job_root / rel).resolve()
        if not str(video).startswith(str(job_root.resolve())):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid video path")
        if video.is_file():
            return video
    return None


def resolve_job_log_path(job_id: str) -> Path:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    return job_root / JOB_LOG


def resolve_job_result_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    job_root = _assert_job_root(_job_dir(validated))
    result = (job_root / JOB_RESULT).resolve()
    if not str(result).startswith(str(job_root.resolve())):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid result path")
    return result if result.is_file() else None


def read_job_log_tail(job_id: str, lines: int = LOG_TAIL_LINES) -> str:
    log_path = resolve_job_log_path(job_id)
    if not log_path.is_file():
        return ""
    try:
        content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except OSError:
        return ""
