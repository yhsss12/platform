from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.services.dataset_naming import persist_manifest_display_fields
from app.core.platform_paths import platform_paths
from app.services.task_config_metadata import build_job_resource_metadata
from app.services.workspace_job_service import (
    record_workspace_job_start,
    sync_workspace_job_from_runtime,
)

PROJECT_ROOT = platform_paths.project_root
WORKING_DIR = PROJECT_ROOT / "integrations" / "NutAssemblyMimicGen"
OUTPUT_ROOT = platform_paths.runs_root / "nut_assembly"
NUT_ASSEMBLY_MVP_PYTHON = Path("/home/ubuntu/miniconda3/envs/nut-assembly-mvp/bin/python")
PYTHON_BIN = Path("/home/ubuntu/miniconda3/envs/cable-threading-mvp/bin/python")
FALLBACK_PYTHON_BIN = PYTHON_BIN
RUN_PY = WORKING_DIR / "run.py"

ALLOWED_ENV_NAMES = frozenset({"Square_D0", "NutAssembly_D0", "NutAssemblySquare", "NutAssembly"})
NUT_ASSEMBLY_DEFAULT_DEMO_DATASET_ID = "nut_assembly_default_demo_dataset"
TIMEOUT_GENERATE = 1800

_NA_JOB_SUFFIX = r"(?:\d{8}_\d{6}_[a-f0-9]{4}|[a-z0-9_]+)"
JOB_ID_PATTERN = re.compile(rf"^na_gen_{_NA_JOB_SUFFIX}$")

JOB_LIVE_STATUS = Path("live") / "status.json"
JOB_ROOT_STATUS = Path("status.json")
JOB_MANIFEST = Path("manifest.json")
JOB_HDF5 = Path("datasets") / "nut_assembly_generated.hdf5"
JOB_SUMMARY = Path("results") / "generation_summary.json"
JOB_LOG = Path("logs") / "generate.log"
JOB_VIDEO = Path("videos") / "generate.mp4"
LOG_STALL_SECONDS = 600
ENV_CHECK_OUTPUT = OUTPUT_ROOT / "debug" / "mimicgen_env_check.json"
ENV_CHECK_SCRIPT = WORKING_DIR / "env_check.py"
SOURCE_DEMO_STATUS_SCRIPT = WORKING_DIR / "scripts" / "source_demo_status.py"


@dataclass
class AsyncJobRecord:
    job_id: str
    job_dir: Path
    command: list[str]
    started_at: str
    process: Any = None


ASYNC_JOBS: dict[str, AsyncJobRecord] = {}


def make_job_id(prefix: str = "na_gen") -> str:
    suffix = secrets.token_hex(2)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{suffix}"


def _job_dir(job_id: str) -> Path:
    return OUTPUT_ROOT / "jobs" / job_id


def validate_job_id(job_id: str) -> str:
    candidate = (job_id or "").strip()
    if ".." in candidate or "/" in candidate or "\\" in candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job ID format")
    if not JOB_ID_PATTERN.match(candidate):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid job ID format")
    return candidate


def _validate_env_name(env_name: str) -> str:
    normalized = (env_name or "NutAssembly_D0").strip()
    if normalized not in ALLOWED_ENV_NAMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"envName must be one of {sorted(ALLOWED_ENV_NAMES)}",
        )
    return normalized


def _prepare_job_dirs(job_id: str, *, include_videos: bool) -> Path:
    root = _job_dir(job_id)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    (root / "results").mkdir(parents=True, exist_ok=True)
    (root / "live").mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    if include_videos:
        (root / "videos").mkdir(parents=True, exist_ok=True)
    return root


CABLE_THREADING_MVP_DIR = PROJECT_ROOT / "integrations" / "CableThreadingMVP"
CABLE_THREADING_PATH_MARKER = "integrations/CableThreadingMVP"
MIMICGEN_ALT = PROJECT_ROOT / "third_party" / "mimicgen"


def _strip_cable_threading_pythonpath(pythonpath: str) -> str:
    if not pythonpath:
        return ""
    cleaned: list[str] = []
    for part in pythonpath.split(os.pathsep):
        if not part:
            continue
        if CABLE_THREADING_PATH_MARKER in part.replace("\\", "/"):
            continue
        cleaned.append(part)
    return os.pathsep.join(cleaned)


def _resolve_mimicgen_root() -> Optional[Path]:
    for candidate in (MIMICGEN_ALT,):
        if (candidate / "mimicgen" / "scripts" / "generate_dataset.py").is_file():
            return candidate
    return None


def _build_env(*, generation_mode: str = "mimicgen_datagen") -> dict[str, str]:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    if generation_mode == "robosuite_rollout":
        env["CONDA_DEFAULT_ENV"] = "cable-threading-mvp"
        env["PYTHONPATH"] = os.pathsep.join([str(CABLE_THREADING_MVP_DIR), str(WORKING_DIR)])
        return env
    env["CONDA_DEFAULT_ENV"] = "nut-assembly-mvp"
    path_parts = [str(WORKING_DIR)]
    mimicgen_root = _resolve_mimicgen_root()
    if mimicgen_root is not None:
        path_parts.insert(0, str(mimicgen_root))
    inherited = _strip_cable_threading_pythonpath(env.get("PYTHONPATH", ""))
    if inherited:
        path_parts.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(path_parts)
    return env


def _format_command(cmd: list[str]) -> str:
    return " ".join(cmd)


def _path_info(path: Path) -> dict[str, Any]:
    exists = path.is_file()
    size_bytes: Optional[int] = None
    if exists:
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
    return {"path": str(path), "exists": exists, "sizeBytes": size_bytes}


def _artifact_paths(job_root: Path) -> dict[str, Path]:
    return {
        "hdf5": job_root / JOB_HDF5,
        "manifest": job_root / JOB_MANIFEST,
        "summary": job_root / JOB_SUMMARY,
        "log": job_root / JOB_LOG,
        "liveStatus": job_root / JOB_LIVE_STATUS,
        "rootStatus": job_root / JOB_ROOT_STATUS,
        "generateVideo": job_root / JOB_VIDEO,
    }


def _resolve_worker_python(generation_mode: str) -> Path:
    """MimicGen datagen requires nut-assembly-mvp; rollout-only uses cable-threading-mvp."""
    if generation_mode == "robosuite_rollout":
        if FALLBACK_PYTHON_BIN.is_file():
            return FALLBACK_PYTHON_BIN
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"cable-threading-mvp python not found: {FALLBACK_PYTHON_BIN}",
        )
    if not NUT_ASSEMBLY_MVP_PYTHON.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"nut-assembly-mvp python required for mimicgen_datagen: {NUT_ASSEMBLY_MVP_PYTHON}",
        )
    return NUT_ASSEMBLY_MVP_PYTHON


def _build_generate_command(
    job_root: Path,
    *,
    episodes: int,
    seed: int,
    env_name: str,
    output_name: str,
    horizon: int,
    render_video: bool,
    source_demo_path: Optional[str],
    source_demo_selection: Optional[str],
    task_template_id: str,
    generation_mode: str,
    physics_enhancement_config: Optional[Path] = None,
) -> list[str]:
    python_bin = _resolve_worker_python(generation_mode)
    cmd = [
        str(python_bin),
        str(RUN_PY),
        "--job-root",
        str(job_root),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--horizon",
        str(horizon),
        "--env-name",
        env_name,
        "--task-template-id",
        task_template_id,
        "--output-name",
        output_name,
        "--generation-mode",
        generation_mode,
    ]
    if render_video:
        cmd.append("--render-video")
    if source_demo_selection:
        cmd.extend(["--source-demo-selection", source_demo_selection])
    if source_demo_path:
        cmd.extend(["--source-demo-path", source_demo_path])
    if physics_enhancement_config and physics_enhancement_config.is_file():
        cmd.extend(["--physics-enhancement-config", str(physics_enhancement_config)])
    return cmd


def get_pinn_model_status(model_id: str = "nut_assembly_pinn_v1") -> dict[str, Any]:
    try:
        sys.path.insert(0, str(WORKING_DIR))
        from utils.pinn_model_registry import check_pinn_model_availability

        return check_pinn_model_availability(model_id)
    except Exception as exc:
        return {
            "modelId": model_id,
            "available": False,
            "error": str(exc),
        }


def _read_live_status_json(job_root: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for rel in (JOB_LIVE_STATUS, JOB_ROOT_STATUS):
        path = job_root / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                merged.update(data)
        except (OSError, json.JSONDecodeError):
            continue
    return merged


def _persist_status_json(job_root: Path, payload: dict[str, Any]) -> None:
    for rel in (JOB_LIVE_STATUS, JOB_ROOT_STATUS):
        path = job_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _find_important_stats(job_root: Path) -> dict[str, Any]:
    output_dir = job_root / "datasets" / "mimicgen_output"
    if not output_dir.is_dir():
        return {}
    for stats_path in output_dir.rglob("important_stats.json"):
        if not stats_path.is_file():
            continue
        try:
            data = json.loads(stats_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            continue
    return {}


def _apply_important_stats_to_live(live: dict[str, Any], stats: dict[str, Any]) -> None:
    if not stats:
        return
    num_success = stats.get("num_success")
    num_attempts = stats.get("num_attempts")
    num_failures = stats.get("num_failures")
    if num_success is not None:
        live["episodesGenerated"] = int(num_success)
    if num_failures is not None:
        live["datagenFailedTrials"] = int(num_failures)
    elif num_attempts is not None and num_success is not None:
        live["datagenFailedTrials"] = max(int(num_attempts) - int(num_success), 0)
    episodes_requested = int(live.get("episodesRequested") or live.get("episodes") or 0)
    if episodes_requested > 0 and live.get("episodesGenerated") is not None:
        live["progress"] = min(99, int(int(live["episodesGenerated"]) / episodes_requested * 100))


def _reconcile_runtime_status(
    job_root: Path,
    live: dict[str, Any],
    record: Optional[AsyncJobRecord],
) -> bool:
    """Reconcile in-memory/disk status from subprocess, logs, and artifacts. Returns True if mutated."""
    changed = False
    artifact_paths = _artifact_paths(job_root)
    log_path = artifact_paths["log"]
    status_value = str(live.get("status") or "running")

    if record and record.process:
        return_code = record.process.poll()
        if return_code is not None and status_value == "running":
            live["status"] = "completed" if return_code == 0 else "failed"
            if return_code != 0 and not live.get("error"):
                live["error"] = f"generate process exited with code {return_code}"
                live["failureReason"] = live.get("failureReason") or "robosuite_runtime_failed"
            live["stage"] = "completed" if return_code == 0 else "failed"
            changed = True
            status_value = str(live["status"])
    elif status_value == "running" and log_path.is_file():
        try:
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            log_text = ""
        if log_text.startswith("error:") or "Traceback (most recent call last)" in log_text:
            live["status"] = "failed"
            live["stage"] = live.get("stage") or "failed"
            if not live.get("error"):
                for line in reversed(log_text.splitlines()):
                    if line.strip().startswith("error:") or "Error" in line or "ModuleNotFoundError" in line:
                        live["error"] = line.strip().removeprefix("error:").strip() or line.strip()
                        break
                live["error"] = live.get("error") or "generate worker failed"
            if not live.get("traceback") and "Traceback" in log_text:
                live["traceback"] = log_text
            live["failureReason"] = live.get("failureReason") or "robosuite_runtime_failed"
            changed = True
            status_value = "failed"

    if status_value == "running" and log_path.is_file():
        log_mtimes: list[float] = []
        for candidate in (
            log_path,
            job_root / "logs" / "mimicgen_attempt.log",
            job_root / "logs" / "prepare_source.log",
        ):
            if candidate.is_file():
                try:
                    log_mtimes.append(candidate.stat().st_mtime)
                except OSError:
                    continue
        log_mtime = max(log_mtimes) if log_mtimes else None
        if log_mtime is not None and time.time() - log_mtime > LOG_STALL_SECONDS:
            live["status"] = "failed"
            live["stage"] = "stalled"
            live["failureReason"] = live.get("failureReason") or "log_stall_timeout"
            live["error"] = (
                live.get("error")
                or f"generate.log 超过 {LOG_STALL_SECONDS // 60} 分钟无更新，任务可能已卡住"
            )
            changed = True
            status_value = "failed"
        else:
            stats = _find_important_stats(job_root)
            if stats:
                before = (live.get("episodesGenerated"), live.get("progress"))
                _apply_important_stats_to_live(live, stats)
                after = (live.get("episodesGenerated"), live.get("progress"))
                if before != after:
                    live["lastHeartbeatAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
                    changed = True

    if status_value == "completed" and not artifact_paths["hdf5"].is_file():
        live["status"] = "failed"
        live["stage"] = "failed"
        live["error"] = live.get("error") or "completed status but HDF5 missing"
        live["failureReason"] = live.get("failureReason") or "hdf5_missing"
        changed = True
        status_value = "failed"

    hdf5_exists = artifact_paths["hdf5"].is_file()
    summary_exists = artifact_paths["summary"].is_file()
    manifest_exists = artifact_paths["manifest"].is_file()

    if hdf5_exists and not summary_exists and status_value in {"running", "completed"}:
        live["status"] = "partial_success" if status_value == "running" else live.get("status", "partial_success")
        live["stage"] = live.get("stage") or "write_summary"
        live["message"] = live.get("message") or "HDF5 已生成，summary 缺失"
        changed = True

    if status_value == "running" and hdf5_exists and manifest_exists:
        live["status"] = "completed"
        live["stage"] = "completed"
        live["progress"] = 100
        changed = True
        status_value = "completed"

    summary_data = _read_json(artifact_paths["summary"])
    manifest_data = _read_json(artifact_paths["manifest"])
    if status_value == "running" and hdf5_exists and summary_data.get("episodesGenerated") is not None:
        live["status"] = "completed"
        live["stage"] = "completed"
        live["progress"] = 100
        live["message"] = live.get("message") or "数据生成已完成"
        changed = True
        status_value = "completed"
    if summary_data:
        _enrich_status_from_summary(live, summary_data, manifest_data)

    if changed:
        live["updatedAt"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _persist_status_json(job_root, live)
    return changed


def _effective_generation_mode(live: dict[str, Any]) -> Optional[str]:
    mode = live.get("generationMode") or live.get("generationModePreference")
    return str(mode) if mode else None


def _effective_policy_mode(live: dict[str, Any]) -> Optional[str]:
    policy = live.get("policyMode")
    if policy:
        return str(policy)
    gen_mode = _effective_generation_mode(live)
    if gen_mode == "mimicgen_datagen":
        return "mimicgen"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _enrich_status_from_summary(live: dict[str, Any], summary: dict[str, Any], manifest: dict[str, Any]) -> None:
    for key in (
        "generationMode",
        "generationModePreference",
        "policyMode",
        "sourceEnvName",
        "runtimeEnvName",
        "sourceDemoPath",
        "sourceDemoOrigin",
        "sourceDemoOriginReason",
        "sourceDemoMd5",
        "sourceDemoEnvName",
        "hasDatagenInfo",
        "hasObjectPoses",
        "objectPoseKeys",
        "fallbackFrom",
        "fallbackReason",
        "mimicgenFallbackReason",
        "successRate",
        "failureDistribution",
        "videoStatus",
        "videoError",
        "hasEpisodeMetadata",
        "hasObjectPoses",
        "validForTrainingEpisodes",
        "graspSuccessEpisodes",
        "liftSuccessEpisodes",
        "alignmentSuccessEpisodes",
        "insertionSuccessEpisodes",
        "averageGraspAttempts",
        "averageFinalXYError",
        "averageFinalHeightError",
        "hasStageStatistics",
        "physicsEnhancementEnabled",
        "enhancementMode",
        "pinnModelId",
        "pinnBackend",
        "modelLoaded",
        "modelPath",
        "pipelineVersion",
        "candidateMode",
        "mimicgenGeneratedDemos",
        "rawDemoCount",
        "repairedDemoCount",
        "finalDemoCount",
        "pinnCandidateCount",
        "pinnRepairAttempted",
        "pinnRepairSucceeded",
        "pinnValidationSucceeded",
        "pinnRepairValidationRate",
        "enhancementGain",
        "enhancementStatus",
    ):
        if live.get(key) is None and summary.get(key) is not None:
            live[key] = summary.get(key)
        if live.get(key) is None and manifest.get(key) is not None:
            live[key] = manifest.get(key)
    status_value = str(live.get("status") or "running")
    for metric_key in ("episodesGenerated", "datagenFailedTrials", "datagenSuccessRate"):
        summary_val = summary.get(metric_key)
        if summary_val is None:
            continue
        live_val = live.get(metric_key)
        if live_val is None or (status_value == "completed" and live_val != summary_val):
            live[metric_key] = summary_val
    if live.get("successfulEpisodes") is None and summary.get("successEpisodes") is not None:
        live["successfulEpisodes"] = summary.get("successEpisodes")
    if live.get("failedEpisodes") is None and summary.get("failedEpisodes") is not None:
        live["failedEpisodes"] = summary.get("failedEpisodes")
    if live.get("episodes") is None and summary.get("episodesRequested") is not None:
        live["episodes"] = summary.get("episodesRequested")


def resolve_job_video_path(job_id: str) -> Optional[Path]:
    validated = validate_job_id(job_id)
    video_path = _job_dir(validated) / JOB_VIDEO
    return video_path if video_path.is_file() else None


def _maybe_persist_nut_assembly_manifest(job_root: Path, job_id: str) -> dict[str, Any]:
    manifest_path = job_root / JOB_MANIFEST
    if not manifest_path.is_file():
        return {}
    enriched = persist_manifest_display_fields(
        manifest_path,
        task_type="nut_assembly",
        source_job_id=job_id,
        simulator_backend="mujoco",
        dataset_format="hdf5",
    )
    return enriched


def start_generate_async(
    *,
    task_template_id: str,
    episodes: int,
    seed: int,
    render_video: bool,
    source_demo_path: Optional[str],
    source_demo_selection: Optional[str] = None,
    env_name: str,
    output_name: str,
    horizon: int,
    task_config_id: Optional[str] = None,
    generation_mode: str = "mimicgen_datagen",
    generation_path: Optional[str] = None,
    generation_metadata: Optional[dict[str, Any]] = None,
    physics_enhancement: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    env_name = _validate_env_name(env_name)
    generation_mode = generation_mode if generation_mode in {"mimicgen_datagen", "robosuite_rollout"} else "mimicgen_datagen"
    if generation_mode == "mimicgen_datagen":
        selection = (source_demo_selection or "official").strip().lower()
        custom_path = (source_demo_path or "").strip()
        if selection == "custom" and custom_path:
            demo_path = Path(custom_path).expanduser()
            if not demo_path.is_file():
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"源示范数据文件不存在: {custom_path}",
                )
            source_demo_selection = "custom"
            source_demo_path = str(demo_path.resolve())
        elif selection in {"local", "auto"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="MimicGen 扩增需指定 workspace 源示范数据集或系统内置示教数据。",
            )
        else:
            try:
                sys.path.insert(0, str(WORKING_DIR))
                from utils.official_assets import is_official_source_validated, official_source_demo_path

                official_path = official_source_demo_path()
                if not is_official_source_validated() or not official_path.is_file():
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail="未检测到系统示教数据，请先通过专家策略生成种子数据。",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"无法校验官方示教数据: {exc}",
                ) from exc
            source_demo_selection = "official"
            source_demo_path = None

    enhancement_payload = physics_enhancement if isinstance(physics_enhancement, dict) else {}
    if enhancement_payload.get("enabled"):
        if generation_mode != "mimicgen_datagen":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PINN 轨迹修复仅支持 MimicGen datagen 模式。",
            )
        model_id = str(enhancement_payload.get("modelId") or "nut_assembly_pinn_v1")
        pinn_status = get_pinn_model_status(model_id)
        if not pinn_status.get("available"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=pinn_status.get("error") or "未检测到 PINN 修复模型，请先完成模型配置。",
            )

    if task_template_id != "nut_assembly_single_arm":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="taskTemplateId must be nut_assembly_single_arm",
        )

    job_id = make_job_id()
    job_root = _prepare_job_dirs(job_id, include_videos=render_video)
    log_path = job_root / JOB_LOG
    physics_config_path: Optional[Path] = None
    if enhancement_payload.get("enabled"):
        physics_config_path = job_root / "configs" / "physics_enhancement.json"
        physics_config_path.parent.mkdir(parents=True, exist_ok=True)
        physics_config_path.write_text(
            json.dumps(enhancement_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    cmd = _build_generate_command(
        job_root,
        episodes=episodes,
        seed=seed,
        env_name=env_name,
        output_name=output_name,
        horizon=horizon,
        render_video=render_video,
        source_demo_path=source_demo_path,
        source_demo_selection=source_demo_selection,
        task_template_id=task_template_id,
        generation_mode=generation_mode,
        physics_enhancement_config=physics_config_path,
    )

    python_bin = Path(cmd[0])
    if not python_bin.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Python interpreter not found: {python_bin}",
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
        env=_build_env(generation_mode=generation_mode),
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
        "stage": "queued",
        "jobType": "generate",
        "taskType": "nut_assembly",
        "taskTemplateId": task_template_id,
        "jobId": job_id,
        "episode": 0,
        "episodes": episodes,
        "episodesRequested": episodes,
        "episodesGenerated": 0,
        "datagenFailedTrials": 0,
        "progress": 0,
        "seed": seed,
        "sourceEnvName": env_name,
        "runtimeEnvName": None,
        "generationMode": generation_mode,
        "generationModePreference": generation_mode,
        "generationPath": generation_path
        or (generation_metadata.get("generationPath") if generation_metadata else None),
        "policyMode": "mimicgen" if generation_mode == "mimicgen_datagen" else "partial_scripted",
        "physicsEnhancementEnabled": bool(enhancement_payload.get("enabled")),
        "message": "NutAssembly 数据生成任务已启动",
        "startedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "lastHeartbeatAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": None,
    }
    if generation_metadata:
        initial_status.update(generation_metadata)
    _persist_status_json(job_root, initial_status)

    record_workspace_job_start(
        job_id=job_id,
        job_type="generate",
        task_type="nut_assembly",
        runtime_path=str(job_root),
        runner="run.py",
        status="running",
        metadata=build_job_resource_metadata(
            task_type="nut_assembly",
            task_config_id=task_config_id,
            extra={
                "episodes": episodes,
                "envName": env_name,
                "seed": seed,
                "outputName": output_name,
                "generationMode": generation_mode,
                **(generation_metadata or {}),
            },
        ),
    )

    return {
        "jobId": job_id,
        "taskType": "nut_assembly",
        "status": "running",
        "statusUrl": f"/api/workspace/nut-assembly/jobs/{job_id}/status",
        "resultUrl": f"/api/workspace/nut-assembly/jobs/{job_id}/result",
        "command": _format_command(cmd),
    }


def get_generate_job_status(job_id: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    sync_workspace_job_from_runtime(validated)
    job_root = _job_dir(validated)
    live = _read_live_status_json(job_root)

    record = ASYNC_JOBS.get(validated)
    _reconcile_runtime_status(job_root, live, record)

    artifact_paths = _artifact_paths(job_root)
    paths = {key: _path_info(path) for key, path in artifact_paths.items()}

    summary = _read_json(job_root / JOB_SUMMARY)
    manifest = _read_json(job_root / JOB_MANIFEST)
    _enrich_status_from_summary(live, summary, manifest)

    if live.get("startedAt"):
        try:
            started = datetime.fromisoformat(str(live["startedAt"]))
            live["elapsedSeconds"] = max(0, int((datetime.now().astimezone() - started).total_seconds()))
        except ValueError:
            pass

    metrics: dict[str, Any] = {}
    if live.get("successfulEpisodes") is not None:
        metrics["successfulEpisodes"] = live.get("successfulEpisodes")
        metrics["successEpisodes"] = live.get("successfulEpisodes")
    if live.get("failedEpisodes") is not None:
        metrics["failedEpisodes"] = live.get("failedEpisodes")
    if live.get("episodes") is not None:
        metrics["episodes"] = live.get("episodes")
    if live.get("episodesRequested") is not None:
        metrics["episodesRequested"] = live.get("episodesRequested")
    if live.get("episodesGenerated") is not None:
        metrics["episodesGenerated"] = live.get("episodesGenerated")
    if live.get("successRate") is not None:
        metrics["successRate"] = live.get("successRate")
    if live.get("failureDistribution") is not None:
        metrics["failureDistribution"] = live.get("failureDistribution")
    if summary:
        metrics["summary"] = summary

    status_value = str(live.get("status") or "running")
    if status_value in {"partial_success", "stalled"}:
        status_value = "failed" if status_value == "stalled" else status_value
    if status_value == "partial_success":
        status_value = "completed"

    if status_value == "completed":
        _maybe_persist_nut_assembly_manifest(job_root, validated)

    video_exists = paths.get("generateVideo", {}).get("exists", False)
    video_url = f"/api/workspace/nut-assembly/jobs/{validated}/video" if video_exists else None

    log_path = artifact_paths["log"]
    log_last_modified_at: Optional[str] = None
    if log_path.is_file():
        try:
            log_last_modified_at = datetime.fromtimestamp(log_path.stat().st_mtime).astimezone().isoformat(
                timespec="seconds"
            )
        except OSError:
            log_last_modified_at = None

    effective_generation_mode = _effective_generation_mode(live)
    effective_policy_mode = _effective_policy_mode(live)

    episodes_requested = live.get("episodesRequested") or live.get("episodes") or metrics.get("episodes")
    episodes_generated = live.get("episodesGenerated")
    if episodes_generated is None and summary.get("episodesGenerated") is not None:
        episodes_generated = summary.get("episodesGenerated")
    datagen_failed_trials = live.get("datagenFailedTrials")
    if datagen_failed_trials is None and summary.get("datagenFailedTrials") is not None:
        datagen_failed_trials = summary.get("datagenFailedTrials")

    datagen_success_rate = summary.get("datagenSuccessRate")
    if datagen_success_rate is None and live.get("datagenSuccessRate") is not None:
        datagen_success_rate = live.get("datagenSuccessRate")
    if datagen_success_rate is None and episodes_requested and episodes_generated is not None:
        try:
            requested_n = int(episodes_requested)
            generated_n = int(episodes_generated)
            if requested_n > 0:
                datagen_success_rate = round(generated_n / requested_n, 4)
        except (TypeError, ValueError):
            datagen_success_rate = None

    return {
        "jobId": validated,
        "taskType": "nut_assembly",
        "status": status_value,
        "live": live,
        "paths": paths,
        "metrics": metrics,
        "command": _format_command(record.command) if record else "",
        "startedAt": live.get("startedAt") or (record.started_at if record else None),
        "stage": live.get("stage"),
        "progress": live.get("progress"),
        "message": live.get("message"),
        "lastHeartbeatAt": live.get("lastHeartbeatAt"),
        "elapsedSeconds": live.get("elapsedSeconds"),
        "logLastModifiedAt": log_last_modified_at,
        "episodesRequested": episodes_requested,
        "episodesGenerated": episodes_generated,
        "datagenFailedTrials": datagen_failed_trials,
        "datagenSuccessRate": datagen_success_rate,
        "traceback": live.get("traceback"),
        "generationMode": effective_generation_mode,
        "policyMode": effective_policy_mode,
        "sourceEnvName": live.get("sourceEnvName"),
        "runtimeEnvName": live.get("runtimeEnvName"),
        "successRate": live.get("successRate"),
        "failureDistribution": live.get("failureDistribution"),
        "sourceDemoPath": live.get("sourceDemoPath") or summary.get("sourceDemoPath"),
        "sourceDemoOrigin": live.get("sourceDemoOrigin") or summary.get("sourceDemoOrigin"),
        "sourceDemoOriginReason": live.get("sourceDemoOriginReason") or summary.get("sourceDemoOriginReason"),
        "hasDatagenInfo": live.get("hasDatagenInfo") if live.get("hasDatagenInfo") is not None else summary.get("hasDatagenInfo"),
        "hasObjectPoses": live.get("hasObjectPoses") if live.get("hasObjectPoses") is not None else summary.get("hasObjectPoses"),
        "objectPoseKeys": live.get("objectPoseKeys") or summary.get("objectPoseKeys"),
        "fallbackFrom": live.get("fallbackFrom") or summary.get("fallbackFrom"),
        "fallbackReason": live.get("fallbackReason") or summary.get("fallbackReason"),
        "videoUrl": video_url,
        "generateVideoExists": video_exists,
        "hdf5Path": paths.get("hdf5", {}).get("path"),
        "videoPath": str(JOB_VIDEO) if video_exists else None,
    }


def get_generate_job_detail(job_id: str, *, log_tail_lines: int = 20) -> dict[str, Any]:
    """Combined job status + log tail for progress modal polling."""
    status_payload = get_generate_job_status(job_id)
    log_tail = read_job_log_tail(job_id, max_lines=log_tail_lines)
    status_payload["logTail"] = log_tail
    return status_payload


def get_generate_job_result(job_id: str) -> dict[str, Any]:
    validated = validate_job_id(job_id)
    job_root = _job_dir(validated)
    summary_path = job_root / JOB_SUMMARY
    manifest_path = job_root / JOB_MANIFEST
    summary: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            summary = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
    status_payload = get_generate_job_status(validated)
    video_exists = bool(status_payload.get("paths", {}).get("generateVideo", {}).get("exists"))
    video_url = f"/api/workspace/nut-assembly/jobs/{validated}/video" if video_exists else None
    return {
        "jobId": validated,
        "status": status_payload["status"],
        "generationMode": manifest.get("generationMode") or summary.get("generationMode"),
        "policyMode": manifest.get("policyMode") or summary.get("policyMode"),
        "sourceEnvName": manifest.get("sourceEnvName") or summary.get("sourceEnvName"),
        "runtimeEnvName": manifest.get("runtimeEnvName") or summary.get("runtimeEnvName"),
        "successRate": summary.get("successRate"),
        "successEpisodes": summary.get("successEpisodes"),
        "validForTrainingEpisodes": summary.get("validForTrainingEpisodes"),
        "failureDistribution": summary.get("failureDistribution"),
        "sourceDemoPath": manifest.get("sourceDemoPath") or summary.get("sourceDemoPath"),
        "sourceDemoOrigin": manifest.get("sourceDemoOrigin") or summary.get("sourceDemoOrigin"),
        "sourceDemoOriginReason": manifest.get("sourceDemoOriginReason") or summary.get("sourceDemoOriginReason"),
        "hasDatagenInfo": manifest.get("hasDatagenInfo") or summary.get("hasDatagenInfo"),
        "hasObjectPoses": manifest.get("hasObjectPoses") or summary.get("hasObjectPoses"),
        "objectPoseKeys": manifest.get("objectPoseKeys") or summary.get("objectPoseKeys"),
        "fallbackFrom": manifest.get("fallbackFrom") or summary.get("fallbackFrom"),
        "fallbackReason": manifest.get("fallbackReason") or summary.get("fallbackReason"),
        "videoStatus": summary.get("videoStatus") or manifest.get("videoStatus"),
        "videoPath": summary.get("videoPath") or (str(JOB_VIDEO) if video_exists else None),
        "videoUrl": video_url,
        "manifest": manifest,
        "summary": summary,
        "paths": status_payload["paths"],
    }


def read_job_log_tail(job_id: str, *, max_lines: int = 80) -> str:
    validated = validate_job_id(job_id)
    log_path = _job_dir(validated) / JOB_LOG
    if not log_path.is_file():
        return ""
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-max_lines:])


def get_mimicgen_env_status(*, refresh: bool = False) -> dict[str, Any]:
    """Read or refresh MimicGen environment check report."""
    if refresh or not ENV_CHECK_OUTPUT.is_file():
        python_bin = NUT_ASSEMBLY_MVP_PYTHON if NUT_ASSEMBLY_MVP_PYTHON.is_file() else FALLBACK_PYTHON_BIN
        if not ENV_CHECK_SCRIPT.is_file():
            return {"overallOk": False, "error": f"env_check.py not found: {ENV_CHECK_SCRIPT}"}
        if not python_bin.is_file():
            return {
                "overallOk": False,
                "error": "nut-assembly-mvp conda env not installed",
                "nutAssemblyMvpPython": str(NUT_ASSEMBLY_MVP_PYTHON),
                "nutAssemblyMvpExists": False,
            }
        ENV_CHECK_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [str(python_bin), str(ENV_CHECK_SCRIPT)],
            cwd=str(WORKING_DIR),
            env=_build_env(generation_mode="mimicgen_datagen"),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if proc.returncode != 0 and not ENV_CHECK_OUTPUT.is_file():
            return {
                "overallOk": False,
                "error": proc.stderr or proc.stdout or f"env_check exit {proc.returncode}",
            }
    if not ENV_CHECK_OUTPUT.is_file():
        return {"overallOk": False, "error": "mimicgen_env_check.json not found"}
    try:
        data = json.loads(ENV_CHECK_OUTPUT.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"overallOk": False, "error": "invalid env check json"}
    except (OSError, json.JSONDecodeError) as exc:
        return {"overallOk": False, "error": str(exc)}


def get_source_demo_status() -> dict[str, Any]:
    """Return official/local source demo catalog for UI selection."""
    python_bin = NUT_ASSEMBLY_MVP_PYTHON if NUT_ASSEMBLY_MVP_PYTHON.is_file() else FALLBACK_PYTHON_BIN
    if not SOURCE_DEMO_STATUS_SCRIPT.is_file():
        return {"error": f"source_demo_status.py not found: {SOURCE_DEMO_STATUS_SCRIPT}"}
    if not python_bin.is_file():
        return {"error": "nut-assembly-mvp python not available", "nutAssemblyMvpExists": False}
    proc = subprocess.run(
        [str(python_bin), str(SOURCE_DEMO_STATUS_SCRIPT)],
        cwd=str(WORKING_DIR),
        env=_build_env(generation_mode="mimicgen_datagen"),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0 and not proc.stdout.strip():
        return {"error": proc.stderr or proc.stdout or f"exit {proc.returncode}"}
    try:
        data = json.loads(proc.stdout)
        return data if isinstance(data, dict) else {"error": "invalid source demo status json"}
    except json.JSONDecodeError as exc:
        return {"error": str(exc), "raw": proc.stdout[:2000]}
