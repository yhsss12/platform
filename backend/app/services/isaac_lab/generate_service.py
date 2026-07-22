"""Isaac Lab Stack Cube 数据生成 job（isaac_gen_*）。"""

from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from app.core.config import settings
from app.services.isaac_lab.cli_runner import IsaacLabCliRunner
from app.services.isaac_lab.generate_cli import (
    ANNOTATE_DEMOS_SCRIPT,
    DEFAULT_MIMIC_TASK_ID,
    DEFAULT_RECORD_TASK_ID,
    DEFAULT_SCRIPTED_EXPERT_TASK_ID,
    EXPERT_POLICY_SCRIPT_BASENAME,
    GENERATE_DATASET_SCRIPT,
    GENERATION_MODES,
    MIMIC_GENERATE_WITH_LIVE_SCRIPT,
    REPLAY_WITH_LIVE_SCRIPT,
    MimicGenerateCliParams,
    RECORD_DEMOS_SCRIPT,
    SCRIPTED_EXPERT_SCRIPT_BASENAME,
    ScriptedExpertCliParams,
    TeleopRecordCliParams,
    build_annotate_demos_cli_args,
    build_expert_policy_cli_args,
    build_generate_dataset_cli_args,
    build_mimic_generate_with_live_cli_args,
    build_record_demos_cli_args,
    build_replay_with_live_cli_args,
    is_expert_policy_mode,
    normalize_generation_mode,
    resolve_num_envs,
)
from app.services.isaac_lab import isaac_dataset_service as dataset_svc
from integrations.isaac_lab.hdf5_image_obs import build_observation_manifest_fields
from integrations.isaac_lab.trajectory_quality import write_trajectory_quality_report
from app.services.isaac_lab.isaac_job_utils import (
    finalize_status,
    make_isaac_job_id,
    read_json,
    utc_now_iso,
    write_json,
)
from app.services.isaac_lab.isaac_runtime_service import (
    RUNTIME_NOT_CONFIGURED_MSG,
    assert_runtime_configured_for_commands,
)
from app.services.isaac_lab.job_paths import (
    isaac_job_artifacts_dir,
    isaac_job_browser_preview_video_path,
    isaac_job_dataset_path,
    isaac_job_generation_manifest_path,
    isaac_job_live_dir,
    isaac_job_live_frames_dir,
    isaac_job_live_latest_frame_path,
    isaac_job_live_status_path,
    isaac_job_metadata_dir,
    isaac_job_metrics_path,
    isaac_job_preview_video_path,
    isaac_job_root,
    isaac_job_status_path,
    isaac_job_stderr_path,
    isaac_job_stdout_path,
    is_isaac_gen_job_id,
)
from app.services.isaac_lab.replay_cli import (
    HDF5_TO_MP4_SCRIPT,
    build_hdf5_to_mp4_cli_args,
)
from app.services.isaac_lab.live_frame_utils import frame_image_is_valid
from app.services.isaac_lab.paths import (
    PROJECT_ROOT,
    mimic_generate_with_live_isaaclab_relative_path,
    mimic_generate_with_live_platform_script,
    replay_with_live_isaaclab_relative_path,
    replay_with_live_platform_script,
    resolve_expert_policy_platform_script,
    resolve_stack_cube_default_seed,
    expert_policy_isaaclab_relative_path,
)
from app.services.isaac_lab.replay_service import resolve_dataset_file
from app.services.isaac_lab.video_compat import ensure_browser_playable_mp4

logger = logging.getLogger(__name__)

DEFAULT_GENERATE_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEFAULT_REPLAY_TASK_ID = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
DEFAULT_SEED_NOT_CONFIGURED_MSG = (
    "Default 物块堆叠 seed HDF5 is not configured. "
    "Please configure ISAACLAB_STACK_CUBE_DEFAULT_SEED or import a seed demo."
)

_ACTIVE_LOCK = threading.Lock()
_ACTIVE_JOBS: set[str] = set()


def make_isaac_gen_job_id() -> str:
    return make_isaac_job_id("isaac_gen")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _job_paths_payload(job_id: str) -> dict[str, str]:
    root = isaac_job_root(job_id)
    artifacts = isaac_job_artifacts_dir(job_id)
    live_dir = isaac_job_live_dir(job_id)
    return {
        "jobRoot": str(root),
        "stdoutLog": str(isaac_job_stdout_path(job_id)),
        "stderrLog": str(isaac_job_stderr_path(job_id)),
        "statusJson": str(isaac_job_status_path(job_id)),
        "artifactsDir": str(artifacts),
        "datasetHdf5": str(isaac_job_dataset_path(job_id)),
        "generationManifest": str(isaac_job_generation_manifest_path(job_id)),
        "metricsJson": str(isaac_job_metrics_path(job_id)),
        "previewVideo": str(isaac_job_preview_video_path(job_id)),
        "liveDir": str(live_dir),
        "latestFrame": str(isaac_job_live_latest_frame_path(job_id)),
        "liveFramesDir": str(isaac_job_live_frames_dir(job_id)),
        "liveStatusJson": str(isaac_job_live_status_path(job_id)),
    }


def _ensure_live_dirs(job_id: str) -> None:
    live_dir = isaac_job_live_dir(job_id)
    live_dir.mkdir(parents=True, exist_ok=True)
    isaac_job_live_frames_dir(job_id).mkdir(parents=True, exist_ok=True)


def _sync_platform_script(runner: IsaacLabCliRunner, source: Path, relative: str) -> Path:
    if runner.root is None:
        raise RuntimeError("Isaac Lab root is not configured")
    dest = runner.root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _sync_mimic_live_scripts(runner: IsaacLabCliRunner) -> None:
    from app.services.isaac_lab.paths import (
        mimic_generate_with_live_isaaclab_relative_path,
        mimic_generate_with_live_platform_script,
        replay_with_live_isaaclab_relative_path,
        replay_with_live_platform_script,
        state_replay_video_isaaclab_relative_path,
        state_replay_video_platform_script,
    )

    gen_src, gen_ok = mimic_generate_with_live_platform_script()
    replay_src, replay_ok = replay_with_live_platform_script()
    state_src, state_ok = state_replay_video_platform_script()
    if not gen_ok or not replay_ok or not state_ok:
        raise FileNotFoundError("Platform Isaac live preview scripts are missing")
    _sync_platform_script(runner, gen_src, mimic_generate_with_live_isaaclab_relative_path())
    _sync_platform_script(runner, replay_src, replay_with_live_isaaclab_relative_path())
    _sync_platform_script(runner, state_src, state_replay_video_isaaclab_relative_path())
    preview_src = PROJECT_ROOT / "backend/integrations/isaac_lab/scripts/preview_video_utils.py"
    if preview_src.is_file() and runner.root is not None:
        preview_dest = runner.root / "scripts/platform/preview_video_utils.py"
        preview_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preview_src, preview_dest)


def _visual_status_fields(
    job_id: str,
    *,
    parallel_num_envs: int | None = None,
    visual_env_index: int = 0,
    visual_phase: str | None = None,
) -> dict[str, Any]:
    parallel = parallel_num_envs
    if parallel is None:
        meta = read_json(isaac_job_metadata_dir(job_id) / "request.json")
        parallel = int(meta.get("numEnvs") or 1)
    parallel = max(1, int(parallel))
    env_index = max(0, int(visual_env_index))
    mode = "single_env" if parallel <= 1 else "single_env"
    if visual_phase == "replay_preview":
        mode = "replay_preview"
    return {
        "visualNumEnvs": 1,
        "parallelNumEnvs": parallel,
        "visualMode": mode,
        "visualEnvIndex": env_index,
    }


def _live_visual_fields(job_id: str, *, enable_cameras: bool, visual_phase: str | None = None) -> dict[str, Any]:
    latest = isaac_job_live_latest_frame_path(job_id)
    preview = isaac_job_preview_video_path(job_id)
    live_file_exists = latest.is_file() and latest.stat().st_size > 64
    live_valid = live_file_exists and frame_image_is_valid(latest)
    preview_available = preview.is_file() and preview.stat().st_size > 0
    phase = visual_phase
    if phase is None:
        if live_valid:
            phase = "live_generate"
        elif preview_available:
            phase = "replay_preview"
        elif enable_cameras:
            phase = "none"
        else:
            phase = "none"
    return {
        "enableCameras": enable_cameras,
        "liveFrameAvailable": live_valid,
        "liveFrameBlack": live_file_exists and not live_valid,
        "latestFramePath": str(latest) if live_valid else None,
        "previewVideoAvailable": preview_available,
        "visualPhase": phase,
        **_visual_status_fields(job_id, visual_phase=phase),
    }


def _run_replay_preview_with_live(
    runner: IsaacLabCliRunner,
    job_id: str,
    *,
    task_id: str,
    dataset_file: Path,
    headless: bool,
    enable_cameras: bool,
    device: str,
    timeout: int,
) -> tuple[bool, str | None]:
    if not enable_cameras:
        return False, "enable_cameras=false"
    live_dir = isaac_job_live_dir(job_id)
    preview_path = isaac_job_preview_video_path(job_id)
    latest = isaac_job_live_latest_frame_path(job_id)
    if (
        preview_path.is_file()
        and preview_path.stat().st_size > 0
        and frame_image_is_valid(latest)
    ):
        return True, None
    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "status": "running",
            "phase": "replay_preview",
            "visualPhase": "replay_preview",
            "message": "Running replay preview for live frames…",
            "paths": _job_paths_payload(job_id),
            **_preview_status_payload(job_id, "generating"),
        },
    )
    result = runner.run_to_files(
        REPLAY_WITH_LIVE_SCRIPT,
        *build_replay_with_live_cli_args(
            task_id=task_id,
            dataset_file=dataset_file,
            live_frame_dir=live_dir,
            headless=headless,
            enable_cameras=enable_cameras,
            device=device,
            live_status_out=isaac_job_live_status_path(job_id),
            preview_video_out=preview_path,
        ),
        stdout_path=isaac_job_artifacts_dir(job_id) / "replay_preview.stdout.log",
        stderr_path=isaac_job_artifacts_dir(job_id) / "replay_preview.stderr.log",
        timeout=min(timeout, 1800),
    )
    latest_ok = isaac_job_live_latest_frame_path(job_id).is_file()
    preview_ok = preview_path.is_file() and preview_path.stat().st_size > 0
    if result.returncode != 0 or result.timed_out:
        return False, f"replay_preview exit={result.returncode} timed_out={result.timed_out}"
    if not latest_ok:
        return False, "replay_preview completed but latest.jpg missing"
    note = None if preview_ok else "preview.mp4 not produced during replay preview"
    return True, note


def _count_hdf5_episodes(dataset_path: Path) -> int:
    try:
        import h5py
    except ImportError:
        return 0
    try:
        with h5py.File(dataset_path, "r") as handle:
            if "data" in handle and hasattr(handle["data"], "keys"):
                return len(list(handle["data"].keys()))
    except Exception as exc:
        logger.debug("episode count skipped for %s: %s", dataset_path, exc)
    return 0


def _attach_trajectory_quality(
    *,
    job_id: str,
    dataset_path: Path,
    generation_mode: str,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Offline QA on generated HDF5; writes artifacts/trajectory_quality_report.json."""
    if not dataset_path.is_file():
        return {}
    report_path = isaac_job_artifacts_dir(job_id) / "trajectory_quality_report.json"
    behavior_report: dict[str, Any] | None = None
    if generation_mode == "expert_policy":
        behavior_path = isaac_job_artifacts_dir(job_id) / "expert_policy_behavior_report.json"
        if behavior_path.is_file():
            try:
                behavior_report = json.loads(behavior_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("behavior report read failed job=%s: %s", job_id, exc)
    try:
        report = write_trajectory_quality_report(
            dataset_path,
            report_path,
            generation_mode=generation_mode,
            behavior_report=behavior_report,
        )
    except Exception as exc:
        logger.warning("trajectory quality report failed job=%s: %s", job_id, exc)
        return {}
    metrics["trajectoryQuality"] = {
        "qualityStatus": report.get("qualityStatus"),
        "qualityDisplayTier": report.get("qualityDisplayTier"),
        "qualityDisplayLabel": report.get("qualityDisplayLabel"),
        "lowQualityDemoCount": report.get("lowQualityDemoCount"),
        "maxActionDelta": report.get("maxActionDelta"),
        "meanActionDelta": report.get("meanActionDelta"),
    }
    manifest["qualityStatus"] = report.get("qualityStatus")
    manifest["qualityWarnings"] = report.get("qualityWarnings") or []
    manifest["qualityDisplayTier"] = report.get("qualityDisplayTier")
    manifest["qualityDisplayLabel"] = report.get("qualityDisplayLabel")
    manifest["qualityDisplayHint"] = report.get("qualityDisplayHint")
    manifest["qualityDisplaySeverity"] = report.get("qualityDisplaySeverity")
    manifest["qualityDisplayDescription"] = report.get("qualityDisplayDescription")
    manifest["qualityDisplayRecommendation"] = report.get("qualityDisplayRecommendation")
    manifest["trajectoryQualityReport"] = str(report_path)
    return report


def _finalize_preview_browser_mp4(job_id: str) -> tuple[bool, Optional[str]]:
    """Transcode preview.mp4 → preview.browser.mp4 when needed."""
    preview_path = isaac_job_preview_video_path(job_id)
    if not preview_path.is_file() or preview_path.stat().st_size <= 0:
        return False, "preview.mp4 missing"
    browser_path = isaac_job_browser_preview_video_path(job_id)
    playable, note = ensure_browser_playable_mp4(preview_path)
    if playable is None:
        return False, note or "browser transcode failed"
    if playable.resolve() != browser_path.resolve() and playable.is_file():
        try:
            if not browser_path.is_file() or browser_path.stat().st_size <= 0:
                shutil.copy2(playable, browser_path)
        except OSError as exc:
            logger.warning("copy browser preview failed job=%s: %s", job_id, exc)
    return True, note


PREVIEW_REL_VIDEO = "artifacts/preview.mp4"
PREVIEW_REL_BROWSER = "artifacts/preview.browser.mp4"


def _preview_status_payload(
    job_id: str,
    preview_status: str,
    *,
    video_note: Optional[str] = None,
) -> dict[str, Any]:
    preview_abs = isaac_job_preview_video_path(job_id)
    payload: dict[str, Any] = {
        "previewStatus": preview_status,
        "previewVideoPath": PREVIEW_REL_VIDEO,
        "browserPreviewVideoPath": PREVIEW_REL_BROWSER,
    }
    if preview_status == "completed":
        payload["videoAvailable"] = preview_abs.is_file() and preview_abs.stat().st_size > 0
    if video_note is not None:
        payload["videoNote"] = video_note
    return payload


def _sync_preview_status(
    job_id: str,
    *,
    preview_status: str,
    video_note: Optional[str] = None,
    sync_manifest: bool = True,
) -> None:
    """Keep previewStatus / paths aligned in status.json and generation_manifest.json."""
    patch = _preview_status_payload(job_id, preview_status, video_note=video_note)
    current = read_json(isaac_job_status_path(job_id)) or {"jobId": job_id}
    current.update(patch)
    finalize_status(job_id, current)
    if sync_manifest:
        manifest_path = isaac_job_generation_manifest_path(job_id)
        manifest = read_json(manifest_path) if manifest_path.is_file() else {"jobId": job_id}
        manifest.update(patch)
        if preview_status == "completed":
            manifest["videoAvailable"] = patch.get("videoAvailable", False)
        write_json(manifest_path, manifest)


def _validate_generated_dataset(dataset_path: Path) -> tuple[bool, int]:
    if not dataset_path.is_file():
        return False, 0
    try:
        if dataset_path.stat().st_size <= 0:
            return False, 0
    except OSError:
        return False, 0
    episode_count = _count_hdf5_episodes(dataset_path)
    return episode_count > 0, episode_count


def _sync_expert_policy_to_isaaclab(runner: IsaacLabCliRunner) -> Path:
    """将平台专家策略脚本同步到 ISAACLAB_ROOT 供 isaaclab.sh 调用。"""
    source, exists = resolve_expert_policy_platform_script()
    if not exists:
        raise FileNotFoundError(
            f"Platform expert policy script not found: {source}"
        )
    if runner.root is None:
        raise RuntimeError("Isaac Lab root is not configured")
    relative = expert_policy_isaaclab_relative_path()
    dest = runner.root / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    return dest


def _sync_scripted_expert_to_isaaclab(runner: IsaacLabCliRunner) -> Path:
    """Deprecated alias."""
    return _sync_expert_policy_to_isaaclab(runner)


def _try_generate_preview_video(
    runner: IsaacLabCliRunner,
    *,
    dataset_file: Path,
    artifacts_dir: Path,
    timeout: int,
) -> tuple[bool, Optional[Path], Optional[str]]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts_dir / "hdf5_to_mp4.stdout.log"
    stderr_path = artifacts_dir / "hdf5_to_mp4.stderr.log"
    try:
        result = runner.run_to_files(
            HDF5_TO_MP4_SCRIPT,
            *build_hdf5_to_mp4_cli_args(input_file=dataset_file, output_dir=artifacts_dir),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            timeout=min(timeout, 600),
        )
    except Exception as exc:
        return False, None, f"hdf5_to_mp4 failed: {exc}"

    if result.returncode != 0 or result.timed_out:
        return False, None, f"hdf5_to_mp4 exit={result.returncode} timed_out={result.timed_out}"

    target = isaac_job_preview_video_path(artifacts_dir.parent.name)
    target.parent.mkdir(parents=True, exist_ok=True)
    mp4_files = sorted(artifacts_dir.glob("*.mp4"))
    if not mp4_files:
        return False, None, "hdf5_to_mp4 completed but no mp4 was produced"
    if mp4_files[0].resolve() != target.resolve():
        shutil.copy2(mp4_files[0], target)
    return True, target, None


def _validate_generation_request(
    *,
    generation_mode: str,
) -> None:
    mode = normalize_generation_mode(generation_mode)
    if mode not in GENERATION_MODES and (generation_mode or "").strip() not in GENERATION_MODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported generationMode: {mode}",
        )
    if mode == "replay_seed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="generationMode=replay_seed is not implemented yet",
        )


def resolve_generation_seed(
    *,
    seed_dataset_file: Optional[str] = None,
    seed_dataset_id: Optional[str] = None,
) -> tuple[Path, str, Optional[str]]:
    """解析 Mimic seed：registry > manual path > platform default seed。"""
    dataset_id = (seed_dataset_id or "").strip()
    if dataset_id:
        dataset = dataset_svc.get_isaac_dataset(dataset_id)
        path = resolve_dataset_file(str(dataset["datasetFile"]))
        return path, "dataset_registry", dataset_id

    manual = (seed_dataset_file or "").strip()
    if manual:
        path = resolve_dataset_file(manual)
        return path, "manual_path", None

    default_path, exists = resolve_stack_cube_default_seed()
    if exists:
        return default_path, "default_seed", None

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=DEFAULT_SEED_NOT_CONFIGURED_MSG,
    )


def _execute_mimic_auto(
    runner: IsaacLabCliRunner,
    job_id: str,
    *,
    mimic_task_id: str,
    seed_path: Path,
    seed_source: str,
    num_demos: int,
    num_envs: int,
    headless: bool,
    enable_cameras: bool,
    device: str,
    timeout: int,
    request_video: bool,
) -> tuple[bool, Path, dict[str, Any]]:
    artifacts = isaac_job_artifacts_dir(job_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    stdout_path = isaac_job_stdout_path(job_id)
    stderr_path = isaac_job_stderr_path(job_id)

    seed_copy = artifacts / "seed.hdf5"
    if seed_path.resolve() != seed_copy.resolve():
        shutil.copy2(seed_path, seed_copy)

    annotated_path = artifacts / "annotated_dataset.hdf5"
    output_path = isaac_job_dataset_path(job_id)

    _append_log(stdout_path, f"=== annotate_demos phase ({utc_now_iso()}) ===")
    annotate_args = build_annotate_demos_cli_args(
        mimic_task_id=mimic_task_id,
        input_file=seed_copy,
        output_file=annotated_path,
        headless=headless,
        enable_cameras=enable_cameras,
        device=device,
    )
    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "status": "running",
            "phase": "annotate",
            "message": "Running annotate_demos.py…",
            "paths": _job_paths_payload(job_id),
        },
    )
    annotate_result = runner.run_to_files(
        ANNOTATE_DEMOS_SCRIPT,
        *annotate_args,
        stdout_path=artifacts / "annotate.stdout.log",
        stderr_path=artifacts / "annotate.stderr.log",
        timeout=timeout,
    )
    if annotate_result.stdout_path.is_file():
        _append_log(stdout_path, annotate_result.stdout_path.read_text(encoding="utf-8", errors="replace"))
    if annotate_result.stderr_path.is_file():
        _append_log(stderr_path, annotate_result.stderr_path.read_text(encoding="utf-8", errors="replace"))

    if annotate_result.returncode != 0 or annotate_result.timed_out or not annotated_path.is_file():
        return False, output_path, {
            "phase": "annotate",
            "exitCode": annotate_result.returncode,
            "timedOut": annotate_result.timed_out,
            "message": "annotate_demos failed",
        }

    mimic_params = MimicGenerateCliParams(
        mimic_task_id=mimic_task_id,
        seed_dataset_file=seed_copy,
        annotated_dataset_file=annotated_path,
        output_dataset_file=output_path,
        num_demos=num_demos,
        num_envs=num_envs,
        headless=headless,
        enable_cameras=enable_cameras,
        device=device,
    )
    _ensure_live_dirs(job_id)
    _sync_mimic_live_scripts(runner)
    live_dir = isaac_job_live_dir(job_id)
    preview_path = isaac_job_preview_video_path(job_id)
    _append_log(stdout_path, f"=== generate_dataset phase ({utc_now_iso()}) ===")
    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "status": "running",
            "phase": "generate",
            "visualPhase": "live_generate" if enable_cameras else "none",
            "message": "Running mimic generate with live preview…" if enable_cameras else "Running generate_dataset.py…",
            "paths": _job_paths_payload(job_id),
            **_live_visual_fields(job_id, enable_cameras=enable_cameras, visual_phase="live_generate" if enable_cameras else "none"),
        },
    )
    generate_result = runner.run_to_files(
        MIMIC_GENERATE_WITH_LIVE_SCRIPT,
        *build_mimic_generate_with_live_cli_args(
            mimic_params,
            live_frame_dir=live_dir,
            live_status_out=isaac_job_live_status_path(job_id),
            preview_video_out=preview_path if request_video or enable_cameras else None,
        ),
        stdout_path=artifacts / "generate.stdout.log",
        stderr_path=artifacts / "generate.stderr.log",
        timeout=timeout,
    )
    if generate_result.stdout_path.is_file():
        _append_log(stdout_path, generate_result.stdout_path.read_text(encoding="utf-8", errors="replace"))
    if generate_result.stderr_path.is_file():
        _append_log(stderr_path, generate_result.stderr_path.read_text(encoding="utf-8", errors="replace"))

    success = (
        generate_result.returncode == 0
        and not generate_result.timed_out
        and output_path.is_file()
    )
    metrics: dict[str, Any] = {
        "generationMode": "mimic_auto",
        "requestedDemos": num_demos,
        "episodeCount": _count_hdf5_episodes(output_path) if output_path.is_file() else 0,
        "annotateExitCode": annotate_result.returncode,
        "generateExitCode": generate_result.returncode,
        "annotateTimedOut": annotate_result.timed_out,
        "generateTimedOut": generate_result.timed_out,
        "finishedAt": utc_now_iso(),
    }
    write_json(isaac_job_metrics_path(job_id), metrics)

    video_available = isaac_job_preview_video_path(job_id).is_file()
    video_note: Optional[str] = None
    if success and enable_cameras and not frame_image_is_valid(isaac_job_live_latest_frame_path(job_id)):
        replay_ok, replay_note = _run_replay_preview_with_live(
            runner,
            job_id,
            task_id=DEFAULT_REPLAY_TASK_ID,
            dataset_file=output_path,
            headless=headless,
            enable_cameras=enable_cameras,
            device=device,
            timeout=timeout,
        )
        if replay_ok:
            video_available = isaac_job_preview_video_path(job_id).is_file()
            video_note = replay_note
        else:
            video_note = replay_note or "replay preview failed"
    elif success and request_video and not video_available:
        video_available, _, video_note = _try_generate_preview_video(
            runner,
            dataset_file=output_path,
            artifacts_dir=artifacts,
            timeout=timeout,
        )
    elif success and not enable_cameras:
        video_note = "当前任务未启用相机输出，无法显示实时画面。"

    manifest = {
        "jobId": job_id,
        "kind": "generate_dataset",
        "generationMode": "mimic_auto",
        "mimicTaskId": mimic_task_id,
        "seedSource": seed_source,
        "seedDatasetFile": str(seed_path.resolve()),
        "seedCopyFile": str(seed_copy),
        "annotatedDatasetFile": str(annotated_path),
        "outputDatasetFile": str(output_path),
        "numDemos": num_demos,
        "numEnvs": num_envs,
        "metrics": metrics,
        "videoAvailable": video_available,
        "videoNote": video_note,
        "finishedAt": utc_now_iso(),
    }
    if success and output_path.is_file():
        _attach_trajectory_quality(
            job_id=job_id,
            dataset_path=output_path,
            generation_mode="mimic_auto",
            metrics=metrics,
            manifest=manifest,
        )
        manifest["metrics"] = metrics
    write_json(isaac_job_generation_manifest_path(job_id), manifest)
    write_json(isaac_job_metrics_path(job_id), metrics)

    return success, output_path, {
        "phase": "generate",
        "exitCode": generate_result.returncode,
        "timedOut": generate_result.timed_out,
        "message": "generate_dataset completed" if success else "generate_dataset failed",
        "metrics": metrics,
        "videoAvailable": video_available,
        "videoNote": video_note,
    }


def _execute_teleop_record(
    runner: IsaacLabCliRunner,
    job_id: str,
    *,
    task_id: str,
    num_demos: int,
    headless: bool,
    enable_cameras: bool,
    device: str,
    timeout: int,
    request_video: bool,
) -> tuple[bool, Path, dict[str, Any]]:
    artifacts = isaac_job_artifacts_dir(job_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    stdout_path = isaac_job_stdout_path(job_id)
    stderr_path = isaac_job_stderr_path(job_id)
    output_path = isaac_job_dataset_path(job_id)

    record_params = TeleopRecordCliParams(
        task_id=task_id,
        dataset_file=output_path,
        num_demos=num_demos,
        headless=headless,
        enable_cameras=enable_cameras,
        device=device,
    )
    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "status": "running",
            "phase": "teleop_record",
            "message": "Running record_demos.py (requires human teleoperation)…",
            "paths": _job_paths_payload(job_id),
        },
    )
    result = runner.run_to_files(
        RECORD_DEMOS_SCRIPT,
        *build_record_demos_cli_args(record_params),
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        timeout=timeout,
    )
    success = result.returncode == 0 and not result.timed_out and output_path.is_file()
    metrics = {
        "generationMode": "teleop_record",
        "requestedDemos": num_demos,
        "episodeCount": _count_hdf5_episodes(output_path) if output_path.is_file() else 0,
        "exitCode": result.returncode,
        "timedOut": result.timed_out,
        "finishedAt": utc_now_iso(),
    }
    write_json(isaac_job_metrics_path(job_id), metrics)
    video_available = False
    video_note: Optional[str] = None
    if success and request_video:
        video_available, _, video_note = _try_generate_preview_video(
            runner,
            dataset_file=output_path,
            artifacts_dir=artifacts,
            timeout=timeout,
        )
    write_json(
        isaac_job_generation_manifest_path(job_id),
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "generationMode": "teleop_record",
            "outputDatasetFile": str(output_path),
            "metrics": metrics,
            "videoAvailable": video_available,
            "finishedAt": utc_now_iso(),
        },
    )
    return success, output_path, {
        "phase": "teleop_record",
        "exitCode": result.returncode,
        "timedOut": result.timed_out,
        "message": "teleop record completed" if success else "teleop record failed",
        "metrics": metrics,
        "videoAvailable": video_available,
        "videoNote": video_note,
    }


def _attach_observation_metadata(
    *,
    dataset_path: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    """Enrich generation manifest with HDF5 observation/camera metadata."""
    if not dataset_path.is_file():
        return
    obs_fields = build_observation_manifest_fields(dataset_path)
    manifest.update(obs_fields)
    metrics.update(
        {
            "observationType": obs_fields.get("observationType"),
            "cameraKeys": obs_fields.get("cameraKeys") or [],
            "imageKeys": obs_fields.get("imageKeys") or [],
            "imageShape": obs_fields.get("imageShape"),
            "obsKeys": obs_fields.get("obsKeys") or [],
        }
    )
    quality = dict(manifest.get("quality") or {})
    quality.update(obs_fields.get("quality") or {})
    manifest["quality"] = quality


def _execute_expert_policy(
    runner: IsaacLabCliRunner,
    job_id: str,
    *,
    task_id: str,
    num_demos: int,
    seed: int,
    max_attempts: int,
    headless: bool,
    enable_cameras: bool,
    device: str,
    timeout: int,
    request_video: bool,
    record_camera_obs: bool = True,
    image_resolution: int = 128,
    include_wrist_camera: bool = False,
) -> tuple[bool, Path, dict[str, Any]]:
    artifacts = isaac_job_artifacts_dir(job_id)
    artifacts.mkdir(parents=True, exist_ok=True)
    stdout_path = isaac_job_stdout_path(job_id)
    stderr_path = isaac_job_stderr_path(job_id)
    output_path = isaac_job_dataset_path(job_id)

    _sync_expert_policy_to_isaaclab(runner)
    script_relative = expert_policy_isaaclab_relative_path()

    expert_params = ScriptedExpertCliParams(
        task_id=task_id,
        dataset_file=output_path,
        num_demos=num_demos,
        seed=seed,
        max_attempts=max_attempts,
        headless=headless,
        enable_cameras=enable_cameras,
        device=device,
        record_camera_obs=record_camera_obs,
        image_resolution=image_resolution,
        include_wrist_camera=include_wrist_camera,
    )

    finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "status": "running",
            "phase": "expert_policy",
            "message": f"Running {EXPERT_POLICY_SCRIPT_BASENAME}…",
            "paths": _job_paths_payload(job_id),
        },
    )
    _append_log(stdout_path, f"=== expert_policy phase ({utc_now_iso()}) ===")
    result = runner.run_to_files(
        script_relative,
        *build_expert_policy_cli_args(expert_params),
        stdout_path=artifacts / "expert_policy.stdout.log",
        stderr_path=artifacts / "expert_policy.stderr.log",
        timeout=timeout,
    )
    if result.stdout_path.is_file():
        _append_log(stdout_path, result.stdout_path.read_text(encoding="utf-8", errors="replace"))
    if result.stderr_path.is_file():
        _append_log(stderr_path, result.stderr_path.read_text(encoding="utf-8", errors="replace"))

    dataset_ok, episode_count = _validate_generated_dataset(output_path)
    success = (
        result.returncode == 0
        and not result.timed_out
        and dataset_ok
    )
    metrics: dict[str, Any] = {
        "generationMode": "expert_policy",
        "requestedDemos": num_demos,
        "episodeCount": episode_count,
        "exitCode": result.returncode,
        "timedOut": result.timed_out,
        "finishedAt": utc_now_iso(),
    }
    write_json(isaac_job_metrics_path(job_id), metrics)

    video_available = False
    video_note: Optional[str] = None
    preview_status: Optional[str] = None
    if success and (request_video or enable_cameras):
        _sync_preview_status(job_id, preview_status="pending")
        _sync_mimic_live_scripts(runner)
        _sync_preview_status(job_id, preview_status="generating")
        replay_ok, replay_note = _run_replay_preview_with_live(
            runner,
            job_id,
            task_id=task_id,
            dataset_file=output_path,
            headless=headless,
            enable_cameras=True,
            device=device,
            timeout=timeout,
        )
        preview_path = isaac_job_preview_video_path(job_id)
        video_available = preview_path.is_file() and preview_path.stat().st_size > 0
        if replay_ok and video_available:
            browser_ok, browser_note = _finalize_preview_browser_mp4(job_id)
            preview_status = "completed"
            video_note = (
                browser_note
                if browser_note and browser_note not in {"transcoded", "transcoded_cache"}
                else replay_note
            )
            _sync_preview_status(job_id, preview_status="completed", video_note=video_note)
        else:
            preview_status = "failed"
            video_note = replay_note or "preview 生成失败，可重新生成回放"
            _sync_preview_status(job_id, preview_status="failed", video_note=video_note)

    manifest = read_json(isaac_job_generation_manifest_path(job_id)) or {
        "jobId": job_id,
        "kind": "generate_dataset",
        "generationMode": "expert_policy",
    }
    manifest.update(
        {
            "generationMode": "expert_policy",
            "taskId": task_id,
            "seedSource": None,
            "expertScript": EXPERT_POLICY_SCRIPT_BASENAME,
            "displayName": "物块堆叠专家策略",
            "requestedDemos": num_demos,
            "outputDatasetFile": str(output_path),
            "numDemos": num_demos,
            "seed": seed,
            "maxAttempts": max_attempts,
            "metrics": metrics,
            "videoAvailable": video_available,
            "videoNote": video_note,
            "finishedAt": utc_now_iso(),
        }
    )
    if preview_status:
        manifest.update(_preview_status_payload(job_id, preview_status, video_note=video_note))
    if success and output_path.is_file():
        _attach_trajectory_quality(
            job_id=job_id,
            dataset_path=output_path,
            generation_mode="expert_policy",
            metrics=metrics,
            manifest=manifest,
        )
        _attach_observation_metadata(
            dataset_path=output_path,
            manifest=manifest,
            metrics=metrics,
        )
        manifest["metrics"] = metrics
    write_json(isaac_job_generation_manifest_path(job_id), manifest)

    if result.returncode != 0 or result.timed_out:
        message = "expert policy CLI failed"
    elif not output_path.is_file():
        message = "expert policy completed but dataset.hdf5 is missing"
    elif episode_count <= 0:
        message = "expert policy completed but HDF5 has no episodes"
    else:
        message = "expert policy completed"

    phase_info: dict[str, Any] = {
        "phase": "expert_policy",
        "exitCode": result.returncode,
        "timedOut": result.timed_out,
        "message": message,
        "metrics": metrics,
        "videoAvailable": video_available,
        "videoNote": video_note,
    }
    if preview_status:
        phase_info.update(_preview_status_payload(job_id, preview_status, video_note=video_note))
    return success, output_path, phase_info


_execute_scripted_expert = _execute_expert_policy


def _execute_generate_job(job_id: str, request: dict[str, Any]) -> None:
    runner = IsaacLabCliRunner.from_settings()
    timeout = int(getattr(settings, "ISAACLAB_GENERATE_TIMEOUT", 7200) or 7200)
    started_at = read_json(isaac_job_status_path(job_id)).get("startedAt")

    generation_mode = str(request.get("generationMode") or "expert_policy")
    task_id = str(request.get("taskId") or DEFAULT_GENERATE_TASK_ID)
    mimic_task_id = str(request.get("mimicTaskId") or DEFAULT_MIMIC_TASK_ID)
    num_demos = int(request.get("numDemos") or 1)
    num_envs = int(request.get("numEnvs") or resolve_num_envs(num_demos))
    headless = bool(request.get("headless", True))
    enable_cameras = bool(request.get("enableCameras", True))
    record_camera_obs = bool(request.get("recordCameraObs", True))
    image_resolution = int(request.get("imageResolution") or 128)
    include_wrist_camera = bool(request.get("includeWristCamera", False))
    request_video = bool(request.get("video", True))
    device = str(request.get("device") or "cpu")
    dataset_name = str(request.get("datasetName") or job_id)
    max_attempts = int(request.get("maxAttempts") or 0)

    dataset_id: Optional[str] = None
    try:
        if generation_mode == "mimic_auto":
            seed_path = resolve_dataset_file(str(request["seedDatasetFile"]))
            seed_source = str(request.get("seedSource") or "manual_path")
            success, output_path, phase_info = _execute_mimic_auto(
                runner,
                job_id,
                mimic_task_id=mimic_task_id,
                seed_path=seed_path,
                seed_source=seed_source,
                num_demos=num_demos,
                num_envs=num_envs,
                headless=headless,
                enable_cameras=enable_cameras,
                device=device,
                timeout=timeout,
                request_video=request_video,
            )
        elif is_expert_policy_mode(generation_mode):
            success, output_path, phase_info = _execute_expert_policy(
                runner,
                job_id,
                task_id=task_id or DEFAULT_SCRIPTED_EXPERT_TASK_ID,
                num_demos=num_demos,
                seed=int(request.get("seed") or 0),
                max_attempts=max_attempts,
                headless=headless,
                enable_cameras=enable_cameras,
                device=device,
                timeout=timeout,
                request_video=request_video,
                record_camera_obs=record_camera_obs,
                image_resolution=image_resolution,
                include_wrist_camera=include_wrist_camera,
            )
        elif generation_mode == "teleop_record":
            success, output_path, phase_info = _execute_teleop_record(
                runner,
                job_id,
                task_id=task_id,
                num_demos=num_demos,
                headless=headless,
                enable_cameras=enable_cameras,
                device=device,
                timeout=timeout,
                request_video=request_video,
            )
        else:
            raise RuntimeError(f"Unsupported generation mode: {generation_mode}")

        if success:
            episode_count = int(phase_info.get("metrics", {}).get("episodeCount") or 0)
            if is_expert_policy_mode(generation_mode) and episode_count <= 0:
                success = False
                phase_info["message"] = "Dataset validation failed: no episodes in HDF5"
            else:
                registered = dataset_svc.register_generated_dataset(
                    job_id=job_id,
                    dataset_name=dataset_name,
                    dataset_file=output_path,
                    task_id=task_id,
                    episode_count=episode_count or num_demos,
                    replay_available=True,
                )
                dataset_id = str(registered.get("id"))

        manifest = read_json(isaac_job_generation_manifest_path(job_id)) or {}
        preview_patch: dict[str, Any] = {}
        for key in ("previewStatus", "previewVideoPath", "browserPreviewVideoPath", "videoNote"):
            value = phase_info.get(key) or manifest.get(key)
            if key == "browserPreviewVideoPath" and value is None:
                value = manifest.get("previewBrowserVideoPath")
            if value is not None:
                preview_patch[key] = value

        finalize_status(
            job_id,
            {
                "jobId": job_id,
                "kind": "generate_dataset",
                "status": "completed" if success else "failed",
                "phase": "done" if success else phase_info.get("phase", "failed"),
                "generationMode": generation_mode,
                "taskId": task_id,
                "datasetName": dataset_name,
                "datasetFile": str(output_path) if success else None,
                "datasetAvailable": success and output_path.is_file(),
                "datasetId": dataset_id,
                "numDemos": num_demos,
                "headless": headless,
                "enableCameras": enable_cameras,
                "videoAvailable": bool(phase_info.get("videoAvailable")),
                "videoNote": phase_info.get("videoNote"),
                "startedAt": started_at,
                "finishedAt": utc_now_iso(),
                "exitCode": phase_info.get("exitCode"),
                "timedOut": phase_info.get("timedOut"),
                "message": (
                    "Dataset generation completed and registered"
                    if success
                    else str(phase_info.get("message") or "Dataset generation failed")
                ),
                "paths": _job_paths_payload(job_id),
                **preview_patch,
            },
        )
    except Exception as exc:
        logger.exception("Isaac Lab generate job %s failed", job_id)
        finalize_status(
            job_id,
            {
                "jobId": job_id,
                "kind": "generate_dataset",
                "status": "failed",
                "phase": "error",
                "generationMode": generation_mode,
                "startedAt": started_at,
                "finishedAt": utc_now_iso(),
                "message": str(exc),
                "paths": _job_paths_payload(job_id),
            },
        )
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_JOBS.discard(job_id)


def start_generate_dataset(
    *,
    task_id: str = DEFAULT_GENERATE_TASK_ID,
    dataset_name: str,
    num_demos: int = 10,
    seed: int = 0,
    headless: bool = True,
    enable_cameras: bool = True,
    record_camera_obs: bool = True,
    image_resolution: int = 128,
    include_wrist_camera: bool = False,
    generation_mode: str = "expert_policy",
    seed_dataset_file: Optional[str] = None,
    seed_dataset_id: Optional[str] = None,
    video: bool = True,
    num_envs: Optional[int] = None,
    max_attempts: Optional[int] = None,
) -> dict[str, Any]:
    assert_runtime_configured_for_commands()
    _validate_generation_request(generation_mode=generation_mode)

    runner = IsaacLabCliRunner.from_settings()
    if not runner.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=RUNTIME_NOT_CONFIGURED_MSG,
        )

    name = (dataset_name or "").strip()
    if not name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="datasetName is required",
        )
    if num_demos < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="numDemos must be >= 1",
        )

    mode = (generation_mode or "expert_policy").strip()
    stored_mode = normalize_generation_mode(mode) if is_expert_policy_mode(mode) else mode
    seed_source: Optional[str] = None
    resolved_seed_id: Optional[str] = None
    resolved_seed_file: Optional[str] = None
    if mode == "mimic_auto":
        seed_path, seed_source, resolved_seed_id = resolve_generation_seed(
            seed_dataset_file=seed_dataset_file,
            seed_dataset_id=seed_dataset_id,
        )
        resolved_seed_file = str(seed_path.resolve())
    elif is_expert_policy_mode(mode):
        platform_script, script_exists = resolve_expert_policy_platform_script()
        if not script_exists:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Expert policy script not found: {platform_script}",
            )

    effective_task_id = (
        DEFAULT_SCRIPTED_EXPERT_TASK_ID if is_expert_policy_mode(mode) else task_id
    )
    effective_max_attempts = int(max_attempts) if max_attempts is not None else 0

    job_id = make_isaac_gen_job_id()
    job_root = isaac_job_root(job_id)
    job_root.mkdir(parents=True, exist_ok=True)
    isaac_job_artifacts_dir(job_id).mkdir(parents=True, exist_ok=True)
    _ensure_live_dirs(job_id)
    meta_dir = isaac_job_metadata_dir(job_id)
    meta_dir.mkdir(parents=True, exist_ok=True)

    request_payload = {
        "kind": "generate_dataset",
        "taskId": effective_task_id,
        "mimicTaskId": DEFAULT_MIMIC_TASK_ID,
        "datasetName": name,
        "numDemos": num_demos,
        "numEnvs": resolve_num_envs(num_demos, num_envs),
        "seed": seed,
        "maxAttempts": effective_max_attempts,
        "headless": headless,
        "enableCameras": enable_cameras,
        "recordCameraObs": record_camera_obs,
        "imageResolution": max(32, int(image_resolution)),
        "includeWristCamera": include_wrist_camera,
        "generationMode": stored_mode,
        "seedSource": seed_source,
        "seedDatasetId": resolved_seed_id,
        "seedDatasetFile": resolved_seed_file,
        "video": video,
        "device": "cpu",
        "submittedAt": utc_now_iso(),
    }
    write_json(meta_dir / "request.json", request_payload)

    started_at = utc_now_iso()
    status_payload = finalize_status(
        job_id,
        {
            "jobId": job_id,
            "kind": "generate_dataset",
            "status": "queued",
            "phase": "queued",
            "generationMode": stored_mode,
            "taskId": effective_task_id,
            "datasetName": name,
            "numDemos": num_demos,
            "headless": headless,
            "enableCameras": enable_cameras,
            "datasetAvailable": False,
            "videoAvailable": False,
            "startedAt": started_at,
            "message": "Dataset generation queued",
            "paths": _job_paths_payload(job_id),
        },
    )

    with _ACTIVE_LOCK:
        _ACTIVE_JOBS.add(job_id)

    thread = threading.Thread(
        target=_execute_generate_job,
        args=(job_id, request_payload),
        name=f"isaac-gen-{job_id}",
        daemon=True,
    )
    thread.start()

    return {
        "jobId": job_id,
        "kind": "generate_dataset",
        "status": status_payload.get("status", "queued"),
        "runtimePath": str(job_root),
        "statusUrl": f"/api/workspace/isaac-lab/jobs/{job_id}/status",
        "logPaths": {
            "stdout": str(isaac_job_stdout_path(job_id)),
            "stderr": str(isaac_job_stderr_path(job_id)),
        },
    }


def get_generate_job_status(job_id: str) -> dict[str, Any]:
    if not is_isaac_gen_job_id(job_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Isaac Lab generate job ID format",
        )
    job_root = isaac_job_root(job_id)
    if not job_root.is_dir():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Isaac Lab generate job not found",
        )
    payload = read_json(isaac_job_status_path(job_id))
    if not payload:
        payload = {"jobId": job_id, "status": "unknown", "message": "status.json missing"}
    payload.setdefault("jobId", job_id)
    payload.setdefault("paths", _job_paths_payload(job_id))
    payload.setdefault("datasetAvailable", isaac_job_dataset_path(job_id).is_file())
    preview_path = isaac_job_preview_video_path(job_id)
    payload.setdefault("videoAvailable", preview_path.is_file())
    artifacts_dir = isaac_job_artifacts_dir(job_id)
    payload.setdefault(
        "artifactStatus",
        {
            "seedHdf5": (artifacts_dir / "seed.hdf5").is_file(),
            "annotatedHdf5": (artifacts_dir / "annotated_dataset.hdf5").is_file(),
            "datasetHdf5": isaac_job_dataset_path(job_id).is_file(),
            "generationManifest": isaac_job_generation_manifest_path(job_id).is_file(),
            "metricsJson": isaac_job_metrics_path(job_id).is_file(),
        },
    )
    meta_request = isaac_job_metadata_dir(job_id) / "request.json"
    enable_cameras = bool(payload.get("enableCameras", True))
    if meta_request.is_file():
        request_data = read_json(meta_request)
        enable_cameras = bool(request_data.get("enableCameras", enable_cameras))
        if request_data.get("seedSource"):
            payload.setdefault("seedSource", request_data.get("seedSource"))
        if payload.get("numDemos") is None and request_data.get("numDemos") is not None:
            payload["numDemos"] = int(request_data["numDemos"])
        payload.setdefault("generationMode", request_data.get("generationMode"))
        payload.setdefault("datasetName", request_data.get("datasetName"))
        if payload.get("seed") is None and request_data.get("seed") is not None:
            payload["seed"] = int(request_data["seed"])
    payload.update(_live_visual_fields(job_id, enable_cameras=enable_cameras))
    payload["previewVideoAvailable"] = preview_path.is_file() and preview_path.stat().st_size > 0
    manifest_path = isaac_job_generation_manifest_path(job_id)
    if manifest_path.is_file():
        manifest = read_json(manifest_path) or {}
        if manifest.get("previewStatus"):
            payload.setdefault("previewStatus", manifest["previewStatus"])
        payload.setdefault("previewVideoPath", manifest.get("previewVideoPath"))
        browser_path = manifest.get("browserPreviewVideoPath") or manifest.get("previewBrowserVideoPath")
        if browser_path:
            payload.setdefault("browserPreviewVideoPath", browser_path)
        if manifest.get("videoNote") and not payload.get("videoNote"):
            payload.setdefault("videoNote", manifest.get("videoNote"))
    payload.update(_enrich_generate_episode_fields(job_id, payload))
    return payload


_MIMIC_DEMO_PROGRESS_RE = re.compile(
    r"(?P<successful>\d+)/(?P<attempts>\d+)\s+\([0-9.]+%\)\s+successful demos generated by mimic"
)


def _parse_mimic_demo_progress(stdout_path: Path) -> tuple[int | None, int | None]:
    if not stdout_path.is_file():
        return None, None
    try:
        text = stdout_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    matches = list(_MIMIC_DEMO_PROGRESS_RE.finditer(text))
    if not matches:
        return None, None
    last = matches[-1]
    return int(last.group("successful")), int(last.group("attempts"))


def _resolve_generate_stdout_path(job_id: str) -> Path:
    artifacts = isaac_job_artifacts_dir(job_id)
    for name in ("generate.stdout.log", "expert_policy.stdout.log", "scripted_expert.stdout.log", "record.stdout.log"):
        candidate = artifacts / name
        if candidate.is_file():
            return candidate
    return isaac_job_stdout_path(job_id)


def _compute_generate_progress_percent(
    *,
    status: str,
    phase: str | None,
    total_episodes: int,
    completed_episodes: int | None,
) -> int:
    if status == "completed":
        return 100
    if status == "failed":
        return 0
    if status == "queued":
        return 5

    phase_key = (phase or "").strip()
    if phase_key == "annotate":
        return 25
    if phase_key in {"done", "register_dataset", "postprocess"}:
        return 95

    if phase_key == "generate" and total_episodes > 0:
        done = max(0, int(completed_episodes or 0))
        ratio = min(1.0, done / total_episodes)
        return min(80, max(26, int(round(25 + ratio * 55))))

    if phase_key == "generate":
        return 55
    if phase_key == "replay_preview":
        return 85
    return 15


def _enrich_generate_episode_fields(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    request_data = read_json(isaac_job_metadata_dir(job_id) / "request.json")
    manifest = read_json(isaac_job_generation_manifest_path(job_id))
    metrics = read_json(isaac_job_metrics_path(job_id))

    total_episodes = payload.get("numDemos")
    if total_episodes is None:
        total_episodes = request_data.get("numDemos")
    if total_episodes is None:
        total_episodes = manifest.get("numDemos")
    if total_episodes is None:
        total_episodes = metrics.get("requestedDemos")
    total_episodes = int(total_episodes) if total_episodes is not None else None

    dataset_path = isaac_job_dataset_path(job_id)
    hdf5_count = _count_hdf5_episodes(dataset_path) if dataset_path.is_file() else 0

    stdout_successful, _stdout_attempts = _parse_mimic_demo_progress(_resolve_generate_stdout_path(job_id))

    completed_episodes: int | None = None
    successful_episodes: int | None = None

    status = str(payload.get("status") or "unknown")
    phase = str(payload.get("phase") or "")

    if status == 'completed':
        manifest_metrics = manifest.get("metrics")
        manifest_episode_count = (
            manifest_metrics.get("episodeCount")
            if isinstance(manifest_metrics, dict)
            else None
        )
        completed_episodes = int(
            metrics.get("episodeCount")
            or manifest_episode_count
            or hdf5_count
            or 0
        )
        successful_episodes = completed_episodes
    elif status == "failed":
        if hdf5_count > 0:
            completed_episodes = hdf5_count
            successful_episodes = hdf5_count
        elif stdout_successful is not None:
            successful_episodes = stdout_successful
            completed_episodes = stdout_successful
    elif status in {"running", "queued"}:
        if hdf5_count > 0:
            completed_episodes = hdf5_count
            successful_episodes = hdf5_count
        elif stdout_successful is not None:
            successful_episodes = stdout_successful
            completed_episodes = stdout_successful

    current_episode: int | None = None
    if total_episodes and completed_episodes is not None and completed_episodes > 0:
        current_episode = min(total_episodes, completed_episodes)
    elif total_episodes and status == "running" and phase == "generate" and stdout_successful is not None:
        current_episode = min(total_episodes, max(1, stdout_successful))

    progress = _compute_generate_progress_percent(
        status=status,
        phase=phase,
        total_episodes=total_episodes or 0,
        completed_episodes=completed_episodes,
    )

    enriched: dict[str, Any] = {
        "progress": progress,
        "episodeCount": total_episodes,
    }
    if total_episodes is not None:
        enriched["totalEpisodes"] = total_episodes
        enriched["numDemos"] = total_episodes
    if completed_episodes is not None:
        enriched["completedEpisodes"] = completed_episodes
    if successful_episodes is not None:
        enriched["successfulEpisodes"] = successful_episodes
    if current_episode is not None:
        enriched["currentEpisode"] = current_episode
    return enriched


def resolve_generate_video_path(job_id: str) -> Optional[Path]:
    if not is_isaac_gen_job_id(job_id):
        return None
    preferred = isaac_job_preview_video_path(job_id)
    if preferred.is_file():
        return preferred
    status_payload = read_json(isaac_job_status_path(job_id))
    if status_payload.get("videoPath"):
        path = Path(str(status_payload["videoPath"]))
        if path.is_file():
            return path
    return None
