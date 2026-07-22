"""SAM3 + SAM3D asset pipeline job orchestration (subprocess)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.services.sam3d_asset_paths import (
    ensure_job_dirs,
    get_asset_pipeline_root,
    get_job_dir,
    get_pipeline_run_script,
    make_job_id,
    normalize_uploaded_image,
    read_job_json,
    resolve_input_image_path,
    safe_join_job_file,
    scan_job_files,
    utc_now_iso,
    write_job_json,
)
from app.services.workspace_job_service import record_workspace_job_start

logger = logging.getLogger(__name__)

ASYNC_PROCS: dict[str, subprocess.Popen] = {}


def _pipeline_enabled() -> bool:
    return bool(settings.SAM3D_PIPELINE_ENABLED)


def _assert_enabled() -> None:
    if not _pipeline_enabled():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SAM3D asset pipeline is disabled (SAM3D_PIPELINE_ENABLED=false)",
        )


def _runner_python() -> str:
    env = os.environ.get("EAI_PYTHON", "").strip()
    if env and Path(env).is_file():
        return env
    return sys.executable


def _sam3_root() -> Path:
    return Path(settings.SAM3_ROOT).expanduser().resolve()


def _sam3d_root() -> Path:
    return Path(settings.SAM3D_OBJECTS_ROOT).expanduser().resolve()


def _sam3_python() -> str:
    return str(Path(settings.SAM3_PYTHON).expanduser().resolve())


def _sam3d_python() -> str:
    return str(Path(settings.SAM3D_OBJECTS_PYTHON).expanduser().resolve())


def _read_live_status(job_dir: Path) -> dict[str, Any]:
    path = job_dir / "live" / "status.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_live_status(job_dir: Path, payload: dict[str, Any]) -> None:
    path = job_dir / "live" / "status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _to_frontend_cutout_item(item: dict[str, Any]) -> dict[str, Any]:
    cutout_path = item.get("cutoutPath") or item.get("previewPath")
    cutout_index = item.get("cutoutIndex")
    if cutout_index is None and item.get("maskIndex") is not None:
        cutout_index = int(item["maskIndex"]) + 1
    if cutout_index is None:
        cutout_index = 0
    label = item.get("label") or str(cutout_index)
    return {
        "cutoutIndex": int(cutout_index),
        "label": label,
        "score": item.get("score"),
        "bbox": item.get("bbox"),
        "cutoutPath": cutout_path,
        "previewPath": cutout_path,
        "originalCutoutPath": item.get("originalCutoutPath"),
        "selectable": bool(cutout_path) and item.get("selectable", True),
    }


def _load_manifest_segmentation(job_dir: Path, segmentation: dict[str, Any] | None) -> dict[str, Any]:
    seg = dict(segmentation or {})
    manifest_path = job_dir / "sam3" / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            seg.setdefault("manifestPath", "sam3/manifest.json")
            seg.setdefault("overlayPath", manifest.get("overlay") or "sam3/overlay.png")
            raw_items = manifest.get("items") or seg.get("items") or []
            seg["items"] = [_to_frontend_cutout_item(item) for item in raw_items if item.get("cutoutPath") or item.get("previewPath")]
        except (OSError, json.JSONDecodeError):
            pass
    elif not seg.get("items"):
        seg["message"] = seg.get("message") or "manifest missing; using file scan fallback"
    return seg


def _merge_status_response(job_dir: Path) -> dict[str, Any]:
    job = read_job_json(job_dir)
    live = _read_live_status(job_dir)
    files = scan_job_files(job_dir)
    segmentation = _load_manifest_segmentation(job_dir, job.get("segmentation"))
    extra = live.get("extra") or {}
    return {
        "jobId": job.get("jobId") or job_dir.name,
        "name": job.get("name"),
        "status": live.get("status") or job.get("status") or "unknown",
        "phase": live.get("phase") or "",
        "progress": live.get("progress", 0.0),
        "message": live.get("message"),
        "error": live.get("error"),
        "updatedAt": live.get("updatedAt") or job.get("updatedAt"),
        "inputImage": job.get("inputImage"),
        "targetEngine": job.get("targetEngine"),
        "assetType": job.get("assetType"),
        "segmentation": segmentation,
        "reconstruction": job.get("reconstruction"),
        "mujocoExport": job.get("mujocoExport"),
        "mujocoVisualization": job.get("mujocoVisualization"),
        "files": files,
        "extra": extra,
        "commandSummary": extra.get("commandSummary"),
    }


def _pipeline_subprocess_env() -> dict[str, str]:
    """Inject HF mirror + pipeline gitconfig into every segment/reconstruct subprocess."""
    env = os.environ.copy()
    hf_endpoint = (settings.SAM3D_HF_ENDPOINT or "https://hf-mirror.com").strip()
    env["HF_ENDPOINT"] = hf_endpoint
    env["SAM3D_HF_ENDPOINT"] = hf_endpoint
    if settings.SAM3D_GIT_GITHUB_SSH_REWRITE:
        gitconfig = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "integrations"
            / "Sam3dAssetPipeline"
            / ".gitconfig.pipeline"
        )
        if gitconfig.is_file():
            env["GIT_CONFIG_GLOBAL"] = str(gitconfig.resolve())
    return env


def _spawn_pipeline(
    cmd: list[str],
    *,
    job_id: str,
    log_path: Path,
    pid_path: Path | None = None,
    launcher_log_path: Path | None = None,
) -> None:
    run_py = get_pipeline_run_script()
    if not run_py.is_file():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"pipeline runner not found: {run_py}",
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if launcher_log_path is not None:
        launcher_log_path.parent.mkdir(parents=True, exist_ok=True)
        launcher = open(launcher_log_path, "a", encoding="utf-8")
        launcher.write(f"[{utc_now_iso()}] spawn: {' '.join(cmd)}\n")
        launcher.flush()
    else:
        launcher = None

    try:
        log_file = open(log_path, "a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            cwd=str(run_py.parent),
            env=_pipeline_subprocess_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as exc:
        if launcher is not None:
            launcher.write(f"[{utc_now_iso()}] spawn failed: {exc}\n")
            launcher.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to start pipeline subprocess: {exc}",
        ) from exc

    ASYNC_PROCS[job_id] = proc
    if pid_path is not None:
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(proc.pid), encoding="utf-8")
    if launcher is not None:
        launcher.write(f"[{utc_now_iso()}] pid={proc.pid}\n")
        launcher.close()


def create_asset_job(*, name: str, uploaded_file: UploadFile) -> dict[str, Any]:
    _assert_enabled()
    job_id = make_job_id()
    job_dir = get_job_dir(job_id)
    ensure_job_dirs(job_dir)

    filename = uploaded_file.filename or "upload.png"
    input_info = normalize_uploaded_image(
        uploaded_file.file,
        filename=filename,
        target_path=job_dir / "input" / "image.png",
    )

    payload = {
        "jobId": job_id,
        "name": (name or "").strip() or job_id,
        "status": "created",
        "assetType": "object",
        "targetEngine": "generic",
        "source": "reconstructed",
        "createdAt": utc_now_iso(),
        "updatedAt": utc_now_iso(),
        **input_info,
    }
    write_job_json(job_dir, payload)

    _write_live_status(
        job_dir,
        {
            "jobId": job_id,
            "status": "created",
            "phase": "created",
            "progress": 0.0,
            "message": "job created",
            "updatedAt": utc_now_iso(),
            "error": None,
            "extra": {},
        },
    )

    try:
        record_workspace_job_start(
            job_id=job_id,
            job_type="asset_pipeline",
            task_type="sam3d_asset_reconstruction",
            runtime_path=str(job_dir),
            runner="Sam3dAssetPipeline/run.py",
            status="pending",
            task_name=payload["name"],
            metadata={"name": payload["name"], "source": "reconstructed"},
        )
    except Exception as exc:
        logger.warning("record_workspace_job_start failed job_id=%s: %s", job_id, exc)

    return {
        "jobId": job_id,
        "status": "created",
        "inputImage": payload.get("inputImage"),
        "name": payload["name"],
    }


def start_segment_job(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    _assert_enabled()
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    image_path = resolve_input_image_path(job_dir)
    prompt = request.get("prompt")
    positive_boxes = request.get("positiveBoxes") or []
    negative_boxes = request.get("negativeBoxes") or []
    confidence = float(request.get("confidenceThreshold", 0.05))
    text_only = bool(request.get("textOnly", False))

    if ASYNC_PROCS.get(job_id) and ASYNC_PROCS[job_id].poll() is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job already running")

    _write_live_status(
        job_dir,
        {
            "jobId": job_id,
            "status": "segmenting",
            "phase": "sam3_segment",
            "progress": 0.15,
            "message": "segmentation queued",
            "updatedAt": utc_now_iso(),
            "error": None,
            "extra": {},
        },
    )

    cmd = [
        _runner_python(),
        str(get_pipeline_run_script()),
        "segment",
        "--job-dir",
        str(job_dir),
        "--sam3-root",
        str(_sam3_root()),
        "--sam3-python",
        _sam3_python(),
        "--image",
        str(image_path),
        "--confidence-threshold",
        str(confidence),
    ]
    cmd.extend(["--prompt", str(prompt or "")])
    if text_only:
        cmd.append("--text-only")
    for box in positive_boxes:
        if isinstance(box, (list, tuple)) and len(box) == 4:
            cmd.extend(["--pos-box", ",".join(str(float(v)) for v in box)])
    for box in negative_boxes:
        if isinstance(box, (list, tuple)) and len(box) == 4:
            cmd.extend(["--neg-box", ",".join(str(float(v)) for v in box)])

    job = read_job_json(job_dir)
    job["segmentationRequest"] = {
        "prompt": prompt,
        "positiveBoxes": positive_boxes,
        "negativeBoxes": negative_boxes,
        "confidenceThreshold": confidence,
        "textOnly": text_only,
    }
    job["updatedAt"] = utc_now_iso()
    write_job_json(job_dir, job)

    _spawn_pipeline(cmd, job_id=job_id, log_path=job_dir / "logs" / "segment.log")
    return _merge_status_response(job_dir)


def start_reconstruct_job(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    _assert_enabled()
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    live = _read_live_status(job_dir)
    if live.get("status") not in {"segmented", "reconstructed"} and not (
        (job_dir / "sam3" / "manifest.json").is_file()
        or (job_dir / "sam3" / "detections.json").is_file()
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="segmentation results required before reconstruction",
        )

    mask_index = request.get("maskIndex")
    cutout_index = request.get("cutoutIndex")
    if cutout_index is None and mask_index is not None:
        cutout_index = int(mask_index) + 1
    if cutout_index is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="cutoutIndex is required",
        )
    cutout_index = int(cutout_index)
    seed = int(request.get("seed", 42))
    prepare_only = bool(request.get("prepareOnly", False))
    image_path = resolve_input_image_path(job_dir)

    if ASYNC_PROCS.get(job_id) and ASYNC_PROCS[job_id].poll() is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="job already running")

    live_status = "segmented" if prepare_only else "reconstructing"
    live_phase = "sam3d_prepare" if prepare_only else "sam3d_reconstruct"
    live_message = "cutout prepare queued" if prepare_only else "reconstruction queued"

    _write_live_status(
        job_dir,
        {
            "jobId": job_id,
            "status": live_status,
            "phase": live_phase,
            "progress": 0.55,
            "message": live_message,
            "updatedAt": utc_now_iso(),
            "error": None,
            "extra": {"cutoutIndex": cutout_index, "prepareOnly": prepare_only},
        },
    )

    cmd = [
        _runner_python(),
        str(get_pipeline_run_script()),
        "reconstruct",
        "--job-dir",
        str(job_dir),
        "--sam3d-root",
        str(_sam3d_root()),
        "--sam3d-python",
        _sam3d_python(),
        "--image",
        str(image_path),
        "--cutout-index",
        str(cutout_index),
        "--seed",
        str(seed),
        "--dinov2-repo",
        str(Path(settings.SAM3D_DINOV2_REPO).expanduser().resolve()),
        "--dinov2-model",
        str(settings.SAM3D_DINOV2_MODEL),
        "--moge-model-path",
        str(Path(settings.SAM3D_MOGE_MODEL_PATH).expanduser().resolve()),
        "--torch-home",
        str(Path(settings.SAM3D_TORCH_HOME).expanduser().resolve()),
        "--hf-home",
        str(Path(settings.SAM3D_HF_HOME).expanduser().resolve()),
        "--timeout-seconds",
        str(int(settings.SAM3D_RECONSTRUCT_TIMEOUT_SECONDS)),
    ]
    if settings.SAM3D_OFFLINE_MODE:
        cmd.append("--offline-mode")
    if prepare_only:
        cmd.append("--prepare-only")

    job = read_job_json(job_dir)
    job["reconstructionRequest"] = {
        "cutoutIndex": cutout_index,
        "seed": seed,
        "prepareOnly": prepare_only,
    }
    job["updatedAt"] = utc_now_iso()
    write_job_json(job_dir, job)

    _spawn_pipeline(
        cmd,
        job_id=job_id,
        log_path=job_dir / "logs" / "reconstruct.log",
        pid_path=job_dir / "live" / "reconstruct.pid",
        launcher_log_path=job_dir / "logs" / "reconstruct_launcher.log",
    )
    return _merge_status_response(job_dir)


def get_asset_job_status(job_id: str) -> dict[str, Any]:
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    return _merge_status_response(job_dir)


def render_mujoco_job(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    _assert_enabled()
    if not settings.MUJOCO_RENDER_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MuJoCo render is disabled (MUJOCO_RENDER_ENABLED=false)",
        )

    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    preview_xml = job_dir / "exports" / "mujoco" / "model_preview.xml"
    if not preview_xml.is_file():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="MuJoCo preview XML not found; run export-mujoco first",
        )

    xml_kind = str(request.get("xmlKind") or "preview")
    width = int(request.get("width") or settings.MUJOCO_RENDER_WIDTH)
    height = int(request.get("height") or settings.MUJOCO_RENDER_HEIGHT)
    render_python = str(Path(settings.MUJOCO_RENDER_PYTHON).expanduser().resolve())
    run_py = get_pipeline_run_script()

    cmd = [
        render_python,
        str(run_py),
        "render-mujoco",
        "--job-dir",
        str(job_dir),
        "--xml-kind",
        xml_kind,
        "--width",
        str(width),
        "--height",
        str(height),
        "--gl",
        settings.MUJOCO_RENDER_GL,
        "--runner-python",
        render_python,
    ]

    log_path = job_dir / "logs" / "reconstruct.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now_iso()}] render-mujoco cmd: {' '.join(cmd)}\n")
        proc = subprocess.run(
            cmd,
            cwd=str(run_py.parent),
            env=_pipeline_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            if proc.stdout:
                handle.write(proc.stdout)
            if proc.stderr:
                handle.write(proc.stderr)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to run MuJoCo render: {exc}",
        ) from exc

    if proc.returncode != 0:
        logger.warning(
            "render-mujoco failed job_id=%s rc=%s stderr=%s",
            job_id,
            proc.returncode,
            proc.stderr,
        )

    return _merge_status_response(job_dir)


def get_asset_job_file(job_id: str, rel_path: str) -> Path:
    job_dir = get_job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")
    try:
        path = safe_join_job_file(job_dir, rel_path)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    return path


def list_asset_jobs(*, limit: int = 50) -> list[dict[str, Any]]:
    from app.services.sam3d_asset_paths import get_asset_pipeline_root

    rows: list[dict[str, Any]] = []
    try:
        from app.core.database import SessionLocal
        from app.models.workspace_job import WorkspaceJob

        with SessionLocal() as db:
            db_rows = (
                db.query(WorkspaceJob)
                .filter(WorkspaceJob.job_type == "asset_pipeline")
                .order_by(WorkspaceJob.updated_at.desc())
                .limit(limit)
                .all()
            )
            for row in db_rows:
                job_dir = Path(row.runtime_path) if row.runtime_path else get_job_dir(row.job_id)
                if job_dir.is_dir():
                    merged = _merge_status_response(job_dir)
                    merged["name"] = merged.get("name") or row.task_name or row.job_id
                    rows.append(merged)
            if rows:
                return rows[:limit]
    except Exception as exc:
        logger.warning("list_asset_jobs db fallback: %s", exc)

    root = get_asset_pipeline_root()
    if not root.is_dir():
        return []

    dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("asset_job_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for job_dir in dirs[:limit]:
        rows.append(_merge_status_response(job_dir))
    return rows


def guess_media_type(rel_path: str) -> str:
    lower = rel_path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".log"):
        return "text/plain; charset=utf-8"
    if lower.endswith(".ply"):
        return "application/octet-stream"
    if lower.endswith(".glb"):
        return "model/gltf-binary"
    if lower.endswith(".obj"):
        return "text/plain; charset=utf-8"
    if lower.endswith(".stl"):
        return "application/octet-stream"
    return "application/octet-stream"


def file_download_headers(rel_path: str, filename: str) -> dict[str, str]:
    lower = rel_path.lower()
    if lower.endswith((".ply", ".log", ".glb", ".obj", ".stl")):
        return {"Content-Disposition": f'attachment; filename="{filename}"'}
    return {}


def _is_asset_job_running(job_id: str, job_dir: Path) -> bool:
    proc = ASYNC_PROCS.get(job_id)
    if proc is not None and proc.poll() is None:
        return True
    live = _read_live_status(job_dir)
    return live.get("status") in {"segmenting", "reconstructing"}


def delete_asset_job(job_id: str) -> dict[str, Any]:
    candidate = (job_id or "").strip()
    if not candidate or "/" in candidate or "\\" in candidate or ".." in candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="invalid job id")

    try:
        job_dir = get_job_dir(candidate)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    root = get_asset_pipeline_root().resolve()
    resolved = job_dir.resolve()
    if resolved != root and root not in resolved.parents:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unsafe job path")

    if not resolved.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="job not found")

    if _is_asset_job_running(candidate, resolved):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="任务运行中，暂不能删除",
        )

    ASYNC_PROCS.pop(candidate, None)

    try:
        from app.core.database import SessionLocal
        from app.models.workspace_job import WorkspaceJob

        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == candidate).one_or_none()
            if row:
                db.delete(row)
                db.commit()
    except Exception as exc:
        logger.warning("delete_asset_job db cleanup failed job_id=%s: %s", candidate, exc)

    try:
        shutil.rmtree(resolved)
    except OSError as exc:
        logger.exception("delete_asset_job rmtree failed job_id=%s path=%s", candidate, resolved)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"failed to delete job directory: {exc}",
        ) from exc

    logger.info("delete_asset_job completed job_id=%s path=%s", candidate, resolved)
    return {"ok": True, "jobId": candidate, "deleted": True}
