from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, status

from integrations.dual_arm_cable.export_il_dataset import IlExportError, export_job, inspect_job
from app.core.platform_paths import platform_paths
from app.services.dataset_naming import persist_manifest_display_fields
from app.services import dual_arm_cable_service as dual_arm_svc

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
DUAL_ARM_ROOT = platform_paths.runs_root / "dual_arm_cable" / "jobs"
DAC_GEN_PATTERN = re.compile(r"^dac_gen_")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _validate_job_id(job_id: str) -> str:
    candidate = (job_id or "").strip()
    if not candidate or not DAC_GEN_PATTERN.match(candidate):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid dual-arm job ID")
    return candidate


def _job_dir(job_id: str) -> Path:
    current = DUAL_ARM_ROOT / job_id
    if current.is_dir():
        return current
    return dual_arm_svc._job_dir(job_id)


def resolve_job_dir(job_id: str) -> Path:
    job_id = _validate_job_id(job_id)
    job_dir = _job_dir(job_id)
    if not job_dir.is_dir():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="dual-arm job not found")
    return job_dir


def probe_il_export(job_id: str) -> dict[str, Any]:
    job_dir = resolve_job_dir(job_id)
    report = inspect_job(job_dir)
    hdf5_path = job_dir / "datasets" / "dataset.hdf5"
    manifest_path = job_dir / "datasets" / "dataset.manifest.json"
    return {
        "jobId": job_id,
        "exportReady": bool(report.get("exportReady")),
        "failureReason": report.get("failureReason"),
        "actionAvailable": bool(report.get("actionAvailable")),
        "observationAvailable": bool(report.get("observationAvailable")),
        "missingFields": report.get("missingFields") or [],
        "hdf5Exists": hdf5_path.is_file(),
        "manifestExists": manifest_path.is_file(),
        "hdf5Path": str(hdf5_path),
        "manifestPath": str(manifest_path),
        "trainable": hdf5_path.is_file() and manifest_path.is_file(),
        "exportReport": report,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _episode_stats(job_dir: Path) -> dict[str, int]:
    episode_result = _read_json(job_dir / "results" / "episode_result.json")
    steps_root = job_dir / "results" / "steps"
    step_count = len([p for p in steps_root.iterdir() if p.is_dir()]) if steps_root.is_dir() else 0
    total = int(episode_result.get("max_cables") or step_count or 0)
    attempted = int(episode_result.get("num_steps_attempted") or step_count or total)
    succeeded = int(episode_result.get("num_cables_succeeded") or 0)
    total_episodes = max(total, attempted, step_count)
    return {
        "totalEpisodes": total_episodes,
        "completedEpisodes": attempted,
        "successfulEpisodesRaw": succeeded,
    }


def _write_generation_manifest(job_dir: Path, job_id: str, *, il_export: dict[str, Any]) -> Path:
    stats = _episode_stats(job_dir)
    episode_result = _read_json(job_dir / "results" / "episode_result.json")
    hdf5_path = job_dir / "datasets" / "dataset.hdf5"
    manifest_path = job_dir / "datasets" / "dataset.manifest.json"
    export_report_path = job_dir / "datasets" / "export_report.json"

    payload: dict[str, Any] = {
        "jobId": job_id,
        "taskType": "dual_arm_cable_manipulation",
        "taskTemplateId": "task_dual_arm_cable_manipulation_v1",
        "simulatorBackend": "mujoco",
        "sourceType": "simulation_generated",
        "sourceJobId": job_id,
        "maxCables": stats["totalEpisodes"],
        "totalEpisodes": stats["totalEpisodes"],
        "completedEpisodes": stats["completedEpisodes"],
        "successfulEpisodes": stats["successfulEpisodesRaw"],
        "episodeSuccess": bool(episode_result.get("episode_success")),
        "datasetFormat": "hdf5" if hdf5_path.is_file() else "manifest",
        "trainable": hdf5_path.is_file() and manifest_path.is_file(),
        "trainingBackends": ["torch_bc"] if hdf5_path.is_file() else [],
        "datasetFile": str(hdf5_path) if hdf5_path.is_file() else None,
        "manifestPath": str(manifest_path) if manifest_path.is_file() else str(job_dir / "results" / "episode_manifest.json"),
        "exportReportPath": str(export_report_path) if export_report_path.is_file() else None,
        "ilExportStatus": il_export.get("status"),
    }
    if il_export.get("exportReport"):
        report = il_export["exportReport"]
        if isinstance(report, dict):
            exported = report.get("exportedEpisodes")
            if isinstance(exported, int):
                payload["successfulEpisodes"] = exported
                payload["trainableDemos"] = exported
    out = job_dir / "generation_manifest.json"
    _write_json(out, payload)
    return out


def _update_status_after_il_export(job_dir: Path, job_id: str, il_export: dict[str, Any]) -> None:
    status_path = job_dir / "status.json"
    status_payload = _read_json(status_path)
    hdf5_path = job_dir / "datasets" / "dataset.hdf5"
    manifest_path = job_dir / "datasets" / "dataset.manifest.json"
    export_status = str(il_export.get("status") or "")
    trainable = hdf5_path.is_file() and manifest_path.is_file()

    status_payload["ilExportStatus"] = export_status
    status_payload["trainable"] = trainable
    status_payload["datasetFormat"] = "hdf5" if trainable else "manifest"
    if trainable:
        status_payload["datasetHdf5Path"] = str(hdf5_path)
        status_payload["datasetManifestPath"] = str(manifest_path)
    if il_export.get("message"):
        status_payload["ilExportMessage"] = il_export["message"]
    report = il_export.get("exportReport")
    if isinstance(report, dict) and report.get("failureReason"):
        status_payload["ilExportFailureReason"] = report["failureReason"]

    _write_json(status_path, status_payload)
    _write_generation_manifest(job_dir, job_id, il_export=il_export)


def _persist_dual_arm_manifest(job_id: str, manifest_path: Path, job_dir: Path) -> None:
    if not manifest_path.is_file():
        return
    stats = _episode_stats(job_dir)
    data = _read_json(manifest_path)
    merged = dict(data) if isinstance(data, dict) else {}
    merged.setdefault("totalEpisodes", stats["totalEpisodes"])
    merged.setdefault("completedEpisodes", stats["completedEpisodes"])
    if merged.get("successfulEpisodes") is None:
        merged["successfulEpisodes"] = stats["successfulEpisodesRaw"]
    manifest_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    persist_manifest_display_fields(
        manifest_path,
        task_type="dual_arm_cable_manipulation",
        source_job_id=job_id,
        simulator_backend="mujoco",
        dataset_format="hdf5",
    )


def build_il_dataset(job_id: str) -> dict[str, Any]:
    job_dir = resolve_job_dir(job_id)
    if (job_dir / "datasets" / "dataset.hdf5").is_file() and (
        job_dir / "datasets" / "dataset.manifest.json"
    ).is_file():
        manifest_path = job_dir / "datasets" / "dataset.manifest.json"
        _persist_dual_arm_manifest(job_id, manifest_path, job_dir)
        il_result = {
            "jobId": job_id,
            "status": "already_built",
            "manifestPath": str(manifest_path),
            "hdf5Path": str(job_dir / "datasets" / "dataset.hdf5"),
            "message": "IL dataset already exists",
        }
        _update_status_after_il_export(job_dir, job_id, il_result)
        return il_result

    try:
        result = export_job(job_dir, job_id=job_id)
    except IlExportError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": str(exc),
                "exportReport": exc.report,
            },
        ) from exc

    manifest_path = Path(str(result["manifestPath"]))
    _persist_dual_arm_manifest(job_id, manifest_path, job_dir)
    il_result = {
        "jobId": job_id,
        "status": "built",
        "manifestPath": result["manifestPath"],
        "hdf5Path": result["hdf5Path"],
        "manifest": result["manifest"],
        "exportReport": result["exportReport"],
        "message": "IL dataset exported successfully",
    }
    _update_status_after_il_export(job_dir, job_id, il_result)
    return il_result


def auto_build_il_dataset_after_generate(job_id: str) -> dict[str, Any]:
    """Generate job 完成后自动导出 IL HDF5（幂等；失败写入 export_report.json）。"""
    candidate = (job_id or "").strip()
    if not candidate or not DAC_GEN_PATTERN.match(candidate):
        return {"jobId": candidate, "status": "skipped", "reason": "not_a_generate_job"}

    job_dir = _job_dir(candidate)
    if not job_dir.is_dir():
        return {"jobId": candidate, "status": "skipped", "reason": "job_dir_missing"}

    hdf5_path = job_dir / "datasets" / "dataset.hdf5"
    manifest_path = job_dir / "datasets" / "dataset.manifest.json"
    if hdf5_path.is_file() and manifest_path.is_file():
        il_result = {
            "jobId": candidate,
            "status": "already_built",
            "manifestPath": str(manifest_path),
            "hdf5Path": str(hdf5_path),
        }
        _update_status_after_il_export(job_dir, candidate, il_result)
        return il_result

    status_payload = _read_json(job_dir / "status.json")
    job_status = str(status_payload.get("status") or "")
    if job_status != "completed":
        return {"jobId": candidate, "status": "skipped", "reason": f"job_status={job_status or 'unknown'}"}

    try:
        result = export_job(job_dir, job_id=candidate)
        manifest_path = Path(str(result["manifestPath"]))
        _persist_dual_arm_manifest(candidate, manifest_path, job_dir)
        il_result = {
            "jobId": candidate,
            "status": "built",
            "manifestPath": result["manifestPath"],
            "hdf5Path": result["hdf5Path"],
            "exportReport": result.get("exportReport"),
        }
        _update_status_after_il_export(job_dir, candidate, il_result)
        logger.info("auto IL export succeeded job=%s hdf5=%s", candidate, result.get("hdf5Path"))
        return il_result
    except IlExportError as exc:
        logger.warning("auto IL export failed job=%s: %s", candidate, exc)
        il_result = {
            "jobId": candidate,
            "status": "export_failed",
            "message": str(exc),
            "exportReport": exc.report,
        }
        _update_status_after_il_export(job_dir, candidate, il_result)
        return il_result
