from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.core.database import SessionLocal
from app.core.platform_paths import is_path_within, platform_paths, resolve_runtime_reference
from app.models.workspace_job import WorkspaceArtifact, WorkspaceJob
from app.services.runtime_job_lifecycle import is_job_deleted

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
RUNTIME_ROOT = platform_paths.runs_root


def _runtime_roots() -> tuple[Path, ...]:
    return (RUNTIME_ROOT,)


def _existing_runtime_path(relative: Path) -> Path:
    return next(
        (root / relative for root in _runtime_roots() if (root / relative).exists()),
        RUNTIME_ROOT / relative,
    )

FORBIDDEN_RUNTIME_DELETE_TARGETS = (
    Path("/"),
    Path("/home"),
    PROJECT_ROOT,
    PROJECT_ROOT / "backend",
    PROJECT_ROOT / "frontend",
    RUNTIME_ROOT,
)


class WorkspaceJobDeleteError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RuntimeDeleteFailedError(WorkspaceJobDeleteError):
    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)

CT_GEN_PREFIX = "ct_gen_"
CT_EVAL_PREFIX = "ct_eval_"
DAC_GEN_PREFIX = "dac_gen_"
EVAL_PREFIX = "eval_"
TRAIN_PREFIX = "train_"


def _looks_like_cable_eval_job(job_root: Path) -> bool:
    if not job_root.is_dir():
        return False
    if (job_root / "results" / "aggregate_result.json").is_file():
        return True
    if (job_root / "results" / "eval.results.json").is_file():
        return True
    if (job_root / "logs" / "run.log").is_file():
        return True
    if (job_root / "live" / "status.json").is_file():
        return True
    return False


TASK_NAME_MAP = {
    "cable_threading": "线缆穿杆",
    "dual_arm_cable_manipulation": "线缆整理",
    "block_stacking": "物块堆叠",
    "isaac_block_stacking": "物块堆叠",
    "unknown": "未知任务",
}

TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled"})


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    if not text:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _normalize_status(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    mapping = {
        "queued": "pending",
        "succeeded": "completed",
        "backend_unavailable": "failed",
    }
    return mapping.get(value, value or "unknown")


def _artifact_counts(artifacts: list[WorkspaceArtifact]) -> dict[str, int]:
    counts = {
        "video": 0,
        "log": 0,
        "manifest": 0,
        "metrics": 0,
        "checkpoint": 0,
        "result": 0,
        "other": 0,
    }
    for art in artifacts:
        t = art.artifact_type or "unknown"
        if t in counts:
            counts[t] += 1
        elif t in {"aggregate_result", "per_episode_result", "episode_result"}:
            counts["result"] += 1
        else:
            counts["other"] += 1
    return counts


def _job_to_summary(job: WorkspaceJob) -> dict[str, Any]:
    artifacts = list(job.artifacts or [])
    counts = _artifact_counts(artifacts)
    metrics = job.metrics_json if isinstance(job.metrics_json, dict) else {}
    meta = job.metadata_json if isinstance(job.metadata_json, dict) else {}
    if job.job_type == "training":
        metrics = {**meta, **metrics}
    elif job.job_type == "evaluation":
        for key in (
            "evaluationType",
            "datasetId",
            "datasetName",
            "modelAssetId",
            "checkpointPath",
            "evaluationMode",
            "modelName",
        ):
            if meta.get(key) is not None:
                metrics[key] = meta.get(key)
    return {
        "jobId": job.job_id,
        "jobType": job.job_type,
        "taskType": job.task_type,
        "taskName": job.task_name,
        "status": job.status,
        "source": job.source,
        "runner": job.runner,
        "createdAt": _iso(job.created_at) or "",
        "updatedAt": _iso(job.updated_at) or "",
        "startedAt": _iso(job.started_at),
        "finishedAt": _iso(job.finished_at),
        "runtimePath": job.runtime_path,
        "metricsSummary": metrics,
        "videoAvailable": counts["video"] > 0,
        "reportAvailable": counts["result"] > 0 or counts["metrics"] > 0,
        "artifactCounts": counts,
    }


def _job_to_detail(job: WorkspaceJob) -> dict[str, Any]:
    summary = _job_to_summary(job)
    summary["metadata"] = job.metadata_json if isinstance(job.metadata_json, dict) else {}
    summary["metrics"] = job.metrics_json if isinstance(job.metrics_json, dict) else {}
    summary["errorMessage"] = job.error_message
    return summary


def _artifact_to_dict(art: WorkspaceArtifact) -> dict[str, Any]:
    return {
        "id": art.id,
        "jobId": art.job_id,
        "artifactType": art.artifact_type,
        "name": art.name,
        "filePath": art.file_path,
        "urlPath": art.url_path,
        "episodeIndex": art.episode_index,
        "createdAt": _iso(art.created_at) or "",
        "metadata": art.metadata_json if isinstance(art.metadata_json, dict) else {},
    }


def _infer_status_from_files(job_root: Path, job_type: str) -> tuple[str, Optional[str], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    error_message: Optional[str] = None

    live_status = _read_json(job_root / "live" / "status.json")
    root_status = _read_json(job_root / "status.json")
    status_payload = live_status or root_status

    if is_job_deleted(live_status) or is_job_deleted(root_status):
        return "deleted", error_message, metrics

    if root_status.get("deleted") is True or str(root_status.get("lifecycleStatus") or "").lower() == "deleted":
        return "deleted", error_message, metrics

    if job_type == "training" and root_status:
        for key in (
            "loss",
            "epoch",
            "totalEpochs",
            "modelAssetId",
            "checkpointPath",
            "datasetId",
            "datasetName",
            "downstreamModelType",
            "trainingBackend",
            "message",
            "startedAt",
            "updatedAt",
        ):
            if root_status.get(key) is not None:
                metrics[key] = root_status.get(key)

    if job_type == "training" and root_status:
        from app.services.training_job_status import resolve_canonical_training_job_status

        train_job_id = job_root.name
        enriched = resolve_canonical_training_job_status(train_job_id, job_root, root_status)
        status = _normalize_status(enriched.get("status"))
        for key in (
            "loss",
            "epoch",
            "totalEpochs",
            "progress",
            "modelAssetId",
            "checkpointPath",
            "checkpointExists",
            "datasetId",
            "datasetName",
            "downstreamModelType",
            "trainingBackend",
            "message",
            "startedAt",
            "updatedAt",
        ):
            if enriched.get(key) is not None:
                metrics[key] = enriched.get(key)
        if enriched.get("message") and status == "failed":
            error_message = str(enriched.get("message"))
        return status, error_message, metrics

    explicit = status_payload.get("status")
    if explicit:
        status = _normalize_status(explicit)
        if status_payload.get("error"):
            error_message = str(status_payload.get("error"))
        if status in TERMINAL_STATUSES:
            return status, error_message, metrics

    if job_type == "evaluation":
        aggregate = _read_json(job_root / "results" / "aggregate_result.json")
        if aggregate:
            metrics.update(aggregate)
            return "completed", error_message, metrics
        ct_results = _read_json(job_root / "results" / "eval.results.json")
        if ct_results:
            metrics.update(
                {
                    "successRate": ct_results.get("success_rate"),
                    "everSuccessRate": ct_results.get("ever_success_rate"),
                    "numEpisodes": ct_results.get("num_episodes"),
                    "aggregate": ct_results.get("aggregate"),
                }
            )
            return "completed", error_message, metrics

    if job_type == "generate":
        if (job_root / "datasets" / "dataset.npz").is_file() or (
            job_root / "results" / "episode_result.json"
        ).is_file():
            episode_result = _read_json(job_root / "results" / "episode_result.json")
            if episode_result:
                metrics.update(episode_result)
            manifest = _read_json(job_root / "datasets" / "dataset.manifest.json")
            if manifest:
                if manifest.get("num_successful") is not None:
                    metrics.setdefault("successfulEpisodes", manifest.get("num_successful"))
                if manifest.get("num_failed") is not None and manifest.get("num_successful") is not None:
                    metrics.setdefault(
                        "episodes",
                        int(manifest.get("num_successful", 0)) + int(manifest.get("num_failed", 0)),
                    )
            live = live_status or {}
            for key in (
                "episodes",
                "successfulEpisodes",
                "frameCount",
                "savedFrameCount",
                "generateVideoSizeBytes",
            ):
                if live.get(key) is not None:
                    metrics[key] = live.get(key)
            if live.get("finalSuccessRate") is not None:
                metrics["finalSuccessRate"] = live.get("finalSuccessRate")
            size_bytes = 0
            for rel in ("datasets/dataset.hdf5", "datasets/dataset.npz"):
                path = job_root / rel
                if path.is_file():
                    try:
                        size_bytes += path.stat().st_size
                    except OSError:
                        pass
            if size_bytes > 0:
                metrics["sizeBytes"] = size_bytes
            return "completed", error_message, metrics

    if status_payload.get("error") or status_payload.get("message", "").lower().find("fail") >= 0:
        if status_payload.get("error"):
            error_message = str(status_payload.get("error"))
        return "failed", error_message, metrics

    if status_payload:
        return _normalize_status(status_payload.get("status") or "running"), error_message, metrics

    return "unknown", error_message, metrics


def _collect_artifact_specs(
    *,
    job_id: str,
    job_type: str,
    task_type: str,
    job_root: Path,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []

    def add(
        artifact_type: str,
        rel_path: Path,
        *,
        url_path: Optional[str] = None,
        episode_index: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        path = (job_root / rel_path).resolve()
        if not path.is_file():
            return
        specs.append(
            {
                "artifact_type": artifact_type,
                "name": rel_path.name,
                "file_path": str(path),
                "url_path": url_path,
                "episode_index": episode_index,
                "metadata_json": metadata or {},
            }
        )

    add("log", Path("logs/run.log"))
    add("log", Path("logs/eval.log"))
    add("log", Path("logs/train.log"))

    if job_type == "generate":
        if task_type == "cable_threading":
            add(
                "video",
                Path("videos/generate.mp4"),
                url_path=f"/api/workspace/cable-threading/jobs/{job_id}/video",
            )
            add("npz", Path("datasets/dataset.npz"))
            add("hdf5", Path("datasets/dataset.hdf5"))
            add("manifest", Path("datasets/dataset.manifest.json"))
            add("metrics", Path("results/collect.csv"))
            add("failures", Path("results/failures.json"))
        elif task_type == "dual_arm_cable_manipulation":
            add(
                "video",
                Path("videos/generate.mp4"),
                url_path=f"/api/workspace/dual-arm-cable/jobs/{job_id}/video",
            )
            add("episode_result", Path("results/episode_result.json"))
            add("hdf5", Path("datasets/dataset.hdf5"))
            add("manifest", Path("datasets/dataset.manifest.json"))
            add("metrics", Path("datasets/export_report.json"))

    if job_type == "evaluation":
        if task_type == "cable_threading":
            from app.services.workspace_runtime_paths import is_imported_workspace_eval_job_id

            video_api = (
                f"/api/workspace/evaluation/jobs/{job_id}/video"
                if is_imported_workspace_eval_job_id(job_id)
                else f"/api/workspace/cable-threading/jobs/{job_id}/video"
            )
            add(
                "video",
                Path("videos/eval.mp4"),
                url_path=video_api,
            )
            add("metrics", Path("results/eval.results.json"))
            add("aggregate_result", Path("results/eval.results.json"))
        elif task_type == "dual_arm_cable_manipulation":
            add("aggregate_result", Path("results/aggregate_result.json"))
            add("per_episode_result", Path("results/per_episode_results.json"))
            add("metrics", Path("results/aggregate_result.json"))
            videos_dir = job_root / "videos"
            if videos_dir.is_dir():
                for video in sorted(videos_dir.glob("episode_*.mp4")):
                    match = re.search(r"episode_(\d+)", video.stem)
                    ep = int(match.group(1)) if match else None
                    add(
                        "video",
                        Path("videos") / video.name,
                        url_path=f"/api/workspace/evaluation/jobs/{job_id}/video"
                        + (f"?episode={ep}" if ep is not None else ""),
                        episode_index=ep,
                    )

    if job_type == "training":
        add("metrics", Path("artifacts/model_manifest.json"))
        add("manifest", Path("artifacts/dataset_manifest.json"))
        checkpoints = job_root / "checkpoints"
        if checkpoints.is_dir():
            for ckpt in sorted(checkpoints.rglob("*.pth")):
                rel = ckpt.relative_to(job_root)
                add("checkpoint", rel)
            for ckpt in sorted(checkpoints.rglob("*.pt")):
                rel = ckpt.relative_to(job_root)
                add("checkpoint", rel)

    return specs


def _upsert_artifacts(
    db: Session,
    *,
    job_id: str,
    specs: list[dict[str, Any]],
    overwrite: bool,
) -> int:
    if overwrite:
        db.query(WorkspaceArtifact).filter(WorkspaceArtifact.job_id == job_id).delete()

    existing_paths = {
        row.file_path
        for row in db.query(WorkspaceArtifact.file_path)
        .filter(WorkspaceArtifact.job_id == job_id)
        .all()
    }
    inserted = 0
    for spec in specs:
        file_path = spec["file_path"]
        if file_path in existing_paths:
            continue
        db.add(
            WorkspaceArtifact(
                job_id=job_id,
                artifact_type=spec["artifact_type"],
                name=spec["name"],
                file_path=file_path,
                url_path=spec.get("url_path"),
                episode_index=spec.get("episode_index"),
                metadata_json=spec.get("metadata_json") or {},
                created_at=_utc_now(),
            )
        )
        inserted += 1
    return inserted


def _sync_job_record(
    db: Session,
    *,
    job_id: str,
    job_type: str,
    task_type: str,
    runtime_path: str,
    runner: Optional[str],
    source: str = "real",
    task_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    created_by: Optional[str] = None,
    overwrite: bool = False,
    artifact_specs: Optional[list[Any]] = None,
) -> tuple[WorkspaceJob, bool, int]:
    job_root = Path(runtime_path)
    status, error_message, metrics = _infer_status_from_files(job_root, job_type)
    now = _utc_now()

    live_status = _read_json(job_root / "live" / "status.json")
    root_status = _read_json(job_root / "status.json")
    started_at = _parse_dt(
        live_status.get("startedAt")
        or root_status.get("startedAt")
        or root_status.get("createdAt")
    )
    finished_at = None
    if status in TERMINAL_STATUSES:
        finished_at = _parse_dt(root_status.get("updatedAt")) or now

    row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
    created = row is None
    if row is None:
        row = WorkspaceJob(
            job_id=job_id,
            job_type=job_type,
            task_type=task_type,
            task_name=task_name or TASK_NAME_MAP.get(task_type, task_type),
            status=status,
            source=source,
            runner=runner,
            created_by=created_by,
            created_at=started_at or now,
            updated_at=now,
            started_at=started_at,
            finished_at=finished_at,
            runtime_path=runtime_path,
            metadata_json=metadata or {},
            metrics_json=metrics,
            error_message=error_message,
        )
        db.add(row)
    else:
        if row.status == "deleted" and not overwrite:
            pass
        elif overwrite or row.status not in TERMINAL_STATUSES:
            row.status = status
            row.metrics_json = metrics or row.metrics_json
            row.error_message = error_message
            row.updated_at = now
            if finished_at:
                row.finished_at = finished_at
            elif status == "running":
                row.finished_at = None
        elif row.status == "completed" and status == "running":
            row.status = status
            row.metrics_json = metrics or row.metrics_json
            row.error_message = error_message
            row.updated_at = now
            row.finished_at = None
        elif row.status in TERMINAL_STATUSES and metrics:
            row.metrics_json = metrics or row.metrics_json
            row.updated_at = now
        if metadata:
            merged = dict(row.metadata_json or {})
            merged.update(metadata)
            row.metadata_json = merged
        if runner and not row.runner:
            row.runner = runner
        if task_name and not row.task_name:
            row.task_name = task_name
        if row.status == "deleted" and not overwrite:
            db.flush()
            return row, created, 0

    if status == "deleted" and not overwrite:
        db.flush()
        return row, created, 0

    db.flush()
    specs = artifact_specs
    if specs is None:
        specs = _collect_artifact_specs(
            job_id=job_id,
            job_type=job_type,
            task_type=task_type,
            job_root=job_root,
        )
    inserted_artifacts = _upsert_artifacts(db, job_id=job_id, specs=specs, overwrite=overwrite)
    return row, created, inserted_artifacts


def patch_workspace_job_metadata(job_id: str, patch: dict[str, Any]) -> None:
    if not patch:
        return
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
            if row is None:
                return
            merged = dict(row.metadata_json or {})
            merged.update(patch)
            row.metadata_json = merged
            row.updated_at = _utc_now()
            db.commit()
    except Exception as exc:
        logger.warning("patch_workspace_job_metadata failed job_id=%s: %s", job_id, exc)


def record_workspace_job_start(
    *,
    job_id: str,
    job_type: str,
    task_type: str,
    runtime_path: str,
    runner: str,
    source: str = "real",
    task_name: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    created_by: Optional[str] = None,
    status: str = "running",
) -> None:
    try:
        with SessionLocal() as db:
            now = _utc_now()
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
            if row is None:
                row = WorkspaceJob(
                    job_id=job_id,
                    job_type=job_type,
                    task_type=task_type,
                    task_name=task_name or TASK_NAME_MAP.get(task_type, task_type),
                    status=_normalize_status(status),
                    source=source,
                    runner=runner,
                    created_by=created_by,
                    created_at=now,
                    updated_at=now,
                    started_at=now,
                    runtime_path=runtime_path,
                    metadata_json=metadata or {},
                    metrics_json={},
                )
                db.add(row)
            else:
                row.status = _normalize_status(status)
                row.updated_at = now
                row.started_at = row.started_at or now
                if metadata:
                    merged = dict(row.metadata_json or {})
                    merged.update(metadata)
                    row.metadata_json = merged
            db.commit()
    except Exception as exc:
        logger.warning("record_workspace_job_start failed job_id=%s: %s", job_id, exc)


def sync_workspace_job_from_runtime(job_id: str, *, overwrite_artifacts: bool = False) -> None:
    candidate = (job_id or "").strip()
    if candidate.startswith(TRAIN_PREFIX):
        from app.services.training_job_sync_service import sync_training_job_from_runtime

        sync_training_job_from_runtime(candidate, overwrite_artifacts=overwrite_artifacts)
        return
    if candidate.startswith(EVAL_PREFIX) or candidate.startswith("isaac_eval_") or candidate.startswith(CT_EVAL_PREFIX):
        from app.services.training_job_sync_service import sync_eval_job_from_runtime

        sync_eval_job_from_runtime(candidate, overwrite_artifacts=overwrite_artifacts)
        return
    if candidate.startswith("eval_"):
        ct_root = _existing_runtime_path(Path("cable_threading/jobs") / candidate)
        if ct_root.is_dir() and _looks_like_cable_eval_job(ct_root):
            from app.services.training_job_sync_service import sync_eval_job_from_runtime

            sync_eval_job_from_runtime(candidate, overwrite_artifacts=overwrite_artifacts)
            return

    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
            if row is None:
                inferred = infer_job_identity(job_id)
                if inferred is None:
                    return
                job_type, task_type, runtime_path, runner = inferred
                _sync_job_record(
                    db,
                    job_id=job_id,
                    job_type=job_type,
                    task_type=task_type,
                    runtime_path=runtime_path,
                    runner=runner,
                    overwrite=overwrite_artifacts,
                )
            else:
                _sync_job_record(
                    db,
                    job_id=row.job_id,
                    job_type=row.job_type,
                    task_type=row.task_type,
                    runtime_path=row.runtime_path,
                    runner=row.runner,
                    source=row.source,
                    task_name=row.task_name,
                    metadata=row.metadata_json if isinstance(row.metadata_json, dict) else {},
                    overwrite=overwrite_artifacts,
                )
            db.commit()
    except Exception as exc:
        logger.warning("sync_workspace_job_from_runtime failed job_id=%s: %s", job_id, exc)
        return

    _maybe_schedule_artifact_upload(candidate)
    _after_workspace_job_sync(candidate)


def _after_workspace_job_sync(job_id: str) -> None:
    try:
        from app.services.platform_stage2_hooks import after_workspace_job_sync

        after_workspace_job_sync(job_id)
    except Exception as exc:
        logger.warning("stage2 hook schedule failed job_id=%s: %s", job_id, exc)


def record_workspace_job_finish(job_id: str) -> None:
    """任务完成收敛：同步 runtime → PostgreSQL，并异步上传 MinIO 产物。"""
    candidate = (job_id or "").strip()
    if not candidate:
        return
    sync_workspace_job_from_runtime(candidate)
    _maybe_schedule_artifact_upload(candidate)


def _maybe_schedule_artifact_upload(job_id: str) -> None:
    try:
        with SessionLocal() as db:
            row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
            if row is None or row.status not in TERMINAL_STATUSES:
                return
        from app.services.artifact_upload_service import schedule_artifact_upload

        schedule_artifact_upload(job_id)
    except Exception as exc:
        logger.warning("artifact upload schedule failed job_id=%s: %s", job_id, exc)


def infer_job_identity(job_id: str) -> Optional[tuple[str, str, str, str]]:
    if job_id.startswith(CT_GEN_PREFIX):
        runtime = str(_existing_runtime_path(Path("cable_threading/jobs") / job_id))
        return "generate", "cable_threading", runtime, "run.py"
    if job_id.startswith(CT_EVAL_PREFIX):
        runtime = str(_existing_runtime_path(Path("cable_threading/jobs") / job_id))
        return "evaluation", "cable_threading", runtime, "run.py"
    if job_id.startswith("eval_"):
        ct_runtime = _existing_runtime_path(Path("cable_threading/jobs") / job_id)
        if ct_runtime.is_dir() and _looks_like_cable_eval_job(ct_runtime):
            return "evaluation", "cable_threading", str(ct_runtime), "run.py"
        runtime = str(RUNTIME_ROOT / "evaluations" / "jobs" / job_id)
        return "evaluation", "dual_arm_cable_manipulation", runtime, "dual_arm_cable_eval_worker.py"
    if job_id.startswith(DAC_GEN_PREFIX):
        runtime = str(_existing_runtime_path(Path("dual_arm_cable/jobs") / job_id))
        return "generate", "dual_arm_cable_manipulation", runtime, "platform_runner.py"
    if job_id.startswith(EVAL_PREFIX):
        runtime = str(RUNTIME_ROOT / "evaluations" / "jobs" / job_id)
        return "evaluation", "dual_arm_cable_manipulation", runtime, "dual_arm_cable_eval_worker.py"
    if job_id.startswith(TRAIN_PREFIX):
        runtime = str(RUNTIME_ROOT / "training" / "jobs" / job_id)
        return "training", "unknown", runtime, "train_bc.py"
    return None


def _iter_runtime_job_dirs(
    *,
    task_type: Optional[str],
    job_type: Optional[str],
) -> list[tuple[str, str, str, Path]]:
    entries: list[tuple[str, str, str, Path]] = []

    def maybe_add(job_id: str, jtype: str, ttype: str, runner: str, root: Path) -> None:
        if job_type and jtype != job_type:
            return
        if task_type and ttype != task_type:
            return
        if not root.is_dir():
            return
        entries.append((job_id, jtype, ttype, runner, root))

    seen_ct: set[str] = set()
    for runtime_root in _runtime_roots():
        ct_jobs = runtime_root / "cable_threading" / "jobs"
        if not ct_jobs.is_dir():
            continue
        for path in ct_jobs.iterdir():
            if not path.is_dir():
                continue
            jid = path.name
            if jid in seen_ct:
                continue
            seen_ct.add(jid)
            if jid.startswith(CT_GEN_PREFIX):
                maybe_add(jid, "generate", "cable_threading", "run.py", path)
            elif jid.startswith(CT_EVAL_PREFIX):
                maybe_add(jid, "evaluation", "cable_threading", "run.py", path)
            elif jid.startswith("eval_") and _looks_like_cable_eval_job(path):
                maybe_add(jid, "evaluation", "cable_threading", "run.py", path)

    seen_dac: set[str] = set()
    for runtime_root in _runtime_roots():
        dac_jobs = runtime_root / "dual_arm_cable" / "jobs"
        if not dac_jobs.is_dir():
            continue
        for path in dac_jobs.iterdir():
            if not path.is_dir() or not path.name.startswith(DAC_GEN_PREFIX):
                continue
            if path.name in seen_dac:
                continue
            seen_dac.add(path.name)
            maybe_add(path.name, "generate", "dual_arm_cable_manipulation", "platform_runner.py", path)

    seen_eval: set[str] = set()
    for runtime_root in _runtime_roots():
        eval_jobs = runtime_root / "evaluations" / "jobs"
        if not eval_jobs.is_dir():
            continue
        for path in eval_jobs.iterdir():
            if not path.is_dir() or not path.name.startswith(EVAL_PREFIX):
                continue
            if path.name in seen_eval:
                continue
            seen_eval.add(path.name)
            req = _read_json(path / "metadata" / "evaluation_request.json")
            ttype = str(req.get("taskType") or "dual_arm_cable_manipulation")
            maybe_add(path.name, "evaluation", ttype, "dual_arm_cable_eval_worker.py", path)

    train_jobs = RUNTIME_ROOT / "training" / "jobs"
    if train_jobs.is_dir():
        for path in train_jobs.iterdir():
            if path.is_dir() and path.name.startswith(TRAIN_PREFIX):
                manifest = _read_json(path / "artifacts" / "dataset_manifest.json")
                ttype = str(manifest.get("taskType") or "unknown")
                maybe_add(path.name, "training", ttype, "train_bc.py", path)

    return entries


def reindex_workspace_runtime_jobs(
    *,
    task_type: Optional[str] = None,
    job_type: Optional[str] = None,
    dry_run: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    result = {
        "scanned": 0,
        "insertedJobs": 0,
        "updatedJobs": 0,
        "insertedArtifacts": 0,
        "skipped": 0,
        "errors": [],
    }

    entries = _iter_runtime_job_dirs(task_type=task_type, job_type=job_type)
    result["scanned"] = len(entries)
    if dry_run:
        return result

    try:
        with SessionLocal() as db:
            for job_id, jtype, ttype, runner, job_root in entries:
                try:
                    metadata: dict[str, Any] = {}
                    if jtype == "evaluation" and ttype == "dual_arm_cable_manipulation":
                        metadata["evaluationRequest"] = _read_json(
                            job_root / "metadata" / "evaluation_request.json"
                        )
                    if jtype == "training":
                        metadata["trainConfig"] = _read_json(job_root / "config" / "train_config.json")

                    _, created, inserted = _sync_job_record(
                        db,
                        job_id=job_id,
                        job_type=jtype,
                        task_type=ttype,
                        runtime_path=str(job_root),
                        runner=runner,
                        metadata=metadata,
                        overwrite=overwrite,
                    )
                    if created:
                        result["insertedJobs"] += 1
                    else:
                        result["updatedJobs"] += 1
                    result["insertedArtifacts"] += inserted
                except Exception as exc:
                    result["errors"].append(f"{job_id}: {exc}")
            db.commit()
    except Exception as exc:
        result["errors"].append(str(exc))

    return result


# Async query helpers for FastAPI routes


async def list_workspace_jobs_async(
    db: Any,
    *,
    job_type: Optional[str],
    task_type: Optional[str],
    status: Optional[str],
    source: Optional[str],
    limit: int,
    offset: int,
) -> tuple[list[dict[str, Any]], int]:
    stmt = select(WorkspaceJob).options(selectinload(WorkspaceJob.artifacts))
    count_stmt = select(func.count()).select_from(WorkspaceJob)

    if job_type:
        stmt = stmt.where(WorkspaceJob.job_type == job_type)
        count_stmt = count_stmt.where(WorkspaceJob.job_type == job_type)
    if task_type:
        stmt = stmt.where(WorkspaceJob.task_type == task_type)
        count_stmt = count_stmt.where(WorkspaceJob.task_type == task_type)
    if status:
        stmt = stmt.where(WorkspaceJob.status == status)
        count_stmt = count_stmt.where(WorkspaceJob.status == status)
    if source:
        stmt = stmt.where(WorkspaceJob.source == source)
        count_stmt = count_stmt.where(WorkspaceJob.source == source)

    total = (await db.scalar(count_stmt)) or 0
    rows = (
        await db.scalars(
            stmt.order_by(WorkspaceJob.created_at.desc()).offset(offset).limit(limit)
        )
    ).all()

    stale_training_ids = [
        row.job_id
        for row in rows
        if row.job_type == "training" and row.status not in TERMINAL_STATUSES
    ]
    if stale_training_ids:
        from app.services.training_job_sync_service import sync_training_job_from_runtime

        for job_id in stale_training_ids:
            sync_training_job_from_runtime(job_id)
        rows = (
            await db.scalars(
                stmt.order_by(WorkspaceJob.created_at.desc()).offset(offset).limit(limit)
            )
        ).all()

    return [_job_to_summary(row) for row in rows], int(total)


async def get_workspace_job_async(db: Any, job_id: str) -> Optional[dict[str, Any]]:
    row = await db.scalar(
        select(WorkspaceJob)
        .options(selectinload(WorkspaceJob.artifacts))
        .where(WorkspaceJob.job_id == job_id)
    )
    if row is None:
        return None
    return _job_to_detail(row)


async def list_workspace_job_artifacts_async(db: Any, job_id: str) -> Optional[list[dict[str, Any]]]:
    row = await db.scalar(select(WorkspaceJob).where(WorkspaceJob.job_id == job_id))
    if row is None:
        return None
    artifacts = (
        await db.scalars(
            select(WorkspaceArtifact)
            .where(WorkspaceArtifact.job_id == job_id)
            .order_by(WorkspaceArtifact.created_at.asc())
        )
    ).all()
    return [_artifact_to_dict(art) for art in artifacts]


def _resolve_runtime_path(runtime_path: str) -> Path:
    text = (runtime_path or "").strip()
    if not text:
        raise WorkspaceJobDeleteError("runtime_path is empty")
    return resolve_runtime_reference(text)


def _validate_runtime_delete_path(runtime_path: str) -> Path:
    target = _resolve_runtime_path(runtime_path)
    roots = (RUNTIME_ROOT.resolve(),)

    forbidden_targets = (*FORBIDDEN_RUNTIME_DELETE_TARGETS, *roots)
    for forbidden in forbidden_targets:
        if target == forbidden.resolve():
            raise WorkspaceJobDeleteError(f"refuse to delete forbidden path: {target}")

    if not any(is_path_within(target, root) for root in roots):
        raise WorkspaceJobDeleteError(f"unsafe path: {target} is not under a runtime root")

    return target


def _delete_runtime_job_directory(job_id: str, runtime_path: str) -> tuple[bool, Optional[str]]:
    """Delete a runs job directory. Returns (runtime_deleted, reason)."""
    text = (runtime_path or "").strip()
    if not text:
        logger.warning(
            "workspace_job_delete skip runtime (empty path): job_id=%s",
            job_id,
        )
        return False, "runtime_path_empty"

    try:
        target = _validate_runtime_delete_path(text)
    except WorkspaceJobDeleteError as exc:
        message = exc.message
        if (
            "unsafe path" in message
            or "forbidden path" in message
            or "runs root" in message
        ):
            logger.warning(
                "workspace_job_delete skip unsafe runtime path: job_id=%s runtime_path=%s reason=%s",
                job_id,
                runtime_path,
                message,
            )
            return False, "unsafe_runtime_path"
        raise

    logger.info(
        "workspace_job_delete runtime before: job_id=%s runtime_path=%s resolved=%s",
        job_id,
        runtime_path,
        target,
    )

    if not target.exists():
        logger.warning(
            "workspace_job_delete runtime missing: job_id=%s resolved=%s",
            job_id,
            target,
        )
        return False, "runtime_path_not_found"

    try:
        shutil.rmtree(target)
    except OSError as exc:
        logger.exception(
            "workspace_job_delete runtime failed: job_id=%s resolved=%s",
            job_id,
            target,
        )
        raise RuntimeDeleteFailedError(f"failed to delete runtime directory: {exc}") from exc

    logger.info(
        "workspace_job_delete runtime after: job_id=%s resolved=%s deleted=true",
        job_id,
        target,
    )
    return True, None


async def delete_workspace_job_async(db: Any, job_id: str) -> Optional[dict[str, Any]]:
    """Delete workspace job, artifacts, and runs directory."""
    row = await db.scalar(
        select(WorkspaceJob)
        .options(selectinload(WorkspaceJob.artifacts))
        .where(WorkspaceJob.job_id == job_id)
    )
    if row is None:
        return None

    runtime_path = row.runtime_path or ""
    artifacts = list(row.artifacts or [])
    artifact_count = len(artifacts)
    deleted_model_assets = 0

    logger.info(
        "workspace_job_delete start: job_id=%s job_type=%s task_type=%s runtime_path=%s artifacts=%s",
        job_id,
        row.job_type,
        row.task_type,
        runtime_path,
        artifact_count,
    )

    if row.job_type == "training":
        from app.services import training_service as training_svc

        training_svc.stop_training_job_if_active(job_id)

    try:
        runtime_deleted, reason = _delete_runtime_job_directory(job_id, runtime_path)
    except RuntimeDeleteFailedError as exc:
        if row.job_type == "training":
            logger.warning(
                "workspace_job_delete runtime failed for training job, continue db delete: job_id=%s error=%s",
                job_id,
                exc.message,
            )
            runtime_deleted, reason = False, exc.message
        else:
            raise

    if row.job_type == "training":
        from app.models.workspace_index import ModelAsset, TrainingMetricSummary

        model_assets = list(
            (await db.scalars(select(ModelAsset).where(ModelAsset.train_job_id == job_id))).all()
        )
        deleted_model_assets = len(model_assets)
        for asset in model_assets:
            await db.delete(asset)
        metric_row = await db.scalar(
            select(TrainingMetricSummary).where(TrainingMetricSummary.job_id == job_id)
        )
        if metric_row is not None:
            await db.delete(metric_row)

    for artifact in artifacts:
        await db.delete(artifact)
    await db.delete(row)
    await db.flush()

    result = {
        "success": True,
        "jobId": job_id,
        "deletedJob": True,
        "deletedArtifacts": artifact_count,
        "deletedModelAssets": deleted_model_assets,
        "runtimeDeleted": runtime_deleted,
        "runtimePath": runtime_path,
        "canReindexRecover": False,
        "jobType": row.job_type,
        "taskType": row.task_type,
    }
    if reason:
        result["reason"] = reason
    if reason in {"unsafe_runtime_path", "runtime_path_not_found", "runtime_path_empty"}:
        result["warning"] = reason

    logger.info(
        "workspace_job_delete done: job_id=%s deleted_artifacts=%s deleted_model_assets=%s runtime_deleted=%s reason=%s",
        job_id,
        artifact_count,
        deleted_model_assets,
        runtime_deleted,
        reason,
    )
    return result
