"""Extended workspace reindex orchestration with dataset backfill and deleted-job safety."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from app.core.platform_paths import platform_paths
from app.services.runtime_job_lifecycle import is_job_deleted, read_json_dict
from app.services.training_job_sync_service import (
    TRAINING_JOBS_ROOT,
    sync_eval_job_from_runtime,
    sync_training_job_from_runtime,
)
from app.services.workspace_dataset_backfill_service import backfill_hdf5_dataset_records

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CABLE_THREADING_JOBS = platform_paths.runs_root / "cable_threading" / "jobs"
EVAL_JOBS_ROOT = platform_paths.evaluation_jobs
CABLE_THREADING_JOB_ROOTS = (CABLE_THREADING_JOBS,)
EVAL_JOB_ROOTS = (EVAL_JOBS_ROOT,)

TRAIN_JOB_DIR_PATTERN = re.compile(r"^train_")


def normalize_reindex_job_type(job_type: Optional[str]) -> Optional[str]:
    if not job_type or job_type in {"all", ""}:
        return None
    mapping = {
        "data_generation": "generate",
        "generate": "generate",
        "training": "training",
        "evaluation": "evaluation",
    }
    return mapping.get(job_type, job_type)


def _runtime_job_deleted(job_root: Path) -> bool:
    for rel in ("status.json", "live/status.json", "metadata/status.json"):
        payload = read_json_dict(job_root / rel)
        if is_job_deleted(payload):
            return True
    deleted_marker = job_root / "deleted.json"
    if deleted_marker.is_file():
        return True
    return False


def _is_train_runtime_dir(job_dir: Path) -> bool:
    if not job_dir.is_dir() or not TRAIN_JOB_DIR_PATTERN.match(job_dir.name):
        return False
    return (job_dir / "status.json").is_file() or (job_dir / "config" / "train_config.json").is_file()


def _looks_like_eval_job(job_root: Path) -> bool:
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


def _iter_eval_job_ids_for_sync() -> list[str]:
    job_ids: list[str] = []
    seen: set[str] = set()

    def add(job_id: str) -> None:
        if job_id and job_id not in seen:
            seen.add(job_id)
            job_ids.append(job_id)

    for jobs_root in CABLE_THREADING_JOB_ROOTS:
        if not jobs_root.is_dir():
            continue
        for path in sorted(jobs_root.iterdir()):
            if not path.is_dir():
                continue
            name = path.name
            if name.startswith("ct_eval_") or (name.startswith("eval_") and _looks_like_eval_job(path)):
                add(name)

    for jobs_root in EVAL_JOB_ROOTS:
        if not jobs_root.is_dir():
            continue
        for path in sorted(jobs_root.iterdir()):
            if path.is_dir() and path.name.startswith("eval_"):
                add(path.name)

    return job_ids


def _iter_train_job_ids_for_sync() -> list[str]:
    if not TRAINING_JOBS_ROOT.is_dir():
        return []
    return sorted(
        path.name
        for path in TRAINING_JOBS_ROOT.iterdir()
        if _is_train_runtime_dir(path)
    )


def reindex_workspace_all(
    *,
    task_type: Optional[str] = None,
    job_type: Optional[str] = None,
    dry_run: bool = False,
    overwrite: bool = False,
    restore_deleted: bool = False,
) -> dict[str, Any]:
    """Full reindex: workspace jobs, training/eval summaries, model assets, dataset tables."""
    from app.services.workspace_job_service import reindex_workspace_runtime_jobs

    normalized_job_type = normalize_reindex_job_type(job_type)
    result: dict[str, Any] = {
        "scanned": 0,
        "insertedJobs": 0,
        "updatedJobs": 0,
        "insertedArtifacts": 0,
        "skipped": 0,
        "skippedDeleted": 0,
        "errors": [],
        "syncedTrainingJobs": 0,
        "syncedTrainingAssets": 0,
        "syncedEvalJobs": 0,
        "syncErrors": [],
        "scannedDatasets": 0,
        "insertedHdf5Datasets": 0,
        "updatedHdf5Datasets": 0,
        "insertedDataAssets": 0,
        "updatedDataAssets": 0,
        "skippedDatasets": 0,
    }

    if dry_run:
        from app.services.workspace_job_service import _iter_runtime_job_dirs

        entries = _iter_runtime_job_dirs(task_type=task_type, job_type=normalized_job_type)
        result["scanned"] = len(entries)
        result["scannedDatasets"] = len(
            __import__(
                "app.services.workspace_dataset_backfill_service",
                fromlist=["discover_dataset_hdf5_paths"],
            ).discover_dataset_hdf5_paths()
        )
        if normalized_job_type is None or normalized_job_type == "training":
            result["syncedTrainingJobs"] = len(_iter_train_job_ids_for_sync())
        if normalized_job_type is None or normalized_job_type == "evaluation":
            result["syncedEvalJobs"] = len(_iter_eval_job_ids_for_sync())
        return result

    if restore_deleted:
        workspace_result = reindex_workspace_runtime_jobs(
            task_type=task_type,
            job_type=normalized_job_type,
            dry_run=False,
            overwrite=overwrite,
        )
    else:
        workspace_result = _reindex_workspace_skip_deleted(
            task_type=task_type,
            job_type=normalized_job_type,
            overwrite=overwrite,
        )
    result.update({k: workspace_result.get(k, result.get(k, 0)) for k in (
        "scanned", "insertedJobs", "updatedJobs", "insertedArtifacts", "skipped", "errors",
    )})
    result["skippedDeleted"] = int(workspace_result.get("skippedDeleted") or 0)

    if normalized_job_type is None or normalized_job_type == "training":
        for job_id in _iter_train_job_ids_for_sync():
            job_dir = TRAINING_JOBS_ROOT / job_id
            if not restore_deleted and _runtime_job_deleted(job_dir):
                result["skippedDeleted"] += 1
                continue
            try:
                sync_training_job_from_runtime(job_id, overwrite_artifacts=overwrite)
                result["syncedTrainingJobs"] += 1
            except Exception as exc:
                result["syncErrors"].append(f"train {job_id}: {exc}")

    if normalized_job_type is None or normalized_job_type == "evaluation":
        for job_id in _iter_eval_job_ids_for_sync():
            job_root = next(
                (
                    root / job_id
                    for root in (*CABLE_THREADING_JOB_ROOTS, *EVAL_JOB_ROOTS)
                    if (root / job_id).is_dir()
                ),
                CABLE_THREADING_JOBS / job_id,
            )
            if not restore_deleted and _runtime_job_deleted(job_root):
                result["skippedDeleted"] += 1
                continue
            try:
                sync_eval_job_from_runtime(job_id, overwrite_artifacts=overwrite)
                result["syncedEvalJobs"] += 1
            except Exception as exc:
                result["syncErrors"].append(f"eval {job_id}: {exc}")

    if normalized_job_type is None or normalized_job_type == "generate":
        dataset_result = backfill_hdf5_dataset_records(dry_run=False, overwrite=overwrite)
        for key in (
            "scannedDatasets",
            "insertedHdf5Datasets",
            "updatedHdf5Datasets",
            "insertedDataAssets",
            "updatedDataAssets",
            "skippedDatasets",
        ):
            result[key] = dataset_result.get(key, 0)
        result["errors"].extend(dataset_result.get("errors") or [])

    return result


def _reindex_workspace_skip_deleted(
    *,
    task_type: Optional[str],
    job_type: Optional[str],
    overwrite: bool,
) -> dict[str, Any]:
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceJob
    from app.services.workspace_job_service import _iter_runtime_job_dirs, _sync_job_record, _read_json

    result = {
        "scanned": 0,
        "insertedJobs": 0,
        "updatedJobs": 0,
        "insertedArtifacts": 0,
        "skipped": 0,
        "skippedDeleted": 0,
        "errors": [],
    }
    entries = _iter_runtime_job_dirs(task_type=task_type, job_type=job_type)
    result["scanned"] = len(entries)

    try:
        with SessionLocal() as db:
            for job_id, jtype, ttype, runner, job_root in entries:
                if _runtime_job_deleted(job_root):
                    row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
                    if row is not None and row.status == "deleted":
                        result["skippedDeleted"] += 1
                        continue
                    if row is None:
                        result["skippedDeleted"] += 1
                        continue
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
