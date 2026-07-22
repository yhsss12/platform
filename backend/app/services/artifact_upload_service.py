"""异步产物上传：checkpoint / evaluation / dataset → MinIO + PostgreSQL 索引。"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.platform_paths import platform_paths, resolve_runtime_reference
from app.models.workspace_index import EvalMetricSummary, ModelAsset
from app.models.workspace_job import WorkspaceArtifact, WorkspaceJob
from app.services.artifact_storage_registry import (
    get_artifact_record,
    mark_artifact_failed,
    mark_artifact_uploaded,
    register_artifact_pending,
    should_skip_upload,
)
from app.services.storage.storage_service import StorageService
from app.services.training_job_sync_service import path_to_storage_uri

logger = logging.getLogger(__name__)

PROJECT_ROOT = platform_paths.project_root
RUNTIME_ROOT = platform_paths.runs_root

EVAL_ARTIFACT_NAMES = (
    ("eval_aggregate", "results/aggregate_result.json"),
    ("eval_report_json", "results/eval.results.json"),
    ("eval_report_md", "report.md"),
)

EVAL_REPORT_GLOBS = (
    ("eval_report_pdf", "results", "*.pdf"),
    ("eval_report_pdf", "reports", "*.pdf"),
)

DATASET_CANDIDATE_PATHS = (
    "datasets/dataset.hdf5",
    "datasets/dataset.npz",
    "datasets/dataset.mcap",
)

def _runtime_roots() -> tuple[Path, ...]:
    return (RUNTIME_ROOT,)


def _runtime_scan_roots() -> tuple[Path, ...]:
    return tuple(
        runtime_root / rel
        for runtime_root in _runtime_roots()
        for rel in (
            Path("training/jobs"),
            Path("evaluations/jobs"),
            Path("cable_threading/jobs"),
            Path("dual_arm_cable/jobs"),
            Path("data_generation/jobs"),
        )
    )

TERMINAL_STATUS_FILES = ("status.json", "live/status.json", "metadata/status.json")


def artifact_upload_enabled() -> bool:
    explicit = getattr(settings, "ARTIFACT_UPLOAD_ENABLED", None)
    if explicit is not None:
        return bool(explicit)
    return StorageService.is_remote_storage_enabled()


def _read_terminal_status(job_root: Path) -> Optional[str]:
    import json

    for rel in TERMINAL_STATUS_FILES:
        path = job_root / rel
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                status = str(data.get("status") or "").strip().lower()
                if status in {"completed", "failed", "canceled", "cancelled", "success", "succeeded"}:
                    return status
        except (OSError, json.JSONDecodeError):
            continue
    return None


def discover_terminal_job_ids(*, limit: int = 50) -> list[str]:
    """扫描 runs 下已终态任务。"""
    candidates: list[tuple[float, str]] = []
    for root in _runtime_scan_roots():
        if not root.is_dir():
            continue
        for job_dir in root.iterdir():
            if not job_dir.is_dir():
                continue
            if _read_terminal_status(job_dir) is None:
                continue
            try:
                mtime = job_dir.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((mtime, job_dir.name))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [job_id for _, job_id in candidates[:limit]]


def discover_runtime_job_ids(*, include_non_terminal: bool = False, limit: int = 10_000) -> list[str]:
    """扫描 runs 发现任务 ID。"""
    if not include_non_terminal:
        return discover_terminal_job_ids(limit=limit)
    ids: set[str] = set()
    for root in _runtime_scan_roots():
        if not root.is_dir():
            continue
        for job_dir in root.iterdir():
            if job_dir.is_dir():
                ids.add(job_dir.name)
    return sorted(ids)[:limit]


def _checkpoint_bucket() -> str:
    return (getattr(settings, "CHECKPOINT_ARCHIVE_BUCKET", None) or "eai-checkpoints").strip()


def _workspace_bucket() -> str:
    return (getattr(settings, "WORKSPACE_ARTIFACT_BUCKET", None) or "eai-workspace-artifacts").strip()


def _upload_local_file(
    *,
    owner_type: str,
    owner_id: str,
    artifact_type: str,
    content_key: str,
    local_path: Path,
    object_key: str,
    bucket: Optional[str] = None,
) -> Optional[str]:
    if not local_path.is_file():
        return None

    existing = get_artifact_record(
        owner_type=owner_type,
        owner_id=owner_id,
        artifact_type=artifact_type,
        content_key=content_key,
    )
    if should_skip_upload(existing):
        return str(existing.storage_uri) if existing else None

    register_artifact_pending(
        owner_type=owner_type,
        owner_id=owner_id,
        artifact_type=artifact_type,
        content_key=content_key,
        local_path=local_path,
    )

    try:
        storage_uri = StorageService.upload_file(
            local_path,
            object_key,
            bucket=bucket or _workspace_bucket(),
        )
        mark_artifact_uploaded(
            owner_type=owner_type,
            owner_id=owner_id,
            artifact_type=artifact_type,
            content_key=content_key,
            storage_uri=storage_uri,
            local_path=local_path,
        )
        if storage_uri.startswith("minio://"):
            try:
                from app.services.platform_stage2_hooks import emit_artifact_uploaded

                emit_artifact_uploaded(
                    job_id=owner_id,
                    artifact_type=artifact_type,
                    storage_uri=storage_uri,
                    content_key=content_key,
                )
            except Exception:
                pass
        return storage_uri
    except Exception as exc:
        mark_artifact_failed(
            owner_type=owner_type,
            owner_id=owner_id,
            artifact_type=artifact_type,
            content_key=content_key,
            error=str(exc),
        )
        logger.warning(
            "artifact upload failed owner=%s/%s type=%s key=%s: %s",
            owner_type,
            owner_id,
            artifact_type,
            content_key,
            exc,
        )
        return None


def upload_training_checkpoints(train_job_id: str) -> dict[str, Any]:
    """上传训练 checkpoint；final 必传，中间 checkpoint 可选。"""
    job_id = (train_job_id or "").strip()
    result: dict[str, Any] = {"jobId": job_id, "uploaded": 0, "skipped": 0, "warnings": []}
    if not job_id or not artifact_upload_enabled():
        return result

    include_intermediate = bool(getattr(settings, "ARTIFACT_UPLOAD_INTERMEDIATE_CHECKPOINTS", False))
    bucket = _checkpoint_bucket()

    with SessionLocal() as db:
        rows = (
            db.query(ModelAsset)
            .filter(
                ModelAsset.train_job_id == job_id,
                ModelAsset.status.in_(["ready", "available", "generating"]),
            )
            .all()
        )

    for row in rows:
        is_final = row.asset_type == "final" or str(row.model_asset_id or "").endswith("_final")
        is_best = row.asset_type == "best"
        if row.asset_type == "epoch" and not include_intermediate and not is_final:
            result["skipped"] += 1
            continue
        if row.asset_type not in ("final", "epoch", "best") and not include_intermediate:
            result["skipped"] += 1
            continue
        if is_best or is_final:
            pass  # always upload

        local_path = StorageService.local_path_from_uri(str(row.storage_uri or ""))
        if local_path is None or not local_path.is_file():
            manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
            cache = manifest.get("localCachePath") or manifest.get("checkpointPath")
            if cache:
                local_path = StorageService.local_path_from_uri(str(cache))
        if local_path is None or not local_path.is_file():
            result["warnings"].append(f"{row.model_asset_id}: missing local checkpoint")
            continue

        content_key = row.model_asset_id or local_path.name
        object_key = f"checkpoints/{job_id}/{local_path.name}"
        storage_uri = _upload_local_file(
            owner_type="train",
            owner_id=job_id,
            artifact_type="checkpoint_final" if is_final else "checkpoint_epoch",
            content_key=content_key,
            local_path=local_path,
            object_key=object_key,
            bucket=bucket,
        )
        if not storage_uri:
            continue
        if storage_uri.startswith("minio://"):
            result["uploaded"] += 1
            with SessionLocal() as db:
                db_row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == row.model_asset_id).one_or_none()
                if db_row is not None:
                    db_row.storage_uri = storage_uri
                    manifest = dict(db_row.manifest_json or {})
                    manifest["localCachePath"] = path_to_storage_uri(local_path)
                    manifest["isFinalCheckpoint"] = is_final
                    db_row.manifest_json = manifest
                    db.commit()
        else:
            result["skipped"] += 1

    return result


def upload_evaluation_artifacts(eval_job_id: str, *, job_root: Optional[Path] = None) -> dict[str, Any]:
    job_id = (eval_job_id or "").strip()
    result: dict[str, Any] = {"jobId": job_id, "uploaded": 0, "artifacts": []}
    if not job_id or not artifact_upload_enabled():
        return result

    root = job_root or _resolve_eval_root(job_id)
    if root is None or not root.is_dir():
        return result

    bucket = _workspace_bucket()
    report_uri: Optional[str] = None
    replay_uri: Optional[str] = None
    summary_paths: dict[str, str] = {}

    for artifact_type, rel in EVAL_ARTIFACT_NAMES:
        path = root / rel
        if not path.is_file():
            alt = root / "artifacts" / Path(rel).name
            path = alt if alt.is_file() else path
        if not path.is_file():
            continue
        storage_uri = _upload_local_file(
            owner_type="eval",
            owner_id=job_id,
            artifact_type=artifact_type,
            content_key=rel,
            local_path=path,
            object_key=f"evaluations/{job_id}/{Path(rel).name}",
            bucket=bucket,
        )
        if storage_uri:
            result["uploaded"] += 1
            result["artifacts"].append({"type": artifact_type, "uri": storage_uri})
            summary_paths[artifact_type] = storage_uri
            if artifact_type == "eval_aggregate":
                report_uri = storage_uri
            if artifact_type.startswith("eval_report"):
                report_uri = report_uri or storage_uri

    for artifact_type, subdir, glob_pat in EVAL_REPORT_GLOBS:
        search_dir = root / subdir
        if not search_dir.is_dir():
            continue
        for pdf in sorted(search_dir.glob(glob_pat)):
            if not pdf.is_file():
                continue
            rel = str(pdf.relative_to(root))
            storage_uri = _upload_local_file(
                owner_type="eval",
                owner_id=job_id,
                artifact_type=artifact_type,
                content_key=rel,
                local_path=pdf,
                object_key=f"evaluations/{job_id}/{pdf.name}",
                bucket=bucket,
            )
            if storage_uri:
                result["uploaded"] += 1
                result["artifacts"].append({"type": artifact_type, "uri": storage_uri})
                summary_paths[rel] = storage_uri
                report_uri = report_uri or storage_uri

    videos_dir = root / "videos"
    if videos_dir.is_dir():
        for video in sorted(videos_dir.glob("*.mp4")):
            rel = f"videos/{video.name}"
            storage_uri = _upload_local_file(
                owner_type="eval",
                owner_id=job_id,
                artifact_type="eval_video",
                content_key=rel,
                local_path=video,
                object_key=f"evaluations/{job_id}/{rel}",
                bucket=bucket,
            )
            if storage_uri:
                result["uploaded"] += 1
                replay_uri = replay_uri or storage_uri
                summary_paths[f"video:{video.name}"] = storage_uri

    _update_eval_db_uris(job_id, report_uri=report_uri, replay_uri=replay_uri, summary_paths=summary_paths, root=root)
    return result


def upload_dataset_artifacts(owner_id: str, job_root: Path) -> dict[str, Any]:
    """上传 generate job 下的 dataset 文件。"""
    result: dict[str, Any] = {"ownerId": owner_id, "uploaded": 0}
    if not artifact_upload_enabled() or not job_root.is_dir():
        return result

    bucket = _workspace_bucket()
    for rel in DATASET_CANDIDATE_PATHS:
        path = job_root / rel
        if not path.is_file():
            continue
        artifact_type = "dataset_hdf5" if rel.endswith(".hdf5") else "dataset_file"
        storage_uri = _upload_local_file(
            owner_type="dataset",
            owner_id=owner_id,
            artifact_type=artifact_type,
            content_key=rel,
            local_path=path,
            object_key=f"datasets/{owner_id}/{path.name}",
            bucket=bucket,
        )
        if storage_uri and storage_uri.startswith("minio://"):
            result["uploaded"] += 1
            _update_workspace_artifact_uri(owner_id, rel, storage_uri, path)
            _update_data_assets_minio_path(owner_id, path, storage_uri)
    return result


def upload_data_asset_file(dataset_id: str, local_path: Path) -> Optional[str]:
    """data_assets 流程：上传 HDF5/MCAP/NPZ 并写 minio_path。"""
    ds_id = (dataset_id or "").strip()
    if not ds_id or not local_path.is_file() or not artifact_upload_enabled():
        return None

    object_key = f"data-assets/{ds_id}/{local_path.name}"
    storage_uri = _upload_local_file(
        owner_type="dataset",
        owner_id=ds_id,
        artifact_type="data_asset_file",
        content_key=local_path.name,
        local_path=local_path,
        object_key=object_key,
        bucket=_workspace_bucket(),
    )
    if storage_uri:
        _update_data_assets_minio_path(ds_id, local_path, storage_uri)
    return storage_uri


def _resolve_eval_root(job_id: str) -> Optional[Path]:
    candidates = [
        root / rel / job_id
        for root in _runtime_roots()
        for rel in (Path("evaluations/jobs"), Path("cable_threading/jobs"))
    ]
    for path in candidates:
        if path.is_dir():
            return path
    with SessionLocal() as db:
        row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
        if row and row.runtime_path:
            path = resolve_runtime_reference(row.runtime_path)
            if path.is_dir():
                return path
    return None


def _update_eval_db_uris(
    job_id: str,
    *,
    report_uri: Optional[str],
    replay_uri: Optional[str],
    summary_paths: dict[str, str],
    root: Path,
) -> None:
    try:
        from app.services.training_job_sync_service import _extract_eval_metric_columns

        with SessionLocal() as db:
            row = db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == job_id).one_or_none()
            summary = dict(row.summary_json or {}) if row is not None else {}
            if summary_paths:
                summary["storagePaths"] = summary_paths
            success_rate, average_score = _extract_eval_metric_columns(summary)
            if row is None:
                db.add(
                    EvalMetricSummary(
                        job_id=job_id,
                        report_uri=report_uri,
                        replay_uri=replay_uri,
                        summary_json=summary or None,
                        success_rate=success_rate,
                        average_score=average_score,
                    )
                )
            else:
                if report_uri:
                    row.report_uri = report_uri
                if replay_uri:
                    row.replay_uri = replay_uri
                row.summary_json = summary or row.summary_json
                row.success_rate = success_rate
                row.average_score = average_score
            for rel, uri in summary_paths.items():
                if rel.startswith("video:"):
                    name = rel.split(":", 1)[1]
                    artifact_type = "eval_video"
                    file_path = str((root / "videos" / name).resolve())
                else:
                    artifact_type = rel.split("/")[0] if "/" in rel else "eval_artifact"
                    file_path = str((root / rel).resolve())
                existing = (
                    db.query(WorkspaceArtifact)
                    .filter(
                        WorkspaceArtifact.job_id == job_id,
                        WorkspaceArtifact.file_path == file_path,
                    )
                    .one_or_none()
                )
                if existing is not None:
                    existing.url_path = uri
                else:
                    db.add(
                        WorkspaceArtifact(
                            job_id=job_id,
                            artifact_type=artifact_type,
                            name=Path(file_path).name,
                            file_path=file_path,
                            url_path=uri,
                        )
                    )
            db.commit()
    except Exception as exc:
        logger.warning("update eval db uris failed job_id=%s: %s", job_id, exc)


def _update_workspace_artifact_uri(job_id: str, rel: str, storage_uri: str, local_path: Path) -> None:
    try:
        with SessionLocal() as db:
            file_path = str(local_path.resolve())
            row = (
                db.query(WorkspaceArtifact)
                .filter(
                    WorkspaceArtifact.job_id == job_id,
                    WorkspaceArtifact.file_path == file_path,
                )
                .one_or_none()
            )
            if row is not None:
                row.url_path = storage_uri
            else:
                db.add(
                    WorkspaceArtifact(
                        job_id=job_id,
                        artifact_type="dataset",
                        name=local_path.name,
                        file_path=file_path,
                        url_path=storage_uri,
                    )
                )
            db.commit()
    except Exception as exc:
        logger.warning("update workspace artifact uri failed job_id=%s: %s", job_id, exc)


def _update_data_assets_minio_path(owner_id: str, local_path: Path, storage_uri: str) -> None:
    try:
        from app.models.data_asset import DataAsset
        from app.services.asset_registration_service import DataAssetsSyncSessionLocal
        from app.services.storage_meta_merge import merge_storage_meta

        with DataAssetsSyncSessionLocal() as db:
            row = (
                db.query(DataAsset)
                .filter(
                    (DataAsset.dataset_id == owner_id)
                    | (DataAsset.file_path == str(local_path.resolve()))
                    | (DataAsset.file_path.contains(str(local_path.name)))
                )
                .order_by(DataAsset.updated_at.desc())
                .first()
            )
            if row is None:
                return
            row.minio_path = storage_uri
            if not str(row.file_path or "").startswith("minio://"):
                row.meta = merge_storage_meta(row.meta, str(local_path.resolve()), storage_uri)
            db.commit()
    except Exception as exc:
        logger.warning("update data_assets minio_path failed owner_id=%s: %s", owner_id, exc)


def infer_owner_type(job_id: str) -> Optional[str]:
    jid = (job_id or "").strip()
    if jid.startswith("train_"):
        return "train"
    if jid.startswith(("eval_", "ct_eval_", "isaac_eval_")):
        return "eval"
    if jid.startswith(("ct_gen_", "dac_gen_", "dg_gen_")):
        return "dataset"
    with SessionLocal() as db:
        row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == jid).one_or_none()
        if row is None:
            return None
        if row.job_type == "training":
            return "train"
        if row.job_type == "evaluation":
            return "eval"
        if row.job_type == "generate":
            return "dataset"
    return None


def process_job_artifact_upload(job_id: str) -> dict[str, Any]:
    owner_type = infer_owner_type(job_id)
    if owner_type == "train":
        return upload_training_checkpoints(job_id)
    if owner_type == "eval":
        return upload_evaluation_artifacts(job_id)
    if owner_type == "dataset":
        root = _resolve_dataset_root(job_id)
        if root is not None:
            return upload_dataset_artifacts(job_id, root)
    return {"jobId": job_id, "uploaded": 0}


def _resolve_dataset_root(job_id: str) -> Optional[Path]:
    for runtime_root in _runtime_roots():
        for prefix, sub in (
            ("ct_gen_", runtime_root / "cable_threading" / "jobs"),
            ("dac_gen_", runtime_root / "dual_arm_cable" / "jobs"),
            ("dg_gen_", runtime_root / "data_generation" / "jobs"),
        ):
            if job_id.startswith(prefix):
                path = sub / job_id
                if path.is_dir():
                    return path
    with SessionLocal() as db:
        row = db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).one_or_none()
        if row and row.runtime_path:
            path = resolve_runtime_reference(row.runtime_path)
            if path.is_dir():
                return path
    return None


def schedule_artifact_upload(job_id: str) -> None:
    """异步上传 worker 入口（不阻塞主流程）。"""
    enqueue_artifact_upload(job_id)


def enqueue_artifact_upload(job_id: str) -> None:
    """将任务加入异步上传队列（daemon thread，不阻塞 training/eval）。"""
    if not artifact_upload_enabled():
        return
    jid = (job_id or "").strip()
    if not jid:
        return

    def _worker() -> None:
        try:
            result = process_job_artifact_upload(jid)
            logger.info("artifact upload done job_id=%s result=%s", jid, result)
        except Exception as exc:
            logger.warning("artifact upload worker failed job_id=%s: %s", jid, exc)

    threading.Thread(target=_worker, name=f"artifact-upload-{jid[-12:]}", daemon=True).start()


def _iter_runtime_artifact_candidates(job_root: Path, owner_type: str, owner_id: str) -> list[tuple[str, str, Path]]:
    """枚举 job 目录下应上传的本地文件。"""
    items: list[tuple[str, str, Path]] = []
    if owner_type == "train":
        for ckpt_dir in (job_root / "checkpoints", job_root / "artifacts" / "checkpoints"):
            if not ckpt_dir.is_dir():
                continue
            for path in sorted(ckpt_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in {".ckpt", ".pt", ".pth", ".safetensors"}:
                    rel = str(path.relative_to(job_root))
                    kind = "checkpoint_final" if "final" in path.name.lower() else "checkpoint_epoch"
                    items.append((kind, rel, path))
    elif owner_type == "eval":
        for artifact_type, rel in EVAL_ARTIFACT_NAMES:
            path = job_root / rel
            if path.is_file():
                items.append((artifact_type, rel, path))
        for artifact_type, subdir, glob_pat in EVAL_REPORT_GLOBS:
            search_dir = job_root / subdir
            if search_dir.is_dir():
                for path in sorted(search_dir.glob(glob_pat)):
                    if path.is_file():
                        items.append((artifact_type, str(path.relative_to(job_root)), path))
        videos_dir = job_root / "videos"
        if videos_dir.is_dir():
            for path in sorted(videos_dir.glob("*.mp4")):
                items.append(("eval_video", f"videos/{path.name}", path))
    elif owner_type == "dataset":
        for rel in DATASET_CANDIDATE_PATHS:
            path = job_root / rel
            if path.is_file():
                artifact_type = "dataset_hdf5" if rel.endswith(".hdf5") else "dataset_file"
                items.append((artifact_type, rel, path))
    return items


def scan_runtime_for_new_artifacts(*, limit: int = 100) -> dict[str, Any]:
    """扫描 runs，检测未登记产物并写入 upload queue（artifact_storage_objects pending）。"""
    enqueued = 0
    scanned_jobs = 0
    job_ids = discover_runtime_job_ids(include_non_terminal=True, limit=limit * 2)

    for job_id in job_ids:
        owner_type = infer_owner_type(job_id)
        if owner_type is None:
            continue
        job_root: Optional[Path] = None
        if owner_type == "train":
            job_root = next(
                (
                    root / "training" / "jobs" / job_id
                    for root in _runtime_roots()
                    if (root / "training" / "jobs" / job_id).is_dir()
                ),
                None,
            )
        elif owner_type == "eval":
            job_root = _resolve_eval_root(job_id)
        else:
            job_root = _resolve_dataset_root(job_id)
        if job_root is None or not job_root.is_dir():
            continue
        scanned_jobs += 1
        for artifact_type, content_key, local_path in _iter_runtime_artifact_candidates(job_root, owner_type, job_id):
            existing = get_artifact_record(
                owner_type=owner_type,
                owner_id=job_id,
                artifact_type=artifact_type,
                content_key=content_key,
            )
            if should_skip_upload(existing):
                continue
            register_artifact_pending(
                owner_type=owner_type,
                owner_id=job_id,
                artifact_type=artifact_type,
                content_key=content_key,
                local_path=local_path,
            )
            enqueued += 1
        if scanned_jobs >= limit:
            break

    return {"scannedJobs": scanned_jobs, "enqueued": enqueued}


def scan_and_upload_pending(limit: int = 20) -> dict[str, Any]:
    """扫描 pending 记录并重试上传。"""
    from app.services.artifact_storage_registry import list_pending_artifacts

    pending = list_pending_artifacts(limit=limit)
    processed = 0
    for item in pending:
        local = Path(item.get("localPath") or "")
        if not local.is_file():
            continue
        owner_type = item["ownerType"]
        owner_id = item["ownerId"]
        if owner_type == "train":
            upload_training_checkpoints(owner_id)
        elif owner_type == "eval":
            upload_evaluation_artifacts(owner_id)
        elif owner_type == "dataset":
            upload_dataset_artifacts(owner_id, local.parent if local.parent.name != "datasets" else local.parent.parent)
        processed += 1
    return {"processed": processed, "pending": len(pending)}


def batch_upload_jobs(job_ids: list[str]) -> dict[str, Any]:
    """批量上传多个任务的产物（幂等）。"""
    results: list[dict[str, Any]] = []
    uploaded_total = 0
    for job_id in job_ids:
        jid = (job_id or "").strip()
        if not jid:
            continue
        try:
            result = process_job_artifact_upload(jid)
            uploaded_total += int(result.get("uploaded", 0) or 0)
            results.append({"jobId": jid, "ok": True, "result": result})
        except Exception as exc:
            logger.warning("batch upload failed job_id=%s: %s", jid, exc)
            results.append({"jobId": jid, "ok": False, "error": str(exc)})
    return {"jobs": len(results), "uploaded": uploaded_total, "results": results}


def run_upload_cycle(*, scan_limit: int = 20, pending_limit: int = 20) -> dict[str, Any]:
    """执行一轮 pending 重试 + 终态任务扫描上传。"""
    if not artifact_upload_enabled():
        return {"enabled": False, "processed": 0}

    summary: dict[str, Any] = {"enabled": True, "discover": {}, "pending": {}, "scanned": []}
    summary["discover"] = scan_runtime_for_new_artifacts(limit=scan_limit)
    summary["pending"] = scan_and_upload_pending(limit=pending_limit)

    for job_id in discover_terminal_job_ids(limit=scan_limit):
        owner_type = infer_owner_type(job_id)
        if owner_type is None:
            continue
        try:
            result = process_job_artifact_upload(job_id)
            summary["scanned"].append({"jobId": job_id, "ownerType": owner_type, "result": result})
            logger.info("artifact upload cycle job_id=%s result=%s", job_id, result)
        except Exception as exc:
            logger.warning("artifact upload cycle failed job_id=%s: %s", job_id, exc)
            summary["scanned"].append({"jobId": job_id, "error": str(exc)})

    summary["processed"] = int(summary["pending"].get("processed", 0)) + len(summary["scanned"])
    return summary
