"""eval_metric_summary / workspace_jobs 评测任务 DB 查询。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.database import SessionLocal
from app.models.workspace_index import EvalMetricSummary
from app.models.workspace_job import WorkspaceJob

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = frozenset({"completed", "failed", "canceled"})
ORPHAN_EVALUATION_STATUSES = frozenset({"unknown", "draft"})


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _metrics_from_row(row: WorkspaceJob, summary: Optional[EvalMetricSummary]) -> dict[str, Any]:
    metrics = dict(row.metrics_json or {}) if isinstance(row.metrics_json, dict) else {}
    meta = dict(row.metadata_json or {}) if isinstance(row.metadata_json, dict) else {}
    eval_request = meta.get("evaluationRequest") if isinstance(meta.get("evaluationRequest"), dict) else {}

    for key in (
        "evaluationMode",
        "evaluationType",
        "evaluationTypeLabel",
        "evaluationObject",
        "productEvaluationMode",
        "datasetId",
        "datasetName",
        "modelAssetId",
        "modelName",
    ):
        if metrics.get(key) is None and eval_request.get(key) is not None:
            metrics[key] = eval_request.get(key)

    if summary and isinstance(summary.summary_json, dict):
        for key, value in summary.summary_json.items():
            if key not in metrics and value is not None:
                metrics[key] = value

    if row.task_name and not metrics.get("modelName"):
        metrics.setdefault("modelName", row.task_name)
    return metrics


def is_valid_evaluation_list_item(item: dict[str, Any]) -> bool:
    from app.services.evaluation.job_paths import is_valid_eval_job_id_format

    eval_job_id = str(item.get("evalJobId") or item.get("jobId") or item.get("job_id") or "").strip()
    workspace_job_id = item.get("workspaceJobId") or item.get("workspace_job_id")
    if is_valid_eval_job_id_format(eval_job_id):
        return True
    return workspace_job_id is not None


def _is_orphan_evaluation_db_row(row: WorkspaceJob) -> bool:
    from app.services.evaluation.job_paths import is_valid_eval_job_id_format

    if row.job_type != "evaluation" or row.status == "deleted":
        return False
    if is_valid_eval_job_id_format(str(row.job_id or "")):
        return False
    return str(row.status or "").strip() in ORPHAN_EVALUATION_STATUSES


def _purge_orphan_evaluation_db_rows(db: Any) -> int:
    removed = 0
    rows = (
        db.query(WorkspaceJob)
        .filter(
            WorkspaceJob.job_type == "evaluation",
            WorkspaceJob.status != "deleted",
        )
        .all()
    )
    for row in rows:
        if not _is_orphan_evaluation_db_row(row):
            continue
        _soft_delete_evaluation_row(db, row)
        removed += 1
    if removed:
        db.commit()
    return removed


def evaluation_job_row_to_list_item(row: WorkspaceJob, summary: Optional[EvalMetricSummary]) -> dict[str, Any]:
    from app.services.evaluation.evaluation_progress import resolve_evaluation_progress
    from app.services.evaluation.evaluation_type import resolve_evaluation_type_from_sources

    metrics = _metrics_from_row(row, summary)
    meta = dict(row.metadata_json or {}) if isinstance(row.metadata_json, dict) else {}
    eval_request = meta.get("evaluationRequest") if isinstance(meta.get("evaluationRequest"), dict) else {}
    summary_json = dict(summary.summary_json or {}) if summary and isinstance(summary.summary_json, dict) else {}

    task_type = row.task_type or eval_request.get("taskType")
    evaluation_mode = metrics.get("evaluationMode") or eval_request.get("evaluationMode")
    video_available = bool(summary and summary.replay_uri) or bool(metrics.get("videoAvailable"))

    progress_info = resolve_evaluation_progress(
        status=str(row.status or ""),
        metrics=metrics,
        summary_json=summary_json,
        job_id=row.job_id,
        runtime_path=row.runtime_path,
    )
    for key, value in progress_info.items():
        if value is not None:
            metrics[key] = value

    type_resolution = resolve_evaluation_type_from_sources(
        evaluation_object=eval_request.get("evaluationObject") or meta.get("evaluationObject"),
        evaluation_mode=evaluation_mode,
        product_evaluation_mode=eval_request.get("productEvaluationMode") or meta.get("productEvaluationMode"),
        model_asset_id=metrics.get("modelAssetId") or eval_request.get("modelAssetId"),
        model_asset_name=metrics.get("modelName") or eval_request.get("modelName"),
        dataset_id=metrics.get("datasetId") or eval_request.get("datasetId"),
        dataset_name=metrics.get("datasetName") or eval_request.get("datasetName"),
        task_type=task_type,
        runner=row.runner,
        task_name=row.task_name,
        metadata=meta,
        metrics=metrics,
        evaluation_request=eval_request,
    )

    success_stats: dict[str, Any]
    try:
        from app.services.evaluation.success_stats import resolve_success_stats

        success_stats = resolve_success_stats(
            row.job_id or "",
            summary_json=summary_json,
            context_json=eval_request,
            runtime_path=row.runtime_path,
            metrics=metrics,
        )
    except Exception:
        success_stats = {
            "successEpisodes": None,
            "totalEpisodes": None,
            "display": "-/-",
            "available": False,
            "reason": "successStats 解析失败",
        }

    return {
        "workspaceJobId": row.id,
        "evalJobId": row.job_id,
        "jobId": row.job_id,
        "taskType": task_type,
        "evaluationMode": evaluation_mode,
        "evaluationObject": type_resolution["evaluationObject"],
        "evaluationType": type_resolution["evaluationType"],
        "evaluationTypeLabel": type_resolution["evaluationTypeLabel"],
        "status": row.status,
        "message": row.error_message or metrics.get("message") or metrics.get("runtimeHealthReason"),
        "errorMessage": row.error_message,
        "createdAt": _iso(row.created_at),
        "updatedAt": _iso(row.updated_at),
        "startedAt": _iso(row.started_at),
        "finishedAt": _iso(row.finished_at),
        "taskName": row.task_name,
        "templateDisplayName": meta.get("templateDisplayName") or meta.get("displayName"),
        "runner": row.runner,
        "runtimePath": row.runtime_path,
        "metrics": metrics,
        "videoAvailable": video_available,
        "reportUri": summary.report_uri if summary else None,
        "replayUri": summary.replay_uri if summary else None,
        "successStats": success_stats,
        **{key: progress_info.get(key) for key in progress_info if progress_info.get(key) is not None},
    }


def list_evaluation_jobs_from_db(*, sync_stale: bool = True) -> list[dict[str, Any]]:
    from app.services.evaluation.evaluation_runtime_health import reconcile_all_running_evaluation_jobs

    rows_out: list[dict[str, Any]] = []
    try:
        if sync_stale:
            reconcile_all_running_evaluation_jobs(limit=200, apply=True)

        with SessionLocal() as db:
            _purge_orphan_evaluation_db_rows(db)

            jobs = (
                db.query(WorkspaceJob)
                .filter(
                    WorkspaceJob.job_type == "evaluation",
                    WorkspaceJob.status != "deleted",
                )
                .order_by(WorkspaceJob.created_at.desc())
                .all()
            )

            for job in jobs:
                try:
                    summary = (
                        db.query(EvalMetricSummary)
                        .filter(EvalMetricSummary.job_id == job.job_id)
                        .one_or_none()
                    )
                    item = evaluation_job_row_to_list_item(job, summary)
                    if is_valid_evaluation_list_item(item):
                        rows_out.append(item)
                except Exception as exc:
                    logger.warning(
                        "evaluation_job_row_to_list_item failed job_id=%s: %s",
                        job.job_id,
                        exc,
                    )
    except Exception as exc:
        logger.exception("list_evaluation_jobs_from_db failed: %s", exc)
        return []
    return rows_out


def get_evaluation_job_from_db(job_id: str) -> Optional[dict[str, Any]]:
    candidate = (job_id or "").strip()
    if not candidate:
        return None
    try:
        with SessionLocal() as db:
            row = (
                db.query(WorkspaceJob)
                .filter(
                    WorkspaceJob.job_id == candidate,
                    WorkspaceJob.job_type == "evaluation",
                    WorkspaceJob.status != "deleted",
                )
                .one_or_none()
            )
            if row is None:
                return None
            summary = (
                db.query(EvalMetricSummary)
                .filter(EvalMetricSummary.job_id == candidate)
                .one_or_none()
            )
            return evaluation_job_row_to_list_item(row, summary)
    except Exception:
        return None


def get_evaluation_result_from_db(job_id: str) -> Optional[dict[str, Any]]:
    candidate = (job_id or "").strip()
    if not candidate:
        return None
    try:
        with SessionLocal() as db:
            summary = (
                db.query(EvalMetricSummary)
                .filter(EvalMetricSummary.job_id == candidate)
                .one_or_none()
            )
            if summary is None:
                return None
            if isinstance(summary.summary_json, dict) and summary.summary_json:
                payload = dict(summary.summary_json)
                payload.setdefault("evalJobId", candidate)
                if summary.report_uri:
                    payload.setdefault("reportUri", summary.report_uri)
                if summary.replay_uri:
                    payload.setdefault("replayUri", summary.replay_uri)
                return payload
    except Exception:
        return None
    return None


PENDING_EVALUATION_STATUSES = frozenset({"pending", "queued", "draft", "待评测", "unknown"})


def _soft_delete_evaluation_row(db: Any, row: Any) -> None:
    from app.models.workspace_index import EvalMetricSummary
    from app.models.workspace_job import WorkspaceArtifact

    job_id = str(row.job_id or "").strip()
    now = datetime.now(timezone.utc)
    row.status = "deleted"
    row.updated_at = now
    merged_meta = dict(row.metadata_json or {}) if isinstance(row.metadata_json, dict) else {}
    merged_meta["deletedAt"] = now.isoformat()
    merged_meta["deletedReason"] = "user_deleted_evaluation"
    row.metadata_json = merged_meta

    if job_id:
        db.query(EvalMetricSummary).filter(EvalMetricSummary.job_id == job_id).delete(
            synchronize_session=False
        )
        db.query(WorkspaceArtifact).filter(WorkspaceArtifact.job_id == job_id).delete(
            synchronize_session=False
        )


def delete_evaluation_job_from_db(job_id: str) -> bool:
    """软删除评测任务：仅 evaluation job_type，并清理摘要与 artifact 索引。"""
    candidate = (job_id or "").strip()
    if not candidate:
        return False
    try:
        from app.models.workspace_job import WorkspaceJob

        with SessionLocal() as db:
            row = (
                db.query(WorkspaceJob)
                .filter(WorkspaceJob.job_id == candidate)
                .one_or_none()
            )
            if row is None:
                return False
            if row.job_type != "evaluation":
                return False

            _soft_delete_evaluation_row(db, row)
            db.commit()
            return True
    except Exception:
        return False


def delete_pending_evaluation_record(workspace_job_id: int | str) -> dict[str, Any]:
    """软删除待启动/排队中的评测记录（按 workspace_jobs.id）。"""
    from fastapi import HTTPException, status
    from app.models.workspace_job import WorkspaceJob

    try:
        record_id = int(workspace_job_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="workspaceJobId 无效",
        ) from exc

    with SessionLocal() as db:
        row = db.query(WorkspaceJob).filter(WorkspaceJob.id == record_id).one_or_none()
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="evaluation record not found",
            )
        if row.job_type != "evaluation":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="只能删除评测任务",
            )
        if row.status == "deleted":
            return {
                "success": True,
                "deleted": True,
                "workspaceJobId": record_id,
                "jobId": row.job_id,
                "status": "deleted",
            }
        if str(row.status or "").strip() not in PENDING_EVALUATION_STATUSES:
            from app.services.evaluation.job_paths import is_valid_eval_job_id_format

            if is_valid_eval_job_id_format(str(row.job_id or "")):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "EVALUATION_RECORD_NOT_PENDING",
                        "message": "该评测任务已启动或已完成，请使用 evalJobId 删除接口。",
                        "jobId": row.job_id,
                        "status": row.status,
                    },
                )
            # legacy smoke / orphan rows with invalid job_id: allow cleanup via workspaceJobId

        _soft_delete_evaluation_row(db, row)
        db.commit()
        return {
            "success": True,
            "deleted": True,
            "workspaceJobId": record_id,
            "jobId": row.job_id,
            "status": "deleted",
        }


def get_evaluation_job_db_row(job_id: str) -> Optional[Any]:
    candidate = (job_id or "").strip()
    if not candidate:
        return None
    try:
        with SessionLocal() as db:
            return (
                db.query(WorkspaceJob)
                .filter(
                    WorkspaceJob.job_id == candidate,
                    WorkspaceJob.job_type == "evaluation",
                )
                .one_or_none()
            )
    except Exception:
        return None
