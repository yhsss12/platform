"""统一资产查询：跨 model / dataset / eval / checkpoint，仅查 PostgreSQL。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.models.artifact_storage_object import ArtifactStorageObject
from app.models.workspace_index import EvalMetricSummary, ModelAsset
from app.models.workspace_job import WorkspaceJob


def search_assets(
    db: Session,
    *,
    asset_type: Optional[str] = None,
    project_id: Optional[str] = None,
    job_id: Optional[str] = None,
    dataset_id: Optional[str] = None,
    min_score: Optional[float] = None,
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    """统一搜索；不直接扫描运行目录。"""
    normalized = (asset_type or "").strip().lower()
    items: list[dict[str, Any]] = []

    if normalized in ("", "model", "checkpoint"):
        items.extend(_search_model_assets(db, project_id, job_id, dataset_id, time_from, time_to, checkpoint_only=normalized == "checkpoint"))
    if normalized in ("", "eval"):
        items.extend(_search_eval_assets(db, project_id, job_id, dataset_id, min_score, time_from, time_to))
    if normalized in ("", "dataset"):
        items.extend(_search_dataset_assets(db, project_id, job_id, dataset_id, time_from, time_to))

    items.sort(key=lambda row: row.get("created_at") or "", reverse=True)
    total = len(items)
    page = items[offset : offset + limit]
    return page, total


def _search_model_assets(
    db: Session,
    project_id: Optional[str],
    job_id: Optional[str],
    dataset_id: Optional[str],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
    *,
    checkpoint_only: bool,
) -> list[dict[str, Any]]:
    q = db.query(ModelAsset, WorkspaceJob).join(WorkspaceJob, WorkspaceJob.job_id == ModelAsset.train_job_id)
    q = q.filter(ModelAsset.status != "deleted")
    if checkpoint_only:
        q = q.filter(or_(ModelAsset.asset_type.in_(["final", "best", "epoch"]), ModelAsset.checkpoint_kind.isnot(None)))
    if project_id:
        q = q.filter(or_(ModelAsset.project_id == project_id, WorkspaceJob.project_id == project_id))
    if job_id:
        q = q.filter(ModelAsset.train_job_id == job_id)
    if dataset_id:
        q = q.filter(ModelAsset.dataset_id == dataset_id)
    if time_from:
        q = q.filter(ModelAsset.created_at >= time_from)
    if time_to:
        q = q.filter(ModelAsset.created_at <= time_to)

    rows: list[dict[str, Any]] = []
    for asset, job in q.limit(500).all():
        kind = asset.checkpoint_kind or asset.asset_type
        atype = "checkpoint" if kind in ("final", "best", "epoch") else "model"
        rows.append(
            {
                "id": asset.model_asset_id,
                "type": atype,
                "job_id": asset.train_job_id,
                "project_id": asset.project_id or job.project_id,
                "dataset_id": asset.dataset_id,
                "storage_uri": asset.storage_uri,
                "summary": {
                    "modelName": asset.model_name,
                    "modelType": asset.model_type,
                    "checkpointKind": kind,
                    "status": asset.status,
                    "metrics": asset.metrics_json,
                },
                "created_at": asset.created_at.isoformat() if asset.created_at else None,
            }
        )
    return rows


def _search_eval_assets(
    db: Session,
    project_id: Optional[str],
    job_id: Optional[str],
    dataset_id: Optional[str],
    min_score: Optional[float],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
) -> list[dict[str, Any]]:
    q = db.query(EvalMetricSummary, WorkspaceJob).join(WorkspaceJob, WorkspaceJob.job_id == EvalMetricSummary.job_id)
    if project_id:
        q = q.filter(WorkspaceJob.project_id == project_id)
    if job_id:
        q = q.filter(EvalMetricSummary.job_id == job_id)
    if min_score is not None:
        q = q.filter(EvalMetricSummary.average_score >= min_score)
    if time_from:
        q = q.filter(EvalMetricSummary.updated_at >= time_from)
    if time_to:
        q = q.filter(EvalMetricSummary.updated_at <= time_to)

    rows: list[dict[str, Any]] = []
    for summary, job in q.limit(500).all():
        summary_json = summary.summary_json if isinstance(summary.summary_json, dict) else {}
        if dataset_id:
            ds = str(summary_json.get("datasetId") or "")
            if ds and ds != dataset_id:
                continue
        rows.append(
            {
                "id": summary.job_id,
                "type": "eval",
                "job_id": summary.job_id,
                "project_id": job.project_id,
                "dataset_id": summary_json.get("datasetId"),
                "storage_uri": summary.report_uri or summary.replay_uri,
                "summary": {
                    "successRate": summary.success_rate,
                    "averageScore": summary.average_score,
                    "modelAssetId": summary.model_asset_id,
                    "metrics": summary_json,
                },
                "created_at": summary.updated_at.isoformat() if summary.updated_at else None,
            }
        )
    return rows


def _search_dataset_assets(
    db: Session,
    project_id: Optional[str],
    job_id: Optional[str],
    dataset_id: Optional[str],
    time_from: Optional[datetime],
    time_to: Optional[datetime],
) -> list[dict[str, Any]]:
    q = db.query(ArtifactStorageObject).filter(ArtifactStorageObject.owner_type == "dataset", ArtifactStorageObject.status == "uploaded")
    if job_id:
        q = q.filter(ArtifactStorageObject.owner_id == job_id)
    if time_from:
        q = q.filter(ArtifactStorageObject.created_at >= time_from)
    if time_to:
        q = q.filter(ArtifactStorageObject.created_at <= time_to)

    rows: list[dict[str, Any]] = []
    for obj in q.limit(500).all():
        oid = obj.owner_id
        if dataset_id and oid != dataset_id and dataset_id not in (obj.content_key or ""):
            continue
        rows.append(
            {
                "id": f"{obj.owner_id}:{obj.content_key}",
                "type": "dataset",
                "job_id": obj.owner_id,
                "project_id": project_id,
                "dataset_id": oid,
                "storage_uri": obj.storage_uri,
                "summary": {
                    "artifactType": obj.artifact_type,
                    "contentKey": obj.content_key,
                    "sha256": obj.sha256,
                    "sizeBytes": obj.size_bytes,
                },
                "created_at": obj.created_at.isoformat() if obj.created_at else None,
            }
        )
    return rows
