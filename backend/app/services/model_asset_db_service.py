"""model_assets 表查询与 API 响应映射。"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.workspace_index import ModelAsset

logger = logging.getLogger(__name__)
from app.services.model_asset_naming import (
    build_checkpoint_asset_display_name,
    resolve_model_asset_context_label,
    resolve_model_asset_display_name,
)

TASK_TYPE_TO_TEMPLATE: dict[str, str] = {
    "cable_threading": "task_cable_threading_v1",
    "dual_arm_cable_manipulation": "dual_arm_cable_manipulation",
    "isaac_block_stacking": "isaac_block_stacking",
}


def _infer_task_template_id(
    *,
    manifest: dict[str, Any],
    dataset_id: Optional[str] = None,
    train_job_id: Optional[str] = None,
) -> Optional[str]:
    task_template_id = str(manifest.get("taskTemplateId") or "").strip()
    if task_template_id:
        return task_template_id
    task_type = str(manifest.get("taskType") or "").strip()
    if task_type:
        return TASK_TYPE_TO_TEMPLATE.get(task_type)
    dataset_key = str(dataset_id or manifest.get("sourceDatasetId") or "").strip()
    if dataset_key.startswith("ds_ct_gen_") or dataset_key.startswith("ds_cable_"):
        return TASK_TYPE_TO_TEMPLATE["cable_threading"]
    if dataset_key.startswith("ds_dac_") or dataset_key.startswith("ds_dual_"):
        return TASK_TYPE_TO_TEMPLATE["dual_arm_cable_manipulation"]
    if dataset_key.startswith("ds_isaac_") or dataset_key.startswith("isaac_ds_"):
        return TASK_TYPE_TO_TEMPLATE["isaac_block_stacking"]
    job_key = str(train_job_id or manifest.get("sourceTrainJobId") or "").strip()
    if job_key.startswith("dac_"):
        return TASK_TYPE_TO_TEMPLATE["dual_arm_cable_manipulation"]
    if job_key.startswith("ct_"):
        return TASK_TYPE_TO_TEMPLATE["cable_threading"]
    return None


def _iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def storage_uri_to_checkpoint_path(storage_uri: Optional[str]) -> str:
    text = (storage_uri or "").strip()
    if not text:
        return ""
    if text.startswith("file://"):
        return text[len("file://") :]
    return text


def _row_to_asset(
    row: ModelAsset,
    *,
    status: Optional[dict[str, Any]] = None,
    for_list: bool = False,
) -> dict[str, Any]:
    from app.services.model_asset_validation import enrich_model_asset

    manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
    status = status or {}
    model_asset_id = row.model_asset_id
    train_job_id = row.train_job_id
    task_type = str(manifest.get("taskType") or status.get("taskType") or "")
    task_template_id = _infer_task_template_id(
        manifest=manifest,
        dataset_id=str(row.dataset_id or manifest.get("sourceDatasetId") or "") or None,
        train_job_id=train_job_id,
    ) or TASK_TYPE_TO_TEMPLATE.get(task_type)
    framework = str(manifest.get("framework") or status.get("trainingBackend") or "unknown")
    model_type = str(row.model_type or manifest.get("modelType") or "unknown")
    training_backend = manifest.get("trainingBackend") or status.get("trainingBackend")

    display_name = str(manifest.get("displayName") or row.model_name or "").strip()
    if row.asset_type in {"final", "best", "epoch"}:
        context_label = resolve_model_asset_context_label(
            training_task_name=str(manifest.get("trainingTaskName") or status.get("taskName") or "") or None,
            dataset_name=str(manifest.get("datasetDisplayName") or status.get("datasetName") or "") or None,
            dataset_id=str(row.dataset_id or status.get("datasetId") or "") or None,
            task_template_id=task_template_id,
            task_type=task_type or None,
        )
        if not display_name or display_name == model_asset_id:
            display_name = build_checkpoint_asset_display_name(
                context_label=context_label,
                kind=row.asset_type,
                epoch=row.epoch,
                metric_name=str(manifest.get("checkpointMetricName") or "") or None,
            )
    elif not display_name:
        display_name = resolve_model_asset_display_name(
            stored_name=row.model_name,
            display_name=str(manifest.get("displayName") or ""),
            training_task_name=str(manifest.get("trainingTaskName") or status.get("taskName") or "") or None,
            dataset_name=str(manifest.get("datasetDisplayName") or status.get("datasetName") or "") or None,
            dataset_id=str(row.dataset_id or "") or None,
            task_template_id=task_template_id,
            task_type=task_type or None,
            framework=framework,
            model_type=model_type,
            training_backend=str(training_backend or "") or None,
            created_at=_iso(row.created_at),
        )

    asset_status = row.status
    if asset_status == "ready":
        asset_status = "available"

    checkpoint_path = storage_uri_to_checkpoint_path(row.storage_uri)
    metrics = row.metrics_json if isinstance(row.metrics_json, dict) else {}

    base = {
        "id": model_asset_id,
        "name": display_name,
        "displayName": display_name,
        "sourceTrainingJobId": train_job_id,
        "sourceDatasetId": row.dataset_id,
        "taskTemplateId": task_template_id,
        "modelType": model_type,
        "framework": framework,
        "trainingBackend": manifest.get("trainingBackend") or training_backend,
        "backendType": manifest.get("backendType") or training_backend,
        "checkpointPath": checkpoint_path,
        "checkpointKind": row.asset_type,
        "checkpointEpoch": row.epoch,
        "checkpointMetricName": metrics.get("checkpointMetricName") or manifest.get("checkpointMetricName"),
        "checkpointMetricValue": metrics.get("checkpointMetricValue") or manifest.get("checkpointMetricValue"),
        "datasetDisplayName": manifest.get("datasetDisplayName") or status.get("datasetName"),
        "manifestPath": str(manifest.get("manifestPath") or ""),
        "version": "v1",
        "status": asset_status,
        "createdAt": _iso(row.created_at),
        "updatedAt": _iso(row.updated_at),
        "structureConfig": manifest.get("structureConfig"),
        "resolvedModelParams": manifest.get("resolvedModelParams"),
        "assetSource": manifest.get("assetSource") or ("imported" if row.asset_type == "imported" else "training"),
        "importMetadata": manifest.get("importMetadata"),
        "taskType": manifest.get("taskType") or task_type or None,
        "validationResult": manifest.get("validationResult"),
        "artifactKind": row.asset_type,
        "evalExecutor": manifest.get("evalExecutor"),
        "trainedActionMode": manifest.get("trainedActionMode") or manifest.get("actionMode"),
        "actionMode": manifest.get("actionMode") or manifest.get("trainedActionMode"),
        "controllerType": manifest.get("controllerType"),
        "actionSchema": manifest.get("actionSchema"),
        "observationSchema": manifest.get("observationSchema"),
        "controllerSchema": manifest.get("controllerSchema"),
        "sideChannelSchema": manifest.get("sideChannelSchema"),
        "actionKey": manifest.get("actionKey"),
        "gripperActionKey": manifest.get("gripperActionKey"),
        "actionDim": manifest.get("actionDim"),
        "fileSizeBytes": int(row.size_bytes or 0),
    }
    return enrich_model_asset(base, for_list=for_list)


def _row_to_detail_asset(row: ModelAsset, *, status: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    base = _row_to_asset(row, status=status)
    manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
    is_placeholder = bool(manifest.get("isPlaceholder"))
    can_evaluate = manifest.get("canEvaluate")
    display_status = str(manifest.get("displayStatus") or row.status or "waiting")

    if can_evaluate is None:
        from app.services.checkpoint_registry import (
            compute_asset_can_evaluate,
            compute_asset_display_status,
        )

        pseudo_entry = {
            "modelAssetId": row.model_asset_id,
            "checkpointKind": row.asset_type,
            "checkpointPath": base.get("checkpointPath"),
            "status": row.status,
            "isPlaceholder": is_placeholder,
        }
        can_evaluate = compute_asset_can_evaluate(pseudo_entry, job_status=status or {})
        display_status = compute_asset_display_status(pseudo_entry, job_status=status or {})

    return {
        **base,
        "isPlaceholder": is_placeholder,
        "canEvaluate": bool(can_evaluate),
        "displayStatus": display_status,
    }


def list_model_assets_from_db(
    *,
    include_deleted: bool = False,
    for_evaluation: bool = False,
    evaluation_task_type: Optional[str] = None,
    for_list: bool = False,
) -> list[dict[str, Any]]:
    from app.services.model_asset_validation import filter_evaluable_model_assets

    with SessionLocal() as db:
        query = db.query(ModelAsset).order_by(ModelAsset.created_at.desc())
        if not include_deleted:
            query = query.filter(ModelAsset.status != "deleted")
        rows = query.all()
        assets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            if row.status in {"deleted", "superseded"}:
                continue
            manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
            if manifest.get("isPlaceholder"):
                continue
            try:
                asset = _row_to_asset(row, for_list=for_list)
            except Exception as exc:
                logger.warning(
                    "Skip model asset row %s during list: %s",
                    row.model_asset_id,
                    exc,
                )
                continue
            if asset["id"] in seen:
                continue
            seen.add(asset["id"])
            assets.append(asset)

        if for_evaluation:
            return filter_evaluable_model_assets(assets, evaluation_task_type=evaluation_task_type)
        return assets


def list_model_assets_for_job_from_db(train_job_id: str) -> list[dict[str, Any]]:
    job_id = (train_job_id or "").strip()
    if not job_id:
        return []
    with SessionLocal() as db:
        rows = (
            db.query(ModelAsset)
            .filter(
                ModelAsset.train_job_id == job_id,
                ModelAsset.status != "deleted",
                ModelAsset.status != "superseded",
            )
            .order_by(ModelAsset.created_at.asc())
            .all()
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            manifest = row.manifest_json if isinstance(row.manifest_json, dict) else {}
            if manifest.get("isPlaceholder"):
                continue
            if row.status == "generating" and not row.storage_uri:
                continue
            asset = _row_to_asset(row)
            if asset.get("checkpointPath") or row.asset_type == "final":
                result.append(asset)
        return result


def list_training_job_model_assets_detail_from_db(
    train_job_id: str,
    *,
    status: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    job_id = (train_job_id or "").strip()
    if not job_id:
        return []
    with SessionLocal() as db:
        rows = (
            db.query(ModelAsset)
            .filter(ModelAsset.train_job_id == job_id, ModelAsset.status != "deleted")
            .order_by(ModelAsset.created_at.asc())
            .all()
        )
        return [_row_to_detail_asset(row, status=status) for row in rows]


def get_model_asset_from_db(model_asset_id: str) -> Optional[dict[str, Any]]:
    candidate = (model_asset_id or "").strip()
    if not candidate:
        return None
    with SessionLocal() as db:
        row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == candidate).one_or_none()
        if row is None or row.status == "deleted":
            return None
        return _row_to_asset(row)


def mark_model_asset_deleted(model_asset_id: str) -> bool:
    candidate = (model_asset_id or "").strip()
    if not candidate:
        return False
    with SessionLocal() as db:
        row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == candidate).one_or_none()
        if row is None:
            return False
        db.delete(row)
        db.commit()
        return True


def count_ready_model_assets_for_job(train_job_id: str) -> int:
    with SessionLocal() as db:
        return (
            db.query(ModelAsset)
            .filter(
                ModelAsset.train_job_id == train_job_id,
                ModelAsset.status.in_(["ready", "available"]),
            )
            .count()
        )
