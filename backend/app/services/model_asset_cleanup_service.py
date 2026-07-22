"""Purge model assets that were revived from deleted training jobs."""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Keep only joint-space DP pipeline assets from deleted-job revival cleanup.
JOINT_PIPELINE_KEEP = {
    "model_joint_dp_20260624_full_final",
}


def purge_model_assets_on_deleted_train_jobs(*, dry_run: bool = False) -> dict[str, Any]:
    from app.core.database import SessionLocal
    from app.models.workspace_index import ModelAsset
    from app.models.workspace_job import WorkspaceJob

    result: dict[str, Any] = {"dryRun": dry_run, "deleted": 0, "kept": 0, "modelAssetIds": []}
    try:
        with SessionLocal() as db:
            deleted_job_ids = [
                row.job_id
                for row in db.query(WorkspaceJob)
                .filter(WorkspaceJob.job_type == "training", WorkspaceJob.status == "deleted")
                .all()
            ]
            if not deleted_job_ids:
                return result

            assets = (
                db.query(ModelAsset)
                .filter(ModelAsset.train_job_id.in_(deleted_job_ids))
                .all()
            )
            for row in assets:
                if row.model_asset_id in JOINT_PIPELINE_KEEP:
                    result["kept"] += 1
                    continue
                result["modelAssetIds"].append(row.model_asset_id)
                if not dry_run:
                    db.delete(row)
                result["deleted"] += 1

            if not dry_run:
                db.commit()
    except Exception as exc:
        logger.warning("purge_model_assets_on_deleted_train_jobs failed: %s", exc)
        result["error"] = str(exc)
    return result


def purge_soft_deleted_model_assets(*, dry_run: bool = False) -> dict[str, Any]:
    from app.core.database import SessionLocal
    from app.models.workspace_index import ModelAsset

    result: dict[str, Any] = {"dryRun": dry_run, "deleted": 0, "modelAssetIds": []}
    try:
        with SessionLocal() as db:
            rows = db.query(ModelAsset).filter(ModelAsset.status == "deleted").all()
            for row in rows:
                if row.model_asset_id in JOINT_PIPELINE_KEEP:
                    continue
                result["modelAssetIds"].append(row.model_asset_id)
                if not dry_run:
                    db.delete(row)
                result["deleted"] += 1
            if not dry_run:
                db.commit()
    except Exception as exc:
        result["error"] = str(exc)
    return result
