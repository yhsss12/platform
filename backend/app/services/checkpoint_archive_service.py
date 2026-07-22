"""训练 checkpoint 异步归档到 MinIO（兼容层；委托 artifact_upload_service）。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from app.services.artifact_upload_service import (
    artifact_upload_enabled,
    schedule_artifact_upload,
    upload_training_checkpoints,
)

logger = logging.getLogger(__name__)


def _checkpoint_archive_enabled() -> bool:
    from app.core.config import settings

    explicit = getattr(settings, "CHECKPOINT_ARCHIVE_ENABLED", None)
    if explicit is not None:
        return bool(explicit)
    return artifact_upload_enabled()


def archive_model_asset_to_minio(
    model_asset_id: str,
    *,
    local_path: Optional[Path] = None,
    bucket: Optional[str] = None,
) -> dict[str, Any]:
    """上传单个 checkpoint；保留旧 API 签名供测试/脚本调用。"""
    from app.core.database import SessionLocal
    from app.models.workspace_index import ModelAsset

    candidate = (model_asset_id or "").strip()
    if not candidate:
        return {"archived": False, "warning": "model_asset_id is empty"}

    with SessionLocal() as db:
        row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == candidate).one_or_none()
        if row is None or row.status == "deleted":
            return {"archived": False, "warning": "model asset not found in database"}
        if str(row.storage_uri or "").startswith("minio://"):
            return {"archived": True, "storageUri": row.storage_uri, "warning": None}
        train_job_id = row.train_job_id

    result = upload_training_checkpoints(train_job_id)
    with SessionLocal() as db:
        row = db.query(ModelAsset).filter(ModelAsset.model_asset_id == candidate).one_or_none()
        if row is not None and str(row.storage_uri or "").startswith("minio://"):
            return {"archived": True, "storageUri": row.storage_uri, "warning": None}

    warning = None
    if result.get("warnings"):
        warning = "; ".join(result["warnings"][:1])
    elif int(result.get("uploaded", 0)) == 0:
        warning = "checkpoint not uploaded"
    return {"archived": False, "warning": warning, "result": result}


def schedule_training_job_checkpoint_archive(train_job_id: str) -> None:
    """训练完成后异步归档 checkpoint（委托统一 artifact upload）。"""
    if not _checkpoint_archive_enabled():
        return
    schedule_artifact_upload(train_job_id)
