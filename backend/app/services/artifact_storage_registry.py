"""artifact_storage_objects 读写与幂等登记。"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.database import SessionLocal
from app.models.artifact_storage_object import ArtifactStorageObject

logger = logging.getLogger(__name__)

MAX_UPLOAD_ATTEMPTS = 5


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def file_digest(path: Path) -> tuple[Optional[str], Optional[int]]:
    if not path.is_file():
        return None, None
    try:
        size = path.stat().st_size
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest(), size
    except OSError:
        return None, None


def get_artifact_record(
    *,
    owner_type: str,
    owner_id: str,
    artifact_type: str,
    content_key: str,
) -> Optional[ArtifactStorageObject]:
    with SessionLocal() as db:
        return (
            db.query(ArtifactStorageObject)
            .filter(
                ArtifactStorageObject.owner_type == owner_type,
                ArtifactStorageObject.owner_id == owner_id,
                ArtifactStorageObject.artifact_type == artifact_type,
                ArtifactStorageObject.content_key == content_key,
            )
            .one_or_none()
        )


def register_artifact_pending(
    *,
    owner_type: str,
    owner_id: str,
    artifact_type: str,
    content_key: str,
    local_path: Path,
) -> ArtifactStorageObject:
    sha256, size_bytes = file_digest(local_path)
    with SessionLocal() as db:
        row = (
            db.query(ArtifactStorageObject)
            .filter(
                ArtifactStorageObject.owner_type == owner_type,
                ArtifactStorageObject.owner_id == owner_id,
                ArtifactStorageObject.artifact_type == artifact_type,
                ArtifactStorageObject.content_key == content_key,
            )
            .one_or_none()
        )
        if row is None:
            row = ArtifactStorageObject(
                owner_type=owner_type,
                owner_id=owner_id,
                artifact_type=artifact_type,
                content_key=content_key,
                local_path=str(local_path.resolve()),
                sha256=sha256,
                size_bytes=size_bytes,
                status="pending",
            )
            db.add(row)
        else:
            if row.status == "uploaded" and row.storage_uri and str(row.storage_uri).startswith("minio://"):
                db.commit()
                return row
            row.local_path = str(local_path.resolve())
            row.sha256 = sha256
            row.size_bytes = size_bytes
            if row.status != "uploaded":
                row.status = "pending"
        db.commit()
        db.refresh(row)
        return row


def mark_artifact_uploaded(
    *,
    owner_type: str,
    owner_id: str,
    artifact_type: str,
    content_key: str,
    storage_uri: str,
    local_path: Optional[Path] = None,
) -> None:
    with SessionLocal() as db:
        row = (
            db.query(ArtifactStorageObject)
            .filter(
                ArtifactStorageObject.owner_type == owner_type,
                ArtifactStorageObject.owner_id == owner_id,
                ArtifactStorageObject.artifact_type == artifact_type,
                ArtifactStorageObject.content_key == content_key,
            )
            .one_or_none()
        )
        if row is None:
            sha256, size_bytes = file_digest(local_path) if local_path else (None, None)
            row = ArtifactStorageObject(
                owner_type=owner_type,
                owner_id=owner_id,
                artifact_type=artifact_type,
                content_key=content_key,
                storage_uri=storage_uri,
                local_path=str(local_path.resolve()) if local_path else None,
                sha256=sha256,
                size_bytes=size_bytes,
                status="uploaded",
                upload_attempts=1,
                last_error=None,
            )
            db.add(row)
        else:
            row.storage_uri = storage_uri
            row.status = "uploaded"
            row.last_error = None
            row.upload_attempts = int(row.upload_attempts or 0) + 1
            row.updated_at = _utc_now()
        db.commit()


def mark_artifact_failed(
    *,
    owner_type: str,
    owner_id: str,
    artifact_type: str,
    content_key: str,
    error: str,
) -> None:
    with SessionLocal() as db:
        row = (
            db.query(ArtifactStorageObject)
            .filter(
                ArtifactStorageObject.owner_type == owner_type,
                ArtifactStorageObject.owner_id == owner_id,
                ArtifactStorageObject.artifact_type == artifact_type,
                ArtifactStorageObject.content_key == content_key,
            )
            .one_or_none()
        )
        if row is None:
            return
        row.status = "failed" if int(row.upload_attempts or 0) >= MAX_UPLOAD_ATTEMPTS else "pending"
        row.upload_attempts = int(row.upload_attempts or 0) + 1
        row.last_error = (error or "")[:2000]
        row.updated_at = _utc_now()
        db.commit()


def should_skip_upload(row: Optional[ArtifactStorageObject]) -> bool:
    if row is None:
        return False
    if row.status == "uploaded" and str(row.storage_uri or "").startswith("minio://"):
        return True
    if int(row.upload_attempts or 0) >= MAX_UPLOAD_ATTEMPTS and row.status == "failed":
        return True
    return False


def list_pending_artifacts(limit: int = 50) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        rows = (
            db.query(ArtifactStorageObject)
            .filter(ArtifactStorageObject.status.in_(["pending", "failed"]))
            .filter(ArtifactStorageObject.upload_attempts < MAX_UPLOAD_ATTEMPTS)
            .order_by(ArtifactStorageObject.updated_at.asc())
            .limit(limit)
            .all()
        )
        return [
            {
                "ownerType": row.owner_type,
                "ownerId": row.owner_id,
                "artifactType": row.artifact_type,
                "contentKey": row.content_key,
                "localPath": row.local_path,
            }
            for row in rows
        ]
