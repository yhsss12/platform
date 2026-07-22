"""统一对象存储索引：artifact_storage_objects。"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from app.db.base import Base


class ArtifactStorageObject(Base):
    __tablename__ = "artifact_storage_objects"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    owner_type = Column(String(32), nullable=False, index=True)
    owner_id = Column(String(128), nullable=False, index=True)
    artifact_type = Column(String(64), nullable=False, index=True)
    content_key = Column(String(512), nullable=False, default="")
    storage_uri = Column(Text, nullable=True)
    local_path = Column(Text, nullable=True)
    sha256 = Column(String(64), nullable=True)
    size_bytes = Column(BigInteger, nullable=True)
    status = Column(String(32), nullable=False, default="pending", index=True)
    upload_attempts = Column(Integer, nullable=False, default=0)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "owner_type",
            "owner_id",
            "artifact_type",
            "content_key",
            name="uq_artifact_storage_owner_content",
        ),
        Index("idx_artifact_storage_owner", "owner_type", "owner_id"),
        Index("idx_artifact_storage_status", "status", "updated_at"),
    )
