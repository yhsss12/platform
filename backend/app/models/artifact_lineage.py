"""数据血缘：dataset / model / eval 关系图。"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base


class ArtifactLineage(Base):
    __tablename__ = "artifact_lineage"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    parent_id = Column(String(128), nullable=False, index=True)
    child_id = Column(String(128), nullable=False, index=True)
    relation_type = Column(String(64), nullable=False, index=True)
    job_id = Column(String(128), nullable=True, index=True)
    metadata_json = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "parent_id",
            "child_id",
            "relation_type",
            name="uq_artifact_lineage_relation",
        ),
        Index("idx_artifact_lineage_job", "job_id", "relation_type"),
    )
