from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base import Base


class WorkspaceJob(Base):
    __tablename__ = "workspace_jobs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(String(128), unique=True, nullable=False, index=True)
    job_type = Column(String(64), nullable=False, index=True)
    task_type = Column(String(64), nullable=False, index=True)
    task_name = Column(String(256), nullable=True)
    status = Column(String(32), nullable=False, default="pending", index=True)
    source = Column(String(32), nullable=False, default="real", index=True)
    runner = Column(String(128), nullable=True)
    project_id = Column(String(128), nullable=True)
    created_by = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
    started_at = Column(DateTime(timezone=True), nullable=True)
    finished_at = Column(DateTime(timezone=True), nullable=True)
    runtime_path = Column(Text, nullable=False)
    metadata_json = Column(JSONB, nullable=True)
    metrics_json = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)

    artifacts = relationship(
        "WorkspaceArtifact",
        back_populates="job",
        cascade="all, delete-orphan",
        foreign_keys="WorkspaceArtifact.job_id",
        primaryjoin="WorkspaceJob.job_id==WorkspaceArtifact.job_id",
    )


class WorkspaceArtifact(Base):
    __tablename__ = "workspace_artifacts"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(
        String(128),
        ForeignKey("workspace_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    artifact_type = Column(String(64), nullable=False, index=True)
    name = Column(String(256), nullable=False)
    file_path = Column(Text, nullable=False)
    url_path = Column(Text, nullable=True)
    episode_index = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    metadata_json = Column(JSONB, nullable=True)

    job = relationship(
        "WorkspaceJob",
        back_populates="artifacts",
        foreign_keys=[job_id],
        primaryjoin="WorkspaceArtifact.job_id==WorkspaceJob.job_id",
    )
