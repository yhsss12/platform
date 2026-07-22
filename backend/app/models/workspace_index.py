"""Workspace 索引表：模型资产、训练/评测指标摘要（PostgreSQL JSONB）。"""

from __future__ import annotations

from sqlalchemy import BigInteger, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base


class ModelAsset(Base):
    __tablename__ = "model_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_asset_id = Column(String(128), unique=True, nullable=False, index=True)
    train_job_id = Column(
        String(128),
        ForeignKey("workspace_jobs.job_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    project_id = Column(String(128), nullable=True, index=True)
    dataset_id = Column(String(128), nullable=True, index=True)
    model_name = Column(String(512), nullable=False, default="")
    model_type = Column(String(64), nullable=True)
    asset_type = Column(String(32), nullable=False, default="epoch", index=True)
    checkpoint_kind = Column(String(32), nullable=True, index=True)
    epoch = Column(Integer, nullable=True)
    storage_uri = Column(Text, nullable=True)
    manifest_json = Column(JSONB, nullable=True)
    metrics_json = Column(JSONB, nullable=True)
    sha256 = Column(String(64), nullable=True)
    size_bytes = Column(BigInteger, nullable=True)
    status = Column(String(32), nullable=False, default="generating", index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        Index("idx_model_assets_train_job_type", "train_job_id", "asset_type"),
        Index("idx_model_assets_status", "status", "created_at"),
    )


class TrainingMetricSummary(Base):
    __tablename__ = "training_metric_summary"

    job_id = Column(
        String(128),
        ForeignKey("workspace_jobs.job_id", ondelete="CASCADE"),
        primary_key=True,
    )
    current_epoch = Column(Integer, nullable=False, default=0)
    total_epochs = Column(Integer, nullable=False, default=0)
    progress = Column(Float, nullable=False, default=0.0)
    current_loss = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    best_loss = Column(Float, nullable=True)
    loss_series = Column(JSONB, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EvalMetricSummary(Base):
    __tablename__ = "eval_metric_summary"

    job_id = Column(
        String(128),
        ForeignKey("workspace_jobs.job_id", ondelete="CASCADE"),
        primary_key=True,
    )
    model_asset_id = Column(String(128), nullable=True, index=True)
    success_rate = Column(Float, nullable=True)
    average_score = Column(Float, nullable=True)
    summary_json = Column(JSONB, nullable=True)
    report_uri = Column(Text, nullable=True)
    replay_uri = Column(Text, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
