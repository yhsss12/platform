"""模型类型定义表：结构配置与训练适配元数据。"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base


class ModelTypeDefinition(Base):
    __tablename__ = "model_type_definitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type_id = Column(String(128), unique=True, nullable=False, index=True)
    name = Column(String(256), nullable=False)
    base_algorithm = Column(String(64), nullable=False, index=True)
    adapter_key = Column(String(64), nullable=False)
    simulator = Column(String(64), nullable=True)
    robot_type = Column(String(64), nullable=True)
    tags = Column(JSONB, nullable=True)
    description = Column(Text, nullable=True)
    structure_config = Column(JSONB, nullable=False, server_default="{}")
    training_defaults = Column(JSONB, nullable=False, server_default="{}")
    status = Column(String(32), nullable=False, default="available", index=True)
    is_builtin = Column(Boolean, nullable=False, default=False)
    training_ready = Column(Boolean, nullable=True)
    training_readiness_status = Column(String(32), nullable=True)
    disabled_reason = Column(Text, nullable=True)
    capability_checked_at = Column(DateTime(timezone=True), nullable=True)
    capability_evidence = Column(JSONB, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
