"""任务模板目录表。"""

from __future__ import annotations

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base


class TaskTemplateCatalog(Base):
    __tablename__ = "task_template_catalog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(String(128), nullable=False, unique=True)
    name = Column(String(512), nullable=False)
    display_name = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    category = Column(String(128), nullable=True)
    simulator = Column(String(64), nullable=True)
    robot_type = Column(String(128), nullable=True)
    task_config_id = Column(String(128), nullable=True)
    metadata_json = Column(JSONB, nullable=False, server_default="{}")
    status = Column(String(32), nullable=False, default="available", index=True)
    is_builtin = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint("template_id", name="uq_task_template_catalog_template_id"),
    )
