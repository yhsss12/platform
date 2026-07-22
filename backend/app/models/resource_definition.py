"""资源中心统一定义表。"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base


class ResourceDefinition(Base):
    __tablename__ = "resource_definitions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    resource_id = Column(String(128), nullable=False, index=True)
    resource_type = Column(String(32), nullable=False, index=True)
    name = Column(String(512), nullable=False, default="")
    display_name = Column(String(512), nullable=True)
    description = Column(Text, nullable=True)
    version = Column(String(64), nullable=False, default="v1")
    status = Column(String(32), nullable=False, default="available", index=True)
    tags = Column(JSONB, nullable=True)
    manifest_json = Column(JSONB, nullable=False, server_default="{}")
    metadata_json = Column(JSONB, nullable=False, server_default="{}")
    manifest_path = Column(Text, nullable=True)
    storage_uri = Column(Text, nullable=True)
    source = Column(String(32), nullable=False, default="registry")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (
        UniqueConstraint(
            "resource_type",
            "resource_id",
            "version",
            name="uq_resource_definitions_type_id_version",
        ),
    )
