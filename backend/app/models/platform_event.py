"""平台事件持久化记录。"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.db.base import Base


class PlatformEventRecord(Base):
    __tablename__ = "platform_events"

    event_id = Column(String(64), primary_key=True)
    event_type = Column(String(64), nullable=False, index=True)
    job_id = Column(String(128), nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), index=True)
    payload = Column(JSONB, nullable=True)
    source = Column(String(64), nullable=False, default="platform")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_platform_events_job_type", "job_id", "event_type"),
        Index("idx_platform_events_type_time", "event_type", "timestamp"),
    )
