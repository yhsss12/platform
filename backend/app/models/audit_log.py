from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import BigInteger, DateTime, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from . import Base


class AuditLog(Base):
    """
    平台统一审计日志（主库 PostgreSQL）。
    说明：users.id 为 UUID 字符串，故 user_id / project_id 使用 VARCHAR 与现网一致。
    """

    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )

    user_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, index=True)
    username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    project_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    project_name: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    team_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)

    action_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    action_label: Mapped[str] = mapped_column(String(200), nullable=False)

    resource_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resource_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    resource_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    result: Mapped[str] = mapped_column(
        String(20), nullable=False, default="SUCCESS", server_default="SUCCESS", index=True
    )

    ip: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    detail_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
