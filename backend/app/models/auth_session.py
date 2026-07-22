import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, backref, mapped_column, relationship
from sqlalchemy.sql import func

from . import Base, User


class AuthSession(Base):
    """
    Session-isolated auth:
    - One browser tab = one session_id
    - refresh_token is opaque and bound to session_id
    - logout revokes only the current session
    """

    __tablename__ = "auth_sessions"

    session_id: Mapped[str] = mapped_column(
        # 兼容前端 sessionId（理论上 UUID=36；这里留更大空间，避免长度异常导致崩溃）
        String(128),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    refresh_token_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped[User] = relationship("User", backref=backref("auth_sessions", passive_deletes=True))

