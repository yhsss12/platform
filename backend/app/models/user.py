import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from . import Base


class UserRole(str, enum.Enum):
    """
    用户角色（四层模型，存库值与 JWT/对外 API 一致）。

    - SUPER_ADMIN：平台超级管理员（全库仅允许一个启用中，见迁移 partial unique index）
    - ADMIN：团队级管理员（users.role；与 team_admins 等业务表配合）
    - OWNER：项目级负责人
    - USER：普通用户

    下列值为历史迁移前可能出现的存库值，ORM 仍须能读取直至数据迁完。
    """

    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    OWNER = "OWNER"
    USER = "USER"

    ADMINISTRATOR = "ADMINISTRATOR"
    MEMBER = "MEMBER"


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    account_id: Mapped[str] = mapped_column(
        String(80),
        unique=True,
        index=True,
        nullable=False,
        comment="登录账号，全局唯一，后端生成，不可修改",
    )
    username: Mapped[str] = mapped_column(
        String(50),
        index=True,
        nullable=False,
        comment="展示名称，可重复，可修改，不用于登录",
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    # 与 alembic 012 一致：列类型为 VARCHAR(32)，非 PostgreSQL 原生 enum（旧库曾用 userrole enum，已迁走）
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, native_enum=False, length=32),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
