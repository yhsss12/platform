"""
团队表、团队管理员表（与 projects / data_assets 同 PostgreSQL 库）
"""
from sqlalchemy import Column, DateTime, Integer, String, Text, UniqueConstraint, Index
from sqlalchemy.sql import func

from app.models.data_asset import Base


class Team(Base):
    __tablename__ = "teams"

    id = Column(String(128), primary_key=True, comment="团队 ID，UUID")
    name = Column(String(256), nullable=False, comment="团队名称")
    code = Column(String(64), nullable=False, comment="团队编码，唯一")
    description = Column(Text, nullable=True, comment="描述")
    status = Column(String(16), nullable=False, default="active", comment="active | inactive")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(String(128), nullable=True, comment="创建者用户名或用户 ID")

    __table_args__ = (
        UniqueConstraint("code", name="uq_teams_code"),
        Index("idx_teams_status", "status"),
        Index("idx_teams_updated", "updated_at"),
    )


class TeamUser(Base):
    """团队普通成员（归属关系）；管理员见 TeamAdmin，二者可并存。"""

    __tablename__ = "team_users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String(128), nullable=False, comment="teams.id")
    user_id = Column(String(36), nullable=False, comment="users.id（主库）")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    created_by = Column(String(128), nullable=True, comment="操作者用户名或 ID")

    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_team_users_team_user"),
        Index("idx_team_users_team_id", "team_id"),
        Index("idx_team_users_user", "user_id"),
    )


class TeamAdmin(Base):
    """团队管理员：用户与团队多对多（仅管理员角色，非项目成员）"""

    __tablename__ = "team_admins"

    id = Column(Integer, primary_key=True, autoincrement=True)
    team_id = Column(String(128), nullable=False, comment="teams.id")
    user_id = Column(String(36), nullable=False, comment="users.id（主库）")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    created_by = Column(String(128), nullable=True, comment="操作者用户名或 ID")

    __table_args__ = (
        UniqueConstraint("team_id", "user_id", name="uq_team_admins_team_user"),
        Index("idx_team_admins_team_id", "team_id"),
        Index("idx_team_admins_user", "user_id"),
    )
