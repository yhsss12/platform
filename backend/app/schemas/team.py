"""团队 API Schema（最小可用）"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_serializer


def _dt_iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if getattr(dt, "tzinfo", None) is not None:
        return dt.isoformat()
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


class TeamCreateBody(BaseModel):
    name: str = Field(..., min_length=1)
    code: str = Field(..., min_length=1)
    description: Optional[str] = None
    status: str = "active"


class TeamUpdateBody(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class TeamListItemOut(BaseModel):
    id: str
    name: str
    code: str
    description: str
    status: str
    admin_count: int
    # team_users 表行数（普通成员；不含 team_admins）
    user_count: int
    project_count: int
    created_at: datetime
    created_by: Optional[str] = None

    @field_serializer("created_at")
    def ser_created(self, dt: datetime) -> str:
        return _dt_iso(dt)


class TeamDetailOut(TeamListItemOut):
    pass


class TeamAdminAddBody(BaseModel):
    user_id: str = Field(..., min_length=1)


class TeamAdminOut(BaseModel):
    id: str
    user_id: str
    username: str
    display_name: str
    email: str
    status: str
    team_id: str


class TeamUserOut(TeamAdminOut):
    """团队普通成员列表行（结构与管理员一致，便于前端复用表格）。"""

    platform_role: str = Field(
        default="",
        description="主库 users.role（如 OWNER/USER），供邀请成员展示；与项目成员表无独立列",
    )


class TeamUserAddBody(BaseModel):
    user_id: str = Field(..., min_length=1)


class TeamProjectOut(BaseModel):
    id: str
    team_id: str
    name: str
    owner: str
    members: int
    assets: int
    updated_at: datetime
    status: str

    @field_serializer("updated_at")
    def ser_u(self, dt: datetime) -> str:
        return _dt_iso(dt)


class UserOptionOut(BaseModel):
    id: str
    username: str


class TeamListPayload(BaseModel):
    items: List[TeamListItemOut]
    total: int
