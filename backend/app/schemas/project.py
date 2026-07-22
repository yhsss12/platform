"""
项目 Schema（projects 表，PostgreSQL）
"""
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, field_serializer


def _datetime_to_iso_utc(dt: datetime) -> str:
    if dt is None:
        return ""
    from datetime import timezone
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    status: str = "进行中"
    owner_id: Optional[str] = None
    team_id: Optional[str] = None


class ProjectCreate(ProjectBase):
    """创建项目：id 可由前端生成 UUID 或由后端生成"""
    id: Optional[str] = None  # 不传则后端生成 UUID


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None
    owner_id: Optional[str] = None
    # 故意不包含 team_id：不允许通过更新接口迁移所属团队


class ProjectResponse(ProjectBase):
    id: str
    created_at: datetime
    updated_at: datetime
    # 列表/详情在带当前用户上下文序列化时使用：是否在 project_members 表中有行（受邀/加入项目）
    viewer_is_project_member: Optional[bool] = None
    viewer_is_project_owner: Optional[bool] = None
    # project_members 表行数，与 GET .../members 列表长度一致
    member_count: Optional[int] = None

    @field_serializer("created_at", "updated_at")
    def serialize_datetime_utc(self, dt: datetime) -> str:
        return _datetime_to_iso_utc(dt)

    class Config:
        from_attributes = True


class ProjectListResponse(BaseModel):
    items: List[ProjectResponse]
    total: int
