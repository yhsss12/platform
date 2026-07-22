from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.roles import normalize_role


class AuditLogOut(BaseModel):
    """id 兼容旧库 UUID 字符串主键与新库 bigint 主键。"""
    id: str
    created_at: datetime
    user_id: Optional[str] = None
    username: Optional[str] = None
    role: Optional[str] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    action_type: str
    action_label: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None
    result: str
    ip: Optional[str] = None
    user_agent: Optional[str] = None
    detail_json: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("id", mode="before")
    @classmethod
    def _id_to_str(cls, v: object) -> str:
        if v is None:
            raise ValueError("audit log id is required")
        return str(v)

    @field_validator("role", mode="before")
    @classmethod
    def _role_canonical(cls, v: object) -> str | None:
        """对外统一四层角色码；历史 ADMINISTRATOR/MEMBER 等在此规范化。"""
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        return normalize_role(s).value


class AuditLogListResponse(BaseModel):
    items: list[AuditLogOut]
    total: int


class AuditLogQueryParams(BaseModel):
    created_from: Optional[datetime] = None
    created_to: Optional[datetime] = None
    username: Optional[str] = None
    project_id: Optional[str] = None
    team_id: Optional[str] = None
    action_type: Optional[str] = None
    result: Optional[str] = None
    q: Optional[str] = Field(None, description="关键字：用户名/动作/资源名/IP 等模糊匹配")
    limit: int = Field(50, ge=1, le=200)
    offset: int = Field(0, ge=0)
