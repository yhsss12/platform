"""
审计日志列表数据范围（管理侧权限第二阶段-A）。

- SUPER_ADMIN：全平台
- ADMIN（users.role=ADMIN）：仅 team_id 属于其在 team_admins 中管辖的团队；
  不含 team_id 的历史日志不在此范围（不展示给其他团队管理员）。
- OWNER / USER：由路由依赖 require_super_admin_or_team_admin 拦截，不应进入列表。

异步列表接口须预先在数据资产库会话中查出 team_admin_team_ids，再传入 audit_log_scope_where，
避免在 ASGI 事件循环内使用 asyncio.run 拉取团队范围。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import ColumnElement, false

from app.core.roles import is_super_admin, is_team_admin_role
from app.models import AuditLog

if TYPE_CHECKING:
    from app.models import User


def audit_log_scope_where(
    current_user: "User",
    *,
    team_admin_team_ids: list[str] | None,
) -> ColumnElement[bool] | None:
    """
    返回需 AND 到 WHERE 的表达式；SUPER_ADMIN 返回 None（不加团队约束）。
    team_admin_team_ids：团队管理员时已预加载的管辖团队 id 列表；非团队管理员可传 None。
    """
    if is_super_admin(current_user.role):
        return None
    if is_team_admin_role(current_user.role):
        ids = team_admin_team_ids or []
        if not ids:
            return false()
        return AuditLog.team_id.in_(ids)
    return false()
