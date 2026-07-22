from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import Column, MetaData, String, Table, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants.audit_actions import ACTION_LABELS_ZH, ALL_ACTION_TYPES
from app.core.audit_log_scope import audit_log_scope_where
from app.core.deps import require_super_admin_or_team_admin_async
from app.core.roles import is_super_admin, is_team_admin_role
from app.crud import team as team_crud
from app.db.data_assets_session import get_data_assets_db
from app.db.session import get_db as get_async_db
from app.models import AuditLog, User
from app.schemas.audit import AuditLogListResponse, AuditLogOut

router = APIRouter(prefix="/audit", tags=["audit"])

_audit_teams_table = Table(
    "teams",
    MetaData(),
    Column("id", String(128), primary_key=True),
    Column("name", String(256)),
)


@router.get("/meta")
async def audit_meta(
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
) -> dict:
    """筛选用：动作类型枚举 + 中文展示名（管理端下拉）。"""
    items = [{"code": c, "label": ACTION_LABELS_ZH.get(c, c)} for c in ALL_ACTION_TYPES]
    return {
        "action_types": list(ALL_ACTION_TYPES),
        "action_items": items,
    }


@router.get("/team-filter-options")
async def audit_team_filter_options(
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
) -> dict:
    """
    审计页「所属团队」下拉：不依赖 /teams（团队 API 仅 SUPER_ADMIN）。
    SUPER_ADMIN：全部团队；ADMIN：仅其在 team_admins 中的团队。
    """
    if is_super_admin(_current_user.role):
        rows, _ = await team_crud.list_teams(assets_db)
        items = [{"id": r.id, "name": r.name} for r in rows]
    elif is_team_admin_role(_current_user.role):
        tids = await team_crud.list_team_ids_where_user_is_team_admin(
            assets_db, str(_current_user.id)
        )
        rows = await team_crud.list_teams_by_ids(assets_db, tids)
        items = [{"id": r.id, "name": r.name} for r in rows]
    else:
        items = []
    return {"items": items, "total": len(items)}


@router.get("", response_model=AuditLogListResponse)
async def list_audit_logs(
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
    created_from: Optional[datetime] = Query(None, description="起始时间（含）"),
    created_to: Optional[datetime] = Query(None, description="结束时间（含）"),
    username: Optional[str] = Query(None, description="用户名模糊匹配"),
    project_id: Optional[str] = Query(None),
    team_id: Optional[str] = Query(None, description="所属团队 id，空为全部"),
    action_type: Optional[str] = Query(None),
    result: Optional[str] = Query(None, description="SUCCESS / FAIL"),
    q: Optional[str] = Query(None, description="关键字"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> AuditLogListResponse:
    """
    统一审计日志列表（超级管理员或团队管理员账号；数据范围见 audit_log_scope）。
    排序：created_at desc
    """
    team_admin_team_ids: list[str] | None = None
    if is_team_admin_role(_current_user.role):
        team_admin_team_ids = await team_crud.list_team_ids_where_user_is_team_admin(
            assets_db, str(_current_user.id)
        )

    wheres = []
    scope = audit_log_scope_where(
        _current_user, team_admin_team_ids=team_admin_team_ids
    )
    if scope is not None:
        wheres.append(scope)

    if created_from is not None:
        wheres.append(AuditLog.created_at >= created_from)
    if created_to is not None:
        wheres.append(AuditLog.created_at <= created_to)
    if username:
        like = f"%{username.strip()}%"
        wheres.append(AuditLog.username.ilike(like))
    if project_id:
        wheres.append(AuditLog.project_id == project_id.strip())
    if team_id and team_id.strip():
        tid = team_id.strip()
        if is_team_admin_role(_current_user.role):
            allowed = set(team_admin_team_ids or [])
            if tid not in allowed:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot filter audit logs by a team outside your admin scope",
                )
        wheres.append(AuditLog.team_id == tid)
    if action_type:
        wheres.append(AuditLog.action_type == action_type.strip())
    if result:
        wheres.append(AuditLog.result == result.strip().upper())

    if q and q.strip():
        pat = f"%{q.strip()}%"
        wheres.append(
            or_(
                AuditLog.username.ilike(pat),
                AuditLog.action_type.ilike(pat),
                AuditLog.action_label.ilike(pat),
                AuditLog.resource_name.ilike(pat),
                AuditLog.resource_id.ilike(pat),
                AuditLog.ip.ilike(pat),
                AuditLog.role.ilike(pat),
                AuditLog.project_name.ilike(pat),
                _audit_teams_table.c.name.ilike(pat),
                AuditLog.error_message.ilike(pat),
            )
        )

    join_on = AuditLog.team_id == _audit_teams_table.c.id
    where_clause = and_(*wheres) if wheres else None

    count_stmt = (
        select(func.count(AuditLog.id))
        .select_from(AuditLog)
        .outerjoin(_audit_teams_table, join_on)
    )
    if where_clause is not None:
        count_stmt = count_stmt.where(where_clause)

    total = int((await db.execute(count_stmt)).scalar() or 0)

    data_stmt = (
        select(AuditLog, _audit_teams_table.c.name.label("team_name"))
        .outerjoin(_audit_teams_table, join_on)
    )
    if where_clause is not None:
        data_stmt = data_stmt.where(where_clause)

    data_stmt = (
        data_stmt.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit)
    )
    rows = (await db.execute(data_stmt)).all()

    items: list[AuditLogOut] = []
    for log_row, tname in rows:
        base = AuditLogOut.model_validate(log_row)
        items.append(base.model_copy(update={"team_name": tname}))
    return AuditLogListResponse(items=items, total=total)
