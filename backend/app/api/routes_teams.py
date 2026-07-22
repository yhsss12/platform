"""
团队管理 API：仅平台超级管理员（SUPER_ADMIN）可访问（管理侧收口）。
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.core.deps import get_current_user, require_super_admin
from app.core.roles import is_super_admin
from app.db.data_assets_session import get_data_assets_db
from app.db.session import get_db
from app.models.user import User, UserRole
from app.crud.user import get_user_by_id
from app.crud import team as team_crud
from app.schemas.common import ApiResponse
from app.services.audit_service import enqueue_audit_log
from app.services.team_delete_service import TeamDeleteError, delete_team_as_super_admin
from app.schemas.team import (
    TeamCreateBody,
    TeamUpdateBody,
    TeamListItemOut,
    TeamDetailOut,
    TeamListPayload,
    TeamAdminAddBody,
    TeamAdminOut,
    TeamUserAddBody,
    TeamUserOut,
    TeamProjectOut,
    UserOptionOut,
)

router = APIRouter()


class DeleteTeamConfirmBody(BaseModel):
    """须与团队名称完全一致（去首尾空白）。"""

    confirmation_name: str


async def _uname_map(udb: AsyncSession, user_ids: List[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for uid in user_ids:
        u = uid.strip()
        if not u or u in out:
            continue
        row = await get_user_by_id(udb, u)
        out[u] = str(getattr(row, "username", "") or u) if row else u
    return out


async def _to_list_item(
    db: AsyncSession,
    udb: AsyncSession,
    row,
) -> TeamListItemOut:
    tid = row.id
    ac = await team_crud.count_team_admins(db, tid)
    uc = await team_crud.count_team_users(db, tid)
    pc = await team_crud.count_team_projects(db, tid)
    desc = getattr(row, "description", None) or ""
    return TeamListItemOut(
        id=tid,
        name=row.name,
        code=row.code,
        description=desc,
        status=str(row.status or "active"),
        admin_count=ac,
        user_count=uc,
        project_count=pc,
        created_at=row.created_at,
        created_by=getattr(row, "created_by", None),
    )


@router.get("/meta/user-options", response_model=ApiResponse)
async def list_user_options_for_teams(
    current_user: User = Depends(require_super_admin),
    exclude_team_id: Optional[str] = Query(
        None,
        description="若传入团队 id，则从候选中排除已在该团队 team_admins 中的用户（避免下拉为空/与列表不一致）",
    ),
    udb: AsyncSession = Depends(get_db),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    供「添加团队管理员」下拉的活跃用户列表。

    仅平台 users.role = ADMIN（团队管理员账号，与 OWNER/USER 区分）；
    可选 exclude_team_id：去掉当前团队已有 team_admins 关系，便于与弹窗「当前管理员」一致。
    """
    from sqlalchemy import select

    already: set[str] = set()
    tid = (exclude_team_id or "").strip()
    if tid:
        if await team_crud.get_team_by_id(db, tid):
            admin_rows = await team_crud.list_team_admin_rows(db, tid)
            already = {str(r.user_id) for r in admin_rows if getattr(r, "user_id", None)}

    r = await udb.execute(
        select(User)
        .where(User.is_active.is_(True), User.role == UserRole.ADMIN)
        .order_by(User.username.asc())
    )
    users = list(r.scalars().all())
    items = [
        UserOptionOut(id=str(u.id), username=u.username)
        for u in users
        if str(u.id) not in already
    ]
    return ApiResponse(ok=True, data={"items": items, "total": len(items)})


@router.get("", response_model=ApiResponse)
async def list_teams_api(
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    rows, total = await team_crud.list_teams(db)
    items: List[TeamListItemOut] = []
    for row in rows:
        items.append(await _to_list_item(db, udb, row))
    return ApiResponse(ok=True, data=TeamListPayload(items=items, total=total).model_dump())


@router.post("", response_model=ApiResponse)
async def create_team_api(
    body: TeamCreateBody,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
):
    code = body.code.strip()
    if await team_crud.get_team_by_code(db, code):
        raise HTTPException(status_code=400, detail="团队编码已存在")
    st = (body.status or "active").strip().lower()
    if st not in ("active", "inactive"):
        raise HTTPException(status_code=400, detail="status 须为 active 或 inactive")
    team = await team_crud.create_team(
        db,
        name=body.name,
        code=code,
        description=body.description,
        status=st,
        created_by=current_user.username,
    )
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.CREATE_TEAM,
        team_id=team.id,
        resource_type=AR.TEAM,
        resource_id=team.id,
        resource_name=team.name,
    )
    ac = await team_crud.count_team_admins(db, team.id)
    uc = await team_crud.count_team_users(db, team.id)
    pc = await team_crud.count_team_projects(db, team.id)
    out = TeamDetailOut(
        id=team.id,
        name=team.name,
        code=team.code,
        description=team.description or "",
        status=str(team.status),
        admin_count=ac,
        user_count=uc,
        project_count=pc,
        created_at=team.created_at,
        created_by=team.created_by,
    )
    return ApiResponse(ok=True, data=out.model_dump())


@router.get("/{team_id}", response_model=ApiResponse)
async def get_team_api(
    team_id: str,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    row = await team_crud.get_team_by_id(db, team_id)
    if not row:
        raise HTTPException(status_code=404, detail="团队不存在")
    item = await _to_list_item(db, udb, row)
    return ApiResponse(ok=True, data=item.model_dump())


@router.patch("/{team_id}", response_model=ApiResponse)
async def patch_team_api(
    team_id: str,
    body: TeamUpdateBody,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if body.status is not None:
        st = body.status.strip().lower()
        if st not in ("active", "inactive"):
            raise HTTPException(status_code=400, detail="status 须为 active 或 inactive")
    obj = await team_crud.update_team(
        db,
        team_id,
        name=body.name,
        description=body.description,
        status=body.status.strip().lower() if body.status is not None else None,
    )
    if not obj:
        raise HTTPException(status_code=404, detail="团队不存在")
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.UPDATE_TEAM,
        team_id=team_id,
        resource_type=AR.TEAM,
        resource_id=team_id,
        resource_name=obj.name,
    )
    item = await _to_list_item(db, udb, obj)
    return ApiResponse(ok=True, data=item.model_dump())


@router.get("/{team_id}/admins", response_model=ApiResponse)
async def list_team_admins_api(
    team_id: str,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    rows = await team_crud.list_team_admin_rows(db, team_id)
    items: List[TeamAdminOut] = []
    for r in rows:
        u = await get_user_by_id(udb, r.user_id)
        uname = u.username if u else r.user_id
        active = bool(getattr(u, "is_active", True)) if u else False
        items.append(
            TeamAdminOut(
                id=str(r.id),
                user_id=r.user_id,
                username=uname,
                display_name=uname,
                email="",
                status="active" if active else "inactive",
                team_id=r.team_id,
            )
        )
    return ApiResponse(ok=True, data={"items": items, "total": len(items)})


@router.post("/{team_id}/admins", response_model=ApiResponse)
async def add_team_admin_api(
    team_id: str,
    body: TeamAdminAddBody,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    uid = body.user_id.strip()
    u = await get_user_by_id(udb, uid)
    if not u:
        raise HTTPException(status_code=404, detail="用户不存在")
    if await team_crud.get_team_admin_by_user(db, team_id, uid):
        raise HTTPException(status_code=400, detail="该用户已是团队管理员")
    row = await team_crud.add_team_admin(db, team_id=team_id, user_id=uid, created_by=current_user.username)
    uname = u.username
    trow = await team_crud.get_team_by_id(db, team_id)
    tname = trow.name if trow else team_id
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.ADD_TEAM_ADMIN,
        team_id=team_id,
        resource_type=AR.TEAM,
        resource_id=team_id,
        resource_name=tname,
        detail_json={"target_user_id": uid, "target_username": uname},
    )
    return ApiResponse(
        ok=True,
        data=TeamAdminOut(
            id=str(row.id),
            user_id=row.user_id,
            username=uname,
            display_name=uname,
            email="",
            status="active" if u.is_active else "inactive",
            team_id=row.team_id,
        ).model_dump(),
    )


@router.delete("/{team_id}/admins/{user_id}", response_model=ApiResponse)
async def remove_team_admin_api(
    team_id: str,
    user_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    ok = await team_crud.remove_team_admin(db, team_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="管理员不存在")
    trow = await team_crud.get_team_by_id(db, team_id)
    tname = trow.name if trow else team_id
    tu = await get_user_by_id(udb, user_id)
    tun = tu.username if tu else user_id
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.REMOVE_TEAM_ADMIN,
        team_id=team_id,
        resource_type=AR.TEAM,
        resource_id=team_id,
        resource_name=tname,
        detail_json={"target_user_id": user_id, "target_username": tun},
    )
    return ApiResponse(ok=True, data={"removed": True})


@router.get("/{team_id}/users", response_model=ApiResponse)
async def list_team_users_api(
    team_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    """
    团队用户只读列表：团队管理页仅 SUPER_ADMIN 会调用到；项目详情邀请成员需本团队用户 id，
    故允许「SUPER_ADMIN」或「team_users / team_admins 中的成员」读取（非团队管理入口收口范围）。
    """
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    if not is_super_admin(current_user.role):
        if not await team_crud.user_is_team_member_or_admin(db, team_id, str(current_user.id)):
            raise HTTPException(status_code=403, detail="Forbidden")
    rows = await team_crud.list_team_user_rows(db, team_id)
    items: List[TeamUserOut] = []
    for r in rows:
        u = await get_user_by_id(udb, r.user_id)
        uname = u.username if u else r.user_id
        active = bool(getattr(u, "is_active", True)) if u else False
        pr = ""
        if u is not None and getattr(u, "role", None) is not None:
            rv = getattr(u.role, "value", None)
            pr = str(rv) if rv is not None else str(u.role)
        items.append(
            TeamUserOut(
                id=str(r.id),
                user_id=r.user_id,
                username=uname,
                display_name=uname,
                email="",
                status="active" if active else "inactive",
                team_id=r.team_id,
                platform_role=pr,
            )
        )
    return ApiResponse(ok=True, data={"items": items, "total": len(items)})


@router.post("/{team_id}/users", response_model=ApiResponse)
async def add_team_user_api(
    team_id: str,
    body: TeamUserAddBody,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    uid = body.user_id.strip()
    u = await get_user_by_id(udb, uid)
    if not u:
        raise HTTPException(status_code=404, detail="用户不存在")
    if await team_crud.get_team_user_by_user(db, team_id, uid):
        raise HTTPException(status_code=400, detail="该用户已在团队中")
    row = await team_crud.add_team_user(db, team_id=team_id, user_id=uid, created_by=current_user.username)
    uname = u.username
    trow = await team_crud.get_team_by_id(db, team_id)
    tname = trow.name if trow else team_id
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.ADD_TEAM_USER,
        team_id=team_id,
        resource_type=AR.TEAM,
        resource_id=team_id,
        resource_name=tname,
        detail_json={"target_user_id": uid, "target_username": uname},
    )
    rv = getattr(u.role, "value", None)
    pr = str(rv) if rv is not None else str(getattr(u, "role", "") or "")
    return ApiResponse(
        ok=True,
        data=TeamUserOut(
            id=str(row.id),
            user_id=row.user_id,
            username=uname,
            display_name=uname,
            email="",
            status="active" if u.is_active else "inactive",
            team_id=row.team_id,
            platform_role=pr,
        ).model_dump(),
    )


@router.delete("/{team_id}/users/{user_id}", response_model=ApiResponse)
async def remove_team_user_api(
    team_id: str,
    user_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    ok = await team_crud.remove_team_user(db, team_id, user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="团队成员不存在")
    trow = await team_crud.get_team_by_id(db, team_id)
    tname = trow.name if trow else team_id
    tu = await get_user_by_id(udb, user_id)
    tun = tu.username if tu else user_id
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.REMOVE_TEAM_USER,
        team_id=team_id,
        resource_type=AR.TEAM,
        resource_id=team_id,
        resource_name=tname,
        detail_json={"target_user_id": user_id, "target_username": tun},
    )
    return ApiResponse(ok=True, data={"removed": True})


@router.get("/{team_id}/projects", response_model=ApiResponse)
async def list_team_projects_api(
    team_id: str,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    if not await team_crud.get_team_by_id(db, team_id):
        raise HTTPException(status_code=404, detail="团队不存在")
    projects = await team_crud.list_projects_for_team(db, team_id)
    owner_ids = [str(p.owner_id) for p in projects if getattr(p, "owner_id", None)]
    unames = await _uname_map(udb, owner_ids)
    items: List[TeamProjectOut] = []
    for p in projects:
        oid = (p.owner_id or "").strip()
        owner_label = unames.get(oid, oid or "—")
        members = await team_crud.count_project_members_distinct(db, p.id, p.owner_id)
        assets = await team_crud.count_project_assets(db, p.id)
        items.append(
            TeamProjectOut(
                id=p.id,
                team_id=team_id,
                name=p.name,
                owner=owner_label,
                members=members,
                assets=assets,
                updated_at=p.updated_at,
                status=str(p.status or "进行中"),
            )
        )
    return ApiResponse(ok=True, data={"items": [i.model_dump() for i in items], "total": len(items)})


@router.post("/{team_id}/delete", response_model=ApiResponse)
async def delete_team_hard_api(
    team_id: str,
    body: DeleteTeamConfirmBody,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
    udb: AsyncSession = Depends(get_db),
):
    """
    物理删除团队：级联删除该团队下所有项目及项目从属数据（与 delete_project 级联范围一致并含 upload_sessions / task_jobs），
    再删除 team_admins / team_users / teams；主库删除 team_account_counter；
    仅当用户除本团队外无其他团队关系且无跨团队项目负责/成员时删除 users 行。
    名称须与 confirmation_name 一致。进行中任务存在时返回 400。
    """
    try:
        summary = await delete_team_as_super_admin(
            db,
            udb,
            team_id=team_id,
            confirmation_name=body.confirmation_name,
        )
    except TeamDeleteError as e:
        raise HTTPException(status_code=400, detail=str(e))

    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.DELETE_TEAM,
        team_id=summary.get("team_id"),
        resource_type=AR.TEAM,
        resource_id=str(summary.get("team_id") or team_id),
        resource_name=str(summary.get("team_name") or ""),
        detail_json=summary,
    )
    return ApiResponse(ok=True, data=summary)
