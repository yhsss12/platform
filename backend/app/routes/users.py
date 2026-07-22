from datetime import datetime, timezone

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import (
    require_super_admin_or_team_admin_async,
    require_user_account_mutations_async,
    require_user_list_access_async,
)
from app.core.roles import (
    CanonicalUserRole,
    is_super_admin,
    is_team_admin_role,
    normalize_role,
)
from app.core.security import hash_password
from app.crud import team as team_crud
from app.services.user_team_access import (
    batch_user_has_any_active_team_async,
    effective_user_is_active,
)
from app.services.user_provision import UserProvisionError, create_user_with_allocated_account_id
from app.db.data_assets_session import get_data_assets_db
from app.db.session import get_db as get_async_db
from app.models import RefreshToken, User, UserRole
from app.models.team import Team, TeamAdmin, TeamUser
from app.models.project_asset import Project, ProjectMember
from app.schemas.user import (
    CreateUserRequest,
    ResetPasswordRequest,
    UpdateUserRoleRequest,
    UserListItemOut,
    UserListPayload,
    UserOut,
)
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import log_audit_safe

router = APIRouter(prefix="/users", tags=["users"])

PLATFORM_SCOPE_LABEL = "平台"


@router.get("/team-options")
async def list_team_options_for_user_create(
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
) -> dict:
    """
    新建用户弹窗用团队下拉：
    - SUPER_ADMIN：返回全部团队
    - ADMIN（团队管理员账号）：返回其在 team_admins 中管辖的团队
    """
    if is_super_admin(_current_user.role):
        rows, _ = await team_crud.list_teams(assets_db)
    else:
        team_ids = await team_crud.list_team_ids_where_user_is_team_admin(
            assets_db, str(_current_user.id)
        )
        rows = await team_crud.list_teams_by_ids(assets_db, team_ids)

    items = [
        {
            "id": str(getattr(t, "id", "") or ""),
            "name": str(getattr(t, "name", "") or ""),
            "code": str(getattr(t, "code", "") or ""),
            "status": str(getattr(t, "status", "") or ""),
        }
        for t in rows
    ]
    return {"items": items, "total": len(items)}


async def _team_scope_for_list_users(
    assets_db: AsyncSession,
    users: list[User],
) -> dict[str, tuple[str | None, str]]:
    """user_id -> (主 team_id 或 None, 列表展示用团队文案)。超管固定「平台」。"""
    out: dict[str, tuple[str | None, str]] = {}
    if not users:
        return out
    for u in users:
        uid = str(u.id)
        if is_super_admin(u.role):
            out[uid] = (None, PLATFORM_SCOPE_LABEL)
    rest = [u for u in users if not is_super_admin(u.role)]
    if not rest:
        return out
    uids = [str(u.id) for u in rest]
    uid_to_tids: dict[str, set[str]] = {uid: set() for uid in uids}
    r1 = await assets_db.execute(
        select(TeamUser.user_id, TeamUser.team_id).where(TeamUser.user_id.in_(uids))
    )
    r2 = await assets_db.execute(
        select(TeamAdmin.user_id, TeamAdmin.team_id).where(TeamAdmin.user_id.in_(uids))
    )
    for uid, tid in r1.all():
        uid_to_tids.setdefault(str(uid), set()).add(str(tid))
    for uid, tid in r2.all():
        uid_to_tids.setdefault(str(uid), set()).add(str(tid))
    all_tids: set[str] = set()
    for s in uid_to_tids.values():
        all_tids.update(s)
    name_by_tid: dict[str, str] = {}
    if all_tids:
        tr = await assets_db.execute(select(Team.id, Team.name).where(Team.id.in_(list(all_tids))))
        for tid, name in tr.all():
            nm = (name or "").strip()
            name_by_tid[str(tid)] = nm or str(tid)
    for u in rest:
        uid = str(u.id)
        tids = sorted(uid_to_tids.get(uid, set()))
        if not tids:
            out[uid] = (None, "—")
            continue
        names = [name_by_tid.get(t, t) for t in sorted(tids, key=lambda x: name_by_tid.get(x, x))]
        label = " / ".join(names)
        primary = tids[0] if len(tids) == 1 else None
        out[uid] = (primary, label)
    return out


async def _assert_may_mutate_user_super_or_team_admin_scoped(
    actor: User,
    target: User,
    assets_db: AsyncSession,
) -> None:
    """超管任意（另由单接口业务规则约束）；团队管理员仅可管辖区内非超管/非团队管理员账号。"""
    if is_super_admin(actor.role):
        return
    if is_team_admin_role(actor.role):
        allowed = await team_crud.list_user_ids_in_teams_administered_by(
            assets_db, str(actor.id)
        )
        if str(target.id) not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not in team admin scope",
            )
        tr = normalize_role(target.role)
        if tr in (CanonicalUserRole.SUPER_ADMIN, CanonicalUserRole.ADMIN):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot modify platform or team admin accounts",
            )
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Forbidden",
    )


async def _assert_user_has_no_project_strong_refs(
    assets_db: AsyncSession,
    user_id: str,
) -> None:
    """删除用户前阻断：仍存在项目成员或项目负责人强关联时不允许删除。"""
    uid = (user_id or "").strip()
    if not uid:
        return

    pm = int(
        (
            await assets_db.execute(
                select(func.count()).select_from(ProjectMember).where(ProjectMember.user_id == uid)
            )
        ).scalar()
        or 0
    )
    po = int(
        (
            await assets_db.execute(
                select(func.count()).select_from(Project).where(Project.owner_id == uid)
            )
        ).scalar()
        or 0
    )

    if pm or po:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="USER_DELETE_BLOCKED_PROJECT_REFS",
        )


async def _remove_team_memberships_for_user(assets_db: AsyncSession, user_id: str) -> None:
    """删除用户在 team_users / team_admins 中的全部行（与主库用户删除配合，避免残留）。"""
    uid = (user_id or "").strip()
    if not uid:
        return
    await assets_db.execute(delete(TeamUser).where(TeamUser.user_id == uid))
    await assets_db.execute(delete(TeamAdmin).where(TeamAdmin.user_id == uid))
    await assets_db.commit()


def _user_search_filter(q: Optional[str]):
    qs = (q or "").strip()
    if not qs:
        return None
    pat = f"%{qs}%"
    return or_(User.username.ilike(pat), User.account_id.ilike(pat))


@router.get("", response_model=UserListPayload)
async def list_users(
    q: Optional[str] = Query(None, description="按登录账号或展示名模糊搜索"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_user_list_access_async),
) -> UserListPayload:
    """
    用户列表（分页）：SUPER_ADMIN 全量；ADMIN 仅管辖团队下 team_users ∪ team_admins。
    OWNER/USER 由依赖拦截；负责人协作请使用项目成员接口。
    """
    search_cond = _user_search_filter(q)

    if is_super_admin(_current_user.role):
        filters = []
        if search_cond is not None:
            filters.append(search_cond)
        cnt_stmt = select(func.count()).select_from(User)
        stmt = select(User)
        for f in filters:
            cnt_stmt = cnt_stmt.where(f)
            stmt = stmt.where(f)
        total = int((await db.execute(cnt_stmt)).scalar() or 0)
        stmt = (
            stmt.order_by(User.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(stmt)
        users = list(result.scalars().all())
    elif is_team_admin_role(_current_user.role):
        allowed = await team_crud.list_user_ids_in_teams_administered_by(
            assets_db, str(_current_user.id)
        )
        if not allowed:
            return UserListPayload(items=[], total=0, page=page, page_size=page_size)
        scope = User.id.in_(list(allowed))
        filters = [scope]
        if search_cond is not None:
            filters.append(search_cond)
        cnt_stmt = select(func.count()).select_from(User).where(*filters)
        total = int((await db.execute(cnt_stmt)).scalar() or 0)
        stmt = (
            select(User)
            .where(*filters)
            .order_by(User.created_at.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await db.execute(stmt)
        users = list(result.scalars().all())
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Management privilege required",
        )
    scope_map = await _team_scope_for_list_users(assets_db, users)
    team_active_map = await batch_user_has_any_active_team_async(db, users)
    items: list[UserListItemOut] = []
    for u in users:
        tid, tname = scope_map.get(str(u.id), (None, "—"))
        base = UserOut.model_validate(u).model_dump()
        uid = str(u.id)
        has_any = team_active_map[uid] if uid in team_active_map else None
        eff = effective_user_is_active(user=u, has_any_active_team=has_any)
        items.append(
            UserListItemOut(**base, team_name=tname, team_id=tid, effective_is_active=eff)
        )
    return UserListPayload(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: CreateUserRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
) -> UserOut:
    """
    创建用户
    - SUPER_ADMIN：可创建 ADMIN / OWNER / USER；team_id 可选（不选则分配平台 Pibot#### 账号且不写入 team_users）；选团队时 account_id 为 teams.code+流水号；创建 ADMIN 时必须指定 team_id
    - ADMIN（团队管理员账号）：仅可创建 OWNER / USER；account_id 取其管辖团队之一（按 team_id 排序首项）的 code+流水号
    - 团队管理员创建的用户：自动加入其管辖的全部团队（team_users）
    """
    display_name = (data.username or "").strip()

    role = normalize_role(data.role)
    team_id_for_account: str | None = None

    if is_super_admin(_current_user.role):
        if role not in (
            CanonicalUserRole.ADMIN,
            CanonicalUserRole.OWNER,
            CanonicalUserRole.USER,
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role for creation",
            )
        tid = (data.team_id or "").strip()
        if role == CanonicalUserRole.ADMIN and not tid:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="team_id is required for ADMIN role",
            )
        if tid:
            team_id_for_account = tid
        else:
            team_id_for_account = None
    elif is_team_admin_role(_current_user.role):
        if role == CanonicalUserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Team admin cannot create ADMIN users",
            )
        if role not in (CanonicalUserRole.OWNER, CanonicalUserRole.USER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Team admin can only create OWNER or USER",
            )
        team_ids = await team_crud.list_team_ids_where_user_is_team_admin(
            assets_db, str(_current_user.id)
        )
        if not team_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No administered team",
            )
        team_id_for_account = team_ids[0]
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privilege to create users",
        )

    try:
        user = await create_user_with_allocated_account_id(
            db,
            assets_db,
            display_username=display_name,
            password=data.password,
            role=UserRole(role.value),
            team_id_for_account=team_id_for_account,
        )
    except UserProvisionError as e:
        key = str(e)
        detail = {
            "USERNAME_EMPTY": "请输入用户名",
            "PASSWORD_EMPTY": "密码不能为空",
            "TEAM_NOT_FOUND": "Team not found",
            "TEAM_INACTIVE": "Team is inactive",
            "ACCOUNT_ID_ALLOCATION_FAILED": "无法分配登录账号，请稍后重试",
            "USER_CREATE_CONFLICT": "创建用户失败（账号分配冲突等），请稍后重试",
        }.get(key, "创建用户失败")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)

    uid = str(user.id)
    creator = str(_current_user.id)

    if is_super_admin(_current_user.role):
        target_team = (data.team_id or "").strip()
        if target_team:
            if role == CanonicalUserRole.ADMIN:
                assets_db.add(
                    TeamAdmin(
                        team_id=target_team,
                        user_id=uid,
                        created_by=creator,
                    )
                )
                assets_db.add(
                    TeamUser(
                        team_id=target_team,
                        user_id=uid,
                        created_by=creator,
                    )
                )
            else:
                assets_db.add(
                    TeamUser(
                        team_id=target_team,
                        user_id=uid,
                        created_by=creator,
                    )
                )
            await assets_db.commit()
    elif is_team_admin_role(_current_user.role):
        team_ids = await team_crud.list_team_ids_where_user_is_team_admin(
            assets_db, str(_current_user.id)
        )
        for tid in team_ids:
            assets_db.add(
                TeamUser(
                    team_id=(tid or "").strip(),
                    user_id=uid,
                    created_by=creator,
                )
            )
        if team_ids:
            await assets_db.commit()

    log_audit_safe(
        user=_current_user,
        action_type=AA.CREATE_USER,
        resource_type=AR.USER,
        resource_id=str(user.id),
        resource_name=display_name,
        detail_json={
            "new_username": display_name,
            "new_account_id": getattr(user, "account_id", None),
            "new_role": data.role,
            "team_id": (data.team_id or "").strip() or team_id_for_account,
        },
        request=request,
    )

    return UserOut.model_validate(user)


@router.patch("/{user_id}/role", response_model=UserOut)
async def update_user_role(
    user_id: str,
    data: UpdateUserRoleRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
) -> UserOut:
    """
    修改用户角色（不含 SUPER_ADMIN）
    - SUPER_ADMIN：可设为 ADMIN / OWNER / USER；最后一个 SUPER_ADMIN 不可被降级
    - ADMIN：仅可改其管辖团队范围内的用户，且新角色只能 OWNER / USER；不可修改 SUPER_ADMIN/ADMIN 账号，不可将任何人设为 ADMIN
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.id == _current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot change your own role",
        )

    new_role = normalize_role(data.role)
    if new_role == CanonicalUserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot assign SUPER_ADMIN via API",
        )

    target_role_before = normalize_role(user.role)

    if is_super_admin(_current_user.role):
        if new_role not in (
            CanonicalUserRole.ADMIN,
            CanonicalUserRole.OWNER,
            CanonicalUserRole.USER,
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid role",
            )
        if target_role_before == CanonicalUserRole.SUPER_ADMIN:
            cnt = await db.execute(
                select(func.count())
                .select_from(User)
                .where(User.role == UserRole.SUPER_ADMIN, User.id != user_id)
            )
            others = int(cnt.scalar() or 0)
            if others == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Cannot demote the last SUPER_ADMIN",
                )
    elif is_team_admin_role(_current_user.role):
        allowed = await team_crud.list_user_ids_in_teams_administered_by(
            assets_db, str(_current_user.id)
        )
        if str(user.id) not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not in team admin scope",
            )
        if target_role_before in (
            CanonicalUserRole.SUPER_ADMIN,
            CanonicalUserRole.ADMIN,
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot change role for platform or team admin accounts",
            )
        if new_role not in (CanonicalUserRole.OWNER, CanonicalUserRole.USER):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Team admin may only assign OWNER or USER",
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privilege to change user role",
        )

    old_val = user.role.value if hasattr(user.role, "value") else str(user.role)
    user.role = UserRole(new_role.value)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    log_audit_safe(
        user=_current_user,
        action_type=AA.UPDATE_USER,
        resource_type=AR.USER,
        resource_id=str(user.id),
        resource_name=user.username,
        detail_json={"operation": "role_change", "old_role": old_val, "new_role": new_role.value},
        request=request,
    )

    return UserOut.model_validate(user)


@router.patch("/{user_id}/disable", response_model=UserOut)
async def disable_user(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_user_account_mutations_async),
) -> UserOut:
    """
    禁用用户（仅 SUPER_ADMIN / 团队管理员账号；辖区内非超管/非团队管理员目标）
    - 设置 is_active = False
    - 撤销该用户的所有 refresh_tokens（强制登出）
    - 记录审计日志
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.id == _current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot disable yourself",
        )

    await _assert_may_mutate_user_super_or_team_admin_scoped(_current_user, user, assets_db)

    if normalize_role(user.role) == CanonicalUserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot disable platform super admin",
        )

    if not user.is_active:
        return UserOut.model_validate(user)

    user.is_active = False
    db.add(user)

    await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked_at.is_(None),
        )
        .values(revoked_at=datetime.now(timezone.utc))
    )

    await db.commit()
    await db.refresh(user)

    log_audit_safe(
        user=_current_user,
        action_type=AA.UPDATE_USER,
        resource_type=AR.USER,
        resource_id=str(user.id),
        resource_name=user.username,
        detail_json={"operation": "disable", "is_active": False},
        request=request,
    )

    return UserOut.model_validate(user)


@router.patch("/{user_id}/enable", response_model=UserOut)
async def enable_user(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_user_account_mutations_async),
) -> UserOut:
    """
    启用用户（仅 SUPER_ADMIN / 团队管理员账号；辖区内目标）
    - 设置 is_active = True
    - 记录审计日志
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    await _assert_may_mutate_user_super_or_team_admin_scoped(_current_user, user, assets_db)

    if user.is_active:
        return UserOut.model_validate(user)

    user.is_active = True
    db.add(user)
    await db.commit()
    await db.refresh(user)

    log_audit_safe(
        user=_current_user,
        action_type=AA.UPDATE_USER,
        resource_type=AR.USER,
        resource_id=str(user.id),
        resource_name=user.username,
        detail_json={"operation": "enable", "is_active": True},
        request=request,
    )

    return UserOut.model_validate(user)


@router.patch("/{user_id}/reset-password", response_model=UserOut)
async def reset_password(
    user_id: str,
    data: ResetPasswordRequest,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_user_account_mutations_async),
) -> UserOut:
    """
    重置用户密码（仅 SUPER_ADMIN / 团队管理员账号；辖区内目标）
    - 更新 password_hash
    - 记录审计日志（不允许记录明文密码）
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.id == _current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot reset your own password from user management",
        )

    await _assert_may_mutate_user_super_or_team_admin_scoped(_current_user, user, assets_db)

    user.password_hash = hash_password(data.new_password)
    db.add(user)
    await db.commit()
    await db.refresh(user)

    log_audit_safe(
        user=_current_user,
        action_type=AA.RESET_PASSWORD,
        resource_type=AR.USER,
        resource_id=str(user.id),
        resource_name=user.username,
        detail_json={"target_username": user.username},
        request=request,
    )

    return UserOut.model_validate(user)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    request: Request,
    db: AsyncSession = Depends(get_async_db),
    assets_db: AsyncSession = Depends(get_data_assets_db),
    _current_user: User = Depends(require_super_admin_or_team_admin_async),
) -> None:
    """
    删除用户
    - SUPER_ADMIN：全库（仍不可删超管账号）
    - 团队管理员：仅管辖团队内用户，且不可删超管/团队管理员账号
    - 不能删除自己
    - 删除前撤销该用户的所有 refresh_tokens
    """
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if user.id == _current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete yourself",
        )

    if normalize_role(user.role) == CanonicalUserRole.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Cannot delete platform super admin",
        )

    if is_super_admin(_current_user.role):
        pass
    elif is_team_admin_role(_current_user.role):
        await _assert_may_mutate_user_super_or_team_admin_scoped(
            _current_user, user, assets_db
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privilege to delete users",
        )

    await _assert_user_has_no_project_strong_refs(assets_db, user_id)
    await _remove_team_memberships_for_user(assets_db, user_id)

    username = user.username

    await db.execute(
        update(RefreshToken)
        .where(
            RefreshToken.user_id == user_id,
            RefreshToken.revoked_at.is_(None),
        )
        .values(revoked_at=datetime.now(timezone.utc))
    )

    log_audit_safe(
        user=_current_user,
        action_type=AA.DELETE_USER,
        resource_type=AR.USER,
        resource_id=user_id,
        resource_name=username,
        detail_json={"deleted_username": username},
        request=request,
    )

    await db.delete(user)
    await db.commit()
