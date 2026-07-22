from typing import Optional

from fastapi import Depends, HTTPException, Request, status, Query
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_token, require_token_type
from app.crud.user import get_user_by_account_id as get_user_by_account_id_async
from app.db.session import get_db as get_db_async
from app.core.roles import (
    CanonicalUserRole,
    is_owner_or_above,
    is_super_admin,
    is_super_admin_or_team_admin,
    is_team_admin_role,
    normalize_role,
)
from app.models import User
from app.services.user_service import get_user_by_account_id
from app.services.user_team_access import (
    user_blocked_by_all_teams_inactive_async,
    user_blocked_by_all_teams_inactive_sync,
)


def _user_from_access_token(token: str, db: Session) -> User:
    try:
        payload = decode_token(token)
        require_token_type(payload, "access")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    account_id: Optional[str] = payload.get("sub")
    if not account_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = get_user_by_account_id(db, account_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User disabled",
        )
    if user_blocked_by_all_teams_inactive_sync(db, user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Team disabled",
        )
    return user


def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """无 Authorization 或解析失败时返回 None，不抛错。"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        return _user_from_access_token(token, db)
    except HTTPException:
        return None


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    """
    获取当前用户（通过 access token）
    - 解析 Bearer token
    - 校验 type=access
    - 查询用户并检查 is_active
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.removeprefix("Bearer ").strip()
    return _user_from_access_token(token, db)


async def get_current_user_ws(
    token: str = Query(...),
) -> User:
    """
    WebSocket 鉴权：不要通过 Depends(get_db) 持有连接整个 WS 生命周期。
    否则每个 WS 连接都会占用一个连接池连接，容易触发 QueuePool timeout 并导致 WS 无故断开。
    """
    t = (token or "").strip()
    if not t:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        return _user_from_access_token(t, db)
    finally:
        db.close()


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """要求高于普通 USER：SUPER_ADMIN / 团队 ADMIN / OWNER。"""
    if not is_owner_or_above(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privilege required",
        )
    return current_user


def require_super_admin(current_user: User = Depends(get_current_user)) -> User:
    """仅平台超级管理员（SUPER_ADMIN）。"""
    if not is_super_admin(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privilege required",
        )
    return current_user


def require_super_admin_or_team_admin(current_user: User = Depends(get_current_user)) -> User:
    """超级管理员或团队管理员账号（users.role=ADMIN）。"""
    if not is_super_admin_or_team_admin(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Management privilege required",
        )
    return current_user


async def get_current_user_async(
    request: Request,
    db: AsyncSession = Depends(get_db_async),
) -> User:
    """
    异步版：获取当前用户（用于 /api/auth/* 等异步路由）。
    使用 AsyncSession，避免 greenlet 混用错误。
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
        require_token_type(payload, "access")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    account_id: Optional[str] = payload.get("sub")
    if not account_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # session 隔离：如果 access_token 含 sid，则必须校验该 session 未被撤销/未过期
    sid: Optional[str] = payload.get("sid")
    if sid:
        from sqlalchemy import select
        from datetime import datetime, timezone
        from app.models.auth_session import AuthSession

        db_sess = (await db.execute(select(AuthSession).where(AuthSession.session_id == sid))).scalar_one_or_none()
        if db_sess is None or db_sess.revoked_at is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session revoked or expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        exp_dt = db_sess.expires_at
        now_ts = datetime.now(timezone.utc).timestamp()
        exp_ts = exp_dt.timestamp() if getattr(exp_dt, "tzinfo", None) else exp_dt.replace(tzinfo=timezone.utc).timestamp()
        if exp_ts <= now_ts:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session revoked or expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        db_sess.last_seen_at = datetime.now(timezone.utc)
        db.add(db_sess)
        await db.commit()

    user = await get_user_by_account_id_async(db, account_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not getattr(user, "is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User disabled",
        )
    if await user_blocked_by_all_teams_inactive_async(db, user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Team disabled",
        )
    # 在 session 未关闭前预加载属性，避免返回后懒加载触发 greenlet 错误
    await db.refresh(user)
    return user


async def require_admin_async(
    current_user: User = Depends(get_current_user_async),
) -> User:
    """异步路由用：与 require_admin 相同，SUPER_ADMIN / 团队 ADMIN / OWNER；普通 USER 禁止。"""
    if not is_owner_or_above(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privilege required",
        )
    return current_user


async def require_super_admin_async(
    current_user: User = Depends(get_current_user_async),
) -> User:
    """仅平台超级管理员（SUPER_ADMIN）；异步路由用。"""
    if not is_super_admin(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Super admin privilege required",
        )
    return current_user


async def require_super_admin_or_team_admin_async(
    current_user: User = Depends(get_current_user_async),
) -> User:
    """超级管理员或团队管理员账号（users.role=ADMIN）；异步路由用。"""
    if not is_super_admin_or_team_admin(current_user.role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Management privilege required",
        )
    return current_user


async def require_user_list_access_async(
    current_user: User = Depends(get_current_user_async),
) -> User:
    """GET /users 列表：仅平台超级管理员与团队管理员账号（与前端用户管理入口一致；OWNER 请用项目成员管理）。"""
    if is_super_admin(current_user.role):
        return current_user
    if is_team_admin_role(current_user.role):
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Management privilege required",
    )


async def require_super_admin_or_project_owner_async(
    current_user: User = Depends(get_current_user_async),
) -> User:
    """
    预留依赖：当前用户账号管理路由已不再使用 OWNER。
    项目负责人能力体现在「项目成员 / 项目编辑」等接口，而非 /users/*。
    """
    if is_super_admin(current_user.role):
        return current_user
    if normalize_role(current_user.role) == CanonicalUserRole.OWNER:
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Super admin or project owner privilege required",
    )


async def require_user_account_mutations_async(
    current_user: User = Depends(get_current_user_async),
) -> User:
    """禁用/启用/重置密码：仅平台超级管理员或团队管理员账号（与 GET/POST 用户管理一致）。"""
    if is_super_admin(current_user.role):
        return current_user
    if is_team_admin_role(current_user.role):
        return current_user
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Insufficient privilege for user account management",
    )
