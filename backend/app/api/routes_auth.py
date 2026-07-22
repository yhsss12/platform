import hashlib
import secrets
import uuid
from datetime import datetime, timedelta, timezone
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.security import (
    create_access_token,
    decode_token,
    hash_password,
    require_token_type,
    verify_password,
)
from app.core.roles import normalize_role
from app.core.deps import get_current_user_async
from app.db.session import get_db
from app.crud.user import get_user_by_account_id
from app.services.user_team_access import user_blocked_by_all_teams_inactive_async
from app.models.user import User
from app.models.auth_session import AuthSession
from app.schemas.auth import (
    AccessTokenOnlyResponse,
    ChangePasswordBody,
    LoginRequest,
    UpdateProfileBody,
    UserResponse,
    TokenResponse,
)
from app.schemas.common import ApiResponse
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import log_audit_safe

router = APIRouter()
logger = logging.getLogger(__name__)


class RefreshRequest(BaseModel):
    refresh_token: str


def _iso_utc(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.isoformat()


@router.post("/login", response_model=ApiResponse)
async def login(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db),
):
    """用户登录：按 account_id 校验；JWT sub = account_id"""
    account_key = (login_data.username or "").strip()
    user = await get_user_by_account_id(db, account_key)
    if not user or not verify_password(login_data.password, user.password_hash):
        return ApiResponse(
            ok=False,
            error="账号或密码错误，请重试",
        )

    if not getattr(user, "is_active", True):
        log_audit_safe(
            user_id=str(user.id),
            username=user.username,
            role=normalize_role(user.role).value,
            action_type=AA.LOGIN_FAIL,
            resource_type=AR.SESSION,
            resource_name=user.username,
            result="FAIL",
            error_message="账号已禁用",
            request=request,
        )
        return ApiResponse(
            ok=False,
            error="账号已禁用",
        )

    if await user_blocked_by_all_teams_inactive_async(db, user):
        log_audit_safe(
            user_id=str(user.id),
            username=user.username,
            role=normalize_role(user.role).value,
            action_type=AA.LOGIN_FAIL,
            resource_type=AR.SESSION,
            resource_name=user.username,
            result="FAIL",
            error_message="所属团队已停用",
            request=request,
        )
        return ApiResponse(
            ok=False,
            error="所属团队已停用，无法登录",
        )

    role_val = normalize_role(user.role).value
    logger.info(
        "[AUTH-TRACE][BACKEND] login account_id=%s id=%s role=%s",
        user.account_id,
        user.id,
        role_val,
    )
    session_id = (request.headers.get("X-Session-Id") or "").strip() or str(uuid.uuid4())

    refresh_token = secrets.token_urlsafe(48)
    refresh_hash = hashlib.sha256(f"{settings.JWT_SECRET}:{refresh_token}".encode("utf-8")).hexdigest()
    expires_at = datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    from sqlalchemy import select

    existing = (await db.execute(select(AuthSession).where(AuthSession.session_id == session_id))).scalar_one_or_none()
    now = datetime.now(timezone.utc)
    if existing:
        existing.user_id = str(user.id)
        existing.refresh_token_hash = refresh_hash
        existing.expires_at = expires_at
        existing.revoked_at = None
        existing.last_seen_at = now
        db.add(existing)
    else:
        db.add(
            AuthSession(
                session_id=session_id,
                user_id=str(user.id),
                refresh_token_hash=refresh_hash,
                expires_at=expires_at,
                last_seen_at=now,
            )
        )

    user.last_login_at = now
    db.add(user)

    await db.commit()
    await db.refresh(user)

    access_token = create_access_token(
        subject=user.account_id,
        role=role_val,
        session_id=session_id,
    )

    log_audit_safe(
        user_id=str(user.id),
        username=user.username,
        role=role_val,
        action_type=AA.LOGIN_SUCCESS,
        resource_type=AR.SESSION,
        resource_id=session_id,
        resource_name=user.username,
        detail_json={"session_id": session_id, "account_id": user.account_id},
        request=request,
    )

    return ApiResponse(
        ok=True,
        data=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            session_id=session_id,
            token_type="bearer",
            role=role_val,
            account_id=user.account_id,
            username=user.username,
        ),
    )


@router.post("/refresh", response_model=AccessTokenOnlyResponse)
async def refresh_access_token(
    request: Request,
    body: RefreshRequest,
    db: AsyncSession = Depends(get_db),
) -> AccessTokenOnlyResponse:
    """使用 refresh_token + X-Session-Id 换取新的 access_token（不使用 Cookie）"""
    from sqlalchemy import select

    session_id = (request.headers.get("X-Session-Id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Session-Id")

    raw_refresh = (body.refresh_token or "").strip()
    if not raw_refresh:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

    refresh_hash = hashlib.sha256(f"{settings.JWT_SECRET}:{raw_refresh}".encode("utf-8")).hexdigest()
    db_sess = (await db.execute(select(AuthSession).where(AuthSession.session_id == session_id))).scalar_one_or_none()
    now_ts = datetime.now(timezone.utc).timestamp()
    if db_sess is None or db_sess.revoked_at is not None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session revoked or expired")
    exp_dt = db_sess.expires_at
    exp_ts = exp_dt.timestamp() if getattr(exp_dt, "tzinfo", None) else exp_dt.replace(tzinfo=timezone.utc).timestamp()
    if exp_ts <= now_ts:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session revoked or expired")
    if db_sess.refresh_token_hash != refresh_hash:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token mismatch")

    user = await db.get(User, db_sess.user_id)
    if user is None or not getattr(user, "is_active", True):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or disabled")

    if await user_blocked_by_all_teams_inactive_async(db, user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or disabled")

    role_val = normalize_role(user.role).value
    logger.info(
        "[AUTH-TRACE][BACKEND] refresh account_id=%s id=%s role=%s sid=%s",
        user.account_id,
        user.id,
        role_val,
        session_id,
    )

    new_refresh_token = secrets.token_urlsafe(48)
    new_refresh_hash = hashlib.sha256(f"{settings.JWT_SECRET}:{new_refresh_token}".encode("utf-8")).hexdigest()
    db_sess.refresh_token_hash = new_refresh_hash
    db_sess.last_seen_at = datetime.now(timezone.utc)
    db.add(db_sess)
    await db.commit()

    access_token = create_access_token(subject=user.account_id, role=role_val, session_id=session_id)
    return AccessTokenOnlyResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        session_id=session_id,
        token_type="bearer",
    )


@router.get("/me", response_model=ApiResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user_async),
):
    """获取当前用户信息"""
    role_val = normalize_role(current_user.role).value
    logger.info(
        "[AUTH-TRACE][BACKEND] me account_id=%s id=%s role=%s",
        current_user.account_id,
        current_user.id,
        role_val,
    )
    created_at = current_user.created_at.isoformat() if getattr(current_user, "created_at", None) else ""
    return ApiResponse(
        ok=True,
        data=UserResponse(
            id=str(current_user.id),
            account_id=current_user.account_id,
            username=current_user.username,
            role=role_val,
            is_active=getattr(current_user, "is_active", True),
            created_at=created_at,
            last_login_at=_iso_utc(getattr(current_user, "last_login_at", None)),
        ),
    )


@router.patch("/profile", response_model=ApiResponse)
async def update_own_profile(
    body: UpdateProfileBody,
    request: Request,
    current_user: User = Depends(get_current_user_async),
    db: AsyncSession = Depends(get_db),
):
    """修改当前用户展示名（username）；不可改 account_id。"""
    un = (body.username or "").strip()
    if not un:
        return ApiResponse(ok=False, error="用户名不能为空")
    current_user.username = un
    db.add(current_user)
    await db.commit()
    await db.refresh(current_user)
    role_val = normalize_role(current_user.role).value
    log_audit_safe(
        user=current_user,
        action_type=AA.UPDATE_USER,
        resource_type=AR.USER,
        resource_id=str(current_user.id),
        resource_name=current_user.username,
        detail_json={"operation": "profile_username", "account_id": current_user.account_id},
        request=request,
    )
    created_at = current_user.created_at.isoformat() if getattr(current_user, "created_at", None) else ""
    return ApiResponse(
        ok=True,
        data=UserResponse(
            id=str(current_user.id),
            account_id=current_user.account_id,
            username=current_user.username,
            role=role_val,
            is_active=getattr(current_user, "is_active", True),
            created_at=created_at,
            last_login_at=_iso_utc(getattr(current_user, "last_login_at", None)),
        ),
    )


@router.post("/change-password", response_model=ApiResponse)
async def change_own_password(
    body: ChangePasswordBody,
    request: Request,
    current_user: User = Depends(get_current_user_async),
    db: AsyncSession = Depends(get_db),
):
    """用户自助修改密码（与管理员重置他人密码接口分离）。"""
    if not verify_password(body.current_password, current_user.password_hash):
        return ApiResponse(ok=False, error="当前密码不正确")
    current_user.password_hash = hash_password(body.new_password)
    db.add(current_user)
    await db.commit()
    log_audit_safe(
        user=current_user,
        action_type=AA.RESET_PASSWORD,
        resource_type=AR.USER,
        resource_id=str(current_user.id),
        resource_name=current_user.username,
        detail_json={"operation": "self_change_password", "account_id": current_user.account_id},
        request=request,
    )
    return ApiResponse(ok=True, data={"detail": "密码已更新"})


@router.post("/logout", response_model=ApiResponse)
async def logout(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> ApiResponse:
    """
    注销（仅当前 session）：
    - 解析 access_token 获取 sid（或使用 X-Session-Id）
    - 撤销该 session，不影响同一用户其他 session
    """
    from sqlalchemy import select

    user: User | None = None
    sub_claim: str | None = None
    session_id: str | None = None

    auth_header = request.headers.get("Authorization") or ""
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        try:
            payload = decode_token(token)
            require_token_type(payload, "access")
            sub_claim = payload.get("sub")
            session_id = payload.get("sid")
        except Exception:
            session_id = None

    if not session_id:
        session_id = (request.headers.get("X-Session-Id") or "").strip() or None

    if session_id:
        db_sess = (await db.execute(select(AuthSession).where(AuthSession.session_id == session_id))).scalar_one_or_none()
        if db_sess and db_sess.revoked_at is None:
            db_sess.revoked_at = datetime.now(timezone.utc)
            db_sess.last_seen_at = datetime.now(timezone.utc)
            db.add(db_sess)
            user = await db.get(User, db_sess.user_id)
            await db.commit()

    log_audit_safe(
        user_id=str(user.id) if user else None,
        username=(user.username if user else sub_claim),
        role=normalize_role(user.role).value if user else None,
        action_type=AA.LOGOUT,
        resource_type=AR.SESSION,
        resource_id=session_id,
        resource_name=(user.username if user else sub_claim),
        detail_json={"session_id": session_id} if session_id else None,
        request=request,
    )

    return ApiResponse(ok=True, data={"detail": "Logged out"})
