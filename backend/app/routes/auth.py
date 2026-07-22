"""
历史认证实现：HttpOnly Cookie + 同步 Session + refresh_tokens 表。

当前线上 API 实际挂载的是 ``app.api.routes_auth``（sessionStorage + auth_sessions）。
请勿在新需求中扩展本文件；保留仅供对照与迁移期排查，删除前需确认无外部依赖。
"""

from datetime import datetime, timezone
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from jose import JWTError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    require_token_type,
    verify_password,
)
from app.models import RefreshToken, User
from app.schemas.auth import (
    AccessTokenOnlyResponse,
    LoginRequest,
    TokenResponse,
    UserResponse,
)
from app.services.user_service import get_user_by_account_id
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import log_audit_safe
from app.core.roles import normalize_role

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_refresh_cookie(response: Response, refresh_token: str, expires_at: datetime) -> None:
    """按照规范设置 HttpOnly refresh_token Cookie"""
    # 确保 expires_at 是 timezone-aware
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    max_age = int((expires_at - now).total_seconds())
    
    # expires 参数应该是 datetime 对象（Starlette/FastAPI 会自动转换）
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
        path="/",
        max_age=max_age,
        expires=expires_at,  # 使用 datetime 对象，不是 timestamp
    )


def _clear_refresh_cookie(response: Response) -> None:
    """清除 refresh_token Cookie"""
    response.set_cookie(
        key="refresh_token",
        value="",
        httponly=True,
        secure=settings.COOKIE_SECURE,
        samesite=settings.COOKIE_SAMESITE,
        domain=settings.COOKIE_DOMAIN,
        path="/",
        max_age=0,
        expires=0,
    )


@router.post("/login", response_model=TokenResponse)
def login(
    data: LoginRequest,
    response: Response,
    request: Request,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """
    用户登录：
    - 校验用户名密码
    - 返回 access_token（JSON）
    - 将 refresh_token 写入 HttpOnly Cookie（不再出现在 JSON）
    """
    user = get_user_by_account_id(db, (data.username or "").strip())
    if not user:
        log_audit_safe(
            username=data.username,
            action_type=AA.LOGIN_FAIL,
            resource_type=AR.SESSION,
            resource_name=data.username,
            result="FAIL",
            error_message="账号或密码错误，请重试",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号或密码错误，请重试",
        )

    if not user.is_active:
        # 登录失败：用户已禁用
        log_audit_safe(
            username=data.username,
            action_type=AA.LOGIN_FAIL,
            resource_type=AR.SESSION,
            resource_name=data.username,
            result="FAIL",
            error_message="用户已禁用",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User disabled",
        )

    if not verify_password(data.password, user.password_hash):
        # 登录失败：密码错误
        log_audit_safe(
            username=data.username,
            action_type=AA.LOGIN_FAIL,
            resource_type=AR.SESSION,
            resource_name=data.username,
            result="FAIL",
            error_message="账号或密码错误，请重试",
            request=request,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="账号或密码错误，请重试",
        )

    access_token = create_access_token(
        subject=user.account_id,
        role=normalize_role(user.role).value,
    )

    # 生成 refresh token（带 jti），并持久化到数据库
    jti = str(uuid.uuid4())
    refresh_data = create_refresh_token(
        subject=user.account_id,
        role=normalize_role(user.role).value,
        jti=jti,
    )
    refresh_token = refresh_data["token"]
    payload = refresh_data["payload"]
    exp = payload.get("exp")
    if isinstance(exp, (int, float)):
        expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
    else:
        # jose 默认会返回 datetime
        expires_at = exp

    db_refresh = RefreshToken(
        user_id=user.id,
        token_jti=jti,
        expires_at=expires_at,
    )
    db.add(db_refresh)
    db.commit()

    # 写入 HttpOnly Cookie
    _set_refresh_cookie(response, refresh_token, expires_at)

    log_audit_safe(
        user=user,
        action_type=AA.LOGIN_SUCCESS,
        resource_type=AR.SESSION,
        resource_id=str(user.id),
        resource_name=user.username,
        detail_json={"jti": jti},
        request=request,
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        role=normalize_role(user.role).value,
        account_id=user.account_id,
        username=user.username,
    )


@router.post("/refresh", response_model=AccessTokenOnlyResponse)
def refresh_access_token(
    request: Request,
    db: Session = Depends(get_db),
) -> AccessTokenOnlyResponse:
    """
    使用 HttpOnly Cookie 中的 refresh_token 换取新的 access_token
    """
    raw_refresh = request.cookies.get("refresh_token")
    if not raw_refresh:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing refresh token",
        )

    try:
        payload = decode_token(raw_refresh)
        require_token_type(payload, "refresh")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    jti = payload.get("jti")
    account_id = payload.get("sub")
    if not jti or not account_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token payload",
        )

    # 校验 refresh token 是否存在且未撤销、未过期
    db_token: RefreshToken | None = (
        db.query(RefreshToken).filter(RefreshToken.token_jti == jti).one_or_none()
    )
    now_ts = datetime.now(timezone.utc).timestamp()
    if db_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or expired",
        )
    # 统一转换为时间戳以避免 offset-naive / offset-aware 比较问题
    exp_dt = db_token.expires_at
    if exp_dt.tzinfo is None:
        exp_ts = exp_dt.replace(tzinfo=timezone.utc).timestamp()
    else:
        exp_ts = exp_dt.timestamp()

    if db_token.revoked_at is not None or exp_ts <= now_ts:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token revoked or expired",
        )

    # 确认用户仍然存在且可用
    user: User | None = get_user_by_account_id(db, account_id)
    if user is None or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or disabled",
        )

    access_token = create_access_token(
        subject=user.account_id,
        role=normalize_role(user.role).value,
    )
    return AccessTokenOnlyResponse(access_token=access_token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
def get_me(
    request: Request,
    db: Session = Depends(get_db),
) -> UserResponse:
    """
    使用 Authorization: Bearer <access_token> 获取当前用户信息
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
        require_token_type(payload, "access")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
        )

    account_id = payload.get("sub")
    if not account_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token payload",
        )

    user: User | None = get_user_by_account_id(db, account_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User disabled",
        )

    return UserResponse(
        id=str(user.id),
        account_id=user.account_id,
        username=user.username,
        role=normalize_role(user.role).value,
        is_active=user.is_active,
        created_at=user.created_at.isoformat(),
        last_login_at=user.last_login_at.isoformat() if getattr(user, "last_login_at", None) else None,
    )


@router.post("/logout")
def logout(
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
) -> dict:
    """
    注销：
    - 清除 refresh_token Cookie
    - 如果解析成功，则在服务端标记当前 refresh token 为 revoked
    - 记录审计日志
    """
    raw_refresh = request.cookies.get("refresh_token")
    user = None
    sub_claim = None
    jti: str | None = None

    if raw_refresh:
        try:
            payload = decode_token(raw_refresh)
            require_token_type(payload, "refresh")
            jti = payload.get("jti")
            sub_claim = payload.get("sub")  # JWT sub = account_id
        except JWTError:
            jti = None

        if jti:
            db_token: RefreshToken | None = (
                db.query(RefreshToken)
                .filter(RefreshToken.token_jti == jti)
                .one_or_none()
            )
            if db_token and db_token.revoked_at is None:
                db_token.revoked_at = datetime.now(timezone.utc)
                db.add(db_token)
                db.commit()
                # 如果有 user_id，获取 user 对象
                if db_token.user_id:
                    user = db.query(User).filter(User.id == db_token.user_id).one_or_none()

    log_audit_safe(
        user=user,
        username=(user.username if user else sub_claim),
        action_type=AA.LOGOUT,
        resource_type=AR.SESSION,
        resource_id=str(user.id) if user else None,
        resource_name=(user.username if user else sub_claim),
        detail_json={"jti": jti} if jti else None,
        request=request,
    )

    _clear_refresh_cookie(response)
    return {"detail": "Logged out"}
