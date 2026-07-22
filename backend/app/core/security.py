from datetime import datetime, timedelta, timezone
from typing import Any, Dict

try:
    import bcrypt  # type: ignore
except ModuleNotFoundError:
    bcrypt = None
from jose import JWTError, jwt

from app.core.config import settings


def hash_password(plain: str) -> str:
    if bcrypt is None:
        raise RuntimeError("bcrypt is required for password hashing")
    if isinstance(plain, str):
        plain_bytes = plain.encode("utf-8")
    else:
        plain_bytes = plain
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_bytes, salt)
    return hashed.decode("utf-8")


# 别名，兼容其他模块的导入
get_password_hash = hash_password


def verify_password(plain: str, hashed: str) -> bool:
    if bcrypt is None:
        raise RuntimeError("bcrypt is required for password verification")
    if isinstance(plain, str):
        plain_bytes = plain.encode("utf-8")
    else:
        plain_bytes = plain
    if isinstance(hashed, str):
        hashed_bytes = hashed.encode("utf-8")
    else:
        hashed_bytes = hashed
    return bcrypt.checkpw(plain_bytes, hashed_bytes)


def _create_token(
    token_type: str,
    subject: str,
    role: str,
    expires_delta: timedelta,
    jti: str | None = None,
    session_id: str | None = None,
) -> Dict[str, Any]:
    """内部工具：创建 JWT，并返回 payload + 编码后的 token"""
    now = datetime.now(timezone.utc)
    payload: Dict[str, Any] = {
        "sub": subject,
        "role": role,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    if jti is not None:
        payload["jti"] = jti
    if session_id is not None:
        payload["sid"] = session_id
    encoded = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALG)
    return {"token": encoded, "payload": payload}


def create_access_token(subject: str, role: str, session_id: str | None = None) -> str:
    """创建 access token，仅返回编码后的 token 字符串"""
    data = _create_token(
        token_type="access",
        subject=subject,
        role=role,
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MIN),
        session_id=session_id,
    )
    return data["token"]


def create_refresh_token(subject: str, role: str, jti: str) -> Dict[str, Any]:
    """
    创建 refresh token，返回包含：
    - token: 编码后的 JWT 字符串
    - payload: 解码前的 payload（含 exp/jti）
    """
    return _create_token(
        token_type="refresh",
        subject=subject,
        role=role,
        expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        jti=jti,
    )


def decode_token(token: str) -> Dict[str, Any]:
    """
    解码并验证任意 JWT。
    - 校验签名
    - 校验 exp（过期会抛出 JWTError）
    """
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALG],
        )
    except JWTError as exc:  # 包含 ExpiredSignatureError 等
        raise exc
    return payload


def require_token_type(payload: Dict[str, Any], expected_type: str) -> None:
    """严格校验 JWT 中的 type 字段"""
    token_type = payload.get("type")
    if token_type != expected_type:
        raise JWTError(f"Invalid token type: expected '{expected_type}', got '{token_type}'")
