from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求：字段名 ``username`` 为历史兼容；语义为登录账号 ``account_id``。"""

    username: str = Field(..., min_length=1, description="登录账号（account_id），非展示名")
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    session_id: str | None = None
    token_type: str = "bearer"
    role: str
    account_id: str
    username: str


class AccessTokenOnlyResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    session_id: str | None = None
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    account_id: str
    username: str
    role: str
    is_active: bool
    created_at: str
    last_login_at: Optional[str] = None

    class Config:
        from_attributes = True


class UpdateProfileBody(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)


class ChangePasswordBody(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=6)
