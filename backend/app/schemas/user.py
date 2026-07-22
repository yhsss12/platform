from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.roles import CanonicalUserRole, normalize_role

PASSWORD_MIN_LEN = 6


class UserOut(BaseModel):
    id: str
    account_id: str
    username: str
    role: str
    is_active: bool
    created_at: datetime
    last_login_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, v: object) -> str:
        """对外统一返回四层角色 code：SUPER_ADMIN / ADMIN / OWNER / USER"""

        return normalize_role(v).value


class CreateUserRequest(BaseModel):
    """展示用户名 + 初始密码 + 角色；登录账号 account_id 由后端按团队/平台规则生成。"""
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=PASSWORD_MIN_LEN)
    """ADMIN / OWNER / USER；具体谁可创建哪种角色由接口按登录者校验。禁止通过 API 创建 SUPER_ADMIN。"""
    role: str
    team_id: str | None = Field(
        default=None,
        description="仅平台超级管理员使用：可选；指定时写入该团队并分配 team_code+流水号 账号。创建 ADMIN 时必须指定。团队管理员创建时忽略（用当前管理员管辖团队）。",
    )

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        role = normalize_role(v)
        if role == CanonicalUserRole.SUPER_ADMIN:
            raise ValueError("Cannot create SUPER_ADMIN via API")
        if role not in (
            CanonicalUserRole.ADMIN,
            CanonicalUserRole.OWNER,
            CanonicalUserRole.USER,
        ):
            raise ValueError("Invalid role for user creation")
        return role.value


class UpdateUserRoleRequest(BaseModel):
    role: str

    @field_validator("role")
    @classmethod
    def _validate_role(cls, v: str) -> str:
        role = normalize_role(v)
        if role == CanonicalUserRole.SUPER_ADMIN:
            raise ValueError("Cannot assign SUPER_ADMIN via API")
        if role not in (
            CanonicalUserRole.ADMIN,
            CanonicalUserRole.OWNER,
            CanonicalUserRole.USER,
        ):
            raise ValueError("Invalid role")
        return role.value


class ResetPasswordRequest(BaseModel):
    new_password: str = Field(min_length=PASSWORD_MIN_LEN)


class UserListItemOut(UserOut):
    """用户列表行：在 UserOut 基础上附带所属团队展示（避免单用户接口强行带团队）。"""

    team_name: str = ""
    team_id: str | None = None
    # 综合可用：账号启用且（无团队关联或至少有一个启用中的团队）
    effective_is_active: bool = True


class UserListPayload(BaseModel):
    """分页用户列表（GET /users）。"""

    items: list[UserListItemOut]
    total: int
    page: int
    page_size: int
