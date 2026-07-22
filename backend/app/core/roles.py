from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Union


class CanonicalUserRole(str, Enum):
    """
    平台统一四层角色（权限体系 v1 基础）。

    - SUPER_ADMIN：平台层，全平台
    - ADMIN：团队层（users.role，与 team_admins 表配合；后续可叠加数据范围）
    - OWNER：项目层负责人
    - USER：项目层普通用户（展示文案可为「用户」）
    """

    SUPER_ADMIN = "SUPER_ADMIN"
    ADMIN = "ADMIN"
    OWNER = "OWNER"
    USER = "USER"


# 历史库存值 → 规范角色（迁移后应仅存右侧四类）
_LEGACY_TO_CANONICAL: dict[str, CanonicalUserRole] = {
    # 旧「平台管理员」→ 超级管理员
    "ADMINISTRATOR": CanonicalUserRole.SUPER_ADMIN,
    # 旧枚举 SUPER_ADMIN 与现定义一致
    "SUPER_ADMIN": CanonicalUserRole.SUPER_ADMIN,
    # 旧「成员」→ USER
    "MEMBER": CanonicalUserRole.USER,
    "USER": CanonicalUserRole.USER,
}


def normalize_role(
    raw: Optional[Union[str, Enum, Any]],
    *,
    default: CanonicalUserRole = CanonicalUserRole.USER,
) -> CanonicalUserRole:
    """
    将任意来源的角色转为 CanonicalUserRole。
    未识别值降级为 default（默认 USER），避免脏数据导致 500。
    """

    if raw is None:
        return default

    if isinstance(raw, Enum):
        value = str(getattr(raw, "value", raw))
    else:
        value = str(raw)

    key = value.strip().upper()
    if not key:
        return default

    try:
        return CanonicalUserRole(key)
    except ValueError:
        pass

    if key in _LEGACY_TO_CANONICAL:
        return _LEGACY_TO_CANONICAL[key]

    return default


def is_super_admin(raw: Optional[Union[str, Enum, Any]]) -> bool:
    return normalize_role(raw) is CanonicalUserRole.SUPER_ADMIN


def is_team_admin_role(raw: Optional[Union[str, Enum, Any]]) -> bool:
    """users.role = ADMIN（团队管理员账号）。"""
    return normalize_role(raw) is CanonicalUserRole.ADMIN


def is_project_owner_role(raw: Optional[Union[str, Enum, Any]]) -> bool:
    return normalize_role(raw) is CanonicalUserRole.OWNER


def is_basic_user_role(raw: Optional[Union[str, Enum, Any]]) -> bool:
    return normalize_role(raw) is CanonicalUserRole.USER


def is_owner_or_above(raw: Optional[Union[str, Enum, Any]]) -> bool:
    """
    项目负责人及以上（含超级管理员、团队管理员账号）。
    注意：团队 ADMIN 不等于项目负责人；此函数用于「非 USER」类操作门槛的过渡判断。
    """

    role = normalize_role(raw)
    return role in {
        CanonicalUserRole.SUPER_ADMIN,
        CanonicalUserRole.ADMIN,
        CanonicalUserRole.OWNER,
    }


def is_super_admin_or_team_admin(raw: Optional[Union[str, Enum, Any]]) -> bool:
    """超级管理员或团队管理员账号（users.role=ADMIN），用于部分原「平台管理员」操作域的过渡。"""
    r = normalize_role(raw)
    return r in (CanonicalUserRole.SUPER_ADMIN, CanonicalUserRole.ADMIN)


def is_administrator(raw: Optional[Union[str, Enum, Any]]) -> bool:
    """
    兼容旧调用点：原「平台管理员」仅指 SUPER_ADMIN。

    团队管理员（ADMIN）请使用 is_team_admin_role / permissions.is_team_admin。
    """

    return is_super_admin(raw)


__all__ = [
    "CanonicalUserRole",
    "normalize_role",
    "is_super_admin",
    "is_team_admin_role",
    "is_project_owner_role",
    "is_basic_user_role",
    "is_super_admin_or_team_admin",
    "is_owner_or_above",
    "is_administrator",
]
