"""
平台权限判断骨架（第一阶段）。

与《权限矩阵 v1》衔接：此处仅提供可复用函数，具体接口/页面拦截在后续阶段逐段接入。
"""
from __future__ import annotations

from typing import Any, Optional

from app.core.roles import (
    CanonicalUserRole,
    is_basic_user_role,
    is_super_admin as is_super_admin_role,
    is_team_admin_role,
    normalize_role,
)
from app.models.user import User


def is_super_admin(user: Optional[User]) -> bool:
    if user is None:
        return False
    return is_super_admin_role(user.role)


def is_team_admin(user: Optional[User]) -> bool:
    if user is None:
        return False
    return is_team_admin_role(user.role)


def is_project_owner(user: Optional[User], project: Any) -> bool:
    """project 须含 owner_id（如 ORM Project）。"""
    if user is None or project is None:
        return False
    oid = str(getattr(project, "owner_id", None) or "").strip()
    return bool(oid) and oid == str(user.id)


def is_basic_user(user: Optional[User]) -> bool:
    if user is None:
        return False
    return is_basic_user_role(user.role)


def can_manage_project_membership(user: Optional[User], project: Any) -> bool:
    """
    同步判断（无 DB）：仅适合粗筛；项目 API 已改用 app.core.project_permissions.can_manage_project_members
    （团队 ADMIN 须命中 team_admins + Project.team_id；USER 即使 owner_id 命中也不可管理成员）。
    """
    if user is None:
        return False
    r = normalize_role(user.role)
    if r is CanonicalUserRole.SUPER_ADMIN:
        return True
    if r is CanonicalUserRole.ADMIN:
        return False
    if r is CanonicalUserRole.OWNER:
        return is_project_owner(user, project)
    return False


def sees_all_projects_without_filter(user: Optional[User]) -> bool:
    """项目列表不做可见性过滤（仅超级管理员）。"""
    return is_super_admin(user)


def can_create_or_delete_projects(user: Optional[User]) -> bool:
    """创建项目入口（粗筛）：超级管理员或团队管理员账号；删除须另用 project_permissions.can_delete_project。"""
    if user is None:
        return False
    r = normalize_role(user.role)
    return r in (CanonicalUserRole.SUPER_ADMIN, CanonicalUserRole.ADMIN)


def is_privileged_above_basic_user(user: Optional[User]) -> bool:
    """高于普通 USER：SUPER_ADMIN / ADMIN / OWNER（用于部分「非成员」能力扩展，按需使用）。"""
    if user is None:
        return False
    return normalize_role(user.role) in {
        CanonicalUserRole.SUPER_ADMIN,
        CanonicalUserRole.ADMIN,
        CanonicalUserRole.OWNER,
    }


__all__ = [
    "is_super_admin",
    "is_team_admin",
    "is_project_owner",
    "is_basic_user",
    "can_manage_project_membership",
    "sees_all_projects_without_filter",
    "can_create_or_delete_projects",
    "is_privileged_above_basic_user",
]
