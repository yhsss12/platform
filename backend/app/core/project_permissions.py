"""
项目写操作权限（第三阶段-A）：与四层角色 + team_admins 辖区 + projects.owner_id 对齐。

- 不改变 is_project_visible_to_user / get_visible_project_ids 的可见性规则，仅约束 PATCH/DELETE/成员增删。
- ADMIN（users.role）：项目在管辖团队下（Project.team_id 命中 team_admins）时可操作；
  无 team_id 的遗留项目仅项目创建者本人可删，避免团队管理员删他人个人项目。
- OWNER：可编辑/管理成员等与 owner_id 对齐；删除仅允许负责人删自己的项目。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import is_project_owner
from app.core.roles import CanonicalUserRole, is_super_admin, is_team_admin_role, normalize_role
from app.crud import team as team_crud
from app.crud.project import project_ids_with_membership_for_user
from app.models.project_asset import ProjectMember

if TYPE_CHECKING:
    from app.models.project_asset import Project
    from app.models.user import User


async def admin_may_manage_team_project(db: AsyncSession, user: "User", project: "Project") -> bool:
    """users.role=ADMIN 且为该项目所属团队在 team_admins 中的管理员。"""
    if not is_team_admin_role(user.role):
        return False
    tid = (getattr(project, "team_id", None) or "").strip()
    if not tid:
        return False
    row = await team_crud.get_team_admin_by_user(db, tid, str(user.id))
    return row is not None


async def can_edit_project(db: AsyncSession, user: "User", project: "Project") -> bool:
    if is_super_admin(user.role):
        return True
    r = normalize_role(user.role)
    if r is CanonicalUserRole.ADMIN:
        return await admin_may_manage_team_project(db, user, project)
    if r is CanonicalUserRole.OWNER:
        return is_project_owner(user, project)
    return False


async def can_manage_project_tasks(db: AsyncSession, user: "User", project: "Project") -> bool:
    """
    项目任务（标注/转换）管理权限：
    - SUPER_ADMIN：允许
    - 项目 owner_id（创建者/负责人）：允许（不依赖 users.role 是否为 OWNER）
    - ADMIN：需命中 team_admins 辖区
    - 其他：拒绝
    """
    if is_super_admin(user.role):
        return True
    if is_project_owner(user, project):
        return True
    uid = str(getattr(user, "id", "") or "").strip()
    pid = str(getattr(project, "id", "") or "").strip()
    if uid and pid:
        # 负责人账号（users.role=OWNER）在项目成员内，也应具备任务管理权限。
        if normalize_role(user.role) is CanonicalUserRole.OWNER:
            member_hits = await project_ids_with_membership_for_user(db, user_id=uid, project_ids=[pid])
            if pid in member_hits:
                return True
        # 项目创建者兼容：项目创建时会写入 project_members 首行，沿用该约定。
        first_uid = (
            await db.execute(
                select(ProjectMember.user_id)
                .where(ProjectMember.project_id == pid)
                .order_by(ProjectMember.created_at.asc(), ProjectMember.id.asc())
                .limit(1)
            )
        ).scalar_one_or_none()
        if str(first_uid or "").strip() == uid:
            return True
    r = normalize_role(user.role)
    if r is CanonicalUserRole.ADMIN:
        return await admin_may_manage_team_project(db, user, project)
    return False


async def can_delete_project(db: AsyncSession, user: "User", project: "Project") -> bool:
    if is_super_admin(user.role):
        return True
    r = normalize_role(user.role)
    if r is CanonicalUserRole.OWNER:
        return is_project_owner(user, project)
    if r is CanonicalUserRole.ADMIN:
        if await admin_may_manage_team_project(db, user, project):
            return True
        tid = (getattr(project, "team_id", None) or "").strip()
        if not tid and is_project_owner(user, project):
            return True
        return False
    return False


async def can_manage_project_members(db: AsyncSession, user: "User", project: "Project") -> bool:
    if is_super_admin(user.role):
        return True
    r = normalize_role(user.role)
    if r is CanonicalUserRole.ADMIN:
        return await admin_may_manage_team_project(db, user, project)
    if r is CanonicalUserRole.OWNER:
        return is_project_owner(user, project)
    return False


__all__ = [
    "admin_may_manage_team_project",
    "can_edit_project",
    "can_manage_project_tasks",
    "can_delete_project",
    "can_manage_project_members",
]
