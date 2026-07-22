"""
标注任务 / 转换任务：列表与执行可见范围、任务管理（创建/改/删/转换执行）权限。

与四层角色 + team_admins 辖区 + projects.owner_id 对齐；复用 can_edit_project / get_visible_project_ids。
"""
from __future__ import annotations

from typing import Optional, Set

import anyio
from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.project_permissions import can_manage_project_tasks
from app.core.roles import is_basic_user_role, is_super_admin, is_team_admin_role
from app.crud import team as team_crud
from app.crud.project import get_project_by_id, get_visible_project_ids
from app.db.data_assets_session import DataAssetsSessionLocal
from app.models.project_asset import Project
from app.models.user import User


async def scoped_project_ids_for_platform_tasks(db: AsyncSession, user: User) -> Optional[Set[str]]:
    """
    列表与「执行层」可见的项目范围。
    None：不限制（SUPER_ADMIN）；否则仅返回集合内 project_id。
    """
    if is_super_admin(user.role):
        return None
    uid = str(getattr(user, "id", "") or "").strip()
    if not uid:
        return set()
    if is_team_admin_role(user.role):
        team_ids = await team_crud.list_team_ids_where_user_is_team_admin(db, uid)
        if not team_ids:
            return set()
        r = await db.execute(select(Project.id).where(Project.team_id.in_(team_ids)))
        return {str(x) for x in r.scalars().all() if x}
    allowed = await get_visible_project_ids(db, user_id=uid, include_owner_projects=True)
    return set(allowed)


async def assert_platform_task_manage_project(db: AsyncSession, user: User, project_id: str) -> Project:
    """创建/编辑/删除标注任务；创建/删除/执行转换任务等管理动作。"""
    pid = (project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="请选择所属项目")
    proj = await get_project_by_id(db, pid)
    if not proj:
        raise HTTPException(status_code=404, detail="项目不存在")
    if (str(getattr(proj, "status", "") or "").strip()) == "已归档":
        raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
    if not await can_manage_project_tasks(db, user, proj):
        raise HTTPException(status_code=403, detail="无权操作该项目下的任务")
    return proj


async def assert_platform_task_execute_project(db: AsyncSession, user: User, project_id: str) -> Project:
    """执行转换任务等执行动作：在平台任务可见项目范围内即可。"""
    pid = (project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="请选择所属项目")
    proj = await get_project_by_id(db, pid)
    if not proj:
        raise HTTPException(status_code=404, detail="项目不存在")
    if (str(getattr(proj, "status", "") or "").strip()) == "已归档":
        raise HTTPException(status_code=403, detail="项目已归档，禁止该操作")
    if is_super_admin(user.role):
        return proj
    scoped = await scoped_project_ids_for_platform_tasks(db, user)
    if scoped is not None and pid not in scoped:
        raise HTTPException(status_code=403, detail="无权操作该项目下的任务")
    return proj


async def assert_label_task_in_execute_scope(db: AsyncSession, user: User, project_id: Optional[str]) -> None:
    """标注执行链：在平台任务可见项目范围内即可（含 USER）。"""
    if is_super_admin(user.role):
        return
    pid = (project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=404, detail="任务不存在")
    scoped = await scoped_project_ids_for_platform_tasks(db, user)
    if scoped is not None and pid not in scoped:
        raise HTTPException(status_code=404, detail="任务不存在")


def _run_assets_coro(coro):
    """在线程池中的同步路由安全执行协程，复用主事件循环。"""

    async def _await() -> object:
        return await coro

    return anyio.from_thread.run(_await)


def scoped_project_ids_sync(user: User) -> Optional[Set[str]]:
    async def _inner() -> Optional[Set[str]]:
        async with DataAssetsSessionLocal() as db:
            return await scoped_project_ids_for_platform_tasks(db, user)

    return _run_assets_coro(_inner())


def assert_platform_task_manage_project_sync(user: User, project_id: str) -> None:
    async def _inner() -> None:
        async with DataAssetsSessionLocal() as db:
            await assert_platform_task_manage_project(db, user, project_id)

    _run_assets_coro(_inner())


def assert_platform_task_execute_project_sync(user: User, project_id: str) -> None:
    async def _inner() -> None:
        async with DataAssetsSessionLocal() as db:
            await assert_platform_task_execute_project(db, user, project_id)

    _run_assets_coro(_inner())


def assert_conversion_analyze_allowed_sync(user: User, *, dataset_project_id: Optional[str]) -> None:
    """转换频率分析：禁止 USER；其余需在平台任务可见项目内。"""
    if is_super_admin(user.role):
        return
    if is_basic_user_role(user.role):
        raise HTTPException(status_code=403, detail="无权限执行该操作")
    pid = (dataset_project_id or "").strip()
    if not pid:
        raise HTTPException(status_code=400, detail="数据集未绑定项目")
    scoped = scoped_project_ids_sync(user)
    if scoped is not None and pid not in scoped:
        raise HTTPException(status_code=403, detail="无权分析该数据")


def assert_conversion_job_in_scope_sync(user: User, *, project_id: Optional[str]) -> None:
    """转换任务列表/详情：与标注列表同一 project 范围（SUPER 不限；其余 scoped）。"""
    if is_super_admin(user.role):
        return
    scoped = scoped_project_ids_sync(user)
    if scoped is None:
        return
    pid = (project_id or "").strip()
    if not pid or pid not in scoped:
        raise HTTPException(status_code=404, detail="Job not found")


def assert_conversion_manage_or_execute_sync(user: User) -> None:
    """
    创建转换任务、启动后台转换、删除任务/产物：禁止 USER。
    OWNER/ADMIN/SUPER_ADMIN 允许（项目级权限由 assert_platform_task_manage_project 再校验）。
    """
    if is_basic_user_role(user.role):
        raise HTTPException(status_code=403, detail="无权限执行转换任务")


__all__ = [
    "scoped_project_ids_for_platform_tasks",
    "assert_platform_task_manage_project",
    "assert_platform_task_execute_project",
    "assert_label_task_in_execute_scope",
    "scoped_project_ids_sync",
    "assert_platform_task_manage_project_sync",
    "assert_platform_task_execute_project_sync",
    "assert_conversion_analyze_allowed_sync",
    "assert_conversion_job_in_scope_sync",
    "assert_conversion_manage_or_execute_sync",
]
