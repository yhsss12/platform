"""
数据资产可见性（与项目/团队权限矩阵一致，供 /data-assets、/hdf5-datasets 等复用）。

- SUPER_ADMIN：不限制 project_id（返回 None 表示全量）。
- ADMIN（users.role=ADMIN）：仅「其在 team_admins 中管辖的团队」下的项目（Project.team_id ∈ 管辖团队）。
- OWNER / USER 等：复用 get_visible_project_ids / is_project_visible_to_user（成员、负责人、团队成员/管理员路径）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.roles import is_basic_user_role, is_super_admin, is_team_admin_role
from app.crud import team as team_crud
from app.crud.project import get_project_by_id, get_visible_project_ids, is_project_visible_to_user
from app.models.project_asset import Project

if TYPE_CHECKING:
    from app.models.data_asset import DataAsset
    from app.models.user import User


class _ProjectIdRef:
    """仅带 project_id，供 data_asset_visible_to_user 复用写入矩阵（超管 / 团队管理员辖区 / 成员可见）。"""

    __slots__ = ("project_id",)

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id


async def user_may_write_data_asset_to_project(
    db: AsyncSession,
    current_user: "User",
    project_id: str,
) -> bool:
    """当前用户是否可按数据资产列表同一矩阵向该项目写入（导入/登记/直传完成）。"""
    pid = (project_id or "").strip()
    if not pid:
        return False
    return await data_asset_visible_to_user(db, current_user, _ProjectIdRef(pid))  # type: ignore[arg-type]


async def resolve_project_for_data_asset_import(
    db: AsyncSession,
    raw: str,
) -> tuple[Optional[Project], Optional[str]]:
    """
    解析导入目标项目：先按 ID，再按唯一名称。
    返回 (Project, None) 或 (None, 中文错误文案)。
    """
    s = (raw or "").strip()
    if not s:
        return None, "请选择所属项目"
    p = await get_project_by_id(db, s)
    if p:
        return p, None
    r = await db.execute(select(Project).where(Project.name == s))
    rows = list(r.scalars().all())
    if len(rows) == 1:
        return rows[0], None
    if len(rows) > 1:
        return None, "项目名重复，请使用项目 ID"
    return None, "项目不存在"


async def assert_may_write_project_for_data_asset_import(
    db: AsyncSession,
    current_user: "User",
    raw: str,
) -> tuple[Optional[Project], Optional[str]]:
    """解析项目 + 归档拦截 + 与列表一致的写入权限。"""
    p, err = await resolve_project_for_data_asset_import(db, raw)
    if err or p is None:
        return None, err
    if (p.status or "").strip() == "已归档":
        return None, "项目已归档，禁止该操作"
    pid = str(p.id)
    if not await user_may_write_data_asset_to_project(db, current_user, pid):
        return None, "无权限向该项目导入或登记数据资产"
    return p, None


async def data_assets_allowed_project_ids(db: AsyncSession, current_user: "User") -> Optional[List[str]]:
    """None 表示不限制；[] 表示无任何可见项目。"""
    if is_super_admin(current_user.role):
        return None
    if is_team_admin_role(current_user.role):
        team_ids = await team_crud.list_team_ids_where_user_is_team_admin(db, str(current_user.id))
        if not team_ids:
            return []
        from app.models.project_asset import Project

        r = await db.execute(select(Project.id).where(Project.team_id.in_(team_ids)))
        return sorted({str(x) for x in r.scalars().all() if x})
    return await get_visible_project_ids(db, user_id=str(current_user.id), include_owner_projects=True)


async def data_asset_visible_to_user(db: AsyncSession, current_user: "User", asset: Optional["DataAsset"]) -> bool:
    if asset is None:
        return False
    if is_super_admin(current_user.role):
        return True
    pid = (asset.project_id or "").strip()
    if not pid:
        return False
    if is_team_admin_role(current_user.role):
        project = await get_project_by_id(db, pid)
        if not project:
            return False
        tid = (getattr(project, "team_id", None) or "").strip()
        if not tid:
            return False
        row = await team_crud.get_team_admin_by_user(db, tid, str(current_user.id))
        return row is not None
    return await is_project_visible_to_user(
        db, project_id=pid, user_id=str(current_user.id), include_owner_projects=True
    )


def user_cannot_delete_data_asset(current_user: "User") -> bool:
    """四层模型中 USER（含历史 MEMBER→USER）不可删数据资产。"""
    return is_basic_user_role(current_user.role)
