"""团队 CRUD（数据资产库 AsyncSession）"""
from __future__ import annotations

import uuid
from typing import List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.team import Team, TeamAdmin, TeamUser
from app.models.project_asset import Project, ProjectMember
from app.models.data_asset import DataAsset


async def get_team_by_id(db: AsyncSession, team_id: str) -> Optional[Team]:
    tid = (team_id or "").strip()
    if not tid:
        return None
    r = await db.execute(select(Team).where(Team.id == tid))
    return r.scalar_one_or_none()


async def get_team_by_code(db: AsyncSession, code: str) -> Optional[Team]:
    c = (code or "").strip()
    if not c:
        return None
    r = await db.execute(select(Team).where(Team.code == c))
    return r.scalar_one_or_none()


async def count_team_admins(db: AsyncSession, team_id: str) -> int:
    tid = (team_id or "").strip()
    if not tid:
        return 0
    q = select(func.count()).select_from(TeamAdmin).where(TeamAdmin.team_id == tid)
    return int((await db.execute(q)).scalar() or 0)


async def count_team_projects(db: AsyncSession, team_id: str) -> int:
    tid = (team_id or "").strip()
    if not tid:
        return 0
    q = select(func.count()).select_from(Project).where(Project.team_id == tid)
    return int((await db.execute(q)).scalar() or 0)


async def count_team_users(db: AsyncSession, team_id: str) -> int:
    tid = (team_id or "").strip()
    if not tid:
        return 0
    q = select(func.count()).select_from(TeamUser).where(TeamUser.team_id == tid)
    return int((await db.execute(q)).scalar() or 0)


async def list_team_user_rows(db: AsyncSession, team_id: str) -> List[TeamUser]:
    tid = (team_id or "").strip()
    if not tid:
        return []
    r = await db.execute(select(TeamUser).where(TeamUser.team_id == tid).order_by(TeamUser.id.asc()))
    return list(r.scalars().all())


async def get_team_user_by_user(db: AsyncSession, team_id: str, user_id: str) -> Optional[TeamUser]:
    r = await db.execute(
        select(TeamUser).where((TeamUser.team_id == team_id) & (TeamUser.user_id == user_id))
    )
    return r.scalar_one_or_none()


async def add_team_user(
    db: AsyncSession,
    *,
    team_id: str,
    user_id: str,
    created_by: Optional[str],
) -> TeamUser:
    obj = TeamUser(
        team_id=team_id.strip(),
        user_id=user_id.strip(),
        created_by=(created_by or "").strip() or None,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def remove_team_user(db: AsyncSession, team_id: str, user_id: str) -> bool:
    row = await get_team_user_by_user(db, team_id, user_id)
    if not row:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def list_teams(db: AsyncSession) -> Tuple[List[Team], int]:
    q = select(Team).order_by(Team.updated_at.desc())
    rows = list((await db.execute(q)).scalars().all())
    return rows, len(rows)


async def create_team(
    db: AsyncSession,
    *,
    name: str,
    code: str,
    description: Optional[str],
    status: str,
    created_by: Optional[str],
) -> Team:
    tid = str(uuid.uuid4())
    obj = Team(
        id=tid,
        name=name.strip(),
        code=code.strip(),
        description=(description or "").strip() or None,
        status=(status or "active").strip().lower() or "active",
        created_by=(created_by or "").strip() or None,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_team(
    db: AsyncSession,
    team_id: str,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[Team]:
    obj = await get_team_by_id(db, team_id)
    if not obj:
        return None
    if name is not None:
        obj.name = name.strip()
    if description is not None:
        obj.description = description.strip() or None
    if status is not None:
        obj.status = (status or "active").strip().lower() or "active"
    await db.commit()
    await db.refresh(obj)
    return obj


async def list_team_admin_rows(db: AsyncSession, team_id: str) -> List[TeamAdmin]:
    tid = (team_id or "").strip()
    if not tid:
        return []
    r = await db.execute(select(TeamAdmin).where(TeamAdmin.team_id == tid).order_by(TeamAdmin.id.asc()))
    return list(r.scalars().all())


async def get_team_admin_by_user(db: AsyncSession, team_id: str, user_id: str) -> Optional[TeamAdmin]:
    r = await db.execute(
        select(TeamAdmin).where((TeamAdmin.team_id == team_id) & (TeamAdmin.user_id == user_id))
    )
    return r.scalar_one_or_none()


async def user_is_team_member_or_admin(db: AsyncSession, team_id: str, user_id: str) -> bool:
    """用户在 team_users 或 team_admins 中即视为对该团队有基础可见性（与平台 ADMINISTRATOR 无关）。"""
    tid = (team_id or "").strip()
    uid = (user_id or "").strip()
    if not tid or not uid:
        return False
    if await get_team_user_by_user(db, tid, uid) is not None:
        return True
    if await get_team_admin_by_user(db, tid, uid) is not None:
        return True
    return False


async def list_team_ids_accessible_by_user(db: AsyncSession, user_id: str) -> List[str]:
    """当前用户作为团队成员或团队管理员所关联的全部 team id（去重）。"""
    uid = (user_id or "").strip()
    if not uid:
        return []
    r1 = await db.execute(select(TeamUser.team_id).where(TeamUser.user_id == uid))
    r2 = await db.execute(select(TeamAdmin.team_id).where(TeamAdmin.user_id == uid))
    tids = {str(x) for x in r1.scalars().all() if x}
    tids.update(str(x) for x in r2.scalars().all() if x)
    return sorted(tids)


async def list_team_ids_where_user_is_team_admin(db: AsyncSession, user_id: str) -> List[str]:
    """当前用户在 team_admins 表中担任管理员的全部 team id（管理侧数据范围口径）。"""
    uid = (user_id or "").strip()
    if not uid:
        return []
    r = await db.execute(select(TeamAdmin.team_id).where(TeamAdmin.user_id == uid))
    return sorted({str(x) for x in r.scalars().all() if x})


async def list_user_ids_in_teams_administered_by(db: AsyncSession, admin_user_id: str) -> set[str]:
    """
    团队管理员（users.role=ADMIN 且须在 team_admins 中有记录）可见的用户 id：
    其管辖团队下出现在 team_users 或 team_admins 的所有用户。
    """
    team_ids = await list_team_ids_where_user_is_team_admin(db, admin_user_id)
    if not team_ids:
        return set()
    r1 = await db.execute(select(TeamUser.user_id).where(TeamUser.team_id.in_(team_ids)))
    r2 = await db.execute(select(TeamAdmin.user_id).where(TeamAdmin.team_id.in_(team_ids)))
    out = {str(x) for x in r1.scalars().all() if x}
    out.update(str(x) for x in r2.scalars().all() if x)
    return out


async def list_user_ids_in_teams_union(db: AsyncSession, team_ids: List[str]) -> set[str]:
    """多个团队下出现在 team_users 或 team_admins 的用户 id 并集（用于项目负责人可见范围）。"""
    ids = [(x or "").strip() for x in team_ids if (x or "").strip()]
    if not ids:
        return set()
    r1 = await db.execute(select(TeamUser.user_id).where(TeamUser.team_id.in_(ids)))
    r2 = await db.execute(select(TeamAdmin.user_id).where(TeamAdmin.team_id.in_(ids)))
    out = {str(x) for x in r1.scalars().all() if x}
    out.update(str(x) for x in r2.scalars().all() if x)
    return out


async def list_distinct_team_ids_for_project_owner(db: AsyncSession, owner_user_id: str) -> List[str]:
    """projects.owner_id = 负责人 的项目所绑定的 team_id（去重、非空）。"""
    uid = (owner_user_id or "").strip()
    if not uid:
        return []
    r = await db.execute(select(Project.team_id).where(Project.owner_id == uid).distinct())
    return [str(x) for x in r.scalars().all() if x and str(x).strip()]


async def list_teams_by_ids(db: AsyncSession, team_ids: List[str]) -> List[Team]:
    ids = [x.strip() for x in team_ids if (x or "").strip()]
    if not ids:
        return []
    r = await db.execute(select(Team).where(Team.id.in_(ids)).order_by(Team.name.asc()))
    return list(r.scalars().all())


async def add_team_admin(
    db: AsyncSession,
    *,
    team_id: str,
    user_id: str,
    created_by: Optional[str],
) -> TeamAdmin:
    obj = TeamAdmin(
        team_id=team_id.strip(),
        user_id=user_id.strip(),
        created_by=(created_by or "").strip() or None,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def remove_team_admin(db: AsyncSession, team_id: str, user_id: str) -> bool:
    row = await get_team_admin_by_user(db, team_id, user_id)
    if not row:
        return False
    await db.delete(row)
    await db.commit()
    return True


async def list_projects_for_team(db: AsyncSession, team_id: str) -> List[Project]:
    tid = (team_id or "").strip()
    if not tid:
        return []
    r = await db.execute(select(Project).where(Project.team_id == tid).order_by(Project.updated_at.desc()))
    return list(r.scalars().all())


async def count_project_members_distinct(db: AsyncSession, project_id: str, owner_id: Optional[str]) -> int:
    pid = (project_id or "").strip()
    if not pid:
        return 0
    r = await db.execute(select(ProjectMember.user_id).where(ProjectMember.project_id == pid))
    uids = {str(x) for x in r.scalars().all() if x}
    oid = (owner_id or "").strip()
    if oid:
        uids.add(oid)
    return len(uids)


async def count_project_assets(db: AsyncSession, project_id: str) -> int:
    pid = (project_id or "").strip()
    if not pid:
        return 0
    q = select(func.count()).select_from(DataAsset).where(DataAsset.project_id == pid)
    return int((await db.execute(q)).scalar() or 0)
