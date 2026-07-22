"""
项目 CRUD（projects 表，PostgreSQL）
"""
import json
import uuid
from typing import List, Optional

from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.project_asset import Project, ProjectMember
from app.schemas.project import ProjectCreate, ProjectUpdate


def _tags_to_db(tags: Optional[List[str]]) -> Optional[str]:
    if not tags:
        return None
    return json.dumps([str(t).strip() for t in tags if str(t).strip()], ensure_ascii=False)


def _tags_from_db(raw: Optional[str]) -> List[str]:
    if not raw or not raw.strip():
        return []
    try:
        parsed = json.loads(raw)
        return list(parsed) if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


async def _ensure_project_members_table(db: AsyncSession) -> None:
    await db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS project_members (
                id SERIAL PRIMARY KEY,
                project_id VARCHAR(128) NOT NULL,
                user_id VARCHAR(36) NOT NULL,
                created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                updated_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
                CONSTRAINT uq_project_members_project_user UNIQUE (project_id, user_id)
            )
            """
        )
    )
    try:
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_project_members_project ON project_members (project_id)"))
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_project_members_user ON project_members (user_id)"))
        await db.execute(text("CREATE INDEX IF NOT EXISTS idx_project_members_updated ON project_members (updated_at)"))
    except Exception:
        pass


async def list_projects(
    db: AsyncSession,
    status: Optional[str] = None,
    allowed_project_ids: Optional[List[str]] = None,
    team_id: Optional[str] = None,
) -> tuple[List[Project], int]:
    """列表；可选按 status、team_id 筛选。按 updated_at 倒序。"""
    query = select(Project)
    count_query = select(func.count()).select_from(Project)
    if allowed_project_ids is not None:
        ids = [str(x).strip() for x in allowed_project_ids if str(x).strip()]
        if not ids:
            return [], 0
        query = query.where(Project.id.in_(ids))
        count_query = count_query.where(Project.id.in_(ids))
    tid = (team_id or "").strip()
    if tid:
        query = query.where(Project.team_id == tid)
        count_query = count_query.where(Project.team_id == tid)
    if status and status.strip():
        query = query.where(Project.status == status.strip())
        count_query = count_query.where(Project.status == status.strip())
    query = query.order_by(Project.updated_at.desc())
    total_r = await db.execute(count_query)
    total = total_r.scalar() or 0
    result = await db.execute(query)
    items = list(result.scalars().all())
    return items, total


async def get_project_by_id(db: AsyncSession, project_id: str) -> Optional[Project]:
    result = await db.execute(select(Project).where(Project.id == project_id))
    return result.scalar_one_or_none()


async def get_project_stats(
    db: AsyncSession,
    project_id: str,
) -> tuple[int, int]:
    """按项目 id 统计：返回 (标注任务数, 数据资产数)。仅按 project_id 匹配，保证表与表通过 project_id 关联。"""
    from app.models.label_task_asset import LabelTask
    from app.models.data_asset import DataAsset

    label_count_r = await db.execute(
        select(func.count()).select_from(LabelTask).where(LabelTask.project_id == project_id)
    )
    data_count_r = await db.execute(
        select(func.count()).select_from(DataAsset).where(DataAsset.project_id == project_id)
    )
    label_count = label_count_r.scalar() or 0
    data_count = data_count_r.scalar() or 0
    return label_count, data_count


async def get_projects_stats_batch(
    db: AsyncSession,
    project_ids: List[str],
) -> dict[str, dict]:
    """批量按项目 id 统计任务数、数据数。返回 { project_id: { label_task_count, dataset_count } }。"""
    from app.models.label_task_asset import LabelTask
    from app.models.data_asset import DataAsset

    if not project_ids:
        return {}

    # 标注任务按 project_id 分组统计
    label_stmt = (
        select(LabelTask.project_id, func.count().label("cnt"))
        .where(LabelTask.project_id.in_(project_ids))
        .group_by(LabelTask.project_id)
    )
    label_rows = (await db.execute(label_stmt)).all()
    label_map = {str(pid): c for pid, c in label_rows if pid}

    # 数据资产按 project_id 分组统计
    data_stmt = (
        select(DataAsset.project_id, func.count().label("cnt"))
        .where(DataAsset.project_id.in_(project_ids))
        .group_by(DataAsset.project_id)
    )
    data_rows = (await db.execute(data_stmt)).all()
    data_map = {str(pid): c for pid, c in data_rows if pid}

    return {
        pid: {
            "label_task_count": label_map.get(pid, 0),
            "dataset_count": data_map.get(pid, 0),
        }
        for pid in project_ids
    }


async def create_project(db: AsyncSession, data: ProjectCreate) -> Project:
    pid = (data.id or "").strip() or str(uuid.uuid4())
    tid = (data.team_id or "").strip() or None if data.team_id is not None else None
    obj = Project(
        id=pid,
        name=data.name.strip(),
        description=(data.description or "").strip() or None,
        tags=_tags_to_db(data.tags),
        status=(data.status or "进行中").strip(),
        owner_id=(data.owner_id or "").strip() or None,
        team_id=tid,
    )
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def update_project(db: AsyncSession, project_id: str, data: ProjectUpdate) -> Optional[Project]:
    obj = await get_project_by_id(db, project_id)
    if not obj:
        return None
    if data.name is not None:
        obj.name = data.name.strip()
    if data.description is not None:
        obj.description = data.description.strip() or None
    if data.tags is not None:
        obj.tags = _tags_to_db(data.tags)
    if data.status is not None:
        obj.status = data.status.strip()
    if data.owner_id is not None:
        obj.owner_id = data.owner_id.strip() or None
    # 不支持通过 PATCH 变更所属团队（不做团队迁移；避免与成员/团队约束不一致）
    await db.commit()
    await db.refresh(obj)
    return obj


async def delete_project(db: AsyncSession, project_id: str) -> bool:
    """仅删除项目记录（不级联）。"""
    obj = await get_project_by_id(db, project_id)
    if not obj:
        return False
    await db.delete(obj)
    await db.commit()
    return True


async def delete_project_children_for_project_id(db: AsyncSession, project_id: str) -> None:
    """
    删除单个项目在库内的从属行（不删 projects 行、不 commit）。
    含 upload_sessions、task_jobs（按 payload 内 project_id/projectId 匹配）。
    """
    from app.models.label_task_asset import LabelTask
    from app.models.data_asset import (
        CollectionTaskAsset,
        CollectionJobAsset,
        ConversionJobAsset,
        DataAsset,
        DataAssetUploadSession,
    )

    pid = (project_id or "").strip()
    if not pid:
        return

    await db.execute(CollectionJobAsset.__table__.delete().where(CollectionJobAsset.project_id == pid))
    await db.execute(CollectionTaskAsset.__table__.delete().where(CollectionTaskAsset.project_id == pid))
    await db.execute(ConversionJobAsset.__table__.delete().where(ConversionJobAsset.project_id == pid))
    await db.execute(ProjectMember.__table__.delete().where(ProjectMember.project_id == pid))
    await db.execute(LabelTask.__table__.delete().where(LabelTask.project_id == pid))
    await db.execute(DataAssetUploadSession.__table__.delete().where(DataAssetUploadSession.project_id == pid))
    # task_jobs.payload JSON：常见路径 params.project_id / params.projectId（及一层嵌套 params）
    await db.execute(
        text(
            """
            DELETE FROM task_jobs
            WHERE (payload->'params'->>'project_id') = :pid
               OR (payload->'params'->>'projectId') = :pid
               OR (payload->'params'->'params'->>'project_id') = :pid
               OR (payload->'params'->'params'->>'projectId') = :pid
            """
        ),
        {"pid": pid},
    )
    await db.execute(DataAsset.__table__.delete().where(DataAsset.project_id == pid))


async def delete_project_with_cascade(db: AsyncSession, project_id: str) -> bool:
    """删除项目并级联删除（仅 DB 记录；磁盘/MinIO 由调用方在适当时机处理）。"""
    obj = await get_project_by_id(db, project_id)
    if not obj:
        return False

    await delete_project_children_for_project_id(db, project_id)
    await db.delete(obj)
    await db.commit()
    return True


async def get_visible_project_ids(db: AsyncSession, *, user_id: str, include_owner_projects: bool = True) -> List[str]:
    """
    返回用户可见的项目 id 列表（非平台 ADMINISTRATOR 的项目范围）。
    - project_members
    - owner_id（历史/负责人）
    - 不因 team_id 关系自动放开可见范围（同团队不代表可见项目）
    """
    uid = (user_id or "").strip()
    if not uid:
        return []

    await _ensure_project_members_table(db)
    member_stmt = select(ProjectMember.project_id).where(ProjectMember.user_id == uid)
    member_rows = (await db.execute(member_stmt)).scalars().all()
    out = {str(pid) for pid in member_rows if pid}

    if include_owner_projects:
        owner_stmt = select(Project.id).where(Project.owner_id == uid)
        owner_rows = (await db.execute(owner_stmt)).scalars().all()
        out.update({str(pid) for pid in owner_rows if pid})

    return sorted(out)


async def is_project_visible_to_user(
    db: AsyncSession, *, project_id: str, user_id: str, include_owner_projects: bool = True
) -> bool:
    pid = (project_id or "").strip()
    uid = (user_id or "").strip()
    if not pid or not uid:
        return False

    await _ensure_project_members_table(db)
    stmt = select(func.count()).select_from(ProjectMember).where(
        (ProjectMember.project_id == pid) & (ProjectMember.user_id == uid)
    )
    r = await db.execute(stmt)
    if (r.scalar() or 0) > 0:
        return True

    if include_owner_projects:
        owner_stmt = select(func.count()).select_from(Project).where((Project.id == pid) & (Project.owner_id == uid))
        r2 = await db.execute(owner_stmt)
        if (r2.scalar() or 0) > 0:
            return True

    return False


async def upsert_project_member(db: AsyncSession, *, project_id: str, user_id: str) -> ProjectMember:
    pid = (project_id or "").strip()
    uid = (user_id or "").strip()
    await _ensure_project_members_table(db)
    existing = (
        await db.execute(select(ProjectMember).where((ProjectMember.project_id == pid) & (ProjectMember.user_id == uid)))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    obj = ProjectMember(project_id=pid, user_id=uid)
    db.add(obj)
    await db.commit()
    await db.refresh(obj)
    return obj


async def delete_project_member(db: AsyncSession, *, project_id: str, user_id: str) -> bool:
    pid = (project_id or "").strip()
    uid = (user_id or "").strip()
    await _ensure_project_members_table(db)
    obj = (
        await db.execute(select(ProjectMember).where((ProjectMember.project_id == pid) & (ProjectMember.user_id == uid)))
    ).scalar_one_or_none()
    if obj is None:
        return False
    await db.delete(obj)
    await db.commit()
    return True


async def list_project_member_ids(db: AsyncSession, *, project_id: str) -> List[str]:
    pid = (project_id or "").strip()
    if not pid:
        return []
    await _ensure_project_members_table(db)
    rows = (await db.execute(select(ProjectMember.user_id).where(ProjectMember.project_id == pid))).scalars().all()
    return [str(x) for x in rows if x]


async def project_member_display_counts_batch(
    db: AsyncSession, projects: List[Project]
) -> dict[str, int]:
    """
    与 GET /projects/{id}/members 返回的 items 条数一致：
    负责人始终占 1 行；project_members 中其余 user_id 各 1 行（与 owner 去重）。
    """
    if not projects:
        return {}
    pids = [str(p.id).strip() for p in projects if p and str(p.id).strip()]
    if not pids:
        return {}
    await _ensure_project_members_table(db)
    stmt = select(ProjectMember.project_id, ProjectMember.user_id).where(
        ProjectMember.project_id.in_(pids)
    )
    rows = (await db.execute(stmt)).all()
    by_pid: dict[str, set[str]] = {}
    for pid, uid in rows:
        sp = str(pid or "").strip()
        su = str(uid or "").strip()
        if not sp or not su:
            continue
        by_pid.setdefault(sp, set()).add(su)

    out: dict[str, int] = {}
    for p in projects:
        spid = str(p.id).strip()
        mids = by_pid.get(spid, set())
        oid = (getattr(p, "owner_id", None) or "").strip()
        if oid:
            out[spid] = 1 + len([u for u in mids if u and u != oid])
        else:
            out[spid] = len([u for u in mids if u])
    return out


async def project_member_display_count_for_project(db: AsyncSession, project_id: str) -> int:
    pid = (project_id or "").strip()
    if not pid:
        return 0
    proj = await get_project_by_id(db, pid)
    if not proj:
        return 0
    m = await project_member_display_counts_batch(db, [proj])
    return m.get(pid, 0)


async def project_ids_with_membership_for_user(
    db: AsyncSession, *, user_id: str, project_ids: List[str]
) -> set[str]:
    """当前用户在 project_members 中拥有行的 project_id 集合（不含仅凭 owner_id/团队可见）。"""
    uid = (user_id or "").strip()
    if not uid or not project_ids:
        return set()
    await _ensure_project_members_table(db)
    pids = [str(x).strip() for x in project_ids if str(x).strip()]
    if not pids:
        return set()
    stmt = select(ProjectMember.project_id).where(
        (ProjectMember.user_id == uid) & (ProjectMember.project_id.in_(pids))
    )
    rows = (await db.execute(stmt)).scalars().all()
    return {str(x) for x in rows if x}
