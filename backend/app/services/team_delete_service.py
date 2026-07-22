"""
超级管理员删除团队：先校验无进行中任务，再在数据资产库事务内级联删项目与团队关系，
主库删除 team_account_counter 与「仅本团队」孤儿用户；最后 best-effort 清理 MinIO 与本地文件。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud import team as team_crud
from app.crud.project import delete_project_children_for_project_id, get_project_by_id
from app.models.project_asset import Project, ProjectMember
from app.models.team import Team, TeamAdmin, TeamUser
from app.models.user import User, UserRole
from app.models.data_asset import (
    CollectionJobAsset,
    ConversionJobAsset,
    DataAsset,
    DataAssetUploadSession,
)
from app.services.minio_service import MinioBucketError, MinioConfigError, remove_project_bucket

logger = logging.getLogger(__name__)


class TeamDeleteError(ValueError):
    """业务可映射为 HTTP 400"""


def _norm(s: str | None) -> str:
    return (s or "").strip()


async def _project_ids_for_team(db: AsyncSession, team_id: str) -> list[str]:
    rows = await team_crud.list_projects_for_team(db, team_id)
    return [str(p.id).strip() for p in rows if p and str(p.id).strip()]


async def _collect_file_paths_for_projects(db: AsyncSession, project_ids: list[str]) -> list[str]:
    if not project_ids:
        return []
    r = await db.execute(select(DataAsset.file_path).where(DataAsset.project_id.in_(project_ids)))
    return [str(x[0]) for x in r.all() if x and x[0]]


async def assert_no_blocking_work_for_projects(db: AsyncSession, project_ids: list[str]) -> None:
    """存在运行中/排队中的任务或未完成的直传会话时拒绝删除。"""
    pids = [x for x in project_ids if x]
    if not pids:
        return

    n = int(
        (
            await db.execute(
                select(func.count())
                .select_from(CollectionJobAsset)
                .where(
                    CollectionJobAsset.project_id.in_(pids),
                    CollectionJobAsset.status == "RUNNING",
                )
            )
        ).scalar()
        or 0
    )
    if n > 0:
        raise TeamDeleteError(f"该团队下存在进行中的采集作业（RUNNING），共 {n} 条，请先结束后再删除团队")

    n2 = int(
        (
            await db.execute(
                select(func.count())
                .select_from(ConversionJobAsset)
                .where(
                    ConversionJobAsset.project_id.in_(pids),
                    func.lower(ConversionJobAsset.status).in_(("queued", "running")),
                )
            )
        ).scalar()
        or 0
    )
    if n2 > 0:
        raise TeamDeleteError(f"该团队下存在排队中或运行中的转换任务，共 {n2} 条，请等待结束或取消后再删除团队")

    n3 = int(
        (
            await db.execute(
                select(func.count())
                .select_from(DataAssetUploadSession)
                .where(
                    DataAssetUploadSession.project_id.in_(pids),
                    DataAssetUploadSession.status == "presigned",
                )
            )
        ).scalar()
        or 0
    )
    if n3 > 0:
        raise TeamDeleteError(f"该团队下存在未完成的上传会话（presigned），共 {n3} 个，请先完成或过期后再删除团队")

    busy_assets = int(
        (
            await db.execute(
                select(func.count())
                .select_from(DataAsset)
                .where(
                    DataAsset.project_id.in_(pids),
                    or_(
                        DataAsset.parse_status == "解析中",
                        DataAsset.sync_status == "syncing",
                    ),
                )
            )
        ).scalar()
        or 0
    )
    if busy_assets > 0:
        raise TeamDeleteError(f"该团队下存在解析中或同步中的数据资产，共 {busy_assets} 条，请等待完成后再删除团队")

    n4 = 0
    for pid in pids:
        r4 = await db.execute(
            text(
                """
                SELECT COUNT(*) FROM task_jobs
                WHERE LOWER(status) IN ('pending', 'queued', 'running')
                  AND (
                        (payload->'params'->>'project_id') = :pid
                     OR (payload->'params'->>'projectId') = :pid
                     OR (payload->'params'->'params'->>'project_id') = :pid
                     OR (payload->'params'->'params'->>'projectId') = :pid
                  )
                """
            ),
            {"pid": pid},
        )
        n4 += int(r4.scalar() or 0)
    if n4 > 0:
        raise TeamDeleteError(
            f"该团队下存在未结束的统一任务（task_jobs：pending/queued/running），共 {n4} 条；"
            "可能含导出等未在 payload 中携带 project_id 的任务，请先结束相关任务后再试"
        )


async def _list_orphan_user_ids_to_delete(
    assets_db: AsyncSession,
    main_db: AsyncSession,
    team_id: str,
    candidate_user_ids: set[str],
) -> list[str]:
    """
    仅删除「除本团队外无任何团队关系、无跨团队项目负责/成员」的非超管用户。
    本团队项目即将全部删除，故不单独按「本团队项目」拦截。
    """
    tid = _norm(team_id)
    out: list[str] = []
    for uid in candidate_user_ids:
        u = uid.strip()
        if not u:
            continue
        row = await main_db.execute(select(User).where(User.id == u))
        user = row.scalar_one_or_none()
        if user is None:
            continue
        if getattr(user, "role", None) == UserRole.SUPER_ADMIN:
            continue

        n_other_tu = int(
            (
                await assets_db.execute(
                    select(func.count())
                    .select_from(TeamUser)
                    .where(TeamUser.user_id == u, TeamUser.team_id != tid)
                )
            ).scalar()
            or 0
        )
        if n_other_tu > 0:
            continue
        n_other_ta = int(
            (
                await assets_db.execute(
                    select(func.count())
                    .select_from(TeamAdmin)
                    .where(TeamAdmin.user_id == u, TeamAdmin.team_id != tid)
                )
            ).scalar()
            or 0
        )
        if n_other_ta > 0:
            continue

        n_own_other = int(
            (
                await assets_db.execute(
                    select(func.count())
                    .select_from(Project)
                    .where(
                        Project.owner_id == u,
                        or_(Project.team_id.is_(None), Project.team_id != tid),
                    )
                )
            ).scalar()
            or 0
        )
        if n_own_other > 0:
            continue

        n_pm_other = int(
            (
                await assets_db.execute(
                    select(func.count())
                    .select_from(ProjectMember)
                    .join(Project, ProjectMember.project_id == Project.id)
                    .where(
                        ProjectMember.user_id == u,
                        or_(Project.team_id.is_(None), Project.team_id != tid),
                    )
                )
            ).scalar()
            or 0
        )
        if n_pm_other > 0:
            continue

        out.append(u)
    return out


def _best_effort_delete_local_path(path_str: str) -> str | None:
    """返回错误信息；成功返回 None。"""
    p = (path_str or "").strip()
    if not p or p.startswith("minio://"):
        return None
    try:
        path = Path(p)
        if not path.exists():
            return None
        if path.is_file() or path.is_symlink():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
    except OSError as e:
        return str(e)
    return None


async def delete_team_as_super_admin(
    assets_db: AsyncSession,
    main_db: AsyncSession,
    *,
    team_id: str,
    confirmation_name: str,
) -> dict[str, Any]:
    """
    删除团队及下属项目（库内级联）。confirmation_name 须与团队名称完全一致（去首尾空白）。
    返回摘要字典供 API 使用。
    """
    tid = _norm(team_id)
    if not tid:
        raise TeamDeleteError("team_id 无效")

    team = await team_crud.get_team_by_id(assets_db, tid)
    if not team:
        raise TeamDeleteError("团队不存在")

    expected = _norm(team.name)
    if _norm(confirmation_name) != expected:
        raise TeamDeleteError("确认名称与团队名称不一致，请重新输入")

    project_ids = await _project_ids_for_team(assets_db, tid)
    project_rows = [await get_project_by_id(assets_db, pid) for pid in project_ids]
    project_rows = [p for p in project_rows if p is not None]
    project_names = [_norm(p.name) or "project" for p in project_rows]

    await assert_no_blocking_work_for_projects(assets_db, project_ids)

    tu_rows = await team_crud.list_team_user_rows(assets_db, tid)
    ta_rows = await team_crud.list_team_admin_rows(assets_db, tid)
    candidate_users = {str(r.user_id) for r in tu_rows if getattr(r, "user_id", None)}
    candidate_users.update(str(r.user_id) for r in ta_rows if getattr(r, "user_id", None))

    local_paths = await _collect_file_paths_for_projects(assets_db, project_ids)

    summary = {
        "team_id": tid,
        "team_name": team.name,
        "projects_deleted": 0,
        "assets_deleted_rows": len(local_paths),
        "team_users_removed": len(tu_rows),
        "team_admins_removed": len(ta_rows),
        "users_deleted": 0,
        "storage_warnings": [],
    }

    try:
        for pid in project_ids:
            await delete_project_children_for_project_id(assets_db, pid)
            obj = await get_project_by_id(assets_db, pid)
            if obj:
                await assets_db.delete(obj)
            summary["projects_deleted"] += 1

        await assets_db.execute(delete(TeamAdmin).where(TeamAdmin.team_id == tid))
        await assets_db.execute(delete(TeamUser).where(TeamUser.team_id == tid))
        await assets_db.execute(delete(Team).where(Team.id == tid))
        await assets_db.commit()
    except Exception:
        await assets_db.rollback()
        raise

    orphan_ids = await _list_orphan_user_ids_to_delete(assets_db, main_db, tid, candidate_users)
    try:
        await main_db.execute(
            text("DELETE FROM team_account_counter WHERE team_id = :tid"),
            {"tid": tid},
        )
        if orphan_ids:
            await main_db.execute(delete(User).where(User.id.in_(orphan_ids)))
        summary["users_deleted"] = len(orphan_ids)
        await main_db.commit()
    except Exception:
        await main_db.rollback()
        raise

    for pname in project_names:
        try:
            remove_project_bucket(pname, force=True)
        except MinioConfigError as e:
            summary["storage_warnings"].append(f"MinIO 未配置，跳过 bucket 删除：{e}")
            break
        except MinioBucketError as e:
            summary["storage_warnings"].append(f"项目「{pname}」MinIO bucket 删除：{e}")

    for fp in local_paths:
        err = _best_effort_delete_local_path(fp)
        if err:
            summary["storage_warnings"].append(f"本地路径清理失败 {fp[:80]}…：{err}")

    return summary
