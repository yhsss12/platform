"""
标注任务执行侧权限：区分「能看见任务」与「可写标注 / 可审核」。

- 标注（执行页、保存 instruction、生成/批量标注等）：SUPER_ADMIN、ADMIN 始终允许；项目 owner 允许；
  否则须与 label_tasks.labeler 或 label_tasks.reviewer 用户名一致（审核员可与标注员同样进入执行并写标注）。
- 审核（PATCH 仅改 verified 等）：SUPER_ADMIN、ADMIN 始终允许；否则须与 label_tasks.reviewer 用户名一致。
- 创建/更新任务时：labeler、reviewer 若非空，须为该项目成员用户名（含负责人在成员接口中的展示）。
"""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.permissions import is_project_owner
from app.core.roles import CanonicalUserRole, is_super_admin, normalize_role
from app.crud.project import get_project_by_id, list_project_member_ids
from app.crud.user import get_user_by_id
from app.db.session import AsyncSessionLocal

if TYPE_CHECKING:
    from app.models.label_task_asset import LabelTask
    from app.models.user import User


def username_field_matches_user(user: "User", field: Optional[str]) -> bool:
    u = (getattr(user, "username", None) or "").strip()
    f = (field or "").strip()
    if not u or not f:
        return False
    return u.casefold() == f.casefold()


async def project_member_username_casefold_set(db: AsyncSession, project_id: str) -> set[str]:
    """当前项目「成员接口」等价全集：负责人 + project_members 的去重用户名字符串（小写）。"""
    pid = (project_id or "").strip()
    if not pid:
        return set()
    proj = await get_project_by_id(db, pid)
    if not proj:
        raise HTTPException(status_code=404, detail="项目不存在")
    member_ids = await list_project_member_ids(db, project_id=pid)
    owner_id = (getattr(proj, "owner_id", None) or "").strip()
    all_ids: list[str] = []
    if owner_id:
        all_ids.append(owner_id)
    for uid in member_ids:
        if uid and uid not in all_ids:
            all_ids.append(uid)
    out: set[str] = set()
    async with AsyncSessionLocal() as udb:
        for uid in all_ids:
            u = await get_user_by_id(udb, uid)
            un = (getattr(u, "username", None) or "").strip()
            if un:
                out.add(un.casefold())
    return out


async def assert_labeler_reviewer_are_project_members(
    db: AsyncSession,
    project_id: str,
    *,
    labeler: Optional[str],
    reviewer: Optional[str],
) -> None:
    lab = (labeler or "").strip()
    rev = (reviewer or "").strip()
    if not lab and not rev:
        return
    allowed = await project_member_username_casefold_set(db, project_id)
    if lab and lab.casefold() not in allowed:
        raise HTTPException(status_code=400, detail="标注员须为当前项目成员")
    if rev and rev.casefold() not in allowed:
        raise HTTPException(status_code=400, detail="审核员须为当前项目成员")


async def user_may_annotate_label_task(db: AsyncSession, user: "User", row: "LabelTask") -> bool:
    if is_super_admin(user.role):
        return True
    if normalize_role(user.role) is CanonicalUserRole.ADMIN:
        return True
    pid = (getattr(row, "project_id", None) or "").strip()
    if pid:
        proj = await get_project_by_id(db, pid)
        if proj is not None and is_project_owner(user, proj):
            return True
    if username_field_matches_user(user, getattr(row, "labeler", None)):
        return True
    return username_field_matches_user(user, getattr(row, "reviewer", None))


async def user_may_review_label_task(db: AsyncSession, user: "User", row: "LabelTask") -> bool:
    if is_super_admin(user.role):
        return True
    if normalize_role(user.role) is CanonicalUserRole.ADMIN:
        return True
    return username_field_matches_user(user, getattr(row, "reviewer", None))


async def assert_user_may_annotate_label_task(db: AsyncSession, user: "User", row: "LabelTask") -> None:
    if not await user_may_annotate_label_task(db, user, row):
        raise HTTPException(status_code=403, detail="无权进行标注")


async def assert_user_may_review_label_task(db: AsyncSession, user: "User", row: "LabelTask") -> None:
    if not await user_may_review_label_task(db, user, row):
        raise HTTPException(status_code=403, detail="无权进行审核")


__all__ = [
    "username_field_matches_user",
    "project_member_username_casefold_set",
    "assert_labeler_reviewer_are_project_members",
    "user_may_annotate_label_task",
    "user_may_review_label_task",
    "assert_user_may_annotate_label_task",
    "assert_user_may_review_label_task",
]
