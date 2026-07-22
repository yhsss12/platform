"""
用户是否因「所属团队全部处于停用」而不可登录/不可调用需鉴权接口。

- 平台 SUPER_ADMIN：不因团队状态限制。
- 在 team_users 或 team_admins 中无任何记录：不因团队限制（与列表「—」团队一致）。
- 有关联团队：仅当存在至少一个 status=active 的团队时，团队维度才放行。
  多团队用户：只要有一个启用团队即可用，避免误伤跨团队账号。
"""
from __future__ import annotations

from typing import Dict, List

from sqlalchemy import bindparam, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from app.core.roles import is_super_admin
from app.models.user import User


def user_blocked_by_all_teams_inactive_sync(db: Session, user: User) -> bool:
    """同步 Session（JWT 解析等）：与主库同 PostgreSQL，可直查 teams / team_* 表。"""
    if is_super_admin(user.role):
        return False
    uid = str(user.id).strip()
    if not uid:
        return False
    q_ids = text(
        """
        SELECT team_id FROM team_users WHERE user_id = :uid
        UNION
        SELECT team_id FROM team_admins WHERE user_id = :uid
        """
    )
    rows = db.execute(q_ids, {"uid": uid}).fetchall()
    tids = [str(r[0]) for r in rows if r[0]]
    if not tids:
        return False
    st = text(
        "SELECT COUNT(*) FROM teams WHERE lower(status) = 'active' AND id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    cnt = db.execute(st, {"ids": tids}).scalar() or 0
    return int(cnt) == 0


async def user_blocked_by_all_teams_inactive_async(db: AsyncSession, user: User) -> bool:
    if is_super_admin(user.role):
        return False
    uid = str(user.id).strip()
    if not uid:
        return False
    q_ids = text(
        """
        SELECT team_id FROM team_users WHERE user_id = :uid
        UNION
        SELECT team_id FROM team_admins WHERE user_id = :uid
        """
    )
    r1 = await db.execute(q_ids, {"uid": uid})
    rows = r1.fetchall()
    tids = [str(r[0]) for r in rows if r[0]]
    if not tids:
        return False
    st = text(
        "SELECT COUNT(*) FROM teams WHERE lower(status) = 'active' AND id IN :ids"
    ).bindparams(bindparam("ids", expanding=True))
    cnt = (await db.execute(st, {"ids": tids})).scalar() or 0
    return int(cnt) == 0


async def batch_user_has_any_active_team_async(
    db: AsyncSession, users: List[User]
) -> Dict[str, bool]:
    """
    返回 user_id -> 是否存在至少一个启用中的关联团队。
    仅包含有 team_users / team_admins 关系的用户；无关系用户不在 dict 中。
    """
    concern = [u for u in users if not is_super_admin(u.role)]
    if not concern:
        return {}
    uids = [str(u.id).strip() for u in concern if str(u.id).strip()]
    if not uids:
        return {}
    stmt = text(
        """
        WITH mu AS (
            SELECT user_id, team_id FROM team_users WHERE user_id IN :uids
            UNION ALL
            SELECT user_id, team_id FROM team_admins WHERE user_id IN :uids
        )
        SELECT mu.user_id, BOOL_OR(lower(t.status) = 'active') AS any_active
        FROM mu
        JOIN teams t ON t.id = mu.team_id
        GROUP BY mu.user_id
        """
    ).bindparams(bindparam("uids", expanding=True))
    result = await db.execute(stmt, {"uids": uids})
    return {str(r[0]): bool(r[1]) for r in result.fetchall() if r[0]}


def effective_user_is_active(*, user: User, has_any_active_team: bool | None) -> bool:
    """
    has_any_active_team:
      None — 无团队关联行，只看 is_active
      True / False — 有关联，需与 is_active 组合
    """
    if is_super_admin(user.role):
        return bool(getattr(user, "is_active", True))
    if not getattr(user, "is_active", True):
        return False
    if has_any_active_team is None:
        return True
    return bool(has_any_active_team)
