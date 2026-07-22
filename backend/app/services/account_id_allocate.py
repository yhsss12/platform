"""登录账号 account_id 分配：平台 Pibot####、团队 {team_code}####（并发安全，依赖计数表 + users 唯一约束）"""
from __future__ import annotations

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.crud.team import get_team_by_id
from app.models.user import User


async def _account_id_taken(db: AsyncSession, account_id: str) -> bool:
    aid = (account_id or "").strip()
    if not aid:
        return True
    r = await db.execute(select(User.id).where(User.account_id == aid).limit(1))
    return r.scalar_one_or_none() is not None


async def allocate_displaced_account_id(db: AsyncSession) -> str:
    """为非超管占用 Pibot 等场景分配新的平台格式账号（Pibot0001 起）。"""
    await db.execute(
        text(
            "INSERT INTO platform_account_counter (id, next_seq) VALUES (1, 0) "
            "ON CONFLICT (id) DO NOTHING"
        )
    )
    for _ in range(500):
        r = await db.execute(
            text(
                "UPDATE platform_account_counter SET next_seq = next_seq + 1 "
                "WHERE id = 1 RETURNING next_seq"
            )
        )
        seq = int(r.scalar_one())
        candidate = f"Pibot{seq:04d}"
        if not await _account_id_taken(db, candidate):
            return candidate
    raise RuntimeError("Failed to allocate platform account_id")


async def allocate_team_scoped_account_id(
    db: AsyncSession,
    assets_db: AsyncSession,
    team_id: str,
) -> str:
    team = await get_team_by_id(assets_db, (team_id or "").strip())
    if team is None:
        raise ValueError("TEAM_NOT_FOUND")
    code = (getattr(team, "code", None) or "").strip()
    if not code:
        raise ValueError("TEAM_CODE_EMPTY")
    tid = (team_id or "").strip()
    await db.execute(
        text(
            "INSERT INTO team_account_counter (team_id, next_seq) VALUES (:tid, 0) "
            "ON CONFLICT (team_id) DO NOTHING"
        ),
        {"tid": tid},
    )
    for _ in range(500):
        r = await db.execute(
            text(
                "UPDATE team_account_counter SET next_seq = next_seq + 1 "
                "WHERE team_id = :tid RETURNING next_seq"
            ),
            {"tid": tid},
        )
        seq = int(r.scalar_one())
        candidate = f"{code}{seq:04d}"
        if not await _account_id_taken(db, candidate):
            return candidate
    raise RuntimeError("Failed to allocate team account_id")
