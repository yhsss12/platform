"""存量非超管用户 account_id 按 teams.code + 四位流水重编号；同步 team_account_counter；超管固定 Pibot0001 兜底

Revision ID: 015_renumber_non_super_account_ids
Revises: 014_super_admin_pibot0001
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "015_renumber_non_super_account_ids"
down_revision: Union[str, None] = "014_super_admin_pibot0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    # 历史 alembic_version.version_num 常为 VARCHAR(32)，本 revision id 较长，stamp 前须加宽
    if insp.has_table("alembic_version"):
        op.execute(
            text("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(128)")
        )
    if not insp.has_table("users"):
        return
    if not insp.has_table("teams"):
        return

    conn = bind

    # 与 014 一致：非超管误占 Pibot0001 时先迁出
    conn.execute(
        text(
            """
            UPDATE users
            SET account_id = '_rel_' || REPLACE(id::text, '-', '')
            WHERE account_id = 'Pibot0001'
              AND UPPER(TRIM(role::text)) <> 'SUPER_ADMIN'
            """
        )
    )

    # 非超管用户（含历史 ADMINISTRATOR / MEMBER 等）
    non_super = conn.execute(
        text(
            """
            SELECT id::text AS uid, created_at, account_id, username
            FROM users
            WHERE UPPER(TRIM(role::text)) <> 'SUPER_ADMIN'
            ORDER BY created_at ASC NULLS LAST, id::text ASC
            """
        )
    ).fetchall()

    if not non_super:
        _ensure_super_admin_pibot0001(conn)
        return

    if not insp.has_table("team_users") and not insp.has_table("team_admins"):
        raise RuntimeError(
            "015: 缺少 team_users / team_admins，无法解析用户所属团队，已中止。"
        )

    orphans: list[str] = []
    by_team: dict[str, list[tuple[str, Any]]] = defaultdict(list)

    for row in non_super:
        uid = (row._mapping["uid"] or "").strip()
        created_at = row.created_at
        tid_row = conn.execute(
            text(
                """
                SELECT team_id FROM (
                    SELECT team_id FROM team_users WHERE user_id = :uid
                    UNION
                    SELECT team_id FROM team_admins WHERE user_id = :uid
                ) x
                ORDER BY team_id ASC
                LIMIT 1
                """
            ),
            {"uid": uid},
        ).fetchone()
        if not tid_row or not (tid_row[0] or "").strip():
            orphans.append(uid)
            continue
        tid = (tid_row[0] or "").strip()
        by_team[tid].append((uid, created_at))

    if orphans:
        raise RuntimeError(
            "015: 以下非超管用户未出现在 team_users / team_admins 中，无法分配团队账号，请先补全团队关系后再迁移："
            + ", ".join(orphans)
        )

    planned: list[tuple[str, str]] = []  # (user_id, new_account_id)

    for tid, members in by_team.items():
        code_row = conn.execute(
            text("SELECT code FROM teams WHERE id = :tid"),
            {"tid": tid},
        ).fetchone()
        if not code_row:
            raise RuntimeError(f"015: 团队 id={tid} 在 teams 表中不存在")
        code = (code_row[0] or "").strip()
        if not code:
            raise RuntimeError(f"015: 团队 id={tid} 的 code 为空，请先清洗 teams 表")

        members_sorted = sorted(members, key=lambda m: (m[1] is None, m[1], m[0]))
        for i, (uid, _) in enumerate(members_sorted):
            new_aid = f"{code}{i + 1:04d}"
            if new_aid == "Pibot0001":
                raise RuntimeError(
                    "015: 分配结果出现 Pibot0001（与超管账号冲突）。请修改相关 teams.code 或数据后重试。"
                )
            planned.append((uid, new_aid))

    new_ids = [p[1] for p in planned]
    if len(new_ids) != len(set(new_ids)):
        raise RuntimeError("015: 内部错误：生成的 account_id 存在重复，已中止")

    # 两阶段：先临时 account_id，避免 unique 冲突
    for uid, _ in planned:
        tmp = "_mig_" + uid.replace("-", "")
        conn.execute(
            text("UPDATE users SET account_id = :tmp WHERE id::text = :uid"),
            {"tmp": tmp, "uid": uid},
        )

    for uid, new_aid in planned:
        conn.execute(
            text("UPDATE users SET account_id = :aid WHERE id::text = :uid"),
            {"aid": new_aid, "uid": uid},
        )

    # team_account_counter：next_seq 至少为本团队已用最大序号
    max_seq_by_team: dict[str, int] = defaultdict(int)
    for tid, members in by_team.items():
        max_seq_by_team[tid] = len(members)

    if insp.has_table("team_account_counter"):
        for tid, mx in max_seq_by_team.items():
            conn.execute(
                text(
                    """
                    INSERT INTO team_account_counter (team_id, next_seq)
                    VALUES (:tid, :mx)
                    ON CONFLICT (team_id) DO UPDATE SET
                        next_seq = GREATEST(team_account_counter.next_seq, EXCLUDED.next_seq)
                    """
                ),
                {"tid": tid, "mx": mx},
            )

    _ensure_super_admin_pibot0001(conn)


def _ensure_super_admin_pibot0001(conn) -> None:
    keeper = conn.execute(
        text(
            """
            SELECT id::text FROM users
            WHERE UPPER(TRIM(role::text)) = 'SUPER_ADMIN' AND is_active = true
            ORDER BY created_at ASC NULLS LAST
            LIMIT 1
            """
        )
    ).fetchone()
    if not keeper:
        return
    kid = keeper[0]
    conn.execute(
        text(
            """
            UPDATE users
            SET account_id = '_rel_' || REPLACE(id::text, '-', '')
            WHERE account_id = 'Pibot0001' AND id::text <> :kid
            """
        ),
        {"kid": kid},
    )
    conn.execute(
        text("UPDATE users SET account_id = 'Pibot0001' WHERE id::text = :kid"),
        {"kid": kid},
    )


def downgrade() -> None:
    # 不可逆：回滚无法恢复旧 account_id
    pass
