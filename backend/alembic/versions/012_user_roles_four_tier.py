"""四层角色模型：users.role 迁移 + 唯一启用超级管理员

Revision ID: 012_user_roles_four_tier
Revises: 011_audit_logs_team_id
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text

revision: str = "012_user_roles_four_tier"
down_revision: Union[str, None] = "011_audit_logs_team_id"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("users"):
        return

    # 放宽 role 列长度（ADMINISTRATOR 等）
    try:
        op.execute(sa.text("ALTER TABLE users ALTER COLUMN role TYPE VARCHAR(32)"))
    except Exception:
        pass

    # 值迁移：旧 → 新四层
    op.execute(
        text("UPDATE users SET role = 'SUPER_ADMIN' WHERE UPPER(TRIM(role)) = 'ADMINISTRATOR'")
    )
    op.execute(text("UPDATE users SET role = 'USER' WHERE UPPER(TRIM(role)) = 'MEMBER'"))
    op.execute(text("UPDATE users SET role = 'OWNER' WHERE UPPER(TRIM(role)) = 'ADMIN'"))

    # 多个启用中的 SUPER_ADMIN：保留 created_at 最早的一个，其余降为团队管理员 ADMIN
    op.execute(
        text(
            """
            WITH ranked AS (
              SELECT id,
                     ROW_NUMBER() OVER (
                       ORDER BY created_at ASC NULLS LAST, username ASC
                     ) AS rn
              FROM users
              WHERE UPPER(TRIM(role)) = 'SUPER_ADMIN' AND is_active = true
            )
            UPDATE users u
            SET role = 'ADMIN', updated_at = NOW()
            FROM ranked r
            WHERE u.id = r.id AND r.rn > 1
            """
        )
    )

    op.execute(
        text(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS uq_users_one_active_super_admin
            ON users ((1))
            WHERE UPPER(TRIM(role)) = 'SUPER_ADMIN' AND is_active = true
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("users"):
        return
    op.execute(text("DROP INDEX IF EXISTS uq_users_one_active_super_admin"))
    op.execute(
        text("UPDATE users SET role = 'ADMINISTRATOR' WHERE UPPER(TRIM(role)) = 'SUPER_ADMIN'")
    )
    op.execute(text("UPDATE users SET role = 'MEMBER' WHERE UPPER(TRIM(role)) = 'USER'"))
    op.execute(text("UPDATE users SET role = 'ADMIN' WHERE UPPER(TRIM(role)) = 'OWNER'"))
