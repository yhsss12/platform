"""将启用中的平台超级管理员登录账号统一为 Pibot0001

Revision ID: 014_super_admin_pibot0001
Revises: 013_user_account_identity
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "014_super_admin_pibot0001"
down_revision: Union[str, None] = "013_user_account_identity"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("users"):
        return

    # 非超管误占 Pibot0001 时先迁出（唯一 account_id）
    op.execute(
        text(
            """
            UPDATE users
            SET account_id = '_rel_' || REPLACE(id::text, '-', '')
            WHERE account_id = 'Pibot0001'
              AND UPPER(TRIM(role)) <> 'SUPER_ADMIN'
            """
        )
    )

    # 当前启用中的唯一超管：Pibot / admin -> Pibot0001（username 保持不动，便于展示名仍为 Pibot）
    op.execute(
        text(
            """
            WITH keeper AS (
                SELECT id FROM users
                WHERE UPPER(TRIM(role)) = 'SUPER_ADMIN' AND is_active = true
                ORDER BY created_at ASC NULLS LAST
                LIMIT 1
            )
            UPDATE users u
            SET account_id = 'Pibot0001'
            FROM keeper k
            WHERE u.id = k.id
              AND u.account_id IN ('Pibot', 'admin')
            """
        )
    )


def downgrade() -> None:
    # 不自动回滚 account_id，避免与登录历史冲突
    pass
