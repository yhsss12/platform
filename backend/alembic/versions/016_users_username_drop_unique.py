"""users.username 去掉全局唯一，仅保留非唯一索引；登录唯一性仍由 account_id 保证

Revision ID: 016_users_username_drop_unique
Revises: 015_renumber_non_super_account_ids
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "016_users_username_drop_unique"
down_revision: Union[str, None] = "015_renumber_non_super_account_ids"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("users"):
        return

    for uq in insp.get_unique_constraints("users") or []:
        cols = list(uq.get("column_names") or [])
        if cols == ["username"]:
            op.drop_constraint(uq["name"], "users", type_="unique")

    for ix in insp.get_indexes("users") or []:
        if ix.get("unique") and list(ix.get("column_names") or []) == ["username"]:
            op.drop_index(ix["name"], table_name="users")

    # 若历史库仅有列级 UNIQUE 而无命名约束暴露，用 PostgreSQL 系统目录兜底
    if getattr(bind.dialect, "name", None) == "postgresql":
        op.execute(
            text(
                """
                DO $$
                DECLARE r record;
                BEGIN
                  FOR r IN
                    SELECT c.conname
                    FROM pg_constraint c
                    JOIN pg_class t ON c.conrelid = t.oid
                    JOIN pg_namespace n ON t.relnamespace = n.oid
                    WHERE t.relname = 'users'
                      AND n.nspname = 'public'
                      AND c.contype = 'u'
                      AND pg_get_constraintdef(c.oid) LIKE '%username%'
                  LOOP
                    EXECUTE format('ALTER TABLE users DROP CONSTRAINT IF EXISTS %I', r.conname);
                  END LOOP;
                END $$;
                """
            )
        )

    insp2 = inspect(bind)
    has_username_ix = any(
        list(ix.get("column_names") or []) == ["username"]
        for ix in (insp2.get_indexes("users") or [])
    )
    if not has_username_ix:
        op.create_index("ix_users_username", "users", ["username"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("users"):
        return
    for ix in insp.get_indexes("users") or []:
        if list(ix.get("column_names") or []) == ["username"]:
            op.drop_index(ix["name"], table_name="users")
    op.create_index("ix_users_username", "users", ["username"], unique=True)
