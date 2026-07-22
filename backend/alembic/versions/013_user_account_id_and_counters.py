"""users.account_id / last_login_at + 平台与团队账号流水计数表

Revision ID: 013_user_account_identity
Revises: 012_user_roles_four_tier
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text

revision: str = "013_user_account_identity"
down_revision: Union[str, None] = "012_user_roles_four_tier"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("users"):
        return

    cols = {c["name"] for c in insp.get_columns("users")}
    if "account_id" not in cols:
        op.add_column("users", sa.Column("account_id", sa.String(length=80), nullable=True))
    if "last_login_at" not in cols:
        op.add_column(
            "users",
            sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        )

    # 回填：登录标识与历史 username 一致（保证存量超管 Pibot / admin 等可继续登录）
    op.execute(text("UPDATE users SET account_id = username WHERE account_id IS NULL OR account_id = ''"))

    op.alter_column("users", "account_id", existing_type=sa.String(length=80), nullable=False)

    indexes = {ix["name"] for ix in insp.get_indexes("users")}
    if "uq_users_account_id" not in indexes:
        try:
            op.create_index("uq_users_account_id", "users", ["account_id"], unique=True)
        except Exception:
            pass

    insp2 = inspect(bind)
    if not insp2.has_table("platform_account_counter"):
        op.create_table(
            "platform_account_counter",
            sa.Column("id", sa.SmallInteger(), primary_key=True, nullable=False),
            sa.Column("next_seq", sa.Integer(), nullable=False, server_default="0"),
        )
        op.execute(text("INSERT INTO platform_account_counter (id, next_seq) VALUES (1, 0)"))

    if not insp2.has_table("team_account_counter"):
        op.create_table(
            "team_account_counter",
            sa.Column("team_id", sa.String(length=128), primary_key=True, nullable=False),
            sa.Column("next_seq", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if insp.has_table("team_account_counter"):
        op.drop_table("team_account_counter")
    if insp.has_table("platform_account_counter"):
        op.drop_table("platform_account_counter")
    if insp.has_table("users"):
        try:
            op.drop_index("uq_users_account_id", table_name="users")
        except Exception:
            pass
        cols = {c["name"] for c in insp.get_columns("users")}
        if "last_login_at" in cols:
            op.drop_column("users", "last_login_at")
        if "account_id" in cols:
            op.drop_column("users", "account_id")
