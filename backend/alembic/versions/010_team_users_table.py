"""team_users：团队与普通用户归属（多对多，与 team_admins 并列）

Revision ID: 010_team_users
Revises: 009_teams_minimal
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "010_team_users"
down_revision: Union[str, None] = "009_teams_minimal"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _index_exists_on_columns(insp, table: str, columns: list[str]) -> bool:
    for ix in insp.get_indexes(table):
        if list(ix.get("column_names") or []) == columns:
            return True
    return False


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "team_users" in insp.get_table_names():
        uq_names = {c["name"] for c in insp.get_unique_constraints("team_users") if c.get("name")}
        if "uq_team_users_team_user" not in uq_names:
            op.create_unique_constraint(
                "uq_team_users_team_user",
                "team_users",
                ["team_id", "user_id"],
            )
        if not _index_exists_on_columns(insp, "team_users", ["team_id"]):
            op.create_index("idx_team_users_team_id", "team_users", ["team_id"], unique=False)
        if not _index_exists_on_columns(insp, "team_users", ["user_id"]):
            op.create_index("idx_team_users_user", "team_users", ["user_id"], unique=False)
        return

    op.create_table(
        "team_users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("team_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("created_by", sa.String(length=128), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("team_id", "user_id", name="uq_team_users_team_user"),
    )
    op.create_index("idx_team_users_team_id", "team_users", ["team_id"], unique=False)
    op.create_index("idx_team_users_user", "team_users", ["user_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "team_users" in insp.get_table_names():
        op.drop_table("team_users")
