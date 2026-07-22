"""teams, team_admins, projects.team_id (与数据资产同库的 PostgreSQL 统一实例)

Revision ID: 009_teams_minimal
Revises: 008_upload_sessions_p2
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "009_teams_minimal"
down_revision: Union[str, None] = "008_upload_sessions_p2"
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
    tables = set(insp.get_table_names())

    if "teams" not in tables:
        op.create_table(
            "teams",
            sa.Column("id", sa.String(length=128), nullable=False),
            sa.Column("name", sa.String(length=256), nullable=False),
            sa.Column("code", sa.String(length=64), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("status", sa.String(length=16), nullable=False, server_default="active"),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
            sa.Column("created_by", sa.String(length=128), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("code", name="uq_teams_code"),
        )
        op.create_index("idx_teams_status", "teams", ["status"], unique=False)
        op.create_index("idx_teams_updated", "teams", ["updated_at"], unique=False)
    else:
        uq_names = {c["name"] for c in insp.get_unique_constraints("teams") if c.get("name")}
        if "uq_teams_code" not in uq_names:
            op.create_unique_constraint("uq_teams_code", "teams", ["code"])

    insp = inspect(bind)
    tables = set(insp.get_table_names())

    if "team_admins" not in tables:
        op.create_table(
            "team_admins",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("team_id", sa.String(length=128), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
            sa.Column("created_by", sa.String(length=128), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("team_id", "user_id", name="uq_team_admins_team_user"),
        )
        op.create_index("idx_team_admins_team_id", "team_admins", ["team_id"], unique=False)
        op.create_index("idx_team_admins_user", "team_admins", ["user_id"], unique=False)
    else:
        uq_names = {c["name"] for c in insp.get_unique_constraints("team_admins") if c.get("name")}
        if "uq_team_admins_team_user" not in uq_names:
            op.create_unique_constraint(
                "uq_team_admins_team_user",
                "team_admins",
                ["team_id", "user_id"],
            )
        if not _index_exists_on_columns(insp, "team_admins", ["team_id"]):
            op.create_index("idx_team_admins_team_id", "team_admins", ["team_id"], unique=False)
        if not _index_exists_on_columns(insp, "team_admins", ["user_id"]):
            op.create_index("idx_team_admins_user", "team_admins", ["user_id"], unique=False)

    insp = inspect(bind)
    if "projects" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("projects")}
    if "team_id" not in cols:
        op.add_column("projects", sa.Column("team_id", sa.String(length=128), nullable=True))
    insp = inspect(bind)
    if not _index_exists_on_columns(insp, "projects", ["team_id"]):
        op.create_index("idx_projects_team_id", "projects", ["team_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    tables = set(insp.get_table_names())

    if "projects" in tables:
        cols = {c["name"] for c in insp.get_columns("projects")}
        if "team_id" in cols:
            for ix in insp.get_indexes("projects"):
                if list(ix.get("column_names") or []) == ["team_id"] and ix.get("name"):
                    op.drop_index(ix["name"], table_name="projects")
            op.drop_column("projects", "team_id")

    if "team_admins" in tables:
        op.drop_table("team_admins")
    if "teams" in tables:
        op.drop_table("teams")
