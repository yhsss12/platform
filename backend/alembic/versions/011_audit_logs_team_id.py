"""audit_logs.team_id（可空，团队维度审计）

Revision ID: 011_audit_logs_team_id
Revises: 010_team_users
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "011_audit_logs_team_id"
down_revision: Union[str, None] = "010_team_users"
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
    if not insp.has_table("audit_logs"):
        return
    cols = {c["name"] for c in insp.get_columns("audit_logs")}
    if "team_id" not in cols:
        op.add_column("audit_logs", sa.Column("team_id", sa.String(length=128), nullable=True))
    insp = inspect(bind)
    if not _index_exists_on_columns(insp, "audit_logs", ["team_id"]):
        op.create_index("ix_audit_logs_team_id", "audit_logs", ["team_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if not insp.has_table("audit_logs"):
        return
    cols = {c["name"] for c in insp.get_columns("audit_logs")}
    if "team_id" in cols:
        for ix in insp.get_indexes("audit_logs"):
            if list(ix.get("column_names") or []) == ["team_id"] and ix.get("name"):
                op.drop_index(ix["name"], table_name="audit_logs")
        op.drop_column("audit_logs", "team_id")
