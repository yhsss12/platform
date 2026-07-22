"""工业级存储索引强化：model_assets / eval_metric_summary 扩展列

Revision ID: 026_industrial_storage_index
Revises: 025_artifact_storage_objects
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "026_industrial_storage_index"
down_revision: Union[str, None] = "025_artifact_storage_objects"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _add_column_if_missing(table: str, column: sa.Column) -> None:
    bind = op.get_bind()
    from sqlalchemy import inspect

    insp = inspect(bind)
    if table not in set(insp.get_table_names()):
        return
    cols = {c["name"] for c in insp.get_columns(table)}
    if column.name not in cols:
        op.add_column(table, column)


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    _add_column_if_missing(
        "model_assets",
        sa.Column("project_id", sa.String(length=128), nullable=True),
    )
    _add_column_if_missing(
        "model_assets",
        sa.Column("checkpoint_kind", sa.String(length=32), nullable=True),
    )
    _add_column_if_missing(
        "eval_metric_summary",
        sa.Column("success_rate", sa.Float(), nullable=True),
    )
    _add_column_if_missing(
        "eval_metric_summary",
        sa.Column("average_score", sa.Float(), nullable=True),
    )

    op.execute(
        """
        UPDATE model_assets
        SET checkpoint_kind = asset_type
        WHERE checkpoint_kind IS NULL AND asset_type IS NOT NULL
        """
    )
    op.execute(
        """
        UPDATE model_assets ma
        SET project_id = wj.project_id
        FROM workspace_jobs wj
        WHERE ma.train_job_id = wj.job_id
          AND ma.project_id IS NULL
          AND wj.project_id IS NOT NULL
        """
    )


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    for table, col in (
        ("eval_metric_summary", "average_score"),
        ("eval_metric_summary", "success_rate"),
        ("model_assets", "checkpoint_kind"),
        ("model_assets", "project_id"),
    ):
        if table in existing:
            cols = {c["name"] for c in insp.get_columns(table)}
            if col in cols:
                op.drop_column(table, col)
