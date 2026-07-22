"""conversion_batch_jobs：补齐 canceled_count（NOT NULL + server default），修正历史 NULL

Revision ID: 018_conversion_batch_jobs_canceled_count
Revises: 017_conversion_batch_jobs
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "018_conversion_batch_jobs_canceled_count"
down_revision: Union[str, None] = "017_conversion_batch_jobs"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    insp = inspect(bind)
    if not insp.has_table("conversion_batch_jobs"):
        return

    cols = {c["name"] for c in insp.get_columns("conversion_batch_jobs")}
    if "canceled_count" not in cols:
        op.execute(
            text(
                "ALTER TABLE conversion_batch_jobs "
                "ADD COLUMN canceled_count INTEGER NOT NULL DEFAULT 0"
            )
        )
        return

    op.execute(text("UPDATE conversion_batch_jobs SET canceled_count = 0 WHERE canceled_count IS NULL"))
    op.execute(text("ALTER TABLE conversion_batch_jobs ALTER COLUMN canceled_count SET DEFAULT 0"))
    op.execute(text("ALTER TABLE conversion_batch_jobs ALTER COLUMN canceled_count SET NOT NULL"))


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    insp = inspect(bind)
    if not insp.has_table("conversion_batch_jobs"):
        return
    cols = {c["name"] for c in insp.get_columns("conversion_batch_jobs")}
    if "canceled_count" in cols:
        op.execute(text("ALTER TABLE conversion_batch_jobs DROP COLUMN IF EXISTS canceled_count"))
