"""conversion_batch_jobs 父任务表；conversion_jobs.batch_id

Revision ID: 017_conversion_batch_jobs
Revises: 016_users_username_drop_unique
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "017_conversion_batch_jobs"
down_revision: Union[str, None] = "016_users_username_drop_unique"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    insp = inspect(bind)
    if not insp.has_table("conversion_jobs"):
        return

    if not insp.has_table("conversion_batch_jobs"):
        op.execute(
            text(
                """
                CREATE TABLE conversion_batch_jobs (
                    batch_id VARCHAR(64) PRIMARY KEY,
                    task_name VARCHAR(256),
                    source_format VARCHAR(32),
                    target_format VARCHAR(64),
                    project_id VARCHAR(128),
                    project_name VARCHAR(256),
                    creator_id VARCHAR(64),
                    total_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    failed_count INTEGER NOT NULL DEFAULT 0,
                    canceled_count INTEGER NOT NULL DEFAULT 0,
                    running_count INTEGER NOT NULL DEFAULT 0,
                    pending_count INTEGER NOT NULL DEFAULT 0,
                    progress_percent DOUBLE PRECISION NOT NULL DEFAULT 0,
                    overall_status VARCHAR(32) NOT NULL DEFAULT 'PENDING',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_conversion_batch_jobs_project ON conversion_batch_jobs (project_id);
                CREATE INDEX IF NOT EXISTS idx_conversion_batch_jobs_creator ON conversion_batch_jobs (creator_id);
                CREATE INDEX IF NOT EXISTS idx_conversion_batch_jobs_created ON conversion_batch_jobs (created_at);
                CREATE INDEX IF NOT EXISTS idx_conversion_batch_jobs_updated ON conversion_batch_jobs (updated_at);
                """
            )
        )

    cols = {c["name"] for c in insp.get_columns("conversion_jobs")}
    if "batch_id" not in cols:
        op.execute(text("ALTER TABLE conversion_jobs ADD COLUMN batch_id VARCHAR(64)"))
        op.execute(text("CREATE INDEX IF NOT EXISTS idx_conversion_jobs_batch_id ON conversion_jobs (batch_id)"))


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    insp = inspect(bind)
    if insp.has_table("conversion_jobs"):
        cols = {c["name"] for c in insp.get_columns("conversion_jobs")}
        if "batch_id" in cols:
            op.execute(text("DROP INDEX IF EXISTS idx_conversion_jobs_batch_id"))
            op.execute(text("ALTER TABLE conversion_jobs DROP COLUMN IF EXISTS batch_id"))
    insp2 = inspect(bind)
    if insp2.has_table("conversion_batch_jobs"):
        op.execute(text("DROP TABLE IF EXISTS conversion_batch_jobs"))
