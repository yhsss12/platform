"""workspace_jobs and workspace_artifacts tables

Revision ID: 020_workspace_jobs_artifacts
Revises: 019_label_tasks_dataset_path_text
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "020_workspace_jobs_artifacts"
down_revision: Union[str, None] = "019_label_tasks_dataset_path_text"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    if "workspace_jobs" not in existing:
        op.create_table(
            "workspace_jobs",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("job_id", sa.String(length=128), nullable=False),
            sa.Column("job_type", sa.String(length=64), nullable=False),
            sa.Column("task_type", sa.String(length=64), nullable=False),
            sa.Column("task_name", sa.String(length=256), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
            sa.Column("source", sa.String(length=32), nullable=False, server_default="real"),
            sa.Column("runner", sa.String(length=128), nullable=True),
            sa.Column("project_id", sa.String(length=128), nullable=True),
            sa.Column("created_by", sa.String(length=128), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("runtime_path", sa.Text(), nullable=False),
            sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("job_id"),
        )
        op.create_index("ix_workspace_jobs_job_type", "workspace_jobs", ["job_type"])
        op.create_index("ix_workspace_jobs_task_type", "workspace_jobs", ["task_type"])
        op.create_index("ix_workspace_jobs_status", "workspace_jobs", ["status"])
        op.create_index("ix_workspace_jobs_source", "workspace_jobs", ["source"])

    if "workspace_artifacts" not in existing:
        op.create_table(
            "workspace_artifacts",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("job_id", sa.String(length=128), nullable=False),
            sa.Column("artifact_type", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=256), nullable=False),
            sa.Column("file_path", sa.Text(), nullable=False),
            sa.Column("url_path", sa.Text(), nullable=True),
            sa.Column("episode_index", sa.Integer(), nullable=True),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.ForeignKeyConstraint(["job_id"], ["workspace_jobs.job_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_workspace_artifacts_job_id", "workspace_artifacts", ["job_id"])
        op.create_index("ix_workspace_artifacts_artifact_type", "workspace_artifacts", ["artifact_type"])


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    op.drop_index("ix_workspace_artifacts_artifact_type", table_name="workspace_artifacts")
    op.drop_index("ix_workspace_artifacts_job_id", table_name="workspace_artifacts")
    op.drop_table("workspace_artifacts")
    op.drop_index("ix_workspace_jobs_source", table_name="workspace_jobs")
    op.drop_index("ix_workspace_jobs_status", table_name="workspace_jobs")
    op.drop_index("ix_workspace_jobs_task_type", table_name="workspace_jobs")
    op.drop_index("ix_workspace_jobs_job_type", table_name="workspace_jobs")
    op.drop_table("workspace_jobs")
