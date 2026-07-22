"""Data Platform Stage II：artifact_lineage + platform_events

Revision ID: 027_data_platform_stage2
Revises: 026_industrial_storage_index
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "027_data_platform_stage2"
down_revision: Union[str, None] = "026_industrial_storage_index"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    if "artifact_lineage" not in existing:
        op.create_table(
            "artifact_lineage",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("parent_id", sa.String(length=128), nullable=False),
            sa.Column("child_id", sa.String(length=128), nullable=False),
            sa.Column("relation_type", sa.String(length=64), nullable=False),
            sa.Column("job_id", sa.String(length=128), nullable=True),
            sa.Column("metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("parent_id", "child_id", "relation_type", name="uq_artifact_lineage_relation"),
        )
        op.create_index("ix_artifact_lineage_parent_id", "artifact_lineage", ["parent_id"])
        op.create_index("ix_artifact_lineage_child_id", "artifact_lineage", ["child_id"])
        op.create_index("ix_artifact_lineage_relation_type", "artifact_lineage", ["relation_type"])
        op.create_index("ix_artifact_lineage_job_id", "artifact_lineage", ["job_id"])
        op.create_index("idx_artifact_lineage_job", "artifact_lineage", ["job_id", "relation_type"])

    if "platform_events" not in existing:
        op.create_table(
            "platform_events",
            sa.Column("event_id", sa.String(length=64), nullable=False),
            sa.Column("event_type", sa.String(length=64), nullable=False),
            sa.Column("job_id", sa.String(length=128), nullable=False),
            sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("source", sa.String(length=64), nullable=False, server_default="platform"),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.PrimaryKeyConstraint("event_id"),
        )
        op.create_index("ix_platform_events_event_type", "platform_events", ["event_type"])
        op.create_index("ix_platform_events_job_id", "platform_events", ["job_id"])
        op.create_index("ix_platform_events_timestamp", "platform_events", ["timestamp"])
        op.create_index("idx_platform_events_job_type", "platform_events", ["job_id", "event_type"])
        op.create_index("idx_platform_events_type_time", "platform_events", ["event_type", "timestamp"])


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())
    if "platform_events" in existing:
        op.drop_table("platform_events")
    if "artifact_lineage" in existing:
        op.drop_table("artifact_lineage")
