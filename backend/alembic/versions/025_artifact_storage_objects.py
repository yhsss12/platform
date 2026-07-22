"""artifact_storage_objects + data_assets.minio_path

Revision ID: 025_artifact_storage_objects
Revises: 024_resource_definition_catalog
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "025_artifact_storage_objects"
down_revision: Union[str, None] = "024_resource_definition_catalog"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    if "artifact_storage_objects" not in existing:
        op.create_table(
            "artifact_storage_objects",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column("owner_type", sa.String(length=32), nullable=False),
            sa.Column("owner_id", sa.String(length=128), nullable=False),
            sa.Column("artifact_type", sa.String(length=64), nullable=False),
            sa.Column("content_key", sa.String(length=512), nullable=False, server_default=""),
            sa.Column("storage_uri", sa.Text(), nullable=True),
            sa.Column("local_path", sa.Text(), nullable=True),
            sa.Column("sha256", sa.String(length=64), nullable=True),
            sa.Column("size_bytes", sa.BigInteger(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
            sa.Column("upload_attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "owner_type",
                "owner_id",
                "artifact_type",
                "content_key",
                name="uq_artifact_storage_owner_content",
            ),
        )
        op.create_index("idx_artifact_storage_owner", "artifact_storage_objects", ["owner_type", "owner_id"])
        op.create_index("idx_artifact_storage_status", "artifact_storage_objects", ["status", "updated_at"])
        op.create_index("ix_artifact_storage_owner_type", "artifact_storage_objects", ["owner_type"])
        op.create_index("ix_artifact_storage_owner_id", "artifact_storage_objects", ["owner_id"])
        op.create_index("ix_artifact_storage_artifact_type", "artifact_storage_objects", ["artifact_type"])
        op.create_index("ix_artifact_storage_status_col", "artifact_storage_objects", ["status"])

    cols = {c["name"] for c in insp.get_columns("data_assets")} if "data_assets" in existing else set()
    if "data_assets" in existing and "minio_path" not in cols:
        op.add_column("data_assets", sa.Column("minio_path", sa.String(length=1024), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())
    if "data_assets" in existing:
        cols = {c["name"] for c in insp.get_columns("data_assets")}
        if "minio_path" in cols:
            op.drop_column("data_assets", "minio_path")
    if "artifact_storage_objects" in existing:
        op.drop_table("artifact_storage_objects")
