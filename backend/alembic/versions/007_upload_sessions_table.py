"""create upload_sessions for MinIO direct single-file upload (Phase 1)

Revision ID: 007_upload_sessions
Revises: 006_audit_bigserial
Create Date: 2026-03-25

与 app.models.data_asset.DataAssetUploadSession 一致。
若表已存在（历史 create_all），upgrade 跳过建表。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "007_upload_sessions"
down_revision: Union[str, None] = "006_audit_bigserial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "upload_sessions" in insp.get_table_names():
        return

    op.create_table(
        "upload_sessions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.String(length=128), nullable=False),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'presigned'"),
        ),
        sa.Column("bucket", sa.String(length=256), nullable=False),
        sa.Column("object_key", sa.String(length=1024), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column(
            "size_bytes",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("content_type", sa.String(length=256), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_upload_sessions_user_id", "upload_sessions", ["user_id"], unique=False)
    op.create_index("ix_upload_sessions_project_id", "upload_sessions", ["project_id"], unique=False)
    op.create_index(
        "idx_upload_sessions_user_project",
        "upload_sessions",
        ["user_id", "project_id"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "upload_sessions" not in insp.get_table_names():
        return
    op.drop_index("idx_upload_sessions_user_project", table_name="upload_sessions")
    op.drop_index("ix_upload_sessions_project_id", table_name="upload_sessions")
    op.drop_index("ix_upload_sessions_user_id", table_name="upload_sessions")
    op.drop_table("upload_sessions")
