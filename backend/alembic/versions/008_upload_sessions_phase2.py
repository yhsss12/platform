"""upload_sessions Phase 2: multi_file / directory direct upload metadata

Revision ID: 008_upload_sessions_p2
Revises: 007_upload_sessions
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision: str = "008_upload_sessions_p2"
down_revision: Union[str, None] = "007_upload_sessions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {c["name"] for c in insp.get_columns("upload_sessions")} if "upload_sessions" in insp.get_table_names() else set()
    if not cols:
        return

    def add(name: str, col):
        if name not in cols:
            op.add_column("upload_sessions", col)

    add(
        "upload_mode",
        sa.Column("upload_mode", sa.String(length=32), server_default=sa.text("'single_file'"), nullable=False),
    )
    add("items_json", sa.Column("items_json", sa.Text(), nullable=True))
    add("manifest_json", sa.Column("manifest_json", sa.Text(), nullable=True))
    add("expected_count", sa.Column("expected_count", sa.Integer(), nullable=True))
    add("expected_total_size", sa.Column("expected_total_size", sa.BigInteger(), nullable=True))
    add("root_dir_name", sa.Column("root_dir_name", sa.String(length=512), nullable=True))
    add("asset_name", sa.Column("asset_name", sa.String(length=512), nullable=True))
    add("result_payload_json", sa.Column("result_payload_json", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    if "upload_sessions" not in insp.get_table_names():
        return
    cols = {c["name"] for c in insp.get_columns("upload_sessions")}
    for name in (
        "result_payload_json",
        "asset_name",
        "root_dir_name",
        "expected_total_size",
        "expected_count",
        "manifest_json",
        "items_json",
        "upload_mode",
    ):
        if name in cols:
            op.drop_column("upload_sessions", name)
