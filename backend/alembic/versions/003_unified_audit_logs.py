"""unified audit_logs schema

Revision ID: 003_unified_audit
Revises: 7d281eb59df1
Create Date: 2026-03-23

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "003_unified_audit"
down_revision: Union[str, None] = "7d281eb59df1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("DROP TABLE IF EXISTS audit_logs CASCADE"))

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("username", sa.String(length=100), nullable=True),
        sa.Column("role", sa.String(length=50), nullable=True),
        sa.Column("project_id", sa.String(length=64), nullable=True),
        sa.Column("project_name", sa.String(length=200), nullable=True),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("action_label", sa.String(length=200), nullable=False),
        sa.Column("resource_type", sa.String(length=100), nullable=True),
        sa.Column("resource_id", sa.String(length=100), nullable=True),
        sa.Column("resource_name", sa.String(length=255), nullable=True),
        sa.Column("result", sa.String(length=20), server_default="SUCCESS", nullable=False),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("detail_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"], unique=False)
    op.create_index("ix_audit_logs_username", "audit_logs", ["username"], unique=False)
    op.create_index("ix_audit_logs_project_id", "audit_logs", ["project_id"], unique=False)
    op.create_index("ix_audit_logs_action_type", "audit_logs", ["action_type"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_action_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_project_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_username", table_name="audit_logs")
    op.drop_index("ix_audit_logs_user_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_table("audit_logs")
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=True),
        sa.Column("username", sa.String(length=50), nullable=True),
        sa.Column("action", sa.String(length=50), nullable=False),
        sa.Column("detail", sa.String(length=500), nullable=True),
        sa.Column("ip", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_audit_logs_username"), "audit_logs", ["username"], unique=False)
    op.create_index(op.f("ix_audit_logs_user_id"), "audit_logs", ["user_id"], unique=False)
    op.create_index(op.f("ix_audit_logs_created_at"), "audit_logs", ["created_at"], unique=False)
    op.create_index(op.f("ix_audit_logs_action"), "audit_logs", ["action"], unique=False)
