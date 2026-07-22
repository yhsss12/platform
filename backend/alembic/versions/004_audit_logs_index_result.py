"""add index on audit_logs.result

Revision ID: 004_audit_result_idx
Revises: 003_unified_audit
Create Date: 2026-03-23

"""
from typing import Sequence, Union

from alembic import op

revision: str = "004_audit_result_idx"
down_revision: Union[str, None] = "003_unified_audit"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index("ix_audit_logs_result", "audit_logs", ["result"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_result", table_name="audit_logs")
