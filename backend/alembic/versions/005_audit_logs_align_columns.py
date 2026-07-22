"""align audit_logs columns with ORM (add missing role, project fields, etc.)

Revision ID: 005_audit_logs_align
Revises: 004_audit_result_idx, 53b08ee59659
Create Date: 2026-03-23

合并 Alembic 双 head，并为已存在但结构偏旧的 audit_logs 表补齐与
app.models.audit_log.AuditLog 一致的列（例如缺 role 时的修复）。

Downgrade 为 no-op：无法安全判断列由本迁移还是 003 全量建表引入。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect
from sqlalchemy.dialects import postgresql

revision: str = "005_audit_logs_align"
down_revision: Union[str, tuple[str, ...], None] = ("004_audit_result_idx", "53b08ee59659")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    insp = inspect(conn)
    if not insp.has_table("audit_logs"):
        return

    existing = {c["name"] for c in insp.get_columns("audit_logs")}

    def add_if_missing(name: str, column: sa.Column) -> None:
        nonlocal existing
        if name not in existing:
            op.add_column("audit_logs", column)
            existing.add(name)

    add_if_missing("role", sa.Column("role", sa.String(50), nullable=True))
    add_if_missing("project_name", sa.Column("project_name", sa.String(200), nullable=True))
    add_if_missing("project_id", sa.Column("project_id", sa.String(64), nullable=True))
    add_if_missing("resource_type", sa.Column("resource_type", sa.String(100), nullable=True))
    add_if_missing("resource_id", sa.Column("resource_id", sa.String(100), nullable=True))
    add_if_missing("resource_name", sa.Column("resource_name", sa.String(255), nullable=True))
    add_if_missing("error_message", sa.Column("error_message", sa.Text(), nullable=True))

    if "detail_json" not in existing:
        op.add_column(
            "audit_logs",
            sa.Column("detail_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        )
        existing.add("detail_json")

    add_if_missing("user_agent", sa.Column("user_agent", sa.Text(), nullable=True))
    add_if_missing("ip", sa.Column("ip", sa.String(64), nullable=True))

    if "result" not in existing:
        op.add_column(
            "audit_logs",
            sa.Column("result", sa.String(20), nullable=False, server_default="SUCCESS"),
        )
        existing.add("result")

    if "action_type" not in existing:
        op.add_column(
            "audit_logs",
            sa.Column(
                "action_type",
                sa.String(100),
                nullable=False,
                server_default=sa.text("'UNKNOWN'"),
            ),
        )
        if "action" in existing:
            op.execute(sa.text("UPDATE audit_logs SET action_type = action"))
        existing.add("action_type")

    if "action_label" not in existing:
        op.add_column(
            "audit_logs",
            sa.Column(
                "action_label",
                sa.String(200),
                nullable=False,
                server_default=sa.text("'-'"),
            ),
        )
        if "detail" in existing and "action" in existing:
            op.execute(
                sa.text(
                    "UPDATE audit_logs SET action_label = LEFT("
                    "COALESCE(NULLIF(TRIM(COALESCE(detail, '')), ''), "
                    "NULLIF(TRIM(COALESCE(action, '')), ''), '-'), 200)"
                )
            )
        elif "action" in existing:
            op.execute(
                sa.text(
                    "UPDATE audit_logs SET action_label = LEFT("
                    "COALESCE(NULLIF(TRIM(COALESCE(action, '')), ''), '-'), 200)"
                )
            )
        elif "detail" in existing:
            op.execute(
                sa.text(
                    "UPDATE audit_logs SET action_label = LEFT("
                    "COALESCE(NULLIF(TRIM(COALESCE(detail, '')), ''), '-'), 200)"
                )
            )
        existing.add("action_label")

    # 刷新列集合与索引
    insp = inspect(conn)
    colset = {c["name"] for c in insp.get_columns("audit_logs")}
    idx_names = {ix["name"] for ix in insp.get_indexes("audit_logs")}

    def ensure_index(name: str, cols: list[str]) -> None:
        if name in idx_names:
            return
        if not all(c in colset for c in cols):
            return
        op.create_index(name, "audit_logs", cols, unique=False)

    ensure_index("ix_audit_logs_project_id", ["project_id"])
    ensure_index("ix_audit_logs_action_type", ["action_type"])
    ensure_index("ix_audit_logs_result", ["result"])


def downgrade() -> None:
    """本迁移为补齐结构，downgrade 不删除列，避免误删由 003 创建的列。"""
    pass
