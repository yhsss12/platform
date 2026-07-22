"""rebuild audit_logs with BIGSERIAL id and migrate legacy rows

Revision ID: 006_audit_bigserial
Revises: 005_audit_logs_align
Create Date: 2026-03-24

问题：历史表 audit_logs.id 为 varchar(36) 且无 DEFAULT，ORM 插入不写 id 导致
NotNullViolation。本迁移将表重建为与 app.models.audit_log 一致的 BIGSERIAL 主键，
并尽量保留旧数据（旧 UUID id 写入 detail_json.legacy_id）。

Downgrade 无法无损恢复，故为 no-op。
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect, text
from sqlalchemy.dialects import postgresql

revision: str = "006_audit_bigserial"
down_revision: Union[str, None] = "005_audit_logs_align"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _id_column_ok(connection) -> bool:
    dt = connection.execute(
        text(
            """
            SELECT data_type FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'audit_logs' AND column_name = 'id'
            """
        )
    ).scalar()
    if dt is None:
        return True
    if dt not in ("bigint", "integer"):
        return False
    seq = connection.execute(
        text("SELECT pg_get_serial_sequence('public.audit_logs', 'id')")
    ).scalar()
    return bool(seq)


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name != "postgresql":
        return

    insp = inspect(conn)
    if not insp.has_table("audit_logs"):
        op.create_table(
            "audit_logs",
            sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
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
        op.create_index("ix_audit_logs_result", "audit_logs", ["result"], unique=False)
        return

    if _id_column_ok(conn):
        return

    legacy_cols = {c["name"] for c in insp.get_columns("audit_logs")}
    has_action = "action" in legacy_cols
    has_detail = "detail" in legacy_cols

    op.execute(text("ALTER TABLE audit_logs RENAME TO audit_logs_legacy"))

    # 索引名在 schema 内全局唯一；legacy 表仍占用 ix_audit_logs_*，需先释放名称
    for idx in (
        "ix_audit_logs_created_at",
        "ix_audit_logs_user_id",
        "ix_audit_logs_username",
        "ix_audit_logs_project_id",
        "ix_audit_logs_action_type",
        "ix_audit_logs_result",
        "ix_audit_logs_action",
    ):
        op.execute(text(f'DROP INDEX IF EXISTS public."{idx}"'))

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
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

    if has_action and has_detail:
        op.execute(
            text(
                """
                INSERT INTO audit_logs (
                    created_at, user_id, username, role, project_id, project_name,
                    action_type, action_label, resource_type, resource_id, resource_name,
                    result, ip, user_agent, detail_json, error_message
                )
                SELECT
                    (COALESCE(l.created_at, NOW()) AT TIME ZONE 'UTC'),
                    l.user_id,
                    l.username,
                    l.role,
                    l.project_id,
                    l.project_name,
                    COALESCE(
                        NULLIF(BTRIM(l.action_type::text), ''),
                        NULLIF(BTRIM(l.action::text), ''),
                        'UNKNOWN'
                    ),
                    COALESCE(
                        NULLIF(BTRIM(l.action_label::text), ''),
                        NULLIF(BTRIM(l.detail::text), ''),
                        '-'
                    ),
                    l.resource_type,
                    l.resource_id,
                    l.resource_name,
                    COALESCE(NULLIF(BTRIM(l.result::text), ''), 'SUCCESS'),
                    CASE
                        WHEN l.ip IS NULL THEN NULL
                        ELSE LEFT(l.ip::text, 64)
                    END,
                    l.user_agent::text,
                    CASE
                        WHEN l.detail_json IS NOT NULL AND l.detail_json::text NOT IN ('null', '')
                            THEN COALESCE(l.detail_json, '{}'::jsonb) || jsonb_build_object('legacy_id', l.id::text)
                        WHEN l.detail IS NOT NULL AND BTRIM(l.detail::text) <> ''
                            THEN jsonb_build_object(
                                'legacy_id', l.id::text,
                                'legacy_detail', l.detail::text
                            )
                        ELSE jsonb_build_object('legacy_id', l.id::text)
                    END,
                    l.error_message
                FROM audit_logs_legacy l
                ORDER BY l.created_at NULLS LAST, l.id::text
                """
            )
        )
    else:
        op.execute(
            text(
                """
                INSERT INTO audit_logs (
                    created_at, user_id, username, role, project_id, project_name,
                    action_type, action_label, resource_type, resource_id, resource_name,
                    result, ip, user_agent, detail_json, error_message
                )
                SELECT
                    (COALESCE(l.created_at, NOW()) AT TIME ZONE 'UTC'),
                    l.user_id,
                    l.username,
                    l.role,
                    l.project_id,
                    l.project_name,
                    COALESCE(NULLIF(BTRIM(l.action_type::text), ''), 'UNKNOWN'),
                    COALESCE(NULLIF(BTRIM(l.action_label::text), ''), '-'),
                    l.resource_type,
                    l.resource_id,
                    l.resource_name,
                    COALESCE(NULLIF(BTRIM(l.result::text), ''), 'SUCCESS'),
                    CASE WHEN l.ip IS NULL THEN NULL ELSE LEFT(l.ip::text, 64) END,
                    l.user_agent::text,
                    CASE
                        WHEN l.detail_json IS NOT NULL AND l.detail_json::text NOT IN ('null', '')
                            THEN COALESCE(l.detail_json, '{}'::jsonb) || jsonb_build_object('legacy_id', l.id::text)
                        ELSE jsonb_build_object('legacy_id', l.id::text)
                    END,
                    l.error_message
                FROM audit_logs_legacy l
                ORDER BY l.created_at NULLS LAST, l.id::text
                """
            )
        )

    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"], unique=False)
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"], unique=False)
    op.create_index("ix_audit_logs_username", "audit_logs", ["username"], unique=False)
    op.create_index("ix_audit_logs_project_id", "audit_logs", ["project_id"], unique=False)
    op.create_index("ix_audit_logs_action_type", "audit_logs", ["action_type"], unique=False)
    op.create_index("ix_audit_logs_result", "audit_logs", ["result"], unique=False)

    op.execute(text("DROP TABLE audit_logs_legacy CASCADE"))


def downgrade() -> None:
    """无法将 BIGSERIAL 数据无损还原为旧版 varchar 主键表结构。"""
    pass
