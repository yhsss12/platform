"""resource_definitions and task_template_catalog tables

Revision ID: 024_resource_definition_catalog
Revises: 023_model_type_readiness_cache
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "024_resource_definition_catalog"
down_revision: Union[str, None] = "023_model_type_readiness_cache"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    if "resource_definitions" not in existing:
        op.create_table(
            "resource_definitions",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("resource_id", sa.String(length=128), nullable=False),
            sa.Column("resource_type", sa.String(length=32), nullable=False),
            sa.Column("name", sa.String(length=512), nullable=False, server_default=""),
            sa.Column("display_name", sa.String(length=512), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("version", sa.String(length=64), nullable=False, server_default="v1"),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="available"),
            sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column(
                "manifest_json",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "metadata_json",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("manifest_path", sa.Text(), nullable=True),
            sa.Column("storage_uri", sa.Text(), nullable=True),
            sa.Column("source", sa.String(length=32), nullable=False, server_default="registry"),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "resource_type",
                "resource_id",
                "version",
                name="uq_resource_definitions_type_id_version",
            ),
        )
        op.create_index(
            "ix_resource_definitions_resource_type",
            "resource_definitions",
            ["resource_type"],
        )
        op.create_index(
            "ix_resource_definitions_resource_id",
            "resource_definitions",
            ["resource_id"],
        )
        op.create_index(
            "ix_resource_definitions_status",
            "resource_definitions",
            ["status"],
        )

    if "task_template_catalog" not in existing:
        op.create_table(
            "task_template_catalog",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("template_id", sa.String(length=128), nullable=False),
            sa.Column("name", sa.String(length=512), nullable=False),
            sa.Column("display_name", sa.String(length=512), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("category", sa.String(length=128), nullable=True),
            sa.Column("simulator", sa.String(length=64), nullable=True),
            sa.Column("robot_type", sa.String(length=128), nullable=True),
            sa.Column("task_config_id", sa.String(length=128), nullable=True),
            sa.Column(
                "metadata_json",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="available"),
            sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
            sa.Column(
                "created_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                nullable=False,
                server_default=sa.text("now()"),
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("template_id", name="uq_task_template_catalog_template_id"),
        )
        op.create_index(
            "ix_task_template_catalog_status",
            "task_template_catalog",
            ["status"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    op.drop_index("ix_task_template_catalog_status", table_name="task_template_catalog")
    op.drop_table("task_template_catalog")
    op.drop_index("ix_resource_definitions_status", table_name="resource_definitions")
    op.drop_index("ix_resource_definitions_resource_id", table_name="resource_definitions")
    op.drop_index("ix_resource_definitions_resource_type", table_name="resource_definitions")
    op.drop_table("resource_definitions")
