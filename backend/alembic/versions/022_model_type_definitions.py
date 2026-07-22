"""model_type_definitions table

Revision ID: 022_model_type_definitions
Revises: 021_workspace_index_tables
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "022_model_type_definitions"
down_revision: Union[str, None] = "021_workspace_index_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    if "model_type_definitions" not in existing:
        op.create_table(
            "model_type_definitions",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("model_type_id", sa.String(length=128), nullable=False),
            sa.Column("name", sa.String(length=256), nullable=False),
            sa.Column("base_algorithm", sa.String(length=64), nullable=False),
            sa.Column("adapter_key", sa.String(length=64), nullable=False),
            sa.Column("simulator", sa.String(length=64), nullable=True),
            sa.Column("robot_type", sa.String(length=64), nullable=True),
            sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column(
                "structure_config",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column(
                "training_defaults",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=False,
                server_default=sa.text("'{}'::jsonb"),
            ),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="available"),
            sa.Column("is_builtin", sa.Boolean(), nullable=False, server_default=sa.text("false")),
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
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("model_type_id"),
        )
        op.create_index(
            "ix_model_type_definitions_model_type_id",
            "model_type_definitions",
            ["model_type_id"],
        )
        op.create_index(
            "ix_model_type_definitions_base_algorithm",
            "model_type_definitions",
            ["base_algorithm"],
        )
        op.create_index(
            "ix_model_type_definitions_status",
            "model_type_definitions",
            ["status"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    if "model_type_definitions" in set(insp.get_table_names()):
        op.drop_table("model_type_definitions")
