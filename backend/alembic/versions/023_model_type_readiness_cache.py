"""model_type_definitions readiness cache columns

Revision ID: 023_model_type_readiness_cache
Revises: 022_model_type_definitions
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "023_model_type_readiness_cache"
down_revision: Union[str, None] = "022_model_type_definitions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    if "model_type_definitions" not in set(insp.get_table_names()):
        return

    columns = {col["name"] for col in insp.get_columns("model_type_definitions")}

    if "training_ready" not in columns:
        op.add_column(
            "model_type_definitions",
            sa.Column("training_ready", sa.Boolean(), nullable=True),
        )
    if "training_readiness_status" not in columns:
        op.add_column(
            "model_type_definitions",
            sa.Column("training_readiness_status", sa.String(length=32), nullable=True),
        )
    if "disabled_reason" not in columns:
        op.add_column(
            "model_type_definitions",
            sa.Column("disabled_reason", sa.Text(), nullable=True),
        )
    if "capability_checked_at" not in columns:
        op.add_column(
            "model_type_definitions",
            sa.Column("capability_checked_at", sa.DateTime(timezone=True), nullable=True),
        )
    if "capability_evidence" not in columns:
        op.add_column(
            "model_type_definitions",
            sa.Column(
                "capability_evidence",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            ),
        )

    op.execute(
        sa.text(
            """
            UPDATE model_type_definitions
            SET
                training_ready = CASE WHEN base_algorithm = 'pi0' THEN false ELSE true END,
                training_readiness_status = CASE
                    WHEN base_algorithm = 'pi0' THEN 'pending'
                    ELSE 'ready'
                END,
                disabled_reason = CASE
                    WHEN base_algorithm = 'pi0' THEN '正在检测 runner'
                    ELSE NULL
                END
            WHERE training_readiness_status IS NULL
            """
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    if "model_type_definitions" not in set(insp.get_table_names()):
        return

    columns = {col["name"] for col in insp.get_columns("model_type_definitions")}
    for name in (
        "capability_evidence",
        "capability_checked_at",
        "disabled_reason",
        "training_readiness_status",
        "training_ready",
    ):
        if name in columns:
            op.drop_column("model_type_definitions", name)
