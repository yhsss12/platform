"""workspace index tables: model_assets, training_metric_summary, eval_metric_summary

Revision ID: 021_workspace_index_tables
Revises: 020_workspace_jobs_artifacts
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "021_workspace_index_tables"
down_revision: Union[str, None] = "020_workspace_jobs_artifacts"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    from sqlalchemy import inspect

    insp = inspect(bind)
    existing = set(insp.get_table_names())

    if "model_assets" not in existing:
        op.create_table(
            "model_assets",
            sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
            sa.Column("model_asset_id", sa.String(length=128), nullable=False),
            sa.Column("train_job_id", sa.String(length=128), nullable=False),
            sa.Column("dataset_id", sa.String(length=128), nullable=True),
            sa.Column("model_name", sa.String(length=512), nullable=False, server_default=""),
            sa.Column("model_type", sa.String(length=64), nullable=True),
            sa.Column("asset_type", sa.String(length=32), nullable=False, server_default="epoch"),
            sa.Column("epoch", sa.Integer(), nullable=True),
            sa.Column("storage_uri", sa.Text(), nullable=True),
            sa.Column("manifest_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("metrics_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("sha256", sa.String(length=64), nullable=True),
            sa.Column("size_bytes", sa.BigInteger(), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="generating"),
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
            sa.ForeignKeyConstraint(["train_job_id"], ["workspace_jobs.job_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("model_asset_id"),
        )
        op.create_index("ix_model_assets_model_asset_id", "model_assets", ["model_asset_id"])
        op.create_index("ix_model_assets_train_job_id", "model_assets", ["train_job_id"])
        op.create_index("ix_model_assets_dataset_id", "model_assets", ["dataset_id"])
        op.create_index("ix_model_assets_asset_type", "model_assets", ["asset_type"])
        op.create_index("ix_model_assets_status", "model_assets", ["status"])
        op.create_index(
            "idx_model_assets_train_job_type",
            "model_assets",
            ["train_job_id", "asset_type"],
        )
        op.create_index(
            "idx_model_assets_status_created",
            "model_assets",
            ["status", "created_at"],
        )

    if "training_metric_summary" not in existing:
        op.create_table(
            "training_metric_summary",
            sa.Column("job_id", sa.String(length=128), nullable=False),
            sa.Column("current_epoch", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("total_epochs", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
            sa.Column("current_loss", sa.Float(), nullable=True),
            sa.Column("final_loss", sa.Float(), nullable=True),
            sa.Column("best_loss", sa.Float(), nullable=True),
            sa.Column("loss_series", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["job_id"], ["workspace_jobs.job_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("job_id"),
        )

    if "eval_metric_summary" not in existing:
        op.create_table(
            "eval_metric_summary",
            sa.Column("job_id", sa.String(length=128), nullable=False),
            sa.Column("model_asset_id", sa.String(length=128), nullable=True),
            sa.Column("summary_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
            sa.Column("report_uri", sa.Text(), nullable=True),
            sa.Column("replay_uri", sa.Text(), nullable=True),
            sa.Column(
                "updated_at",
                sa.DateTime(timezone=True),
                server_default=sa.text("now()"),
                nullable=False,
            ),
            sa.ForeignKeyConstraint(["job_id"], ["workspace_jobs.job_id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("job_id"),
        )
        op.create_index("ix_eval_metric_summary_model_asset_id", "eval_metric_summary", ["model_asset_id"])


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return

    op.drop_index("ix_eval_metric_summary_model_asset_id", table_name="eval_metric_summary")
    op.drop_table("eval_metric_summary")
    op.drop_table("training_metric_summary")
    op.drop_index("idx_model_assets_status_created", table_name="model_assets")
    op.drop_index("idx_model_assets_train_job_type", table_name="model_assets")
    op.drop_index("ix_model_assets_status", table_name="model_assets")
    op.drop_index("ix_model_assets_asset_type", table_name="model_assets")
    op.drop_index("ix_model_assets_dataset_id", table_name="model_assets")
    op.drop_index("ix_model_assets_train_job_id", table_name="model_assets")
    op.drop_index("ix_model_assets_model_asset_id", table_name="model_assets")
    op.drop_table("model_assets")
