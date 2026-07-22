"""label_tasks.dataset_path 扩容为 TEXT，支持大批量数据集路径

Revision ID: 019_label_tasks_dataset_path_text
Revises: 018_conversion_batch_jobs_canceled_count
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
from sqlalchemy import inspect, text

revision: str = "019_label_tasks_dataset_path_text"
down_revision: Union[str, None] = "018_conversion_batch_jobs_canceled_count"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    insp = inspect(bind)
    if not insp.has_table("label_tasks"):
        return
    cols = {c["name"] for c in insp.get_columns("label_tasks")}
    if "dataset_path" not in cols:
        return
    op.execute(
        text(
            "ALTER TABLE label_tasks "
            "ALTER COLUMN dataset_path TYPE TEXT"
        )
    )


def downgrade() -> None:
    bind = op.get_bind()
    if getattr(bind.dialect, "name", None) != "postgresql":
        return
    insp = inspect(bind)
    if not insp.has_table("label_tasks"):
        return
    cols = {c["name"] for c in insp.get_columns("label_tasks")}
    if "dataset_path" not in cols:
        return
    op.execute(
        text(
            "ALTER TABLE label_tasks "
            "ALTER COLUMN dataset_path TYPE VARCHAR(1024)"
        )
    )
