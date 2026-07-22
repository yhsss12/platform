"""add source column to hdf5_datasets

Revision ID: 002
Revises: 001
Create Date: 2025-03-04

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("hdf5_datasets", sa.Column("source", sa.String(), nullable=True))
    op.execute("UPDATE hdf5_datasets SET source = 'local' WHERE source IS NULL")


def downgrade() -> None:
    pass
