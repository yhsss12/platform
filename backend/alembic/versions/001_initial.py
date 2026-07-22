"""initial migration

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users table
    op.create_table(
        'users',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('username', sa.String(), nullable=False, unique=True),
        sa.Column('password_hash', sa.String(), nullable=False),
        sa.Column('role', sa.Enum('ADMIN', 'OPERATOR', 'QC', name='userrole'), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
    )
    op.create_index('ix_users_username', 'users', ['username'])

    # Tasks table
    op.create_table(
        'tasks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.String(), nullable=True),
        sa.Column('status', sa.Enum('DRAFT', 'READY', 'RUNNING', 'COMPLETED', 'ARCHIVED', name='taskstatus'), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    # Jobs table
    op.create_table(
        'jobs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id'), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELED', name='jobstatus'), nullable=False),
        sa.Column('operator_name', sa.String(), nullable=True),
        sa.Column('mcap_path', sa.String(), nullable=True),
        sa.Column('mcap_size_bytes', sa.Integer(), nullable=True),
        sa.Column('duration_sec', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.String(), nullable=True),
        sa.Column('finished_at', sa.String(), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    # Runs table
    op.create_table(
        'runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('task_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('tasks.id'), nullable=False),
        sa.Column('status', sa.Enum('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED', 'CANCELED', name='runstatus'), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )

    # Datasets table
    op.create_table(
        'datasets',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('status', sa.Enum('ACTIVE', 'ARCHIVED', name='datasetstatus'), nullable=False),
        sa.Column('created_at', sa.String(), nullable=False),
        sa.Column('updated_at', sa.String(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('datasets')
    op.drop_table('runs')
    op.drop_table('jobs')
    op.drop_table('tasks')
    op.drop_table('users')
    op.execute('DROP TYPE IF EXISTS datasetstatus')
    op.execute('DROP TYPE IF EXISTS runstatus')
    op.execute('DROP TYPE IF EXISTS jobstatus')
    op.execute('DROP TYPE IF EXISTS taskstatus')
    op.execute('DROP TYPE IF EXISTS userrole')
