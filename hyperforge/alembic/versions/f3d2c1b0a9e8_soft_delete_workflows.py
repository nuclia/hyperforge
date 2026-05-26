"""add soft delete columns to workflows

Revision ID: f3d2c1b0a9e8
Revises: 9c6f6a1b4e7f
Create Date: 2026-05-11 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f3d2c1b0a9e8"
down_revision: Union[str, None] = "9c6f6a1b4e7f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


WORKFLOW_TABLE = "retrieval_agent_workflow"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        WORKFLOW_TABLE,
        sa.Column(
            "is_deleted",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        WORKFLOW_TABLE,
        sa.Column("deleted_by", sa.String(), nullable=True),
    )
    op.add_column(
        WORKFLOW_TABLE,
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_retrieval_agent_workflow_is_deleted"),
        WORKFLOW_TABLE,
        ["is_deleted"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_workflow_deleted_at"),
        WORKFLOW_TABLE,
        ["deleted_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_retrieval_agent_workflow_deleted_at"), table_name=WORKFLOW_TABLE
    )
    op.drop_index(
        op.f("ix_retrieval_agent_workflow_is_deleted"), table_name=WORKFLOW_TABLE
    )
    op.drop_column(WORKFLOW_TABLE, "deleted_at")
    op.drop_column(WORKFLOW_TABLE, "deleted_by")
    op.drop_column(WORKFLOW_TABLE, "is_deleted")
