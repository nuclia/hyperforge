"""Add encrypted Sync OAuth credentials.

Revision ID: 6a23d8ef74c1
Revises: f3d2c1b0a9e8
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "6a23d8ef74c1"
down_revision: Union[str, None] = "f3d2c1b0a9e8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "sync_oauth_credentials",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("user_id", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("provider", sa.String(), nullable=False),
        sa.Column("sync_config_id", sa.String(), nullable=False),
        sa.Column("encrypted_credentials", sa.Text(), nullable=False),
        sa.Column(
            "created",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "modified",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["account", "agent_id"],
            ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "account", "user_id", "agent_id", "provider", "sync_config_id"
        ),
    )


def downgrade() -> None:
    op.drop_table("sync_oauth_credentials")
