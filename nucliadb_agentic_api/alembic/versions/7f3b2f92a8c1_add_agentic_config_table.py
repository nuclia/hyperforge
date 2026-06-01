"""Add agentic config table

Revision ID: 7f3b2f92a8c1
Revises: None
Create Date: 2026-05-30 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "7f3b2f92a8c1"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agentic_config_table",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("kbid", sa.String(), nullable=False),
        sa.Column("agentic_id", sa.String(), nullable=False),
        sa.Column("created", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), nullable=True),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.PrimaryKeyConstraint("account", "kbid", "agentic_id"),
    )
    op.create_index(
        op.f("ix_agentic_config_table_account"),
        "agentic_config_table",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agentic_config_table_kbid"),
        "agentic_config_table",
        ["kbid"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_agentic_config_table_kbid"), table_name="agentic_config_table"
    )
    op.drop_index(
        op.f("ix_agentic_config_table_account"), table_name="agentic_config_table"
    )
    op.drop_table("agentic_config_table")
