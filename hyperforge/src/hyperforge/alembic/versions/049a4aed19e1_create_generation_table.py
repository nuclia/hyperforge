"""create generation table

Revision ID: 049a4aed19e1
Revises: b8edb72295e0
Create Date: 2025-05-13 22:31:23.774015

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "049a4aed19e1"
down_revision: Union[str, None] = "b8edb72295e0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "retrieval_agent_generation",
        sa.Column(
            "id",
            postgresql.UUID(),
            server_default=sa.func.uuid_generate_v4(),
            nullable=False,
        ),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("kbid", sa.String(), nullable=False),
        sa.Column(
            "generation", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("created", sa.DateTime(), default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["account", "kbid"],
            ["retrieval_agent_config.account", "retrieval_agent_config.kbid"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_retrieval_agent_generation_account"),
        "retrieval_agent_generation",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_generation_kbid"),
        "retrieval_agent_generation",
        ["kbid"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("retrieval_agent_generation")
