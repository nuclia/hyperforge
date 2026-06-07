"""memory

Revision ID: 4d89a36c8bda
Revises: 2847934e2d59
Create Date: 2025-11-24 22:39:41.424068

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "4d89a36c8bda"
down_revision: Union[str, None] = "2847934e2d59"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "retrieval_agent_config",
        sa.Column(
            "memory",
            postgresql.JSONB(astext_type=sa.Text()),
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("retrieval_agent_config", "memory")
