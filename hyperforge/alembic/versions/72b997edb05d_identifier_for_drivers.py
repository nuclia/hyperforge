"""identifier for drivers

Revision ID: 72b997edb05d
Revises: 049a4aed19e1
Create Date: 2025-06-02 13:50:46.393105

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "72b997edb05d"
down_revision: Union[str, None] = "049a4aed19e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("retrieval_agents_drivers", sa.Column("identifier", sa.String))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("retrieval_agents_drivers", "identifier")
