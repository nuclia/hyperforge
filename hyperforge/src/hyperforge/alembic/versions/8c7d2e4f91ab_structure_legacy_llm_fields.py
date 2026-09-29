"""Structure legacy string model fields in stored agent configurations.

Revision ID: 8c7d2e4f91ab
Revises: 6a23d8ef74c1
"""

from typing import Sequence, Union

from alembic import op

from hyperforge.db.legacy_llm_migration import migrate_legacy_llm_fields

revision: str = "8c7d2e4f91ab"
down_revision: Union[str, None] = "6a23d8ef74c1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    migrate_legacy_llm_fields(op.get_bind())


def downgrade() -> None:
    pass
