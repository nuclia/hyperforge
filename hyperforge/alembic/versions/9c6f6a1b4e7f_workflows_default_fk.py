"""Add workflows table and backfill default workflow references

Revision ID: 9c6f6a1b4e7f
Revises: 1416cb41bd49
Create Date: 2026-01-27 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "9c6f6a1b4e7f"
down_revision: Union[str, None] = "1416cb41bd49"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


WORKFLOW_TABLE = "retrieval_agent_workflow"
PROMPTS_TABLE = "retrieval_agent_prompts"
DEFAULT_WORKFLOW_ID = "default"


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column("retrieval_agent_config", "kbid", new_column_name="agent_id")
    op.alter_column("retrieval_agent_preprocess", "kbid", new_column_name="agent_id")
    op.alter_column("retrieval_agent_postprocess", "kbid", new_column_name="agent_id")
    op.alter_column("retrieval_agent_context", "kbid", new_column_name="agent_id")
    op.alter_column("retrieval_agent_generation", "kbid", new_column_name="agent_id")
    op.alter_column("retrieval_agents_drivers", "kbid", new_column_name="agent_id")

    op.add_column(
        "retrieval_agent_config",
        sa.Column("description", sa.String(), nullable=True),
    )

    op.add_column(
        "retrieval_agent_config",
        sa.Column("title", sa.String(), nullable=True),
    )

    op.add_column(
        "retrieval_agent_config",
        sa.Column("instructions", sa.String(), nullable=True),
    )

    op.create_table(
        PROMPTS_TABLE,
        sa.Column(
            "id",
            postgresql.UUID,
            primary_key=True,
            server_default=sa.func.uuid_generate_v4(),
        ),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("prompt", sa.String(), nullable=False),
        sa.Column(
            "arguments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "icons",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "meta",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created", sa.DateTime(), default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), onupdate=sa.func.now()),
    )
    op.create_index(
        op.f("ix_retrieval_agent_prompts_account"),
        PROMPTS_TABLE,
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_prompts_agent_id"),
        PROMPTS_TABLE,
        ["agent_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_retrieval_agent_workflow_prompts",
        PROMPTS_TABLE,
        "retrieval_agent_config",
        ["account", "agent_id"],
        ["account", "agent_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )

    op.create_table(
        WORKFLOW_TABLE,
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column(
            "parameters",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "required",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "rules",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created", sa.DateTime(), default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint("account", "agent_id", "workflow_id"),
    )
    op.create_index(
        op.f("ix_retrieval_agent_workflow_account"),
        WORKFLOW_TABLE,
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_workflow_agent_id"),
        WORKFLOW_TABLE,
        ["agent_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_workflow_workflow_id"),
        WORKFLOW_TABLE,
        ["workflow_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_retrieval_agent_workflow_config",
        WORKFLOW_TABLE,
        "retrieval_agent_config",
        ["account", "agent_id"],
        ["account", "agent_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )

    # Add workflow_id columns as nullable first so we can backfill safely.
    op.add_column(
        "retrieval_agent_preprocess",
        sa.Column("workflow_id", sa.String(), nullable=True),
    )
    op.add_column(
        "retrieval_agent_postprocess",
        sa.Column("workflow_id", sa.String(), nullable=True),
    )
    op.add_column(
        "retrieval_agent_generation",
        sa.Column("workflow_id", sa.String(), nullable=True),
    )
    op.add_column(
        "retrieval_agent_context",
        sa.Column("workflow_id", sa.String(), nullable=True),
    )

    # Ensure a default workflow row exists for every existing config.
    op.execute(
        sa.text(
            """
            INSERT INTO retrieval_agent_workflow (
                account,
                agent_id,
                workflow_id,
                name,
                description,
                parameters,
                rules,
                created,
                modified
            )
            SELECT
                rac.account,
                rac.agent_id,
                :default_workflow_id,
                :default_workflow_id,
                'Default workflow',
                '{}'::jsonb,
                rac.rules,
                NOW(),
                NOW()
            FROM retrieval_agent_config AS rac
            ON CONFLICT (account, agent_id, workflow_id) DO NOTHING
            """
        ).bindparams(default_workflow_id=DEFAULT_WORKFLOW_ID)
    )

    # Backfill workflow_id on existing agent rows before enforcing FKs.
    for table_name in (
        "retrieval_agent_preprocess",
        "retrieval_agent_postprocess",
        "retrieval_agent_generation",
        "retrieval_agent_context",
    ):
        op.execute(
            sa.text(
                f"""
                UPDATE {table_name}
                SET workflow_id = :default_workflow_id
                WHERE workflow_id IS NULL
                """
            ).bindparams(default_workflow_id=DEFAULT_WORKFLOW_ID)
        )

    # Enforce non-nullable workflow_id and add indexes.
    op.alter_column(
        "retrieval_agent_preprocess",
        "workflow_id",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "retrieval_agent_postprocess",
        "workflow_id",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "retrieval_agent_generation",
        "workflow_id",
        existing_type=sa.String(),
        nullable=False,
    )
    op.alter_column(
        "retrieval_agent_context",
        "workflow_id",
        existing_type=sa.String(),
        nullable=False,
    )

    op.create_index(
        op.f("ix_retrieval_agent_preprocess_workflow_id"),
        "retrieval_agent_preprocess",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_postprocess_workflow_id"),
        "retrieval_agent_postprocess",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_generation_workflow_id"),
        "retrieval_agent_generation",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_retrieval_agent_context_workflow_id"),
        "retrieval_agent_context",
        ["workflow_id"],
        unique=False,
    )

    # Finally, add the composite foreign keys.
    op.create_foreign_key(
        "fk_preprocess_workflow",
        "retrieval_agent_preprocess",
        WORKFLOW_TABLE,
        ["account", "agent_id", "workflow_id"],
        ["account", "agent_id", "workflow_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_postprocess_workflow",
        "retrieval_agent_postprocess",
        WORKFLOW_TABLE,
        ["account", "agent_id", "workflow_id"],
        ["account", "agent_id", "workflow_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_generation_workflow",
        "retrieval_agent_generation",
        WORKFLOW_TABLE,
        ["account", "agent_id", "workflow_id"],
        ["account", "agent_id", "workflow_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_context_workflow",
        "retrieval_agent_context",
        WORKFLOW_TABLE,
        ["account", "agent_id", "workflow_id"],
        ["account", "agent_id", "workflow_id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "fk_context_workflow", "retrieval_agent_context", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_generation_workflow", "retrieval_agent_generation", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_postprocess_workflow", "retrieval_agent_postprocess", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_preprocess_workflow", "retrieval_agent_preprocess", type_="foreignkey"
    )

    op.drop_index(
        op.f("ix_retrieval_agent_context_workflow_id"),
        table_name="retrieval_agent_context",
    )
    op.drop_index(
        op.f("ix_retrieval_agent_generation_workflow_id"),
        table_name="retrieval_agent_generation",
    )
    op.drop_index(
        op.f("ix_retrieval_agent_postprocess_workflow_id"),
        table_name="retrieval_agent_postprocess",
    )
    op.drop_index(
        op.f("ix_retrieval_agent_preprocess_workflow_id"),
        table_name="retrieval_agent_preprocess",
    )

    op.drop_column("retrieval_agent_context", "workflow_id")
    op.drop_column("retrieval_agent_generation", "workflow_id")
    op.drop_column("retrieval_agent_postprocess", "workflow_id")
    op.drop_column("retrieval_agent_preprocess", "workflow_id")

    op.drop_constraint(
        "fk_retrieval_agent_workflow_config", WORKFLOW_TABLE, type_="foreignkey"
    )
    op.drop_index(
        op.f("ix_retrieval_agent_workflow_workflow_id"), table_name=WORKFLOW_TABLE
    )
    op.drop_index(
        op.f("ix_retrieval_agent_workflow_agent_id"), table_name=WORKFLOW_TABLE
    )
    op.drop_index(
        op.f("ix_retrieval_agent_workflow_account"), table_name=WORKFLOW_TABLE
    )
    op.drop_table(WORKFLOW_TABLE)
    op.drop_table(PROMPTS_TABLE)

    op.drop_column("retrieval_agent_config", "description")

    op.drop_column("retrieval_agent_config", "title")

    op.drop_column("retrieval_agent_config", "instructions")
