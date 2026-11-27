"""Unify workflow stages into Harness engine, toolkits, skills, and storage tables.

Revision ID: c4e8a91b2d30
Revises: 8c7d2e4f91ab
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import context, op
from sqlalchemy.dialects import postgresql

from hyperforge.configure import GLOBAL_REGISTRY, load_entry_point_plugins
from hyperforge.db.workflow_migration import (
    migrate_database_workflows_to_harness,
    restore_database_workflows_from_snapshot,
)

revision: str = "c4e8a91b2d30"
down_revision: Union[str, None] = "8c7d2e4f91ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "retrieval_agent_workflow",
        sa.Column(
            "execution_mode",
            sa.String(),
            nullable=False,
            server_default="loop",
        ),
    )
    op.add_column(
        "retrieval_agent_workflow",
        sa.Column(
            "harness_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "retrieval_agent_workflow",
        sa.Column("workflow_code", sa.Text(), nullable=True),
    )
    op.add_column(
        "retrieval_agent_workflow",
        sa.Column(
            "legacy_stages_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )

    op.create_table(
        "agent_workflow_toolkits",
        sa.Column(
            "id",
            postgresql.UUID(),
            server_default=sa.func.uuid_generate_v4(),
            nullable=False,
        ),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("namespace", sa.String(), nullable=False),
        sa.Column("toolkit", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column(
            "lazy_load",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column(
            "ingest_skills",
            sa.Boolean(),
            nullable=False,
            server_default=sa.true(),
        ),
        sa.Column(
            "config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["account", "agent_id", "workflow_id"],
            [
                "retrieval_agent_workflow.account",
                "retrieval_agent_workflow.agent_id",
                "retrieval_agent_workflow.workflow_id",
            ],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "account",
            "agent_id",
            "workflow_id",
            "namespace",
            name="uq_agent_workflow_toolkits_namespace",
        ),
    )
    op.create_index(
        op.f("ix_agent_workflow_toolkits_account"),
        "agent_workflow_toolkits",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_workflow_toolkits_agent_id"),
        "agent_workflow_toolkits",
        ["agent_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_agent_workflow_toolkits_workflow_id"),
        "agent_workflow_toolkits",
        ["workflow_id"],
        unique=False,
    )

    op.create_table(
        "agent_skills",
        sa.Column(
            "id",
            postgresql.UUID(),
            server_default=sa.func.uuid_generate_v4(),
            nullable=False,
        ),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("skill_id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("instructions", sa.Text(), nullable=False),
        sa.Column(
            "recommended_tools",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "code_examples",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "preload",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("meta", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("modified", sa.DateTime(), onupdate=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["account", "agent_id"],
            ["retrieval_agent_config.account", "retrieval_agent_config.agent_id"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "account", "agent_id", "skill_id", name="uq_agent_skills_skill_id"
        ),
    )
    op.create_index(
        op.f("ix_agent_skills_account"), "agent_skills", ["account"], unique=False
    )
    op.create_index(
        op.f("ix_agent_skills_agent_id"), "agent_skills", ["agent_id"], unique=False
    )

    op.create_table(
        "harness_conversations",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("account", "agent_id", "id"),
    )
    op.create_index(
        op.f("ix_harness_conversations_account"),
        "harness_conversations",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_harness_conversations_agent_id"),
        "harness_conversations",
        ["agent_id"],
        unique=False,
    )

    op.create_table(
        "harness_events",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("seq", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("turn_id", sa.String(), nullable=True),
        sa.Column("harness_agent_id", sa.String(), nullable=True),
        sa.Column("parent_agent_id", sa.String(), nullable=True),
        sa.Column("parent_call_id", sa.String(), nullable=True),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("created_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.ForeignKeyConstraint(
            ["account", "agent_id", "conversation_id"],
            [
                "harness_conversations.account",
                "harness_conversations.agent_id",
                "harness_conversations.id",
            ],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("account", "agent_id", "id"),
    )
    op.create_index(
        op.f("ix_harness_events_seq"), "harness_events", ["seq"], unique=False
    )
    op.create_index(
        op.f("ix_harness_events_conversation_id"),
        "harness_events",
        ["conversation_id"],
        unique=False,
    )

    op.create_table(
        "harness_memories",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("seq", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("scope", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_datetime", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("account", "agent_id", "id"),
    )
    op.create_index(
        op.f("ix_harness_memories_seq"), "harness_memories", ["seq"], unique=False
    )
    op.create_index(
        op.f("ix_harness_memories_scope"), "harness_memories", ["scope"], unique=False
    )

    if not context.is_offline_mode():
        load_entry_point_plugins()
        ext_migrations = {
            key: entry.handler
            for key, entry in GLOBAL_REGISTRY.migrations.items()
            if entry.handler is not None
        }
        migrate_database_workflows_to_harness(
            op.get_bind(), external_migrations=ext_migrations
        )

    op.drop_table("retrieval_agent_preprocess")
    op.drop_table("retrieval_agent_context")
    op.drop_table("retrieval_agent_generation")
    op.drop_table("retrieval_agent_postprocess")


def downgrade() -> None:
    for table, col in (
        ("retrieval_agent_preprocess", "preprocess"),
        ("retrieval_agent_context", "context"),
        ("retrieval_agent_generation", "generation"),
        ("retrieval_agent_postprocess", "postprocess"),
    ):
        op.create_table(
            table,
            sa.Column(
                "id",
                postgresql.UUID(),
                server_default=sa.func.uuid_generate_v4(),
                nullable=False,
            ),
            sa.Column("account", sa.String(), nullable=False),
            sa.Column("agent_id", sa.String(), nullable=False),
            sa.Column("workflow_id", sa.String(), nullable=False),
            sa.Column(col, postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("created", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("modified", sa.DateTime(), onupdate=sa.func.now()),
            sa.ForeignKeyConstraint(
                ["account", "agent_id", "workflow_id"],
                [
                    "retrieval_agent_workflow.account",
                    "retrieval_agent_workflow.agent_id",
                    "retrieval_agent_workflow.workflow_id",
                ],
                onupdate="CASCADE",
                ondelete="CASCADE",
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(f"ix_{table}_account", table, ["account"], unique=False)
        op.create_index(f"ix_{table}_agent_id", table, ["agent_id"], unique=False)
        op.create_index(f"ix_{table}_workflow_id", table, ["workflow_id"], unique=False)

    if not context.is_offline_mode():
        restore_database_workflows_from_snapshot(op.get_bind())

    op.drop_table("harness_memories")
    op.drop_table("harness_events")
    op.drop_table("harness_conversations")
    op.drop_table("agent_skills")
    op.drop_table("agent_workflow_toolkits")
    op.drop_column("retrieval_agent_workflow", "legacy_stages_snapshot")
    op.drop_column("retrieval_agent_workflow", "workflow_code")
    op.drop_column("retrieval_agent_workflow", "harness_config")
    op.drop_column("retrieval_agent_workflow", "execution_mode")
