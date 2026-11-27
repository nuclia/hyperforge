"""Add Procedural Graph version storage, evolution runs, and Harness decision audit tables.

Revision ID: d7f31b8e4c21
Revises: c4e8a91b2d30
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "d7f31b8e4c21"
down_revision: Union[str, None] = "c4e8a91b2d30"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "procedural_graphs",
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("active_version_id", postgresql.UUID(), nullable=True),
        sa.Column(
            "guidance_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "evolution_config",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("modified_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.PrimaryKeyConstraint("account", "agent_id", "workflow_id"),
    )
    op.create_index(
        op.f("ix_procedural_graphs_account"),
        "procedural_graphs",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graphs_agent_id"),
        "procedural_graphs",
        ["agent_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graphs_workflow_id"),
        "procedural_graphs",
        ["workflow_id"],
        unique=False,
    )

    op.create_table(
        "procedural_graph_versions",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("parent_version_id", postgresql.UUID(), nullable=True),
        sa.Column(
            "graph",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "edits_from_parent",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("validation_score", sa.Float(), nullable=True),
        sa.Column(
            "validation_metrics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("diagnostics", sa.Text(), nullable=True),
        sa.Column("evolution_run_id", postgresql.UUID(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["account", "agent_id", "workflow_id"],
            [
                "procedural_graphs.account",
                "procedural_graphs.agent_id",
                "procedural_graphs.workflow_id",
            ],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "account",
            "agent_id",
            "workflow_id",
            "version_number",
            name="uq_procedural_graph_versions_number",
        ),
    )
    op.create_index(
        op.f("ix_procedural_graph_versions_account"),
        "procedural_graph_versions",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graph_versions_agent_id"),
        "procedural_graph_versions",
        ["agent_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graph_versions_workflow_id"),
        "procedural_graph_versions",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graph_versions_fingerprint"),
        "procedural_graph_versions",
        ["fingerprint"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graph_versions_status"),
        "procedural_graph_versions",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_graph_versions_evolution_run_id"),
        "procedural_graph_versions",
        ["evolution_run_id"],
        unique=False,
    )

    op.create_table(
        "procedural_evolution_runs",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("base_version_id", postgresql.UUID(), nullable=False),
        sa.Column("retained_version_id", postgresql.UUID(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("mode", sa.String(), nullable=False),
        sa.Column(
            "provenance",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("rounds_completed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_rounds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["account", "agent_id", "workflow_id"],
            [
                "procedural_graphs.account",
                "procedural_graphs.agent_id",
                "procedural_graphs.workflow_id",
            ],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_procedural_evolution_runs_account"),
        "procedural_evolution_runs",
        ["account"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_evolution_runs_agent_id"),
        "procedural_evolution_runs",
        ["agent_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_evolution_runs_workflow_id"),
        "procedural_evolution_runs",
        ["workflow_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_procedural_evolution_runs_status"),
        "procedural_evolution_runs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "harness_audit_episodes",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("turn_id", sa.String(), nullable=False),
        sa.Column("execution_mode", sa.String(), nullable=False),
        sa.Column("graph_version_id", postgresql.UUID(), nullable=True),
        sa.Column("graph_fingerprint", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("final_output", sa.Text(), nullable=True),
        sa.Column("outcome_status", sa.String(), nullable=False),
        sa.Column("total_steps", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_tool_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("input_tokens", sa.Float(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reward", sa.Float(), nullable=True),
        sa.Column("reward_source", sa.String(), nullable=True),
        sa.Column(
            "reward_breakdown",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    for col in (
        "account",
        "agent_id",
        "workflow_id",
        "conversation_id",
        "turn_id",
        "graph_version_id",
        "graph_fingerprint",
        "outcome_status",
        "reward",
        "created_at",
    ):
        op.create_index(
            op.f(f"ix_harness_audit_episodes_{col}"),
            "harness_audit_episodes",
            [col],
            unique=False,
        )

    op.create_table(
        "harness_audit_decisions",
        sa.Column("id", postgresql.UUID(), nullable=False),
        sa.Column("episode_id", postgresql.UUID(), nullable=False),
        sa.Column("account", sa.String(), nullable=False),
        sa.Column("agent_id", sa.String(), nullable=False),
        sa.Column("workflow_id", sa.String(), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("decision_id", sa.String(), nullable=False),
        sa.Column("decision_type", sa.String(), nullable=False),
        sa.Column("active_node", sa.String(), nullable=True),
        sa.Column("graph_scope", sa.String(), nullable=False),
        sa.Column(
            "candidate_node_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "traversed_edge",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("guidance_text", sa.Text(), nullable=True),
        sa.Column("guidance_status", sa.String(), nullable=True),
        sa.Column("action_name", sa.String(), nullable=True),
        sa.Column(
            "action_arguments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("observation_summary", sa.Text(), nullable=True),
        sa.Column("action_status", sa.String(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("step_reward", sa.Float(), nullable=True),
        sa.Column("input_tokens", sa.Float(), nullable=False, server_default="0"),
        sa.Column("output_tokens", sa.Float(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Float(), nullable=False, server_default="0"),
        sa.Column("decision_cache_key", sa.String(), nullable=True),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["episode_id"],
            ["harness_audit_episodes.id"],
            onupdate="CASCADE",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    for col in (
        "episode_id",
        "account",
        "agent_id",
        "workflow_id",
        "decision_type",
        "active_node",
        "action_name",
        "created_at",
    ):
        op.create_index(
            op.f(f"ix_harness_audit_decisions_{col}"),
            "harness_audit_decisions",
            [col],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("harness_audit_decisions")
    op.drop_table("harness_audit_episodes")
    op.drop_table("procedural_evolution_runs")
    op.drop_table("procedural_graph_versions")
    op.drop_table("procedural_graphs")
