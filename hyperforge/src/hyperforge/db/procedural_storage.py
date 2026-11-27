"""PostgreSQL storage implementations for Procedural Graphs, version history, and Harness Audit."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any, Literal, cast
from uuid import uuid4

import databases
import sqlalchemy as sa
from pydantic import BaseModel
from sqlalchemy.dialects import postgresql as pg_dialect
from sqlalchemy.dialects.postgresql import JSONB

from hyperforge.database import metadata
from hyperforge.harness_sdk.audit import (
    AuditDecisionRecord,
    AuditEpisode,
    AuditTrajectory,
    AuditTraversedEdge,
    DecisionType,
    GraphScope,
    OutcomeStatus,
)
from hyperforge.procedural.graph import GraphEdits, ProceduralGraph, prepare_candidate
from hyperforge.procedural.guidance import ProceduralGuidanceConfig
from hyperforge.procedural.storage import (
    EvolutionMode,
    EvolutionRunStatus,
    EvolutionSettings,
    GraphVersionSource,
    GraphVersionStatus,
    GuidanceRuntimeSettings,
    ProceduralEvolutionRun,
    ProceduralGraphBinding,
    ProceduralGraphVersion,
    format_rejection_memory,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _decode_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    return _utcnow()


procedural_graphs = sa.Table(
    "procedural_graphs",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("active_version_id", pg_dialect.UUID, nullable=True),
    sa.Column("guidance_config", JSONB, nullable=False),
    sa.Column("evolution_config", JSONB, nullable=False),
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
)

procedural_graph_versions = sa.Table(
    "procedural_graph_versions",
    metadata,
    sa.Column("id", pg_dialect.UUID, primary_key=True, nullable=False),
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, nullable=False, index=True),
    sa.Column("version_number", sa.Integer, nullable=False),
    sa.Column("fingerprint", sa.String, nullable=False, index=True),
    sa.Column("parent_version_id", pg_dialect.UUID, nullable=True),
    sa.Column("graph", JSONB, nullable=False),
    sa.Column("edits_from_parent", JSONB, nullable=True),
    sa.Column("status", sa.String, nullable=False, index=True),
    sa.Column("source", sa.String, nullable=False),
    sa.Column("validation_score", sa.Float, nullable=True),
    sa.Column("validation_metrics", JSONB, nullable=False),
    sa.Column("diagnostics", sa.Text, nullable=True),
    sa.Column("evolution_run_id", pg_dialect.UUID, nullable=True, index=True),
    sa.Column("created_by", sa.String, nullable=True),
    sa.Column("comment", sa.Text, nullable=True),
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
    sa.UniqueConstraint(
        "account",
        "agent_id",
        "workflow_id",
        "version_number",
        name="uq_procedural_graph_versions_number",
    ),
)

procedural_evolution_runs = sa.Table(
    "procedural_evolution_runs",
    metadata,
    sa.Column("id", pg_dialect.UUID, primary_key=True, nullable=False),
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, nullable=False, index=True),
    sa.Column("base_version_id", pg_dialect.UUID, nullable=False),
    sa.Column("retained_version_id", pg_dialect.UUID, nullable=True),
    sa.Column("status", sa.String, nullable=False, index=True),
    sa.Column("mode", sa.String, nullable=False),
    sa.Column("provenance", JSONB, nullable=False),
    sa.Column("rounds_completed", sa.Integer, nullable=False, default=0),
    sa.Column("total_rounds", sa.Integer, nullable=False, default=0),
    sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    sa.Column("error", sa.Text, nullable=True),
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
)

harness_audit_episodes = sa.Table(
    "harness_audit_episodes",
    metadata,
    sa.Column("id", pg_dialect.UUID, primary_key=True, nullable=False),
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, nullable=False, index=True),
    sa.Column("conversation_id", sa.String, nullable=False, index=True),
    sa.Column("turn_id", sa.String, nullable=False, index=True),
    sa.Column("execution_mode", sa.String, nullable=False),
    sa.Column("graph_version_id", pg_dialect.UUID, nullable=True, index=True),
    sa.Column("graph_fingerprint", sa.String, nullable=True, index=True),
    sa.Column("model", sa.String, nullable=False),
    sa.Column("query", sa.Text, nullable=False),
    sa.Column("final_output", sa.Text, nullable=True),
    sa.Column("outcome_status", sa.String, nullable=False, index=True),
    sa.Column("total_steps", sa.Integer, nullable=False, default=0),
    sa.Column("total_tool_calls", sa.Integer, nullable=False, default=0),
    sa.Column("input_tokens", sa.Float, nullable=False, default=0.0),
    sa.Column("output_tokens", sa.Float, nullable=False, default=0.0),
    sa.Column("latency_ms", sa.Float, nullable=False, default=0.0),
    sa.Column("reward", sa.Float, nullable=True, index=True),
    sa.Column("reward_source", sa.String, nullable=True),
    sa.Column("reward_breakdown", JSONB, nullable=False),
    sa.Column("metadata", JSONB, nullable=False),
    sa.Column("error", sa.Text, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
)

harness_audit_decisions = sa.Table(
    "harness_audit_decisions",
    metadata,
    sa.Column("id", pg_dialect.UUID, primary_key=True, nullable=False),
    sa.Column("episode_id", pg_dialect.UUID, nullable=False, index=True),
    sa.Column("account", sa.String, nullable=False, index=True),
    sa.Column("agent_id", sa.String, nullable=False, index=True),
    sa.Column("workflow_id", sa.String, nullable=False, index=True),
    sa.Column("step_index", sa.Integer, nullable=False),
    sa.Column("decision_id", sa.String, nullable=False),
    sa.Column("decision_type", sa.String, nullable=False, index=True),
    sa.Column("active_node", sa.String, nullable=True, index=True),
    sa.Column("graph_scope", sa.String, nullable=False),
    sa.Column("candidate_node_ids", JSONB, nullable=False),
    sa.Column("traversed_edge", JSONB, nullable=True),
    sa.Column("guidance_text", sa.Text, nullable=True),
    sa.Column("guidance_status", sa.String, nullable=True),
    sa.Column("action_name", sa.String, nullable=True, index=True),
    sa.Column("action_arguments", JSONB, nullable=True),
    sa.Column("observation_summary", sa.Text, nullable=True),
    sa.Column("action_status", sa.String, nullable=False),
    sa.Column("error", sa.Text, nullable=True),
    sa.Column("step_reward", sa.Float, nullable=True),
    sa.Column("input_tokens", sa.Float, nullable=False, default=0.0),
    sa.Column("output_tokens", sa.Float, nullable=False, default=0.0),
    sa.Column("latency_ms", sa.Float, nullable=False, default=0.0),
    sa.Column("decision_cache_key", sa.String, nullable=True),
    sa.Column("trace_id", sa.String, nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, index=True),
    sa.ForeignKeyConstraint(
        ["episode_id"],
        ["harness_audit_episodes.id"],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)


def _row_to_version(row: Any) -> ProceduralGraphVersion:
    return ProceduralGraphVersion(
        id=str(row["id"]),
        account=row["account"],
        agent_id=row["agent_id"],
        workflow_id=row["workflow_id"],
        version_number=int(row["version_number"]),
        fingerprint=row["fingerprint"],
        parent_version_id=(
            str(row["parent_version_id"]) if row["parent_version_id"] else None
        ),
        graph=ProceduralGraph.model_validate(_decode_json(row["graph"], {})),
        edits_from_parent=_decode_json(row["edits_from_parent"], None),
        status=cast(GraphVersionStatus, row["status"]),
        source=cast(GraphVersionSource, row["source"]),
        validation_score=(
            float(row["validation_score"])
            if row["validation_score"] is not None
            else None
        ),
        validation_metrics=_decode_json(row["validation_metrics"], {}),
        diagnostics=row["diagnostics"],
        evolution_run_id=(
            str(row["evolution_run_id"]) if row["evolution_run_id"] else None
        ),
        created_by=row["created_by"],
        comment=row["comment"],
        created_at=_parse_dt(row["created_at"]),
    )


def _row_to_run(row: Any) -> ProceduralEvolutionRun:
    return ProceduralEvolutionRun(
        id=str(row["id"]),
        account=row["account"],
        agent_id=row["agent_id"],
        workflow_id=row["workflow_id"],
        base_version_id=str(row["base_version_id"]),
        retained_version_id=(
            str(row["retained_version_id"]) if row["retained_version_id"] else None
        ),
        status=cast(EvolutionRunStatus, row["status"]),
        mode=cast(EvolutionMode, row["mode"]),
        provenance=_decode_json(row["provenance"], {}),
        rounds_completed=int(row["rounds_completed"]),
        total_rounds=int(row["total_rounds"]),
        started_at=_parse_dt(row["started_at"]),
        completed_at=(_parse_dt(row["completed_at"]) if row["completed_at"] else None),
        error=row["error"],
    )


class PostgresProceduralGraphStore:
    """Durable PostgreSQL `ProceduralGraphStoreProtocol` implementation."""

    def __init__(self, database: databases.Database) -> None:
        self.database = database

    async def _ensure_binding_row(
        self, account: str, agent_id: str, workflow_id: str
    ) -> None:
        existing = await self.database.fetch_one(
            procedural_graphs.select().where(
                procedural_graphs.c.account == account,
                procedural_graphs.c.agent_id == agent_id,
                procedural_graphs.c.workflow_id == workflow_id,
            )
        )
        if existing is None:
            await self.database.execute(
                procedural_graphs.insert().values(
                    account=account,
                    agent_id=agent_id,
                    workflow_id=workflow_id,
                    active_version_id=None,
                    guidance_config=GuidanceRuntimeSettings().model_dump(),
                    evolution_config=EvolutionSettings().model_dump(),
                    created_at=_utcnow(),
                    modified_at=None,
                )
            )

    async def get_binding(
        self, account: str, agent_id: str, workflow_id: str
    ) -> ProceduralGraphBinding | None:
        row = await self.database.fetch_one(
            procedural_graphs.select().where(
                procedural_graphs.c.account == account,
                procedural_graphs.c.agent_id == agent_id,
                procedural_graphs.c.workflow_id == workflow_id,
            )
        )
        if row is None:
            return None
        active_id = str(row["active_version_id"]) if row["active_version_id"] else None
        active_ver = await self.get_version(active_id) if active_id else None
        return ProceduralGraphBinding(
            account=row["account"],
            agent_id=row["agent_id"],
            workflow_id=row["workflow_id"],
            active_version_id=active_id,
            active_version=active_ver,
            guidance_settings=GuidanceRuntimeSettings.model_validate(
                _decode_json(row["guidance_config"], {})
            ),
            evolution_settings=EvolutionSettings.model_validate(
                _decode_json(row["evolution_config"], {})
            ),
            created_at=_parse_dt(row["created_at"]),
            modified_at=(_parse_dt(row["modified_at"]) if row["modified_at"] else None),
        )

    async def upsert_settings(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        guidance_settings: GuidanceRuntimeSettings | None = None,
        evolution_settings: EvolutionSettings | None = None,
    ) -> ProceduralGraphBinding:
        await self._ensure_binding_row(account, agent_id, workflow_id)
        values: dict[str, Any] = {"modified_at": _utcnow()}
        if guidance_settings is not None:
            values["guidance_config"] = guidance_settings.model_dump()
        if evolution_settings is not None:
            values["evolution_config"] = evolution_settings.model_dump()
        await self.database.execute(
            procedural_graphs.update()
            .where(
                procedural_graphs.c.account == account,
                procedural_graphs.c.agent_id == agent_id,
                procedural_graphs.c.workflow_id == workflow_id,
            )
            .values(**values)
        )
        binding = await self.get_binding(account, agent_id, workflow_id)
        assert binding is not None
        return binding

    async def get_active_guidance_config(
        self, account: str, agent_id: str, workflow_id: str
    ) -> tuple[ProceduralGraphVersion, ProceduralGuidanceConfig] | None:
        binding = await self.get_binding(account, agent_id, workflow_id)
        if binding is None or binding.active_version is None:
            return None
        cfg = binding.guidance_settings.to_guidance_config(binding.active_version.graph)
        return binding.active_version, cfg

    async def get_version(self, version_id: str) -> ProceduralGraphVersion | None:
        row = await self.database.fetch_one(
            procedural_graph_versions.select().where(
                procedural_graph_versions.c.id == str(version_id)
            )
        )
        return _row_to_version(row) if row is not None else None

    async def get_version_by_fingerprint(
        self, account: str, agent_id: str, workflow_id: str, fingerprint: str
    ) -> ProceduralGraphVersion | None:
        stmt = (
            procedural_graph_versions.select()
            .where(
                procedural_graph_versions.c.account == account,
                procedural_graph_versions.c.agent_id == agent_id,
                procedural_graph_versions.c.workflow_id == workflow_id,
                procedural_graph_versions.c.fingerprint == fingerprint,
                procedural_graph_versions.c.status != "rejected",
            )
            .order_by(procedural_graph_versions.c.version_number.desc())
            .limit(1)
        )
        row = await self.database.fetch_one(stmt)
        return _row_to_version(row) if row is not None else None

    async def list_versions(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        status: GraphVersionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ProceduralGraphVersion]:
        stmt = procedural_graph_versions.select().where(
            procedural_graph_versions.c.account == account,
            procedural_graph_versions.c.agent_id == agent_id,
            procedural_graph_versions.c.workflow_id == workflow_id,
        )
        if status is not None:
            stmt = stmt.where(procedural_graph_versions.c.status == status)
        stmt = (
            stmt.order_by(procedural_graph_versions.c.version_number.desc())
            .offset(max(0, offset))
            .limit(max(1, limit))
        )
        rows = await self.database.fetch_all(stmt)
        return [_row_to_version(r) for r in rows]

    async def create_version(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        graph: ProceduralGraph,
        *,
        parent_version_id: str | None = None,
        edits_from_parent: GraphEdits | dict[str, Any] | None = None,
        status: GraphVersionStatus = "draft",
        source: GraphVersionSource = "manual",
        validation_score: float | None = None,
        validation_metrics: dict[str, Any] | None = None,
        diagnostics: str | None = None,
        evolution_run_id: str | None = None,
        created_by: str | None = None,
        comment: str | None = None,
        promote: bool = False,
    ) -> ProceduralGraphVersion:
        await self._ensure_binding_row(account, agent_id, workflow_id)
        latest_row = await self.database.fetch_one(
            procedural_graph_versions.select()
            .where(
                procedural_graph_versions.c.account == account,
                procedural_graph_versions.c.agent_id == agent_id,
                procedural_graph_versions.c.workflow_id == workflow_id,
            )
            .order_by(procedural_graph_versions.c.version_number.desc())
            .limit(1)
        )
        next_num = int(latest_row["version_number"]) + 1 if latest_row else 1
        ver_id = str(uuid4())
        edits_payload = (
            edits_from_parent.model_dump(mode="json")
            if isinstance(edits_from_parent, GraphEdits)
            else dict(edits_from_parent)
            if isinstance(edits_from_parent, dict)
            else None
        )
        now = _utcnow()
        await self.database.execute(
            procedural_graph_versions.insert().values(
                id=ver_id,
                account=account,
                agent_id=agent_id,
                workflow_id=workflow_id,
                version_number=next_num,
                fingerprint=graph.fingerprint,
                parent_version_id=parent_version_id,
                graph=graph.model_dump(mode="json"),
                edits_from_parent=edits_payload,
                status=status,
                source=source,
                validation_score=validation_score,
                validation_metrics=dict(validation_metrics or {}),
                diagnostics=diagnostics,
                evolution_run_id=evolution_run_id,
                created_by=created_by,
                comment=comment,
                created_at=now,
            )
        )
        if promote:
            return await self.promote_version(account, agent_id, workflow_id, ver_id)
        loaded = await self.get_version(ver_id)
        assert loaded is not None
        return loaded

    async def apply_edits(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        parent_version_id: str,
        edits: GraphEdits,
        available_tools: Iterable[str] | None = None,
        source: GraphVersionSource = "manual",
        created_by: str | None = None,
        comment: str | None = None,
        promote: bool = False,
    ) -> ProceduralGraphVersion:
        parent = await self.get_version(parent_version_id)
        if (
            parent is None
            or parent.account != account
            or parent.agent_id != agent_id
            or parent.workflow_id != workflow_id
        ):
            raise ValueError(f"Parent graph version not found: {parent_version_id}")
        candidate = prepare_candidate(
            parent.graph, edits, available_tools=available_tools
        )
        return await self.create_version(
            account,
            agent_id,
            workflow_id,
            candidate,
            parent_version_id=parent.id,
            edits_from_parent=edits,
            status="candidate",
            source=source,
            created_by=created_by,
            comment=comment,
            promote=promote,
        )

    async def record_rejected_proposal(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        parent_version_id: str,
        fallback_graph: ProceduralGraph,
        raw_edits: Any,
        diagnostics: str | None = None,
        candidate_graph: ProceduralGraph | None = None,
        validation_score: float | None = None,
        validation_metrics: dict[str, Any] | None = None,
        evolution_run_id: str | None = None,
        source: GraphVersionSource = "evolution",
        created_by: str | None = None,
    ) -> ProceduralGraphVersion:
        stored_graph = (
            candidate_graph if candidate_graph is not None else fallback_graph
        )
        edits_dict = (
            raw_edits.model_dump(mode="json")
            if isinstance(raw_edits, BaseModel)
            else raw_edits
            if isinstance(raw_edits, dict)
            else {"raw": repr(raw_edits)}
        )
        return await self.create_version(
            account,
            agent_id,
            workflow_id,
            stored_graph,
            parent_version_id=parent_version_id,
            edits_from_parent=edits_dict,
            status="rejected",
            source=source,
            validation_score=validation_score,
            validation_metrics=validation_metrics,
            diagnostics=diagnostics,
            evolution_run_id=evolution_run_id,
            created_by=created_by,
            promote=False,
        )

    async def promote_version(
        self, account: str, agent_id: str, workflow_id: str, version_id: str
    ) -> ProceduralGraphVersion:
        target = await self.get_version(version_id)
        if (
            target is None
            or target.account != account
            or target.agent_id != agent_id
            or target.workflow_id != workflow_id
        ):
            raise ValueError(f"Procedural graph version not found: {version_id}")
        if target.status == "rejected":
            raise ValueError("Cannot promote a rejected procedural graph version")

        await self._ensure_binding_row(account, agent_id, workflow_id)
        async with self.database.transaction():
            await self.database.execute(
                procedural_graph_versions.update()
                .where(
                    procedural_graph_versions.c.account == account,
                    procedural_graph_versions.c.agent_id == agent_id,
                    procedural_graph_versions.c.workflow_id == workflow_id,
                    procedural_graph_versions.c.status == "active",
                )
                .values(status="archived")
            )
            await self.database.execute(
                procedural_graph_versions.update()
                .where(procedural_graph_versions.c.id == str(version_id))
                .values(status="active")
            )
            await self.database.execute(
                procedural_graphs.update()
                .where(
                    procedural_graphs.c.account == account,
                    procedural_graphs.c.agent_id == agent_id,
                    procedural_graphs.c.workflow_id == workflow_id,
                )
                .values(active_version_id=str(version_id), modified_at=_utcnow())
            )
        promoted = await self.get_version(version_id)
        assert promoted is not None
        return promoted

    async def rollback_to_version(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        target_version_id: str,
        *,
        created_by: str | None = None,
        comment: str | None = None,
    ) -> ProceduralGraphVersion:
        target = await self.get_version(target_version_id)
        if (
            target is None
            or target.account != account
            or target.agent_id != agent_id
            or target.workflow_id != workflow_id
        ):
            raise ValueError(f"Target graph version not found: {target_version_id}")
        if target.status == "rejected":
            raise ValueError("Cannot rollback to a rejected procedural graph version")
        binding = await self.get_binding(account, agent_id, workflow_id)
        parent_id = (
            binding.active_version_id
            if binding and binding.active_version_id
            else target.id
        )
        rollback_comment = comment or f"Rollback to version #{target.version_number}"
        return await self.create_version(
            account,
            agent_id,
            workflow_id,
            target.graph,
            parent_version_id=parent_id,
            status="active",
            source="rollback",
            validation_score=target.validation_score,
            validation_metrics=target.validation_metrics,
            created_by=created_by,
            comment=rollback_comment,
            promote=True,
        )

    async def get_rejection_memory(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        limit: int = 10,
        max_chars: int = 16000,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or max_chars <= 0:
            return []
        rejected = await self.list_versions(
            account, agent_id, workflow_id, status="rejected", limit=limit
        )
        return format_rejection_memory(rejected, limit=limit, max_chars=max_chars)

    async def create_evolution_run(
        self, run: ProceduralEvolutionRun
    ) -> ProceduralEvolutionRun:
        await self._ensure_binding_row(run.account, run.agent_id, run.workflow_id)
        await self.database.execute(
            procedural_evolution_runs.insert().values(
                id=run.id,
                account=run.account,
                agent_id=run.agent_id,
                workflow_id=run.workflow_id,
                base_version_id=run.base_version_id,
                retained_version_id=run.retained_version_id,
                status=run.status,
                mode=run.mode,
                provenance=run.provenance,
                rounds_completed=run.rounds_completed,
                total_rounds=run.total_rounds,
                started_at=run.started_at,
                completed_at=run.completed_at,
                error=run.error,
            )
        )
        row = await self.database.fetch_one(
            procedural_evolution_runs.select().where(
                procedural_evolution_runs.c.id == run.id
            )
        )
        assert row is not None
        return _row_to_run(row)

    async def update_evolution_run(
        self, run_id: str, **updates: Any
    ) -> ProceduralEvolutionRun:
        await self.database.execute(
            procedural_evolution_runs.update()
            .where(procedural_evolution_runs.c.id == str(run_id))
            .values(**updates)
        )
        row = await self.database.fetch_one(
            procedural_evolution_runs.select().where(
                procedural_evolution_runs.c.id == str(run_id)
            )
        )
        if row is None:
            raise ValueError(f"Evolution run not found: {run_id}")
        return _row_to_run(row)

    async def list_evolution_runs(
        self, account: str, agent_id: str, workflow_id: str, *, limit: int = 20
    ) -> list[ProceduralEvolutionRun]:
        stmt = (
            procedural_evolution_runs.select()
            .where(
                procedural_evolution_runs.c.account == account,
                procedural_evolution_runs.c.agent_id == agent_id,
                procedural_evolution_runs.c.workflow_id == workflow_id,
            )
            .order_by(procedural_evolution_runs.c.started_at.desc())
            .limit(max(1, limit))
        )
        rows = await self.database.fetch_all(stmt)
        return [_row_to_run(r) for r in rows]


def _row_to_episode(row: Any) -> AuditEpisode:
    return AuditEpisode(
        id=str(row["id"]),
        account=row["account"],
        agent_id=row["agent_id"],
        workflow_id=row["workflow_id"],
        conversation_id=row["conversation_id"],
        turn_id=row["turn_id"],
        execution_mode="code" if row["execution_mode"] == "code" else "loop",
        graph_version_id=(
            str(row["graph_version_id"]) if row["graph_version_id"] else None
        ),
        graph_fingerprint=row["graph_fingerprint"],
        model=row["model"],
        query=row["query"],
        final_output=row["final_output"],
        outcome_status=cast(OutcomeStatus, row["outcome_status"]),
        total_steps=int(row["total_steps"]),
        total_tool_calls=int(row["total_tool_calls"]),
        input_tokens=float(row["input_tokens"]),
        output_tokens=float(row["output_tokens"]),
        latency_ms=float(row["latency_ms"]),
        reward=float(row["reward"]) if row["reward"] is not None else None,
        reward_source=row["reward_source"],
        reward_breakdown=_decode_json(row["reward_breakdown"], {}),
        metadata=_decode_json(row["metadata"], {}),
        error=row["error"],
        created_at=_parse_dt(row["created_at"]),
        completed_at=(_parse_dt(row["completed_at"]) if row["completed_at"] else None),
    )


def _row_to_decision(row: Any) -> AuditDecisionRecord:
    raw_edge = _decode_json(row["traversed_edge"], None)
    return AuditDecisionRecord(
        id=str(row["id"]),
        episode_id=str(row["episode_id"]),
        account=row["account"],
        agent_id=row["agent_id"],
        workflow_id=row["workflow_id"],
        step_index=int(row["step_index"]),
        decision_id=row["decision_id"],
        decision_type=cast(DecisionType, row["decision_type"]),
        active_node=row["active_node"],
        graph_scope=cast(GraphScope, row["graph_scope"]),
        candidate_node_ids=list(_decode_json(row["candidate_node_ids"], [])),
        traversed_edge=(
            AuditTraversedEdge.model_validate(raw_edge)
            if isinstance(raw_edge, dict)
            else None
        ),
        guidance_text=row["guidance_text"],
        guidance_status=row["guidance_status"],
        action_name=row["action_name"],
        action_arguments=_decode_json(row["action_arguments"], None),
        observation_summary=row["observation_summary"],
        action_status=row["action_status"],
        error=row["error"],
        step_reward=(
            float(row["step_reward"]) if row["step_reward"] is not None else None
        ),
        input_tokens=float(row["input_tokens"]),
        output_tokens=float(row["output_tokens"]),
        latency_ms=float(row["latency_ms"]),
        decision_cache_key=row["decision_cache_key"],
        trace_id=row["trace_id"],
        created_at=_parse_dt(row["created_at"]),
    )


class PostgresHarnessAuditStore:
    """Durable PostgreSQL `HarnessAuditProtocol` implementation."""

    def __init__(self, database: databases.Database) -> None:
        self.database = database

    async def start_episode(self, episode: AuditEpisode) -> str:
        await self.database.execute(
            harness_audit_episodes.insert().values(
                id=episode.id,
                account=episode.account,
                agent_id=episode.agent_id,
                workflow_id=episode.workflow_id,
                conversation_id=episode.conversation_id,
                turn_id=episode.turn_id,
                execution_mode=episode.execution_mode,
                graph_version_id=episode.graph_version_id,
                graph_fingerprint=episode.graph_fingerprint,
                model=episode.model,
                query=episode.query,
                final_output=episode.final_output,
                outcome_status=episode.outcome_status,
                total_steps=episode.total_steps,
                total_tool_calls=episode.total_tool_calls,
                input_tokens=episode.input_tokens,
                output_tokens=episode.output_tokens,
                latency_ms=episode.latency_ms,
                reward=episode.reward,
                reward_source=episode.reward_source,
                reward_breakdown=dict(episode.reward_breakdown),
                metadata=dict(episode.metadata),
                error=episode.error,
                created_at=episode.created_at,
                completed_at=episode.completed_at,
            )
        )
        return episode.id

    async def record_decision(self, decision: AuditDecisionRecord) -> None:
        await self.database.execute(
            harness_audit_decisions.insert().values(
                id=decision.id,
                episode_id=decision.episode_id,
                account=decision.account,
                agent_id=decision.agent_id,
                workflow_id=decision.workflow_id,
                step_index=decision.step_index,
                decision_id=decision.decision_id,
                decision_type=decision.decision_type,
                active_node=decision.active_node,
                graph_scope=decision.graph_scope,
                candidate_node_ids=list(decision.candidate_node_ids),
                traversed_edge=(
                    decision.traversed_edge.model_dump(mode="json")
                    if decision.traversed_edge is not None
                    else None
                ),
                guidance_text=decision.guidance_text,
                guidance_status=decision.guidance_status,
                action_name=decision.action_name,
                action_arguments=decision.action_arguments,
                observation_summary=decision.observation_summary,
                action_status=decision.action_status,
                error=decision.error,
                step_reward=decision.step_reward,
                input_tokens=decision.input_tokens,
                output_tokens=decision.output_tokens,
                latency_ms=decision.latency_ms,
                decision_cache_key=decision.decision_cache_key,
                trace_id=decision.trace_id,
                created_at=decision.created_at,
            )
        )

    async def complete_episode(
        self,
        episode_id: str,
        *,
        outcome_status: Literal["completed", "failed", "interrupted"],
        final_output: str | None = None,
        total_steps: int = 0,
        total_tool_calls: int = 0,
        input_tokens: float = 0.0,
        output_tokens: float = 0.0,
        latency_ms: float = 0.0,
        error: str | None = None,
        reward: float | None = None,
        reward_source: str | None = None,
        reward_breakdown: dict[str, Any] | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "outcome_status": outcome_status,
            "final_output": final_output,
            "total_steps": total_steps,
            "total_tool_calls": total_tool_calls,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "error": error,
            "completed_at": _utcnow(),
        }
        if reward is not None:
            values["reward"] = reward
            values["reward_source"] = reward_source
            if reward_breakdown is not None:
                values["reward_breakdown"] = dict(reward_breakdown)
        await self.database.execute(
            harness_audit_episodes.update()
            .where(harness_audit_episodes.c.id == str(episode_id))
            .values(**values)
        )

    async def attach_reward(
        self,
        episode_id: str,
        *,
        reward: float,
        source: str = "external",
        breakdown: dict[str, Any] | None = None,
        step_rewards: Mapping[int, float] | None = None,
    ) -> None:
        existing = await self.database.fetch_one(
            harness_audit_episodes.select().where(
                harness_audit_episodes.c.id == str(episode_id)
            )
        )
        if existing is None:
            raise ValueError(f"Unknown audit episode: {episode_id}")
        values: dict[str, Any] = {"reward": reward, "reward_source": source}
        if breakdown is not None:
            values["reward_breakdown"] = dict(breakdown)
        async with self.database.transaction():
            await self.database.execute(
                harness_audit_episodes.update()
                .where(harness_audit_episodes.c.id == str(episode_id))
                .values(**values)
            )
            if step_rewards:
                for step_idx, s_rew in step_rewards.items():
                    await self.database.execute(
                        harness_audit_decisions.update()
                        .where(
                            harness_audit_decisions.c.episode_id == str(episode_id),
                            harness_audit_decisions.c.step_index == int(step_idx),
                        )
                        .values(step_reward=float(s_rew))
                    )

    async def get_episode(self, episode_id: str) -> AuditTrajectory | None:
        ep_row = await self.database.fetch_one(
            harness_audit_episodes.select().where(
                harness_audit_episodes.c.id == str(episode_id)
            )
        )
        if ep_row is None:
            return None
        dec_rows = await self.database.fetch_all(
            harness_audit_decisions.select()
            .where(harness_audit_decisions.c.episode_id == str(episode_id))
            .order_by(harness_audit_decisions.c.step_index.asc())
        )
        return AuditTrajectory(
            episode=_row_to_episode(ep_row),
            decisions=[_row_to_decision(r) for r in dec_rows],
        )

    async def query_episodes(
        self,
        *,
        account: str | None = None,
        agent_id: str | None = None,
        workflow_id: str | None = None,
        graph_fingerprint: str | None = None,
        outcome_status: str | None = None,
        min_reward: float | None = None,
        max_reward: float | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AuditTrajectory]:
        stmt = harness_audit_episodes.select()
        if account is not None:
            stmt = stmt.where(harness_audit_episodes.c.account == account)
        if agent_id is not None:
            stmt = stmt.where(harness_audit_episodes.c.agent_id == agent_id)
        if workflow_id is not None:
            stmt = stmt.where(harness_audit_episodes.c.workflow_id == workflow_id)
        if graph_fingerprint is not None:
            stmt = stmt.where(
                harness_audit_episodes.c.graph_fingerprint == graph_fingerprint
            )
        if outcome_status is not None:
            stmt = stmt.where(harness_audit_episodes.c.outcome_status == outcome_status)
        if min_reward is not None:
            stmt = stmt.where(harness_audit_episodes.c.reward >= min_reward)
        if max_reward is not None:
            stmt = stmt.where(harness_audit_episodes.c.reward <= max_reward)
        stmt = (
            stmt.order_by(harness_audit_episodes.c.created_at.desc())
            .offset(max(0, offset))
            .limit(max(1, limit))
        )
        ep_rows = await self.database.fetch_all(stmt)
        results: list[AuditTrajectory] = []
        for ep_row in ep_rows:
            ep = _row_to_episode(ep_row)
            dec_rows = await self.database.fetch_all(
                harness_audit_decisions.select()
                .where(harness_audit_decisions.c.episode_id == ep.id)
                .order_by(harness_audit_decisions.c.step_index.asc())
            )
            results.append(
                AuditTrajectory(
                    episode=ep, decisions=[_row_to_decision(r) for r in dec_rows]
                )
            )
        return results


__all__ = [
    "PostgresHarnessAuditStore",
    "PostgresProceduralGraphStore",
    "harness_audit_decisions",
    "harness_audit_episodes",
    "procedural_evolution_runs",
    "procedural_graph_versions",
    "procedural_graphs",
]
