"""Versioned Procedural Graph storage protocol, lineage models, and diff helpers."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .graph import GraphEdits, ProceduralGraph, prepare_candidate
from .guidance import ProceduralGuidanceConfig

GraphVersionStatus = Literal["draft", "candidate", "active", "archived", "rejected"]
GraphVersionSource = Literal[
    "manual", "skeleton", "evolution", "rl_policy", "rollback", "import"
]
EvolutionRunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
EvolutionMode = Literal["llm_refiner", "rl_offline", "hybrid"]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class GuidanceRuntimeSettings(BaseModel):
    """Runtime guidance hyperparameters stored alongside a workflow's active graph."""

    model_config = ConfigDict(extra="forbid")

    hops: int = Field(default=2, ge=0, le=10)
    window: int = Field(default=3, ge=1, le=100)
    model: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, allow_inf_nan=False)
    failure_policy: Literal["unguided", "raise"] = "unguided"
    single_action: bool = False
    max_guidance_chars: int = Field(default=8000, ge=1)
    max_observation_chars: int = Field(default=8000, ge=1)
    max_graph_chars: int = Field(default=100000, ge=1)

    def to_guidance_config(self, graph: ProceduralGraph) -> ProceduralGuidanceConfig:
        return ProceduralGuidanceConfig(
            graph=graph,
            hops=self.hops,
            window=self.window,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            failure_policy=self.failure_policy,
            single_action=self.single_action,
            max_guidance_chars=self.max_guidance_chars,
            max_observation_chars=self.max_observation_chars,
            max_graph_chars=self.max_graph_chars,
        )

    @classmethod
    def from_guidance_config(
        cls, config: ProceduralGuidanceConfig
    ) -> GuidanceRuntimeSettings:
        dumped = config.model_dump(exclude={"graph"})
        return cls.model_validate(dumped)


class EvolutionSettings(BaseModel):
    """Default offline self-evolution & RL tuning parameters for a workflow."""

    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(default=5, ge=0, le=500)
    batch_size: int = Field(default=4, ge=1, le=1000)
    max_context_chars: int = Field(default=16000, ge=1)
    rejection_memory_limit: int = Field(default=10, ge=0, le=200)
    max_rejection_chars: int = Field(default=16000, ge=1)
    auto_promote: bool = False
    validation_split_ratio: float = Field(default=0.25, gt=0.0, lt=1.0)


class ProceduralGraphVersion(BaseModel):
    """Immutable historical snapshot of a workflow's Procedural Graph."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    account: str
    agent_id: str
    workflow_id: str
    version_number: int = Field(ge=1)
    fingerprint: str
    parent_version_id: str | None = None
    graph: ProceduralGraph
    edits_from_parent: dict[str, Any] | None = None
    status: GraphVersionStatus = "draft"
    source: GraphVersionSource = "manual"
    validation_score: float | None = None
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    diagnostics: str | None = None
    evolution_run_id: str | None = None
    created_by: str | None = None
    comment: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class ProceduralGraphBinding(BaseModel):
    """Workflow-level Procedural Graph attachment and runtime settings."""

    model_config = ConfigDict(extra="forbid")

    account: str
    agent_id: str
    workflow_id: str
    active_version_id: str | None = None
    active_version: ProceduralGraphVersion | None = None
    guidance_settings: GuidanceRuntimeSettings = Field(
        default_factory=GuidanceRuntimeSettings
    )
    evolution_settings: EvolutionSettings = Field(default_factory=EvolutionSettings)
    created_at: datetime = Field(default_factory=_utcnow)
    modified_at: datetime | None = None


class ProceduralEvolutionRun(BaseModel):
    """Record of an offline self-evolution or RL graph optimization job."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    account: str
    agent_id: str
    workflow_id: str
    base_version_id: str
    retained_version_id: str | None = None
    status: EvolutionRunStatus = "pending"
    mode: EvolutionMode = "llm_refiner"
    provenance: dict[str, Any] = Field(default_factory=dict)
    rounds_completed: int = 0
    total_rounds: int = 0
    started_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    error: str | None = None


class GraphVersionDiff(BaseModel):
    """Structural and attribute diff between two `ProceduralGraph` versions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_version_id: str
    to_version_id: str
    from_fingerprint: str
    to_fingerprint: str
    added_nodes: list[dict[str, Any]]
    removed_nodes: list[dict[str, Any]]
    modified_nodes: list[dict[str, Any]]
    added_edges: list[dict[str, Any]]
    removed_edges: list[dict[str, Any]]
    modified_edges: list[dict[str, Any]]


def diff_procedural_graphs(
    from_version: ProceduralGraphVersion,
    to_version: ProceduralGraphVersion,
) -> GraphVersionDiff:
    """Compute canonical node and edge differences between two graph versions."""
    old_nodes = {n.id: n.model_dump(mode="json") for n in from_version.graph.nodes}
    new_nodes = {n.id: n.model_dump(mode="json") for n in to_version.graph.nodes}

    added_nodes = [new_nodes[k] for k in sorted(new_nodes.keys() - old_nodes.keys())]
    removed_nodes = [old_nodes[k] for k in sorted(old_nodes.keys() - new_nodes.keys())]
    modified_nodes = [
        {"id": k, "before": old_nodes[k], "after": new_nodes[k]}
        for k in sorted(old_nodes.keys() & new_nodes.keys())
        if old_nodes[k] != new_nodes[k]
    ]

    old_edges = {
        (e.source, e.target, e.relation): e.model_dump(mode="json")
        for e in from_version.graph.edges
    }
    new_edges = {
        (e.source, e.target, e.relation): e.model_dump(mode="json")
        for e in to_version.graph.edges
    }

    added_edges = [new_edges[k] for k in sorted(new_edges.keys() - old_edges.keys())]
    removed_edges = [old_edges[k] for k in sorted(old_edges.keys() - new_edges.keys())]
    modified_edges = [
        {
            "triplet": {"source": k[0], "target": k[1], "relation": k[2]},
            "before": old_edges[k],
            "after": new_edges[k],
        }
        for k in sorted(old_edges.keys() & new_edges.keys())
        if old_edges[k] != new_edges[k]
    ]

    return GraphVersionDiff(
        from_version_id=from_version.id,
        to_version_id=to_version.id,
        from_fingerprint=from_version.fingerprint,
        to_fingerprint=to_version.fingerprint,
        added_nodes=added_nodes,
        removed_nodes=removed_nodes,
        modified_nodes=modified_nodes,
        added_edges=added_edges,
        removed_edges=removed_edges,
        modified_edges=modified_edges,
    )


def format_rejection_memory(
    rejected_versions: Iterable[ProceduralGraphVersion],
    *,
    limit: int = 10,
    max_chars: int = 16000,
) -> list[dict[str, Any]]:
    """Format newest rejected candidates within count and serialized char budgets.

    Matches arXiv:2609.09153v1 Section 3.3 Step 4: retains edits, candidate
    fingerprints, aggregate validation scores, and structural diagnostics without
    exposing validation task examples.
    """
    if limit <= 0 or max_chars <= 0:
        return []
    ordered = sorted(rejected_versions, key=lambda v: v.version_number)
    recent = ordered[-limit:]
    summaries: list[dict[str, Any]] = []
    for item in reversed(recent):
        entry: dict[str, Any] = {
            "version_number": item.version_number,
            "reason": (
                "proposal_error"
                if item.diagnostics and item.validation_score is None
                else "validation_worse"
            ),
            "candidate_score": item.validation_score,
            "edits": item.edits_from_parent,
            "candidate_fingerprint": item.fingerprint,
        }
        if item.diagnostics:
            entry["structural_diagnostic"] = item.diagnostics
        candidate_list = [entry, *summaries]
        encoded = json.dumps(
            candidate_list, ensure_ascii=True, sort_keys=True, indent=2
        )
        if len(encoded) <= max_chars:
            summaries = candidate_list
    return summaries


@runtime_checkable
class ProceduralGraphStoreProtocol(Protocol):
    """Storage interface for workflow-bound Procedural Graphs and version history."""

    async def get_binding(
        self, account: str, agent_id: str, workflow_id: str
    ) -> ProceduralGraphBinding | None: ...

    async def upsert_settings(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        guidance_settings: GuidanceRuntimeSettings | None = None,
        evolution_settings: EvolutionSettings | None = None,
    ) -> ProceduralGraphBinding: ...

    async def get_active_guidance_config(
        self, account: str, agent_id: str, workflow_id: str
    ) -> tuple[ProceduralGraphVersion, ProceduralGuidanceConfig] | None: ...

    async def get_version(self, version_id: str) -> ProceduralGraphVersion | None: ...

    async def get_version_by_fingerprint(
        self, account: str, agent_id: str, workflow_id: str, fingerprint: str
    ) -> ProceduralGraphVersion | None: ...

    async def list_versions(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        status: GraphVersionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[ProceduralGraphVersion]: ...

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
    ) -> ProceduralGraphVersion: ...

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
    ) -> ProceduralGraphVersion: ...

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
    ) -> ProceduralGraphVersion: ...

    async def promote_version(
        self, account: str, agent_id: str, workflow_id: str, version_id: str
    ) -> ProceduralGraphVersion: ...

    async def rollback_to_version(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        target_version_id: str,
        *,
        created_by: str | None = None,
        comment: str | None = None,
    ) -> ProceduralGraphVersion: ...

    async def get_rejection_memory(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        limit: int = 10,
        max_chars: int = 16000,
    ) -> list[dict[str, Any]]: ...

    async def create_evolution_run(
        self, run: ProceduralEvolutionRun
    ) -> ProceduralEvolutionRun: ...

    async def update_evolution_run(
        self, run_id: str, **updates: Any
    ) -> ProceduralEvolutionRun: ...

    async def list_evolution_runs(
        self, account: str, agent_id: str, workflow_id: str, *, limit: int = 20
    ) -> list[ProceduralEvolutionRun]: ...


class InMemoryProceduralGraphStore:
    """In-memory `ProceduralGraphStoreProtocol` for standalone mode and unit tests."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, str, str], ProceduralGraphBinding] = {}
        self._versions_by_id: dict[str, ProceduralGraphVersion] = {}
        self._versions_by_wf: dict[tuple[str, str, str], list[str]] = {}
        self._runs: dict[str, ProceduralEvolutionRun] = {}

    def _ensure_binding(
        self, account: str, agent_id: str, workflow_id: str
    ) -> ProceduralGraphBinding:
        key = (account, agent_id, workflow_id)
        binding = self._bindings.get(key)
        if binding is None:
            binding = ProceduralGraphBinding(
                account=account, agent_id=agent_id, workflow_id=workflow_id
            )
            self._bindings[key] = binding
        return binding

    async def get_binding(
        self, account: str, agent_id: str, workflow_id: str
    ) -> ProceduralGraphBinding | None:
        binding = self._bindings.get((account, agent_id, workflow_id))
        if binding is None:
            return None
        result = binding.model_copy(deep=True)
        if result.active_version_id:
            active = self._versions_by_id.get(result.active_version_id)
            result.active_version = active.model_copy(deep=True) if active else None
        return result

    async def upsert_settings(
        self,
        account: str,
        agent_id: str,
        workflow_id: str,
        *,
        guidance_settings: GuidanceRuntimeSettings | None = None,
        evolution_settings: EvolutionSettings | None = None,
    ) -> ProceduralGraphBinding:
        binding = self._ensure_binding(account, agent_id, workflow_id)
        if guidance_settings is not None:
            binding.guidance_settings = guidance_settings.model_copy(deep=True)
        if evolution_settings is not None:
            binding.evolution_settings = evolution_settings.model_copy(deep=True)
        binding.modified_at = _utcnow()
        res = await self.get_binding(account, agent_id, workflow_id)
        assert res is not None
        return res

    async def get_active_guidance_config(
        self, account: str, agent_id: str, workflow_id: str
    ) -> tuple[ProceduralGraphVersion, ProceduralGuidanceConfig] | None:
        binding = await self.get_binding(account, agent_id, workflow_id)
        if binding is None or binding.active_version is None:
            return None
        cfg = binding.guidance_settings.to_guidance_config(binding.active_version.graph)
        return binding.active_version, cfg

    async def get_version(self, version_id: str) -> ProceduralGraphVersion | None:
        ver = self._versions_by_id.get(version_id)
        return ver.model_copy(deep=True) if ver is not None else None

    async def get_version_by_fingerprint(
        self, account: str, agent_id: str, workflow_id: str, fingerprint: str
    ) -> ProceduralGraphVersion | None:
        ids = self._versions_by_wf.get((account, agent_id, workflow_id), [])
        for vid in reversed(ids):
            ver = self._versions_by_id[vid]
            if ver.fingerprint == fingerprint and ver.status != "rejected":
                return ver.model_copy(deep=True)
        return None

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
        ids = self._versions_by_wf.get((account, agent_id, workflow_id), [])
        items = [self._versions_by_id[vid] for vid in reversed(ids)]
        if status is not None:
            items = [v for v in items if v.status == status]
        sliced = items[max(0, offset) : max(0, offset) + max(1, limit)]
        return [v.model_copy(deep=True) for v in sliced]

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
        self._ensure_binding(account, agent_id, workflow_id)
        key = (account, agent_id, workflow_id)
        seq_list = self._versions_by_wf.setdefault(key, [])
        next_num = len(seq_list) + 1
        edits_payload = (
            edits_from_parent.model_dump(mode="json")
            if isinstance(edits_from_parent, GraphEdits)
            else dict(edits_from_parent)
            if isinstance(edits_from_parent, dict)
            else None
        )
        ver = ProceduralGraphVersion(
            account=account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            version_number=next_num,
            fingerprint=graph.fingerprint,
            parent_version_id=parent_version_id,
            graph=graph.model_copy(deep=True),
            edits_from_parent=edits_payload,
            status=status,
            source=source,
            validation_score=validation_score,
            validation_metrics=dict(validation_metrics or {}),
            diagnostics=diagnostics,
            evolution_run_id=evolution_run_id,
            created_by=created_by,
            comment=comment,
        )
        self._versions_by_id[ver.id] = ver
        seq_list.append(ver.id)
        if promote:
            return await self.promote_version(account, agent_id, workflow_id, ver.id)
        return ver.model_copy(deep=True)

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
        ver = self._versions_by_id.get(version_id)
        if (
            ver is None
            or ver.account != account
            or ver.agent_id != agent_id
            or ver.workflow_id != workflow_id
        ):
            raise ValueError(f"Procedural graph version not found: {version_id}")
        if ver.status == "rejected":
            raise ValueError("Cannot promote a rejected procedural graph version")

        binding = self._ensure_binding(account, agent_id, workflow_id)
        if binding.active_version_id and binding.active_version_id != version_id:
            prev = self._versions_by_id.get(binding.active_version_id)
            if prev is not None and prev.status == "active":
                prev.status = "archived"

        ver.status = "active"
        binding.active_version_id = ver.id
        binding.modified_at = _utcnow()
        return ver.model_copy(deep=True)

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
        ids = self._versions_by_wf.get((account, agent_id, workflow_id), [])
        rejected = [
            self._versions_by_id[vid]
            for vid in ids
            if self._versions_by_id[vid].status == "rejected"
        ]
        return format_rejection_memory(rejected, limit=limit, max_chars=max_chars)

    async def create_evolution_run(
        self, run: ProceduralEvolutionRun
    ) -> ProceduralEvolutionRun:
        stored = run.model_copy(deep=True)
        self._runs[stored.id] = stored
        return stored.model_copy(deep=True)

    async def update_evolution_run(
        self, run_id: str, **updates: Any
    ) -> ProceduralEvolutionRun:
        run = self._runs.get(run_id)
        if run is None:
            raise ValueError(f"Evolution run not found: {run_id}")
        for k, v in updates.items():
            setattr(run, k, v)
        return run.model_copy(deep=True)

    async def list_evolution_runs(
        self, account: str, agent_id: str, workflow_id: str, *, limit: int = 20
    ) -> list[ProceduralEvolutionRun]:
        matched = [
            r.model_copy(deep=True)
            for r in self._runs.values()
            if r.account == account
            and r.agent_id == agent_id
            and r.workflow_id == workflow_id
        ]
        matched.sort(key=lambda item: item.started_at, reverse=True)
        return matched[: max(1, limit)]


__all__ = [
    "EvolutionMode",
    "EvolutionRunStatus",
    "EvolutionSettings",
    "GraphVersionDiff",
    "GraphVersionSource",
    "GraphVersionStatus",
    "GuidanceRuntimeSettings",
    "InMemoryProceduralGraphStore",
    "ProceduralEvolutionRun",
    "ProceduralGraphBinding",
    "ProceduralGraphStoreProtocol",
    "ProceduralGraphVersion",
    "diff_procedural_graphs",
    "format_rejection_memory",
]
