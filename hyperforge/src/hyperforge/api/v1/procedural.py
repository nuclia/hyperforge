"""REST API v1 endpoints for versioned Procedural Graphs, Decision Audit, and RL tuning."""

from __future__ import annotations

from typing import Any

from fastapi import Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request
from starlette.responses import PlainTextResponse

from hyperforge.api.authentication import requires_one
from hyperforge.api.models import StashRoles
from hyperforge.api.v1.router import router
from hyperforge.db import exceptions
from hyperforge.db.agents import AgentManager
from hyperforge.harness_sdk import (
    AuditTrajectory,
    EdgeEmpiricalMetrics,
    compute_edge_empirical_metrics,
    export_rl_transitions_jsonl,
)
from hyperforge.procedural import (
    EvolutionSettings,
    GraphEdits,
    GraphVersionDiff,
    GraphVersionStatus,
    GuidanceRuntimeSettings,
    NodeGuidancePolicy,
    ProceduralEvolutionRun,
    ProceduralGraph,
    ProceduralGraphBinding,
    ProceduralGraphVersion,
    SelectiveGuidanceBandit,
    diff_procedural_graphs,
)


class PutProceduralGraphRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    graph: ProceduralGraph | None = None
    skeleton: bool = False
    guidance_settings: GuidanceRuntimeSettings | None = None
    evolution_settings: EvolutionSettings | None = None
    parent_version_id: str | None = None
    comment: str | None = None
    promote: bool = True


class PatchProceduralSettingsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    guidance_settings: GuidanceRuntimeSettings | None = None
    evolution_settings: EvolutionSettings | None = None


class ApplyGraphEditsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    edits: GraphEdits
    parent_version_id: str | None = None
    available_tools: list[str] | None = None
    comment: str | None = None
    promote: bool = False


class RollbackGraphRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    comment: str | None = None


class AttachRewardRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reward: float
    source: str = "external"
    breakdown: dict[str, Any] = Field(default_factory=dict)
    step_rewards: dict[int, float] | None = None


async def _ensure_workflow(
    agent_manager: AgentManager, account: str, agent_id: str, workflow_id: str
) -> None:
    try:
        await agent_manager.ensure_workflow_active(account, agent_id, workflow_id)
    except exceptions.NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph",
    status_code=200,
    description="Get Workflow Procedural Graph Binding and Active Version",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_procedural_graph(
    request: Request,
    agent_id: str,
    workflow_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphBinding | None:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    return await agent_manager.procedural_store.get_binding(
        x_stf_account, agent_id, workflow_id
    )


@router.put(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph",
    status_code=200,
    description="Create and Optionally Activate a Procedural Graph Version",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def put_workflow_procedural_graph(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: PutProceduralGraphRequest,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphVersion:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    if item.graph is None and not item.skeleton:
        raise HTTPException(
            status_code=400,
            detail="Either `graph` must be provided or `skeleton` must be true",
        )
    graph = item.graph if item.graph is not None else ProceduralGraph.skeleton()
    if item.guidance_settings is not None or item.evolution_settings is not None:
        await agent_manager.procedural_store.upsert_settings(
            x_stf_account,
            agent_id,
            workflow_id,
            guidance_settings=item.guidance_settings,
            evolution_settings=item.evolution_settings,
        )
    binding = await agent_manager.procedural_store.get_binding(
        x_stf_account, agent_id, workflow_id
    )
    parent_id = item.parent_version_id or (
        binding.active_version_id if binding else None
    )
    return await agent_manager.procedural_store.create_version(
        x_stf_account,
        agent_id,
        workflow_id,
        graph,
        parent_version_id=parent_id,
        status="active" if item.promote else "draft",
        source="skeleton" if item.graph is None else "manual",
        created_by=x_stf_user,
        comment=item.comment,
        promote=item.promote,
    )


@router.patch(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/settings",
    status_code=200,
    description="Update Procedural Guidance & Evolution Settings",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def patch_workflow_procedural_settings(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: PatchProceduralSettingsRequest,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphBinding:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    return await agent_manager.procedural_store.upsert_settings(
        x_stf_account,
        agent_id,
        workflow_id,
        guidance_settings=item.guidance_settings,
        evolution_settings=item.evolution_settings,
    )


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/edits",
    status_code=200,
    description="Apply Validated GraphEdits to Create a Candidate Procedural Graph Version",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def apply_workflow_procedural_edits(
    request: Request,
    agent_id: str,
    workflow_id: str,
    item: ApplyGraphEditsRequest,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphVersion:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    parent_id = item.parent_version_id
    if parent_id is None:
        binding = await agent_manager.procedural_store.get_binding(
            x_stf_account, agent_id, workflow_id
        )
        parent_id = binding.active_version_id if binding else None
    if parent_id is None:
        raise HTTPException(
            status_code=400,
            detail="No parent_version_id provided and workflow has no active graph version",
        )
    try:
        return await agent_manager.procedural_store.apply_edits(
            x_stf_account,
            agent_id,
            workflow_id,
            parent_version_id=parent_id,
            edits=item.edits,
            available_tools=item.available_tools,
            source="manual",
            created_by=x_stf_user,
            comment=item.comment,
            promote=item.promote,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/versions",
    status_code=200,
    description="List Historical Procedural Graph Versions",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def list_workflow_procedural_versions(
    request: Request,
    agent_id: str,
    workflow_id: str,
    status: GraphVersionStatus | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[ProceduralGraphVersion]:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    return await agent_manager.procedural_store.list_versions(
        x_stf_account,
        agent_id,
        workflow_id,
        status=status,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/versions/{version_id}",
    status_code=200,
    description="Get Specific Procedural Graph Version",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_procedural_version(
    request: Request,
    agent_id: str,
    workflow_id: str,
    version_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphVersion:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    ver = await agent_manager.procedural_store.get_version(version_id)
    if (
        ver is None
        or ver.account != x_stf_account
        or ver.agent_id != agent_id
        or ver.workflow_id != workflow_id
    ):
        raise HTTPException(
            status_code=404, detail="Procedural graph version not found"
        )
    return ver


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/versions/{version_id}/diff",
    status_code=200,
    description="Diff a Procedural Graph Version Against Its Parent or Another Version",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def diff_workflow_procedural_version(
    request: Request,
    agent_id: str,
    workflow_id: str,
    version_id: str,
    from_version_id: str | None = Query(default=None),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> GraphVersionDiff:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    target = await agent_manager.procedural_store.get_version(version_id)
    if (
        target is None
        or target.account != x_stf_account
        or target.agent_id != agent_id
        or target.workflow_id != workflow_id
    ):
        raise HTTPException(status_code=404, detail="Target version not found")
    source_id = from_version_id or target.parent_version_id
    if source_id is None:
        raise HTTPException(
            status_code=400,
            detail="Version has no parent_version_id; specify from_version_id explicitly",
        )
    base = await agent_manager.procedural_store.get_version(source_id)
    if (
        base is None
        or base.account != x_stf_account
        or base.agent_id != agent_id
        or base.workflow_id != workflow_id
    ):
        raise HTTPException(status_code=404, detail="Base version not found")
    return diff_procedural_graphs(base, target)


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/versions/{version_id}/promote",
    status_code=200,
    description="Promote a Procedural Graph Version to Active",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def promote_workflow_procedural_version(
    request: Request,
    agent_id: str,
    workflow_id: str,
    version_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphVersion:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    try:
        return await agent_manager.procedural_store.promote_version(
            x_stf_account, agent_id, workflow_id, version_id
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/versions/{version_id}/rollback",
    status_code=200,
    description="Rollback Workflow Procedural Graph to a Previous Historical Version",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def rollback_workflow_procedural_version(
    request: Request,
    agent_id: str,
    workflow_id: str,
    version_id: str,
    item: RollbackGraphRequest | None = None,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> ProceduralGraphVersion:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    try:
        return await agent_manager.procedural_store.rollback_to_version(
            x_stf_account,
            agent_id,
            workflow_id,
            version_id,
            created_by=x_stf_user,
            comment=item.comment if item else None,
        )
    except ValueError as exc:
        code = 404 if "not found" in str(exc).lower() else 400
        raise HTTPException(status_code=code, detail=str(exc)) from exc


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/rejection-memory",
    status_code=200,
    description="Get Bounded Rejection Memory for Self-Evolution Refiner",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_rejection_memory(
    request: Request,
    agent_id: str,
    workflow_id: str,
    limit: int = Query(default=10, ge=0, le=100),
    max_chars: int = Query(default=16000, ge=1, le=200000),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[dict[str, Any]]:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    return await agent_manager.procedural_store.get_rejection_memory(
        x_stf_account, agent_id, workflow_id, limit=limit, max_chars=max_chars
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/procedural-graph/evolution-runs",
    status_code=200,
    description="List Procedural Graph Self-Evolution Runs",
    tags=["Procedural Graphs"],
)
@requires_one([StashRoles.OWNER])
async def list_workflow_evolution_runs(
    request: Request,
    agent_id: str,
    workflow_id: str,
    limit: int = Query(default=20, ge=1, le=100),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[ProceduralEvolutionRun]:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    return await agent_manager.procedural_store.list_evolution_runs(
        x_stf_account, agent_id, workflow_id, limit=limit
    )


# ---------------------------------------------------------------------------
# Harness Decision Audit & RL Analytics Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/audit/episodes",
    status_code=200,
    description="Query Audited Harness Decision Episodes",
    tags=["Harness Audit & RL"],
)
@requires_one([StashRoles.OWNER])
async def list_workflow_audit_episodes(
    request: Request,
    agent_id: str,
    workflow_id: str,
    graph_fingerprint: str | None = Query(default=None),
    outcome_status: str | None = Query(default=None),
    min_reward: float | None = Query(default=None),
    max_reward: float | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[AuditTrajectory]:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    return await agent_manager.audit_store.query_episodes(
        account=x_stf_account,
        agent_id=agent_id,
        workflow_id=workflow_id,
        graph_fingerprint=graph_fingerprint,
        outcome_status=outcome_status,
        min_reward=min_reward,
        max_reward=max_reward,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/audit/episodes/{episode_id}",
    status_code=200,
    description="Get Full Audited Episode Trajectory",
    tags=["Harness Audit & RL"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_audit_episode(
    request: Request,
    agent_id: str,
    workflow_id: str,
    episode_id: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AuditTrajectory:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    traj = await agent_manager.audit_store.get_episode(episode_id)
    if (
        traj is None
        or traj.episode.account != x_stf_account
        or traj.episode.agent_id != agent_id
        or traj.episode.workflow_id != workflow_id
    ):
        raise HTTPException(status_code=404, detail="Audit episode not found")
    return traj


@router.post(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/audit/episodes/{episode_id}/reward",
    status_code=200,
    description="Attach Scalar and Step-Level RL Rewards to an Audited Episode",
    tags=["Harness Audit & RL"],
)
@requires_one([StashRoles.OWNER])
async def attach_workflow_episode_reward(
    request: Request,
    agent_id: str,
    workflow_id: str,
    episode_id: str,
    item: AttachRewardRequest,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> AuditTrajectory:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    traj = await agent_manager.audit_store.get_episode(episode_id)
    if (
        traj is None
        or traj.episode.account != x_stf_account
        or traj.episode.agent_id != agent_id
        or traj.episode.workflow_id != workflow_id
    ):
        raise HTTPException(status_code=404, detail="Audit episode not found")
    await agent_manager.audit_store.attach_reward(
        episode_id,
        reward=item.reward,
        source=item.source,
        breakdown=item.breakdown,
        step_rewards=item.step_rewards,
    )
    updated = await agent_manager.audit_store.get_episode(episode_id)
    assert updated is not None
    return updated


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/audit/export",
    status_code=200,
    description="Export Audited Trajectories as JSONL RL Transition Tuples",
    tags=["Harness Audit & RL"],
)
@requires_one([StashRoles.OWNER])
async def export_workflow_rl_dataset(
    request: Request,
    agent_id: str,
    workflow_id: str,
    gamma: float = Query(default=0.99, ge=0.0, le=1.0),
    graph_fingerprint: str | None = Query(default=None),
    min_reward: float | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> PlainTextResponse:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    trajectories = await agent_manager.audit_store.query_episodes(
        account=x_stf_account,
        agent_id=agent_id,
        workflow_id=workflow_id,
        graph_fingerprint=graph_fingerprint,
        min_reward=min_reward,
        limit=limit,
    )
    jsonl_body = export_rl_transitions_jsonl(trajectories, gamma=gamma)
    return PlainTextResponse(jsonl_body, media_type="application/x-ndjson")


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/audit/edge-metrics",
    status_code=200,
    description="Compute Per-Edge Empirical Reward, Failure, and Token Metrics",
    tags=["Harness Audit & RL"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_edge_metrics(
    request: Request,
    agent_id: str,
    workflow_id: str,
    graph_fingerprint: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[EdgeEmpiricalMetrics]:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    trajectories = await agent_manager.audit_store.query_episodes(
        account=x_stf_account,
        agent_id=agent_id,
        workflow_id=workflow_id,
        graph_fingerprint=graph_fingerprint,
        limit=limit,
    )
    return compute_edge_empirical_metrics(trajectories)


@router.get(
    "/api/v1/agent/{agent_id}/workflow/{workflow_id}/audit/guidance-policy",
    status_code=200,
    description="Estimate Selective Guidance Bandit Policy per Graph Node",
    tags=["Harness Audit & RL"],
)
@requires_one([StashRoles.OWNER])
async def get_workflow_guidance_policy(
    request: Request,
    agent_id: str,
    workflow_id: str,
    token_cost_weight: float = Query(default=0.00002, ge=0.0),
    min_samples_per_arm: int = Query(default=2, ge=1),
    gamma: float = Query(default=0.99, ge=0.0, le=1.0),
    graph_fingerprint: str | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
) -> list[NodeGuidancePolicy]:
    agent_manager: AgentManager = request.app.agent_manager
    await _ensure_workflow(agent_manager, x_stf_account, agent_id, workflow_id)
    trajectories = await agent_manager.audit_store.query_episodes(
        account=x_stf_account,
        agent_id=agent_id,
        workflow_id=workflow_id,
        graph_fingerprint=graph_fingerprint,
        limit=limit,
    )
    bandit = SelectiveGuidanceBandit(
        token_cost_weight=token_cost_weight,
        min_samples_per_arm=min_samples_per_arm,
    )
    bandit.fit_trajectories(trajectories, gamma=gamma)
    return bandit.summary()
