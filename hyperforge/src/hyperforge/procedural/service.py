"""Database-backed Procedural Graph self-evolution, contrastive audit mining, and RL gating."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hyperforge.harness_sdk.audit import (
    AuditTrajectory,
    HarnessAuditProtocol,
    RLTransitionTuple,
    compute_edge_empirical_metrics,
    trajectories_to_rl_transitions,
)

from .evolution import EvaluationTask, Refiner, Runner, ScoredRollout, TokenCodec
from .graph import GraphEdits, ProceduralGraph, prepare_candidate
from .storage import (
    ProceduralEvolutionRun,
    ProceduralGraphStoreProtocol,
    ProceduralGraphVersion,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2
    )


def _summarize_rollout_metrics(traces: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not traces:
        return {}
    count = len(traces)
    mean_score = sum(float(t.get("score", 0.0)) for t in traces) / count
    summary: dict[str, Any] = {
        "episodes": count,
        "mean_score": round(mean_score, 6),
    }
    metric_keys: set[str] = set()
    for t in traces:
        if isinstance(t.get("metrics"), dict):
            metric_keys.update(t["metrics"].keys())
    for key in sorted(metric_keys):
        vals = [
            float(t["metrics"][key])
            for t in traces
            if isinstance(t.get("metrics"), dict) and key in t["metrics"]
        ]
        if vals:
            summary[f"mean_{key}"] = round(sum(vals) / len(vals), 4)
    return summary


async def evolve_with_store(
    store: ProceduralGraphStoreProtocol,
    *,
    account: str,
    agent_id: str,
    workflow_id: str,
    train_tasks: Sequence[EvaluationTask],
    validation_tasks: Sequence[EvaluationTask],
    runner: Runner,
    refiner: Refiner,
    available_tools: Sequence[str],
    provenance: dict[str, Any],
    base_version_id: str | None = None,
    initial_graph: ProceduralGraph | None = None,
    rounds: int = 5,
    batch_size: int = 2,
    max_context_chars: int = 16000,
    max_context_tokens: int | None = None,
    token_codec: TokenCodec | None = None,
    rejection_memory_limit: int = 10,
    max_rejection_chars: int = 16000,
    auto_promote: bool = False,
    created_by: str | None = "evolution",
) -> tuple[ProceduralEvolutionRun, ProceduralGraphVersion]:
    """Execute validation-gated Procedural Graph self-evolution backed by `store`.

    Implements Algorithm 1 of arXiv:2609.09153v1 against persistent version history:
    - Evaluates base version once on `validation_tasks` to cache reference score.
    - Across `rounds`, rolls out training batches, tail-truncates trajectory context,
      supplies persisted rejection memory, validates proposed `GraphEdits`, and gates
      structural candidates on `S_val(G_cand) >= S_val(G_retained)` (accepting ties).
    - Every candidate (accepted or rejected) is recorded as an immutable
      `ProceduralGraphVersion` with full parent lineage, edits, scores, and diagnostics.
    """
    if rounds < 0 or batch_size < 1 or rejection_memory_limit < 0:
        raise ValueError(
            "rounds/rejection_memory_limit must be nonnegative; batch_size >= 1"
        )
    if not train_tasks or not validation_tasks:
        raise ValueError("train_tasks and validation_tasks must be nonempty")
    if not {"model", "evaluator", "tools"} <= provenance.keys():
        raise ValueError("provenance requires model, evaluator, and tools")

    tools = tuple(available_tools)
    base_version: ProceduralGraphVersion | None = None
    if base_version_id is not None:
        base_version = await store.get_version(base_version_id)
        if base_version is None:
            raise ValueError(f"Base graph version not found: {base_version_id}")
    else:
        binding = await store.get_binding(account, agent_id, workflow_id)
        if binding and binding.active_version is not None:
            base_version = binding.active_version
        else:
            seed_graph = initial_graph or ProceduralGraph.skeleton()
            base_version = await store.create_version(
                account,
                agent_id,
                workflow_id,
                seed_graph,
                status="candidate",
                source="skeleton" if initial_graph is None else "manual",
                created_by=created_by,
                comment="Initial seed graph for self-evolution",
            )

    base_version.graph.check_tools(tools)

    evo_run = await store.create_evolution_run(
        ProceduralEvolutionRun(
            account=account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            base_version_id=base_version.id,
            retained_version_id=base_version.id,
            status="running",
            mode="llm_refiner",
            provenance=dict(provenance),
            total_rounds=rounds,
        )
    )

    async def _evaluate_split(
        graph: ProceduralGraph, tasks: Sequence[EvaluationTask]
    ) -> tuple[float, list[dict[str, Any]]]:
        traces: list[dict[str, Any]] = []
        for task in tasks:
            res = await runner(graph.model_copy(deep=True), task.model_copy(deep=True))
            rollout = ScoredRollout.model_validate(
                res.model_dump() if isinstance(res, BaseModel) else res
            )
            if rollout.task_id != task.id:
                raise ValueError(
                    f"Runner task ID mismatch: expected {task.id!r}, got {rollout.task_id!r}"
                )
            trace = rollout.model_dump(mode="json")
            trace["query"] = task.query
            _canonical_json(trace)
            traces.append(trace)
        mean = sum(float(t["score"]) for t in traces) / len(traces)
        return mean, traces

    try:
        retained_version = base_version
        retained_graph = base_version.graph
        if base_version.validation_score is not None:
            retained_score = float(base_version.validation_score)
        else:
            retained_score, base_val_traces = await _evaluate_split(
                retained_graph, validation_tasks
            )
            retained_version = await store.create_version(
                account,
                agent_id,
                workflow_id,
                retained_graph,
                parent_version_id=base_version.id,
                status="candidate",
                source="evolution",
                validation_score=retained_score,
                validation_metrics=_summarize_rollout_metrics(base_val_traces),
                evolution_run_id=evo_run.id,
                created_by=created_by,
                comment="Baseline validation checkpoint",
            )
            await store.update_evolution_run(
                evo_run.id, retained_version_id=retained_version.id
            )

        cursor = 0
        for round_idx in range(1, rounds + 1):
            batch = list(train_tasks[cursor : cursor + batch_size])
            cursor = (cursor + len(batch)) % len(train_tasks)

            proposal: Any = None
            candidate_graph: ProceduralGraph | None = None
            phase = "training"
            try:
                _, train_traces = await _evaluate_split(retained_graph, batch)
                training_ctx = "".join(_canonical_json(t) + "\n" for t in train_traces)
                if max_context_tokens is not None:
                    assert token_codec is not None
                    bounded_ctx = token_codec.decode(
                        token_codec.tokenize(training_ctx)[-max_context_tokens:]
                    )
                else:
                    bounded_ctx = training_ctx[-max_context_chars:]

                rejections = await store.get_rejection_memory(
                    account,
                    agent_id,
                    workflow_id,
                    limit=rejection_memory_limit,
                    max_chars=max_rejection_chars,
                )
                phase = "refiner"
                proposal = await refiner(
                    retained_graph.model_copy(deep=True),
                    bounded_ctx,
                    rejections,
                )
                phase = "proposal"
                edits = GraphEdits.model_validate(
                    proposal.model_dump()
                    if isinstance(proposal, BaseModel)
                    else proposal
                )
                candidate_graph = prepare_candidate(
                    retained_graph, edits, available_tools=tools
                )
            except Exception as exc:
                await store.record_rejected_proposal(
                    account,
                    agent_id,
                    workflow_id,
                    parent_version_id=retained_version.id,
                    fallback_graph=retained_graph,
                    raw_edits=proposal,
                    diagnostics=f"{phase}_error: {type(exc).__name__}: {exc}",
                    evolution_run_id=evo_run.id,
                    created_by=created_by,
                )
                await store.update_evolution_run(evo_run.id, rounds_completed=round_idx)
                continue

            try:
                cand_score, val_traces = await _evaluate_split(
                    candidate_graph, validation_tasks
                )
            except Exception as exc:
                await store.record_rejected_proposal(
                    account,
                    agent_id,
                    workflow_id,
                    parent_version_id=retained_version.id,
                    fallback_graph=retained_graph,
                    raw_edits=edits,
                    candidate_graph=candidate_graph,
                    diagnostics=f"validation_error: {type(exc).__name__}: {exc}",
                    evolution_run_id=evo_run.id,
                    created_by=created_by,
                )
                await store.update_evolution_run(evo_run.id, rounds_completed=round_idx)
                continue

            val_metrics = _summarize_rollout_metrics(val_traces)
            if cand_score >= retained_score:
                retained_version = await store.create_version(
                    account,
                    agent_id,
                    workflow_id,
                    candidate_graph,
                    parent_version_id=retained_version.id,
                    edits_from_parent=edits,
                    status="candidate",
                    source="evolution",
                    validation_score=cand_score,
                    validation_metrics=val_metrics,
                    evolution_run_id=evo_run.id,
                    created_by=created_by,
                    comment=f"Accepted evolution round {round_idx} (score={cand_score:.4f})",
                )
                retained_graph = candidate_graph
                retained_score = cand_score
                await store.update_evolution_run(
                    evo_run.id,
                    retained_version_id=retained_version.id,
                    rounds_completed=round_idx,
                )
            else:
                await store.record_rejected_proposal(
                    account,
                    agent_id,
                    workflow_id,
                    parent_version_id=retained_version.id,
                    fallback_graph=retained_graph,
                    raw_edits=edits,
                    candidate_graph=candidate_graph,
                    validation_score=cand_score,
                    validation_metrics=val_metrics,
                    diagnostics=(
                        f"validation_worse: candidate={cand_score:.4f} < retained={retained_score:.4f}"
                    ),
                    evolution_run_id=evo_run.id,
                    created_by=created_by,
                )
                await store.update_evolution_run(evo_run.id, rounds_completed=round_idx)

        if auto_promote:
            retained_version = await store.promote_version(
                account, agent_id, workflow_id, retained_version.id
            )

        completed_run = await store.update_evolution_run(
            evo_run.id,
            status="completed",
            retained_version_id=retained_version.id,
            completed_at=_utcnow(),
        )
        return completed_run, retained_version

    except Exception as exc:
        await store.update_evolution_run(
            evo_run.id,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
            completed_at=_utcnow(),
        )
        raise


def audit_trajectories_to_contrastive_context(
    trajectories: Sequence[AuditTrajectory],
    *,
    high_reward_threshold: float = 0.7,
    low_reward_threshold: float = 0.4,
    max_chars: int = 16000,
) -> str:
    """Build refiner training context contrasting high-reward and low-reward audit episodes.

    Includes empirical edge reward/failure statistics computed from the audited
    decisions so the refiner or RL policy tuner sees exact transition bottlenecks.
    """
    scored = [t for t in trajectories if t.episode.reward is not None]
    high = [
        t for t in scored if float(t.episode.reward or 0.0) >= high_reward_threshold
    ]
    low = [t for t in scored if float(t.episode.reward or 0.0) <= low_reward_threshold]
    edge_stats = [
        m.model_dump(mode="json") for m in compute_edge_empirical_metrics(scored)
    ]

    def _compact(traj: AuditTrajectory) -> dict[str, Any]:
        ep = traj.episode
        return {
            "episode_id": ep.id,
            "query": ep.query,
            "score": ep.reward,
            "outcome_status": ep.outcome_status,
            "final_output": (ep.final_output or "")[:1000],
            "steps": [
                {
                    "step": d.step_index,
                    "type": d.decision_type,
                    "active_node": d.active_node,
                    "action": d.action_name,
                    "edge": (
                        d.traversed_edge.model_dump(mode="json")
                        if d.traversed_edge
                        else None
                    ),
                    "status": d.action_status,
                    "step_reward": d.step_reward,
                    "error": d.error,
                }
                for d in traj.decisions
                if d.decision_type != "procedural_guidance"
            ],
        }

    payload = {
        "empirical_edge_metrics": edge_stats,
        "high_reward_trajectories": [_compact(t) for t in high],
        "low_reward_trajectories": [_compact(t) for t in low],
    }
    serialized = _canonical_json(payload)
    return serialized[-max_chars:]


async def propose_candidate_from_audit(
    store: ProceduralGraphStoreProtocol,
    audit_store: HarnessAuditProtocol,
    *,
    account: str,
    agent_id: str,
    workflow_id: str,
    refiner: Refiner,
    available_tools: Sequence[str],
    base_version_id: str | None = None,
    episode_limit: int = 100,
    high_reward_threshold: float = 0.7,
    low_reward_threshold: float = 0.4,
    max_context_chars: int = 16000,
    rejection_memory_limit: int = 10,
    promote: bool = False,
    created_by: str | None = "rl_audit_refiner",
) -> ProceduralGraphVersion:
    """Mine audited decision episodes and propose a validated `ProceduralGraphVersion`."""
    if base_version_id is not None:
        parent = await store.get_version(base_version_id)
    else:
        binding = await store.get_binding(account, agent_id, workflow_id)
        parent = binding.active_version if binding else None
    if parent is None:
        parent = await store.create_version(
            account,
            agent_id,
            workflow_id,
            ProceduralGraph.skeleton(),
            status="candidate",
            source="skeleton",
            created_by=created_by,
        )

    trajectories = await audit_store.query_episodes(
        account=account,
        agent_id=agent_id,
        workflow_id=workflow_id,
        graph_fingerprint=parent.fingerprint,
        limit=episode_limit,
    )
    if not trajectories:
        trajectories = await audit_store.query_episodes(
            account=account,
            agent_id=agent_id,
            workflow_id=workflow_id,
            limit=episode_limit,
        )
    if not trajectories:
        raise ValueError("No audited episodes available to propose graph updates")

    context_str = audit_trajectories_to_contrastive_context(
        trajectories,
        high_reward_threshold=high_reward_threshold,
        low_reward_threshold=low_reward_threshold,
        max_chars=max_context_chars,
    )
    rejections = await store.get_rejection_memory(
        account, agent_id, workflow_id, limit=rejection_memory_limit
    )

    proposal: Any = None
    try:
        proposal = await refiner(
            parent.graph.model_copy(deep=True), context_str, rejections
        )
        edits = GraphEdits.model_validate(
            proposal.model_dump() if isinstance(proposal, BaseModel) else proposal
        )
        candidate = prepare_candidate(
            parent.graph, edits, available_tools=available_tools
        )
    except Exception as exc:
        await store.record_rejected_proposal(
            account,
            agent_id,
            workflow_id,
            parent_version_id=parent.id,
            fallback_graph=parent.graph,
            raw_edits=proposal,
            diagnostics=f"audit_proposal_error: {type(exc).__name__}: {exc}",
            source="rl_policy",
            created_by=created_by,
        )
        raise

    edge_metrics = [
        m.model_dump(mode="json") for m in compute_edge_empirical_metrics(trajectories)
    ]
    return await store.create_version(
        account,
        agent_id,
        workflow_id,
        candidate,
        parent_version_id=parent.id,
        edits_from_parent=edits,
        status="candidate",
        source="rl_policy",
        validation_metrics={
            "source_episodes": len(trajectories),
            "parent_edge_metrics": edge_metrics,
        },
        created_by=created_by,
        comment=f"Proposed from {len(trajectories)} audited decision episodes",
        promote=promote,
    )


class NodeGuidancePolicy(BaseModel):
    """Learned contextual-bandit / RL gating statistics for a single graph node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    node_id: str
    guided_samples: int
    unguided_samples: int
    guided_mean_utility: float
    unguided_mean_utility: float
    recommend_guidance: bool = Field(
        description="True when E[G_t - token_cost_weight * tokens | guided] >= unguided."
    )


class SelectiveGuidanceBandit:
    """Contextual bandit learned from `RLTransitionTuple`s to gate guidance calls.

    Addresses Section 5.5 & Section 6 of arXiv:2609.09153v1: estimates per-node
    expected token-penalized return-to-go `G_t - token_cost_weight * tokens_t` to
    decide when online generative guidance improves net utility vs. skipping it.
    """

    def __init__(
        self,
        *,
        token_cost_weight: float = 0.00002,
        min_samples_per_arm: int = 2,
    ) -> None:
        self.token_cost_weight = token_cost_weight
        self.min_samples_per_arm = min_samples_per_arm
        self._guided: dict[str, list[float]] = defaultdict(list)
        self._unguided: dict[str, list[float]] = defaultdict(list)

    def fit_trajectories(
        self, trajectories: Sequence[AuditTrajectory], *, gamma: float = 0.99
    ) -> None:
        self.fit(trajectories_to_rl_transitions(trajectories, gamma=gamma))

    def fit(self, transitions: Sequence[RLTransitionTuple]) -> None:
        self._guided.clear()
        self._unguided.clear()
        for tr in transitions:
            if tr.decision_type != "llm_action":
                continue
            node = tr.active_node or "__full__"
            utility = tr.return_to_go - self.token_cost_weight * tr.tokens_used
            if tr.guidance_invoked:
                self._guided[node].append(utility)
            else:
                self._unguided[node].append(utility)

    def should_invoke_guidance(self, active_node: str | None) -> bool:
        node = active_node or "__full__"
        g_vals = self._guided.get(node, [])
        u_vals = self._unguided.get(node, [])
        if (
            len(g_vals) < self.min_samples_per_arm
            or len(u_vals) < self.min_samples_per_arm
        ):
            return True
        return (sum(g_vals) / len(g_vals)) >= (sum(u_vals) / len(u_vals))

    def summary(self) -> list[NodeGuidancePolicy]:
        nodes = sorted(set(self._guided.keys()) | set(self._unguided.keys()))
        policies: list[NodeGuidancePolicy] = []
        for node in nodes:
            g_vals = self._guided.get(node, [])
            u_vals = self._unguided.get(node, [])
            g_mean = sum(g_vals) / len(g_vals) if g_vals else 0.0
            u_mean = sum(u_vals) / len(u_vals) if u_vals else 0.0
            policies.append(
                NodeGuidancePolicy(
                    node_id=node,
                    guided_samples=len(g_vals),
                    unguided_samples=len(u_vals),
                    guided_mean_utility=round(g_mean, 6),
                    unguided_mean_utility=round(u_mean, 6),
                    recommend_guidance=self.should_invoke_guidance(
                        None if node == "__full__" else node
                    ),
                )
            )
        return policies


__all__ = [
    "NodeGuidancePolicy",
    "SelectiveGuidanceBandit",
    "audit_trajectories_to_contrastive_context",
    "evolve_with_store",
    "propose_candidate_from_audit",
]
