"""Harness decision audit protocol, trajectory models, and RL transition exporters."""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .models import HarnessEvent, HarnessEventType, utcnow

if TYPE_CHECKING:
    from hyperforge.procedural.graph import ProceduralGraph

DecisionType = Literal[
    "procedural_guidance",
    "llm_action",
    "tool_call",
    "skill_load",
    "feedback",
    "codemode_step",
]
OutcomeStatus = Literal["running", "completed", "failed", "interrupted"]
GraphScope = Literal["local", "full", "none"]


class AuditTraversedEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    relation: str


class AuditDecisionRecord(BaseModel):
    """Single step-level decision inside an audited `AgentHarness` episode."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    episode_id: str
    account: str = "default"
    agent_id: str = "default"
    workflow_id: str = "default"
    step_index: int = Field(ge=0)
    decision_id: str = ""
    decision_type: DecisionType

    # Procedural Graph state at decision step t
    active_node: str | None = None
    graph_scope: GraphScope = "none"
    candidate_node_ids: list[str] = Field(default_factory=list)
    traversed_edge: AuditTraversedEdge | None = None
    guidance_text: str | None = None
    guidance_status: Literal["completed", "failed", "skipped"] | None = None

    # Action & observation
    action_name: str | None = None
    action_arguments: dict[str, Any] | None = None
    observation_summary: str | None = None
    action_status: Literal["completed", "failed", "rejected"] = "completed"
    error: str | None = None

    # RL & cost signals
    step_reward: float | None = None
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    latency_ms: float = 0.0
    decision_cache_key: str | None = None
    trace_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)


class AuditEpisode(BaseModel):
    """Turn-level episode header linking workflow, Procedural Graph version, and reward."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    account: str = "default"
    agent_id: str = "default"
    workflow_id: str = "default"
    conversation_id: str
    turn_id: str
    execution_mode: Literal["loop", "code"] = "loop"
    graph_version_id: str | None = None
    graph_fingerprint: str | None = None
    model: str = ""
    query: str = ""
    final_output: str | None = None
    outcome_status: OutcomeStatus = "running"
    total_steps: int = 0
    total_tool_calls: int = 0
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    latency_ms: float = 0.0
    reward: float | None = None
    reward_source: str | None = None
    reward_breakdown: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None


class AuditTrajectory(BaseModel):
    """Complete episode paired with its ordered step-level decisions."""

    model_config = ConfigDict(extra="forbid")

    episode: AuditEpisode
    decisions: list[AuditDecisionRecord] = Field(default_factory=list)


class RLTransitionTuple(BaseModel):
    """Standardized `(s_t, g_t, a_t, o_t, r_t, G_t, s_{t+1}, done)` tuple for offline RL."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    step_index: int
    workflow_id: str
    graph_fingerprint: str | None
    query: str
    prior_actions: tuple[str, ...]
    active_node: str | None
    graph_scope: GraphScope
    candidate_node_ids: tuple[str, ...]
    guidance_invoked: bool
    guidance_text: str | None
    decision_type: DecisionType
    action_name: str | None
    action_arguments: dict[str, Any] | None
    traversed_edge: AuditTraversedEdge | None
    observation: str | None
    action_status: str
    step_reward: float
    episode_reward: float | None
    return_to_go: float
    tokens_used: float
    latency_ms: float
    next_active_node: str | None
    done: bool


class EdgeEmpiricalMetrics(BaseModel):
    """Empirical reward, reliability, and token-cost statistics per Procedural Graph edge."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    relation: str
    traversal_count: int
    episode_count: int
    mean_episode_reward: float | None
    mean_step_reward: float
    tool_failure_rate: float
    mean_tokens: float
    mean_latency_ms: float


@runtime_checkable
class HarnessAuditProtocol(Protocol):
    """Pluggable sink and query interface for `AgentHarness` decision auditing."""

    async def start_episode(self, episode: AuditEpisode) -> str: ...

    async def record_decision(self, decision: AuditDecisionRecord) -> None: ...

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
    ) -> None: ...

    async def attach_reward(
        self,
        episode_id: str,
        *,
        reward: float,
        source: str = "external",
        breakdown: dict[str, Any] | None = None,
        step_rewards: Mapping[int, float] | None = None,
    ) -> None: ...

    async def get_episode(self, episode_id: str) -> AuditTrajectory | None: ...

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
    ) -> list[AuditTrajectory]: ...


class NoopAuditSink:
    """Zero-overhead default audit sink when auditing is not configured."""

    async def start_episode(self, episode: AuditEpisode) -> str:
        return episode.id

    async def record_decision(self, decision: AuditDecisionRecord) -> None:
        return None

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
        return None

    async def attach_reward(
        self,
        episode_id: str,
        *,
        reward: float,
        source: str = "external",
        breakdown: dict[str, Any] | None = None,
        step_rewards: Mapping[int, float] | None = None,
    ) -> None:
        return None

    async def get_episode(self, episode_id: str) -> AuditTrajectory | None:
        del episode_id
        return None

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
        return []


class InMemoryAuditStore:
    """In-memory `HarnessAuditProtocol` implementation for standalone mode and tests."""

    def __init__(self) -> None:
        self._episodes: dict[str, AuditEpisode] = {}
        self._decisions: dict[str, list[AuditDecisionRecord]] = defaultdict(list)

    async def start_episode(self, episode: AuditEpisode) -> str:
        stored = episode.model_copy(deep=True)
        self._episodes[stored.id] = stored
        self._decisions.setdefault(stored.id, [])
        return stored.id

    async def record_decision(self, decision: AuditDecisionRecord) -> None:
        if decision.episode_id not in self._episodes:
            raise ValueError(f"Unknown audit episode: {decision.episode_id}")
        self._decisions[decision.episode_id].append(decision.model_copy(deep=True))

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
        ep = self._episodes.get(episode_id)
        if ep is None:
            raise ValueError(f"Unknown audit episode: {episode_id}")
        ep.outcome_status = outcome_status
        ep.final_output = final_output
        ep.total_steps = total_steps
        ep.total_tool_calls = total_tool_calls
        ep.input_tokens = input_tokens
        ep.output_tokens = output_tokens
        ep.latency_ms = latency_ms
        ep.error = error
        if reward is not None:
            ep.reward = reward
            ep.reward_source = reward_source
            if reward_breakdown is not None:
                ep.reward_breakdown = dict(reward_breakdown)
        ep.completed_at = utcnow()

    async def attach_reward(
        self,
        episode_id: str,
        *,
        reward: float,
        source: str = "external",
        breakdown: dict[str, Any] | None = None,
        step_rewards: Mapping[int, float] | None = None,
    ) -> None:
        ep = self._episodes.get(episode_id)
        if ep is None:
            raise ValueError(f"Unknown audit episode: {episode_id}")
        ep.reward = reward
        ep.reward_source = source
        if breakdown is not None:
            ep.reward_breakdown = dict(breakdown)
        if step_rewards:
            for dec in self._decisions.get(episode_id, ()):
                if dec.step_index in step_rewards:
                    dec.step_reward = float(step_rewards[dec.step_index])

    async def get_episode(self, episode_id: str) -> AuditTrajectory | None:
        ep = self._episodes.get(episode_id)
        if ep is None:
            return None
        decs = sorted(
            (d.model_copy(deep=True) for d in self._decisions.get(episode_id, ())),
            key=lambda item: item.step_index,
        )
        return AuditTrajectory(episode=ep.model_copy(deep=True), decisions=decs)

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
        matched: list[AuditEpisode] = []
        for ep in self._episodes.values():
            if account is not None and ep.account != account:
                continue
            if agent_id is not None and ep.agent_id != agent_id:
                continue
            if workflow_id is not None and ep.workflow_id != workflow_id:
                continue
            if (
                graph_fingerprint is not None
                and ep.graph_fingerprint != graph_fingerprint
            ):
                continue
            if outcome_status is not None and ep.outcome_status != outcome_status:
                continue
            if min_reward is not None and (ep.reward is None or ep.reward < min_reward):
                continue
            if max_reward is not None and (ep.reward is None or ep.reward > max_reward):
                continue
            matched.append(ep)
        matched.sort(key=lambda item: item.created_at, reverse=True)
        sliced = matched[max(0, offset) : max(0, offset) + max(1, limit)]
        results: list[AuditTrajectory] = []
        for ep in sliced:
            traj = await self.get_episode(ep.id)
            if traj is not None:
                results.append(traj)
        return results


def trajectories_to_rl_transitions(
    trajectories: Sequence[AuditTrajectory],
    *,
    gamma: float = 0.99,
) -> list[RLTransitionTuple]:
    """Convert audited trajectories into discounted RL transition tuples."""
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0.0, 1.0]")
    transitions: list[RLTransitionTuple] = []
    for traj in trajectories:
        ep = traj.episode
        actionable = [
            dec
            for dec in sorted(traj.decisions, key=lambda d: d.step_index)
            if dec.decision_type != "procedural_guidance"
        ]
        if not actionable:
            continue

        n = len(actionable)
        raw_rewards = [float(dec.step_reward or 0.0) for dec in actionable]
        if ep.reward is not None:
            raw_rewards[-1] += float(ep.reward)

        returns = [0.0] * n
        running = 0.0
        for idx in range(n - 1, -1, -1):
            running = raw_rewards[idx] + gamma * running
            returns[idx] = running

        prior_actions: list[str] = []
        for idx, dec in enumerate(actionable):
            is_last = idx == n - 1
            next_node = actionable[idx + 1].active_node if not is_last else None
            transitions.append(
                RLTransitionTuple(
                    episode_id=ep.id,
                    step_index=dec.step_index,
                    workflow_id=ep.workflow_id,
                    graph_fingerprint=ep.graph_fingerprint,
                    query=ep.query,
                    prior_actions=tuple(prior_actions),
                    active_node=dec.active_node,
                    graph_scope=dec.graph_scope,
                    candidate_node_ids=tuple(dec.candidate_node_ids),
                    guidance_invoked=dec.guidance_status == "completed",
                    guidance_text=dec.guidance_text,
                    decision_type=dec.decision_type,
                    action_name=dec.action_name,
                    action_arguments=dec.action_arguments,
                    traversed_edge=dec.traversed_edge,
                    observation=dec.observation_summary,
                    action_status=dec.action_status,
                    step_reward=float(dec.step_reward or 0.0),
                    episode_reward=ep.reward,
                    return_to_go=round(returns[idx], 6),
                    tokens_used=dec.input_tokens + dec.output_tokens,
                    latency_ms=dec.latency_ms,
                    next_active_node=next_node,
                    done=is_last,
                )
            )
            if dec.action_name:
                prior_actions.append(dec.action_name)
    return transitions


def export_rl_transitions_jsonl(
    trajectories: Sequence[AuditTrajectory],
    *,
    gamma: float = 0.99,
) -> str:
    """Serialize RL transition tuples as newline-delimited JSON (JSONL)."""
    tuples = trajectories_to_rl_transitions(trajectories, gamma=gamma)
    return "".join(item.model_dump_json() + "\n" for item in tuples)


def compute_edge_empirical_metrics(
    trajectories: Sequence[AuditTrajectory],
) -> list[EdgeEmpiricalMetrics]:
    """Aggregate empirical reward, error rate, and token cost per traversed graph edge."""
    buckets: dict[tuple[str, str, str], dict[str, Any]] = {}
    for traj in trajectories:
        ep = traj.episode
        seen_in_ep: set[tuple[str, str, str]] = set()
        for dec in traj.decisions:
            edge = dec.traversed_edge
            if edge is None:
                continue
            key = (edge.source, edge.target, edge.relation)
            bucket = buckets.setdefault(
                key,
                {
                    "count": 0,
                    "episodes": 0,
                    "ep_rewards": [],
                    "step_rewards": [],
                    "failures": 0,
                    "tokens": [],
                    "latencies": [],
                },
            )
            bucket["count"] += 1
            bucket["step_rewards"].append(float(dec.step_reward or 0.0))
            if dec.action_status == "failed":
                bucket["failures"] += 1
            bucket["tokens"].append(dec.input_tokens + dec.output_tokens)
            bucket["latencies"].append(dec.latency_ms)
            if key not in seen_in_ep:
                seen_in_ep.add(key)
                bucket["episodes"] += 1
                if ep.reward is not None:
                    bucket["ep_rewards"].append(float(ep.reward))

    metrics: list[EdgeEmpiricalMetrics] = []
    for (src, dst, rel), b in sorted(buckets.items()):
        cnt = b["count"]
        ep_rews = b["ep_rewards"]
        metrics.append(
            EdgeEmpiricalMetrics(
                source=src,
                target=dst,
                relation=rel,
                traversal_count=cnt,
                episode_count=b["episodes"],
                mean_episode_reward=(
                    round(sum(ep_rews) / len(ep_rews), 6) if ep_rews else None
                ),
                mean_step_reward=round(sum(b["step_rewards"]) / cnt, 6),
                tool_failure_rate=round(b["failures"] / cnt, 6),
                mean_tokens=round(sum(b["tokens"]) / cnt, 2),
                mean_latency_ms=round(sum(b["latencies"]) / cnt, 2),
            )
        )
    return metrics


def _resolve_edge(
    graph: ProceduralGraph | None, source_node: str | None, action_name: str | None
) -> AuditTraversedEdge | None:
    if graph is None or not source_node or not action_name:
        return None
    target_node = graph.locate(action_name)
    if target_node is None:
        return None
    for edge in graph.edges:
        if edge.source == source_node and edge.target == target_node:
            return AuditTraversedEdge(
                source=edge.source, target=edge.target, relation=edge.relation
            )
    return None


def _truncate_text(raw: Any, max_chars: int = 4000) -> str:
    if raw is None:
        return ""
    text = (
        raw
        if isinstance(raw, str)
        else json.dumps(raw, default=str, ensure_ascii=False)
    )
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


class HarnessAuditObserver:
    """Translates live `AgentHarness` events into `HarnessAuditProtocol` records."""

    def __init__(
        self,
        sink: HarnessAuditProtocol,
        *,
        account: str = "default",
        agent_id: str = "default",
        workflow_id: str = "default",
        graph: ProceduralGraph | None = None,
        graph_version_id: str | None = None,
        max_observation_chars: int = 4000,
    ) -> None:
        self.sink = sink
        self.account = account
        self.agent_id = agent_id
        self.workflow_id = workflow_id
        self.graph = graph
        self.graph_version_id = graph_version_id
        self.max_observation_chars = max_observation_chars

        self._episode_id: str | None = None
        self._last_episode_id: str | None = None
        self._turn_started_mono: float = 0.0
        self._step_index: int = 0
        self._current_node: str | None = "Start" if graph is not None else None
        self._pending_guidance: dict[str, dict[str, Any]] = {}
        self._pending_tools: dict[str, tuple[dict[str, Any], float, str | None]] = {}
        self._llm_starts: dict[str, float] = {}

    @property
    def current_episode_id(self) -> str | None:
        return self._episode_id

    @property
    def last_episode_id(self) -> str | None:
        return self._last_episode_id

    async def on_event(
        self,
        event: HarnessEvent,
        *,
        execution_mode: Literal["loop", "code"] = "loop",
        model: str = "",
        available_tools: Sequence[str] = (),
        skills_loaded: Sequence[str] = (),
        usage: Any = None,
    ) -> None:
        etype = event.type
        if etype in {HarnessEventType.TEXT_DELTA, HarnessEventType.REASONING_DELTA}:
            return
        if event.parent_agent_id is not None:
            return

        payload = event.payload
        if etype == HarnessEventType.TURN_STARTED:
            self._turn_started_mono = time.monotonic()
            self._step_index = 0
            self._current_node = "Start" if self.graph is not None else None
            self._pending_guidance.clear()
            self._pending_tools.clear()
            self._llm_starts.clear()
            ep = AuditEpisode(
                account=self.account,
                agent_id=self.agent_id,
                workflow_id=self.workflow_id,
                conversation_id=event.conversation_id,
                turn_id=event.turn_id or "",
                execution_mode=execution_mode,
                graph_version_id=self.graph_version_id,
                graph_fingerprint=self.graph.fingerprint if self.graph else None,
                model=model,
                query=str(payload.get("prompt", "")),
                metadata={
                    "available_tools": list(available_tools),
                    "skills_loaded": list(skills_loaded),
                    "tags": list(event.tags),
                },
            )
            self._episode_id = await self.sink.start_episode(ep)
            self._last_episode_id = self._episode_id
            return

        if self._episode_id is None:
            return

        if etype == HarnessEventType.PROCEDURAL_GUIDANCE:
            dec_id = str(payload.get("decision_id", ""))
            active_node = payload.get("active_node")
            raw_scope = payload.get("scope")
            scope: GraphScope = raw_scope if raw_scope in {"local", "full"} else "none"
            status = "failed" if payload.get("status") == "failed" else "completed"
            g_info = {
                "active_node": active_node,
                "scope": scope,
                "node_ids": [str(n) for n in (payload.get("node_ids") or [])],
                "text": str(payload.get("text", "")) or None,
                "status": status,
                "input_tokens": float(payload.get("input_tokens", 0.0)),
                "output_tokens": float(payload.get("output_tokens", 0.0)),
                "latency_ms": float(payload.get("latency_seconds", 0.0)) * 1000.0,
                "trace_id": payload.get("trace_id"),
                "error": payload.get("error"),
            }
            if dec_id:
                self._pending_guidance[dec_id] = g_info
            if isinstance(active_node, str):
                self._current_node = active_node
            await self._emit_decision(
                decision_id=dec_id,
                decision_type="procedural_guidance",
                active_node=active_node if isinstance(active_node, str) else None,
                graph_scope=scope,
                candidate_node_ids=g_info["node_ids"],
                guidance_text=g_info["text"],
                guidance_status=status,
                action_name="generate_guidance",
                action_status="failed" if status == "failed" else "completed",
                error=g_info["error"],
                input_tokens=g_info["input_tokens"],
                output_tokens=g_info["output_tokens"],
                latency_ms=g_info["latency_ms"],
                trace_id=g_info["trace_id"],
            )
            return

        if etype == HarnessEventType.LLM_STARTED:
            call_id = str(payload.get("call_id", ""))
            if call_id:
                self._llm_starts[call_id] = time.monotonic()
            return

        if etype in {HarnessEventType.LLM_COMPLETED, HarnessEventType.LLM_FAILED}:
            call_id = str(payload.get("call_id", ""))
            started = self._llm_starts.pop(call_id, None)
            elapsed_ms = (
                (time.monotonic() - started) * 1000.0 if started is not None else 0.0
            )
            g_ctx = self._pending_guidance.pop(call_id, None)
            dc = payload.get("decision_cache")
            cache_key = (
                str(dc.get("key"))
                if isinstance(dc, dict) and dc.get("key") is not None
                else None
            )
            failed = etype == HarnessEventType.LLM_FAILED
            await self._emit_decision(
                decision_id=call_id,
                decision_type="llm_action",
                active_node=(g_ctx["active_node"] if g_ctx else self._current_node),
                graph_scope=g_ctx["scope"] if g_ctx else "none",
                candidate_node_ids=g_ctx["node_ids"] if g_ctx else [],
                guidance_text=g_ctx["text"] if g_ctx else None,
                guidance_status=g_ctx["status"] if g_ctx else "skipped",
                action_name="llm_generate",
                observation_summary=_truncate_text(
                    payload.get("text"), self.max_observation_chars
                )
                or None,
                action_status="failed" if failed else "completed",
                error=str(payload["error"]) if payload.get("error") else None,
                step_reward=-0.2 if failed else 0.0,
                input_tokens=float(payload.get("input_tokens", 0.0)),
                output_tokens=float(payload.get("output_tokens", 0.0)),
                latency_ms=elapsed_ms,
                decision_cache_key=cache_key,
                trace_id=(
                    str(payload["trace_id"]) if payload.get("trace_id") else None
                ),
            )
            return

        if etype == HarnessEventType.TOOL_REQUESTED:
            call = payload.get("call")
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                self._pending_tools[call["id"]] = (
                    call,
                    time.monotonic(),
                    self._current_node,
                )
            return

        if etype in {HarnessEventType.TOOL_COMPLETED, HarnessEventType.TOOL_FAILED}:
            if payload.get("code_workflow"):
                return
            call_id = str(payload.get("call_id", ""))
            tool_name = str(payload.get("tool", "tool"))
            req_entry = self._pending_tools.pop(call_id, None)
            args_dict = (
                dict(req_entry[0].get("arguments") or {})
                if req_entry is not None
                else None
            )
            elapsed_ms = (
                (time.monotonic() - req_entry[1]) * 1000.0
                if req_entry is not None
                else float(payload.get("duration_ms", 0.0))
            )
            source_node = req_entry[2] if req_entry is not None else self._current_node
            traversed = _resolve_edge(self.graph, source_node, tool_name)
            if self.graph is not None:
                located = self.graph.locate(tool_name)
                if located is not None:
                    self._current_node = located
            failed = etype == HarnessEventType.TOOL_FAILED
            raw_res = payload.get("result")
            err_str = (
                str(raw_res.get("error"))
                if failed and isinstance(raw_res, dict) and raw_res.get("error")
                else ("Tool execution failed" if failed else None)
            )
            dtype: DecisionType = (
                "codemode_step" if payload.get("nested") else "tool_call"
            )
            await self._emit_decision(
                decision_id=call_id,
                decision_type=dtype,
                active_node=source_node,
                graph_scope="local" if source_node and self.graph else "none",
                traversed_edge=traversed,
                action_name=tool_name,
                action_arguments=args_dict,
                observation_summary=_truncate_text(raw_res, self.max_observation_chars),
                action_status="failed" if failed else "completed",
                error=err_str,
                step_reward=-0.1 if failed else 0.0,
                latency_ms=elapsed_ms,
            )
            return

        if etype == HarnessEventType.SKILLS_LOADED:
            ids = [str(x) for x in (payload.get("ids") or [])]
            await self._emit_decision(
                decision_id=event.id,
                decision_type="skill_load",
                active_node=self._current_node,
                action_name="load_skill",
                action_arguments={"ids": ids},
                observation_summary=", ".join(ids),
                step_reward=0.0,
            )
            return

        if etype == HarnessEventType.FEEDBACK_RESOLVED:
            await self._emit_decision(
                decision_id=str(payload.get("request_id", event.id)),
                decision_type="feedback",
                active_node=self._current_node,
                action_name="request_feedback",
                observation_summary=_truncate_text(
                    payload.get("response"), self.max_observation_chars
                ),
                step_reward=0.0,
            )
            return

        if etype in {
            HarnessEventType.TURN_COMPLETED,
            HarnessEventType.TURN_FAILED,
            HarnessEventType.TURN_INTERRUPTED,
        }:
            ep_id = self._episode_id
            self._episode_id = None
            elapsed_ms = (time.monotonic() - self._turn_started_mono) * 1000.0
            status_map: dict[
                HarnessEventType, Literal["completed", "failed", "interrupted"]
            ] = {
                HarnessEventType.TURN_COMPLETED: "completed",
                HarnessEventType.TURN_FAILED: "failed",
                HarnessEventType.TURN_INTERRUPTED: "interrupted",
            }
            auto_reward: float | None = None
            auto_source: str | None = None
            auto_breakdown: dict[str, Any] | None = None
            meta = payload.get("metadata")
            if isinstance(meta, dict) and isinstance(meta.get("remi"), dict):
                remi = meta["remi"]
                rel = remi.get("answer_relevance")
                gnd = remi.get("groundedness")
                parts = [
                    float(v) / 5.0 for v in (rel, gnd) if isinstance(v, (int, float))
                ]
                if parts:
                    auto_reward = round(sum(parts) / len(parts), 6)
                    auto_source = "remi"
                    auto_breakdown = {
                        "answer_relevance": rel,
                        "groundedness": gnd,
                    }

            await self.sink.complete_episode(
                ep_id,
                outcome_status=status_map[etype],
                final_output=(
                    str(payload["text"]) if payload.get("text") is not None else None
                ),
                total_steps=self._step_index,
                total_tool_calls=int(getattr(usage, "tool_calls", 0)),
                input_tokens=float(getattr(usage, "input_tokens", 0.0)),
                output_tokens=float(getattr(usage, "output_tokens", 0.0)),
                latency_ms=elapsed_ms,
                error=str(payload["error"]) if payload.get("error") else None,
                reward=auto_reward,
                reward_source=auto_source,
                reward_breakdown=auto_breakdown,
            )

    async def _emit_decision(self, **kwargs: Any) -> None:
        if self._episode_id is None:
            return
        idx = self._step_index
        self._step_index += 1
        rec = AuditDecisionRecord(
            episode_id=self._episode_id,
            account=self.account,
            agent_id=self.agent_id,
            workflow_id=self.workflow_id,
            step_index=idx,
            **kwargs,
        )
        await self.sink.record_decision(rec)


__all__ = [
    "AuditDecisionRecord",
    "AuditEpisode",
    "AuditTrajectory",
    "AuditTraversedEdge",
    "DecisionType",
    "EdgeEmpiricalMetrics",
    "GraphScope",
    "HarnessAuditObserver",
    "HarnessAuditProtocol",
    "InMemoryAuditStore",
    "NoopAuditSink",
    "OutcomeStatus",
    "RLTransitionTuple",
    "compute_edge_empirical_metrics",
    "export_rl_transitions_jsonl",
    "trajectories_to_rl_transitions",
]
