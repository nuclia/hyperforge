"""Tests for versioned Procedural Graph storage, Harness Decision Audit, RL export, and DB evolution."""

from __future__ import annotations

import io
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import alembic.command
import alembic.config
import pytest
import sqlalchemy as sa
from fastapi import FastAPI, UploadFile
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from starlette.authentication import AuthCredentials
from starlette.middleware.authentication import AuthenticationMiddleware

import hyperforge
from hyperforge.api.authentication import User
from hyperforge.api.models import StashRoles
from hyperforge.api.v1.router import router as v1_router
from hyperforge.db.agents import AgentManager
from hyperforge.db.procedural_storage import (
    PostgresProceduralGraphStore,
)
from hyperforge.db.settings import DataManagerSettings
from hyperforge.harness_sdk import (
    AgentHarness,
    HarnessToolCall,
    InMemoryAuditStore,
    ModelDelta,
    ToolCallContext,
    compute_edge_empirical_metrics,
    export_rl_transitions_jsonl,
    tool,
    trajectories_to_rl_transitions,
)
from hyperforge.models import MemoryConfig, Rules
from hyperforge.procedural import (
    EdgeEndpoints,
    EvolutionSettings,
    GraphEdits,
    GuidanceRuntimeSettings,
    InMemoryProceduralGraphStore,
    ProceduralGraph,
    ProceduralGuidanceConfig,
    ProcedureEdge,
    ProcedureNode,
    SelectiveGuidanceBandit,
    diff_procedural_graphs,
    evolve_with_store,
    propose_candidate_from_audit,
)
from hyperforge.procedural.evolution import EvaluationTask, ScoredRollout

# ---------------------------------------------------------------------------
# Lightweight SQLite async DB adapter for Docker-free SQL integration tests
# ---------------------------------------------------------------------------


class _RowMapping:
    def __init__(self, data: dict[str, Any]) -> None:
        self._mapping = data

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return list(self._mapping.values())[key]
        return self._mapping[key]


class _SQLiteAsyncDatabase:
    def __init__(self, engine: sa.Engine) -> None:
        self.engine = engine
        self._conn = engine.connect()

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        self._conn.close()

    @asynccontextmanager
    async def transaction(self):
        yield

    def _compile_stmt(self, query: Any) -> tuple[str, dict[str, Any]]:
        if isinstance(query, str):
            return query, {}
        named_dialect = sa.dialects.sqlite.dialect(paramstyle="named")
        compiled = query.compile(
            dialect=named_dialect, compile_kwargs={"render_postcompile": True}
        )
        params: dict[str, Any] = {}
        for k, v in (compiled.params or {}).items():
            if isinstance(v, (dict, list)):
                params[k] = json.dumps(v)
            elif hasattr(v, "isoformat"):
                params[k] = v.isoformat()
            else:
                params[k] = v
        return str(compiled), params

    async def execute(self, query: Any, values: dict[str, Any] | None = None) -> Any:
        if (
            isinstance(query, sa.sql.Insert)
            and "id" not in query.compile().params
            and "id" in query.table.c
            and isinstance(query.table.c.id.type, sa.dialects.postgresql.UUID)
        ):
            gen_id = str(uuid4())
            query = query.values(id=gen_id)
            sql, params = self._compile_stmt(query)
            self._conn.execute(text(sql), params)
            self._conn.commit()
            return gen_id
        sql, params = self._compile_stmt(query)
        if values:
            params.update(
                {
                    k: (
                        json.dumps(v)
                        if isinstance(v, (dict, list))
                        else v.isoformat()
                        if hasattr(v, "isoformat")
                        else v
                    )
                    for k, v in values.items()
                }
            )
        self._conn.execute(text(sql), params)
        self._conn.commit()
        return None

    async def fetch_one(
        self, query: Any, values: dict[str, Any] | None = None
    ) -> _RowMapping | None:
        rows = await self.fetch_all(query, values)
        return rows[0] if rows else None

    async def fetch_all(
        self, query: Any, values: dict[str, Any] | None = None
    ) -> list[_RowMapping]:
        sql, params = self._compile_stmt(query)
        if values:
            params.update(values)
        res = self._conn.execute(text(sql), params)
        keys = list(res.keys())
        return [_RowMapping(dict(zip(keys, row))) for row in res.fetchall()]


def _init_sqlite_tables(conn: sa.Connection) -> None:
    conn.connection.executescript(
        """
        CREATE TABLE retrieval_agent_config (
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            rules TEXT NOT NULL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified TIMESTAMP,
            memory TEXT NOT NULL,
            description TEXT,
            title TEXT,
            instructions TEXT,
            PRIMARY KEY (account, agent_id)
        );
        CREATE TABLE retrieval_agent_workflow (
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT,
            parameters TEXT,
            required TEXT,
            rules TEXT,
            execution_mode TEXT NOT NULL DEFAULT 'loop',
            harness_config TEXT NOT NULL DEFAULT '{}',
            workflow_code TEXT,
            legacy_stages_snapshot TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified TIMESTAMP,
            is_deleted BOOLEAN NOT NULL DEFAULT 0,
            deleted_by TEXT,
            deleted_at TIMESTAMP,
            PRIMARY KEY (account, agent_id, workflow_id)
        );
        CREATE TABLE agent_workflow_toolkits (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            namespace TEXT NOT NULL,
            toolkit TEXT NOT NULL,
            title TEXT,
            lazy_load BOOLEAN NOT NULL DEFAULT 0,
            ingest_skills BOOLEAN NOT NULL DEFAULT 1,
            config TEXT NOT NULL DEFAULT '{}',
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified TIMESTAMP
        );
        CREATE TABLE agent_skills (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            instructions TEXT NOT NULL,
            recommended_tools TEXT NOT NULL DEFAULT '[]',
            code_examples TEXT NOT NULL DEFAULT '[]',
            preload BOOLEAN NOT NULL DEFAULT 0,
            meta TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified TIMESTAMP
        );
        CREATE TABLE retrieval_agent_prompts (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            name TEXT NOT NULL,
            description TEXT NOT NULL,
            prompt TEXT NOT NULL,
            arguments TEXT,
            icons TEXT,
            meta TEXT,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified TIMESTAMP
        );
        CREATE TABLE retrieval_agents_drivers (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            driver TEXT NOT NULL,
            provider TEXT NOT NULL,
            identifier TEXT NOT NULL,
            config TEXT NOT NULL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            modified TIMESTAMP
        );
        CREATE TABLE procedural_graphs (
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            active_version_id TEXT,
            guidance_config TEXT NOT NULL DEFAULT '{}',
            evolution_config TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            modified_at TEXT,
            PRIMARY KEY (account, agent_id, workflow_id)
        );
        CREATE TABLE procedural_graph_versions (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            fingerprint TEXT NOT NULL,
            parent_version_id TEXT,
            graph TEXT NOT NULL,
            edits_from_parent TEXT,
            status TEXT NOT NULL,
            source TEXT NOT NULL,
            validation_score REAL,
            validation_metrics TEXT NOT NULL DEFAULT '{}',
            diagnostics TEXT,
            evolution_run_id TEXT,
            created_by TEXT,
            comment TEXT,
            created_at TEXT NOT NULL,
            UNIQUE (account, agent_id, workflow_id, version_number)
        );
        CREATE TABLE procedural_evolution_runs (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            base_version_id TEXT NOT NULL,
            retained_version_id TEXT,
            status TEXT NOT NULL,
            mode TEXT NOT NULL,
            provenance TEXT NOT NULL DEFAULT '{}',
            rounds_completed INTEGER NOT NULL DEFAULT 0,
            total_rounds INTEGER NOT NULL DEFAULT 0,
            started_at TEXT NOT NULL,
            completed_at TEXT,
            error TEXT
        );
        CREATE TABLE harness_audit_episodes (
            id TEXT PRIMARY KEY,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            conversation_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            execution_mode TEXT NOT NULL,
            graph_version_id TEXT,
            graph_fingerprint TEXT,
            model TEXT NOT NULL,
            query TEXT NOT NULL,
            final_output TEXT,
            outcome_status TEXT NOT NULL,
            total_steps INTEGER NOT NULL DEFAULT 0,
            total_tool_calls INTEGER NOT NULL DEFAULT 0,
            input_tokens REAL NOT NULL DEFAULT 0,
            output_tokens REAL NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0,
            reward REAL,
            reward_source TEXT,
            reward_breakdown TEXT NOT NULL DEFAULT '{}',
            metadata TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT
        );
        CREATE TABLE harness_audit_decisions (
            id TEXT PRIMARY KEY,
            episode_id TEXT NOT NULL,
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            decision_id TEXT NOT NULL,
            decision_type TEXT NOT NULL,
            active_node TEXT,
            graph_scope TEXT NOT NULL,
            candidate_node_ids TEXT NOT NULL DEFAULT '[]',
            traversed_edge TEXT,
            guidance_text TEXT,
            guidance_status TEXT,
            action_name TEXT,
            action_arguments TEXT,
            observation_summary TEXT,
            action_status TEXT NOT NULL,
            error TEXT,
            step_reward REAL,
            input_tokens REAL NOT NULL DEFAULT 0,
            output_tokens REAL NOT NULL DEFAULT 0,
            latency_ms REAL NOT NULL DEFAULT 0,
            decision_cache_key TEXT,
            trace_id TEXT,
            created_at TEXT NOT NULL
        );
        """
    )


def _sample_two_tool_graph() -> ProceduralGraph:
    return ProceduralGraph(
        nodes=(
            ProcedureNode(id="Start", type="STATUS"),
            ProcedureNode(id="search_kb", type="ACTION", description="Search KB"),
            ProcedureNode(id="verify_facts", type="ACTION", description="Verify facts"),
            ProcedureNode(id="End", type="STATUS"),
        ),
        edges=(
            ProcedureEdge(
                source="Start",
                target="search_kb",
                relation="LEADS_TO",
                guidance="Search the KB for primary evidence first.",
                pitfalls="Do not answer before searching.",
            ),
            ProcedureEdge(
                source="search_kb",
                target="verify_facts",
                relation="PROVIDES_INPUT_FOR",
                condition="Evidence retrieved",
                guidance="Verify dates and numbers before responding.",
            ),
            ProcedureEdge(
                source="verify_facts",
                target="End",
                relation="CONVERGES_TO",
                guidance="Synthesize final verified answer.",
            ),
        ),
    )


# ---------------------------------------------------------------------------
# 1. Alembic PostgreSQL Offline SQL Verification
# ---------------------------------------------------------------------------


def test_alembic_procedural_and_audit_ddl_upgrade_and_downgrade(capsys):
    ini_path = Path(hyperforge.__file__).parent / "alembic.ini"
    cfg_up = alembic.config.Config(str(ini_path))
    cfg_up.set_main_option(
        "sqlalchemy.url", "postgresql://postgres:postgres@localhost:5432/test_db"
    )
    alembic.command.upgrade(cfg_up, "c4e8a91b2d30:d7f31b8e4c21", sql=True)
    up_sql = capsys.readouterr().out
    assert "CREATE TABLE procedural_graphs" in up_sql
    assert "CREATE TABLE procedural_graph_versions" in up_sql
    assert "CREATE TABLE procedural_evolution_runs" in up_sql
    assert "CREATE TABLE harness_audit_episodes" in up_sql
    assert "CREATE TABLE harness_audit_decisions" in up_sql

    cfg_down = alembic.config.Config(str(ini_path))
    cfg_down.set_main_option(
        "sqlalchemy.url", "postgresql://postgres:postgres@localhost:5432/test_db"
    )
    alembic.command.downgrade(cfg_down, "d7f31b8e4c21:c4e8a91b2d30", sql=True)
    down_sql = capsys.readouterr().out
    assert "DROP TABLE harness_audit_decisions" in down_sql
    assert "DROP TABLE harness_audit_episodes" in down_sql
    assert "DROP TABLE procedural_evolution_runs" in down_sql
    assert "DROP TABLE procedural_graph_versions" in down_sql
    assert "DROP TABLE procedural_graphs" in down_sql


# ---------------------------------------------------------------------------
# 2. Versioned Procedural Graph Store (Lineage, Diff, Rejection Memory, Export)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "postgres_sqlite"])
async def test_procedural_graph_versioning_lineage_diff_and_rollback(backend: str):
    db: _SQLiteAsyncDatabase | None = None
    if backend == "memory":
        store = InMemoryProceduralGraphStore()
    else:
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            _init_sqlite_tables(conn)
            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_config (account, agent_id, rules, memory) "
                    "VALUES ('acc', 'ag', '{\"rules\": []}', '{}')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_workflow (account, agent_id, workflow_id, name) "
                    "VALUES ('acc', 'ag', 'default', 'Default')"
                )
            )
        db = _SQLiteAsyncDatabase(engine)
        store = PostgresProceduralGraphStore(db)  # type: ignore[arg-type]

    try:
        v1 = await store.create_version(
            "acc",
            "ag",
            "default",
            ProceduralGraph.skeleton(),
            source="skeleton",
            promote=True,
            comment="Initial skeleton",
        )
        assert v1.version_number == 1
        assert v1.status == "active"

        edits = GraphEdits(
            delete_edges=(EdgeEndpoints(source="Start", target="End"),),
            add_nodes=(
                ProcedureNode(id="search_kb", type="ACTION", description="Search"),
            ),
            add_edges=(
                ProcedureEdge(
                    source="Start",
                    target="search_kb",
                    guidance="Call search_kb first.",
                ),
                ProcedureEdge(
                    source="search_kb",
                    target="End",
                    guidance="Summarize search results.",
                ),
            ),
        )
        v2 = await store.apply_edits(
            "acc",
            "ag",
            "default",
            parent_version_id=v1.id,
            edits=edits,
            available_tools=["search_kb"],
            promote=True,
        )
        assert v2.version_number == 2
        assert v2.status == "active"
        assert v2.parent_version_id == v1.id

        # v1 was automatically archived when v2 was promoted
        v1_reloaded = await store.get_version(v1.id)
        assert v1_reloaded is not None and v1_reloaded.status == "archived"

        # Structural diff between v1 and v2
        diff = diff_procedural_graphs(v1_reloaded, v2)
        assert [n["id"] for n in diff.added_nodes] == ["search_kb"]
        assert len(diff.added_edges) == 2
        assert len(diff.removed_edges) == 1

        # Record a rejected proposal into rejection memory
        rej = await store.record_rejected_proposal(
            "acc",
            "ag",
            "default",
            parent_version_id=v2.id,
            fallback_graph=v2.graph,
            raw_edits={"delete_nodes": ["Start"]},
            diagnostics="ValueError: Graph must contain Start",
        )
        assert rej.status == "rejected"
        rej_mem = await store.get_rejection_memory("acc", "ag", "default", limit=5)
        assert len(rej_mem) == 1
        assert rej_mem[0]["reason"] == "proposal_error"
        assert "Start" in rej_mem[0]["structural_diagnostic"]

        # Rollback creates a new linear version #4 pointing back to v1's graph
        v4 = await store.rollback_to_version("acc", "ag", "default", v1.id)
        assert v4.version_number == 4
        assert v4.source == "rollback"
        assert v4.status == "active"
        assert v4.fingerprint == v1.fingerprint

        history = await store.list_versions("acc", "ag", "default")
        assert [item.version_number for item in history] == [4, 3, 2, 1]
    finally:
        if db is not None:
            await db.disconnect()


# ---------------------------------------------------------------------------
# 3. Live AgentHarness Decision Audit + RL Transitions + Selective Bandit
# ---------------------------------------------------------------------------


class _QueryInput(BaseModel):
    q: str


class _ToolOut(BaseModel):
    result: str


class _ScriptedGuidedModelClient:
    """Deterministic model client handling both procedural guidance and solver turns."""

    def __init__(self) -> None:
        self._solver_step = 0

    async def stream(
        self,
        *,
        model: str,
        messages: list[Any],
        tools: list[Any],
        reasoning_effort: Any = None,
        response_format: Any = None,
        execution_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[ModelDelta]:
        del model, tools, reasoning_effort, response_format
        if (execution_context or {}).get("purpose") == "procedural_guidance":
            yield ModelDelta(
                text="Follow the graph: search_kb -> verify_facts -> answer.",
                input_tokens=40,
                output_tokens=15,
            )
            return

        self._solver_step += 1
        if self._solver_step == 1:
            yield ModelDelta(
                tool_calls=[
                    HarnessToolCall(
                        id="c1", name="search_kb", arguments={"q": "runway"}
                    )
                ],
                input_tokens=80,
                output_tokens=20,
            )
        elif self._solver_step == 2:
            yield ModelDelta(
                tool_calls=[
                    HarnessToolCall(
                        id="c2", name="verify_facts", arguments={"q": "18 months"}
                    )
                ],
                input_tokens=110,
                output_tokens=25,
            )
        else:
            yield ModelDelta(
                text="Verified runway is 18 months.",
                input_tokens=130,
                output_tokens=18,
            )


@pytest.mark.asyncio
async def test_live_harness_audits_guided_decisions_edges_and_exports_rl_dataset():
    @tool(name="search_kb", description="Search KB")
    async def search_kb(_ctx: ToolCallContext, val: _QueryInput) -> _ToolOut:
        return _ToolOut(result=f"Found doc for {val.q}: 18 months")

    @tool(name="verify_facts", description="Verify facts")
    async def verify_facts(_ctx: ToolCallContext, val: _QueryInput) -> _ToolOut:
        del val
        return _ToolOut(result="verified")

    audit_store = InMemoryAuditStore()
    graph = _sample_two_tool_graph()
    client = _ScriptedGuidedModelClient()

    harness = AgentHarness(
        model="test-model",
        model_client=client,  # type: ignore[arg-type]
        tools=[search_kb, verify_facts],
        procedural_guidance=ProceduralGuidanceConfig(graph=graph, hops=2, window=3),
        procedural_graph_version_id="ver-123",
        audit_sink=audit_store,
        execution_context={
            "conversation_metadata": {
                "account": "acc1",
                "agent_id": "cfo-agent",
                "workflow_id": "finance",
            }
        },
    )

    async with harness:
        events = [ev async for ev in harness.run("What is our cash runway?")]
    assert any(
        ev.payload.get("text") == "Verified runway is 18 months." for ev in events
    )

    ep_id = harness.last_audit_episode_id
    assert ep_id is not None

    await audit_store.attach_reward(
        ep_id,
        reward=1.0,
        source="evaluator",
        breakdown={"accuracy": 1.0},
        step_rewards={2: 0.1, 5: 0.2},
    )

    traj = await audit_store.get_episode(ep_id)
    assert traj is not None
    assert traj.episode.outcome_status == "completed"
    assert traj.episode.graph_version_id == "ver-123"
    assert traj.episode.graph_fingerprint == graph.fingerprint
    assert traj.episode.reward == 1.0

    tool_decisions = [d for d in traj.decisions if d.decision_type == "tool_call"]
    assert [d.action_name for d in tool_decisions] == ["search_kb", "verify_facts"]
    assert tool_decisions[0].traversed_edge is not None
    assert tool_decisions[0].traversed_edge.source == "Start"
    assert tool_decisions[0].traversed_edge.target == "search_kb"
    assert tool_decisions[1].traversed_edge is not None
    assert tool_decisions[1].traversed_edge.source == "search_kb"
    assert tool_decisions[1].traversed_edge.target == "verify_facts"

    llm_decisions = [d for d in traj.decisions if d.decision_type == "llm_action"]
    assert len(llm_decisions) == 3
    assert all(d.guidance_status == "completed" for d in llm_decisions)
    assert llm_decisions[0].active_node == "Start"
    assert llm_decisions[1].active_node == "search_kb"
    assert llm_decisions[2].active_node == "verify_facts"

    # Empirical edge metrics
    edge_metrics = compute_edge_empirical_metrics([traj])
    assert len(edge_metrics) == 2
    assert edge_metrics[0].mean_episode_reward == 1.0
    assert edge_metrics[0].tool_failure_rate == 0.0

    # RL transition tuple export
    rl_tuples = trajectories_to_rl_transitions([traj], gamma=0.9)
    assert len(rl_tuples) == 5  # 3 llm_action + 2 tool_call
    assert rl_tuples[-1].done is True
    assert rl_tuples[0].return_to_go > 0.7

    jsonl = export_rl_transitions_jsonl([traj], gamma=0.9)
    lines = [json.loads(line) for line in jsonl.strip().splitlines()]
    assert len(lines) == 5

    # SelectiveGuidanceBandit test: compare high-token guided vs cheaper equal-reward unguided on Start
    bandit = SelectiveGuidanceBandit(token_cost_weight=0.001, min_samples_per_arm=1)
    unguided_copy = rl_tuples[0].model_copy(
        update={"guidance_invoked": False, "tokens_used": 10.0, "return_to_go": 1.0}
    )
    guided_expensive = rl_tuples[0].model_copy(
        update={"guidance_invoked": True, "tokens_used": 500.0, "return_to_go": 1.0}
    )
    bandit.fit([guided_expensive, unguided_copy])
    assert bandit.should_invoke_guidance("Start") is False


# ---------------------------------------------------------------------------
# 4. DB-Backed Self-Evolution (`evolve_with_store`) & Audit Mining
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evolve_with_store_tracks_accepts_rejections_and_promotes_winner():
    store = InMemoryProceduralGraphStore()
    train = [
        EvaluationTask(id="t1", query="Query 1"),
        EvaluationTask(id="t2", query="Query 2"),
    ]
    val = [
        EvaluationTask(id="v1", query="Val 1"),
        EvaluationTask(id="v2", query="Val 2"),
    ]

    async def fake_runner(
        graph: ProceduralGraph, task: EvaluationTask
    ) -> ScoredRollout:
        has_search = any(n.id == "search_kb" for n in graph.nodes)
        has_verify = any(n.id == "verify_facts" for n in graph.nodes)
        has_bad = any(n.id == "bad_shortcut" for n in graph.nodes)
        score = 0.2
        if has_search:
            score = 0.7
        if has_search and has_verify:
            score = 0.95
        if has_bad:
            score = 0.1
        return ScoredRollout(
            task_id=task.id,
            score=score,
            output="ans",
            metrics={"steps": 2.0, "tokens": 120.0},
        )

    round_counter = 0
    observed_rejections: list[list[dict[str, Any]]] = []

    async def scripted_refiner(
        graph: ProceduralGraph, _ctx: str, rejections: list[dict[str, Any]]
    ) -> GraphEdits:
        nonlocal round_counter
        round_counter += 1
        observed_rejections.append(rejections)
        if round_counter == 1:
            # Round 1: Structural error (deletes Start!) -> rejected before validation
            return GraphEdits(delete_nodes=("Start",))
        if round_counter == 2:
            # Round 2: Valid edit adding search_kb -> improves 0.2 -> 0.7 -> accepted!
            return GraphEdits(
                delete_edges=(EdgeEndpoints(source="Start", target="End"),),
                add_nodes=(ProcedureNode(id="search_kb", type="ACTION"),),
                add_edges=(
                    ProcedureEdge(
                        source="Start", target="search_kb", guidance="Search first"
                    ),
                    ProcedureEdge(source="search_kb", target="End", guidance="Finish"),
                ),
            )
        if round_counter == 3:
            # Round 3: Valid edit adding bad_shortcut -> lowers 0.7 -> 0.1 -> rejected by gate!
            return GraphEdits(
                add_nodes=(ProcedureNode(id="bad_shortcut", type="REASONING"),),
                add_edges=(
                    ProcedureEdge(
                        source="Start", target="bad_shortcut", guidance="Skip search"
                    ),
                    ProcedureEdge(
                        source="bad_shortcut", target="End", guidance="Guess"
                    ),
                ),
            )
        # Round 4: Valid edit adding verify_facts -> improves 0.7 -> 0.95 -> accepted!
        assert any(n.id == "search_kb" for n in graph.nodes)
        return GraphEdits(
            delete_edges=(EdgeEndpoints(source="search_kb", target="End"),),
            add_nodes=(ProcedureNode(id="verify_facts", type="ACTION"),),
            add_edges=(
                ProcedureEdge(
                    source="search_kb", target="verify_facts", guidance="Verify"
                ),
                ProcedureEdge(source="verify_facts", target="End", guidance="Answer"),
            ),
        )

    evo_run, final_ver = await evolve_with_store(
        store,
        account="acc",
        agent_id="ag",
        workflow_id="default",
        train_tasks=train,
        validation_tasks=val,
        runner=fake_runner,
        refiner=scripted_refiner,
        available_tools=["search_kb", "verify_facts"],
        provenance={
            "model": "test",
            "evaluator": "exact",
            "tools": ["search_kb", "verify_facts"],
        },
        rounds=4,
        batch_size=1,
        auto_promote=True,
    )

    assert evo_run.status == "completed"
    assert evo_run.rounds_completed == 4
    assert final_ver.status == "active"
    assert final_ver.validation_score == 0.95
    assert {n.id for n in final_ver.graph.nodes} == {
        "Start",
        "search_kb",
        "verify_facts",
        "End",
    }
    # Round 2 saw Round 1's structural rejection; Round 4 saw both Round 1 & Round 3 rejections!
    assert len(observed_rejections[1]) == 1
    assert len(observed_rejections[3]) == 2


# ---------------------------------------------------------------------------
# 5. FastAPI REST v1 Endpoints + AgentManager Runtime & Export Integration
# ---------------------------------------------------------------------------


class _OwnerAuthBackend:
    async def authenticate(self, conn):
        return AuthCredentials([StashRoles.OWNER]), User(username="owner")


@pytest.mark.asyncio
async def test_rest_api_procedural_graph_audit_and_export_roundtrip():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _init_sqlite_tables(conn)

    db = _SQLiteAsyncDatabase(engine)
    manager = AgentManager(
        database=db,  # type: ignore[arg-type]
        settings=DataManagerSettings(postgresql_dsn="postgresql://localhost/unused"),
    )
    await manager.add_agent(
        account="acc", agent_id="ag1", memory=MemoryConfig(), rules=Rules()
    )

    app = FastAPI()
    app.agent_manager = manager  # type: ignore[attr-defined]
    app.include_router(v1_router)
    app.add_middleware(AuthenticationMiddleware, backend=_OwnerAuthBackend())
    headers = {
        "X-STF-USER": "alice",
        "X-STF-ACCOUNT": "acc",
        "X-STF-ACCOUNT-TYPE": "enterprise",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 1) Initialize skeleton via PUT
        put_resp = await client.put(
            "/api/v1/agent/ag1/workflow/default/procedural-graph",
            headers=headers,
            json={
                "skeleton": True,
                "guidance_settings": GuidanceRuntimeSettings(
                    hops=2, window=4
                ).model_dump(),
                "evolution_settings": EvolutionSettings(rounds=3).model_dump(),
                "comment": "Seed skeleton",
                "promote": True,
            },
        )
        assert put_resp.status_code == 200
        v1_id = put_resp.json()["id"]

        # 2) Apply GraphEdits via POST .../edits
        edit_resp = await client.post(
            "/api/v1/agent/ag1/workflow/default/procedural-graph/edits",
            headers=headers,
            json={
                "edits": {
                    "delete_edges": [{"source": "Start", "target": "End"}],
                    "add_nodes": [{"id": "lookup", "type": "ACTION"}],
                    "add_edges": [
                        {
                            "source": "Start",
                            "target": "lookup",
                            "guidance": "Lookup record first.",
                        },
                        {
                            "source": "lookup",
                            "target": "End",
                            "guidance": "Summarize record.",
                        },
                    ],
                },
                "available_tools": ["lookup"],
                "promote": True,
            },
        )
        assert edit_resp.status_code == 200
        v2_id = edit_resp.json()["id"]

        # 3) Inspect diff via GET .../diff
        diff_resp = await client.get(
            f"/api/v1/agent/ag1/workflow/default/procedural-graph/versions/{v2_id}/diff",
            headers=headers,
        )
        assert diff_resp.status_code == 200
        assert [n["id"] for n in diff_resp.json()["added_nodes"]] == ["lookup"]

        # 4) Verify AgentManager.get_agent_config() automatically wires active ProceduralGraph!
        runtime_cfg = await manager.get_agent_config(
            "acc", "ag1", workflow_id="default"
        )
        assert runtime_cfg.procedural_guidance is not None
        assert runtime_cfg.procedural_guidance.hops == 2
        assert runtime_cfg.procedural_guidance.window == 4
        assert runtime_cfg.procedural_graph_version_id == v2_id

        # 5) Record an audited episode in PostgresHarnessAuditStore, attach reward via API, and export JSONL
        from hyperforge.harness_sdk.audit import (
            AuditDecisionRecord,
            AuditEpisode,
            AuditTraversedEdge,
        )

        ep_id = await manager.audit_store.start_episode(
            AuditEpisode(
                account="acc",
                agent_id="ag1",
                workflow_id="default",
                conversation_id="conv-99",
                turn_id="turn-1",
                graph_version_id=v2_id,
                graph_fingerprint=runtime_cfg.procedural_guidance.graph.fingerprint,
                model="claude-sonnet-4",
                query="Find customer 42",
            )
        )
        await manager.audit_store.record_decision(
            AuditDecisionRecord(
                episode_id=ep_id,
                account="acc",
                agent_id="ag1",
                workflow_id="default",
                step_index=0,
                decision_id="d1",
                decision_type="tool_call",
                active_node="Start",
                graph_scope="local",
                traversed_edge=AuditTraversedEdge(
                    source="Start", target="lookup", relation="LEADS_TO"
                ),
                action_name="lookup",
                action_arguments={"id": 42},
                observation_summary="Customer 42 active",
                input_tokens=50,
                output_tokens=12,
                latency_ms=35.0,
            )
        )
        await manager.audit_store.complete_episode(
            ep_id,
            outcome_status="completed",
            final_output="Customer 42 is active.",
            total_steps=1,
            total_tool_calls=1,
            input_tokens=50,
            output_tokens=12,
            latency_ms=40.0,
        )

        rew_resp = await client.post(
            f"/api/v1/agent/ag1/workflow/default/audit/episodes/{ep_id}/reward",
            headers=headers,
            json={"reward": 0.9, "source": "human_label", "step_rewards": {"0": 0.05}},
        )
        assert rew_resp.status_code == 200
        assert rew_resp.json()["episode"]["reward"] == 0.9

        metrics_resp = await client.get(
            "/api/v1/agent/ag1/workflow/default/audit/edge-metrics",
            headers=headers,
        )
        assert metrics_resp.status_code == 200
        assert len(metrics_resp.json()) == 1
        assert metrics_resp.json()[0]["mean_episode_reward"] == 0.9

        export_resp = await client.get(
            "/api/v1/agent/ag1/workflow/default/audit/export",
            headers=headers,
        )
        assert export_resp.status_code == 200
        exported_lines = [
            json.loads(line) for line in export_resp.text.strip().splitlines()
        ]
        assert len(exported_lines) == 1
        assert exported_lines[0]["return_to_go"] == 0.95

        # 6) Propose a new graph candidate directly from the audited trajectories!
        async def audit_refiner(
            _graph: ProceduralGraph, ctx_str: str, _rejs: list[dict[str, Any]]
        ) -> GraphEdits:
            assert "empirical_edge_metrics" in ctx_str
            return GraphEdits(
                delete_edges=(EdgeEndpoints(source="lookup", target="End"),),
                add_nodes=(ProcedureNode(id="format_reply", type="REASONING"),),
                add_edges=(
                    ProcedureEdge(
                        source="lookup",
                        target="format_reply",
                        guidance="Format concise reply",
                    ),
                    ProcedureEdge(
                        source="format_reply",
                        target="End",
                        guidance="Deliver final answer",
                    ),
                ),
            )

        proposed = await propose_candidate_from_audit(
            manager.procedural_store,
            manager.audit_store,
            account="acc",
            agent_id="ag1",
            workflow_id="default",
            refiner=audit_refiner,
            available_tools=["lookup"],
            promote=True,
        )
        assert proposed.source == "rl_policy"
        assert proposed.status == "active"

        # 7) Rollback via REST API to v1
        rb_resp = await client.post(
            f"/api/v1/agent/ag1/workflow/default/procedural-graph/versions/{v1_id}/rollback",
            headers=headers,
            json={"comment": "Revert to skeleton"},
        )
        assert rb_resp.status_code == 200
        assert rb_resp.json()["source"] == "rollback"

    # 8) Verify encrypted V2 Export -> Import preserves ProceduralGraph!
    passphrase = "procedural-export-passphrase-1234"
    exported_bytes, _ = await manager.export("acc", "ag1", passphrase)
    await manager.add_agent(
        account="acc", agent_id="ag-imported", memory=MemoryConfig(), rules=Rules()
    )
    await manager.import_config(
        account="acc",
        agent_id="ag-imported",
        import_file=UploadFile(filename="pg.bin", file=io.BytesIO(exported_bytes)),
        passphrase=passphrase,
        overwrite=True,
    )
    imported_cfg = await manager.get_agent_config("acc", "ag-imported")
    assert imported_cfg.procedural_guidance is not None
    assert imported_cfg.procedural_guidance.hops == 2

    await db.disconnect()
