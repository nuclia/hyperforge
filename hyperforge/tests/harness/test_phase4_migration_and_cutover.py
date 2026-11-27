"""Phase 4 tests: 4-stage-to-Harness transpiler, Alembic DDL & data migration, AgentManager, API, Standalone, and Engine cutover."""

from __future__ import annotations

import io
import json
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
from RestrictedPython import compile_restricted
from sqlalchemy import create_engine, text
from starlette.authentication import AuthCredentials
from starlette.middleware.authentication import AuthenticationMiddleware

import hyperforge
from hyperforge.api.authentication import User
from hyperforge.api.models import StashRoles
from hyperforge.api.v1.router import router as v1_router
from hyperforge.configure import GLOBAL_REGISTRY, load_entry_point_plugins
from hyperforge.db.agents import AgentManager
from hyperforge.db.harness_storage import PostgresHarnessStorage
from hyperforge.db.settings import DataManagerSettings
from hyperforge.db.workflow_migration import (
    migrate_database_workflows_to_harness,
    restore_database_workflows_from_snapshot,
    transpile_stages_to_harness,
)
from hyperforge.engine import main as engine_main
from hyperforge.harness.config import HarnessAgentConfig
from hyperforge.harness_sdk import (
    HarnessConversation,
    HarnessEvent,
    HarnessEventType,
    HarnessMemory,
)
from hyperforge.models import MemoryConfig, Rules
from hyperforge.prompts import PromptConfig
from hyperforge.retrieval.config import RetrievalAgentConfig, RetrievalAgentExportV1
from hyperforge.standalone.agent import StaticAgentManager
from hyperforge.standalone.config import StandAloneAgentConfig, WorkflowConfig
from hyperforge.workflows import WorkflowInput

# ---------------------------------------------------------------------------
# Lightweight async SQLite adapter matching `databases.Database` interface
# ---------------------------------------------------------------------------


class _RowMapping:
    def __init__(self, data: dict[str, Any]) -> None:
        self._mapping = data

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return list(self._mapping.values())[key]
        return self._mapping[key]


class _SQLiteAsyncDatabase:
    """In-memory SQLite wrapper implementing `fetch_one`, `fetch_all`, `execute`, `transaction`."""

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
            else:
                params[k] = v
        sql = str(compiled)
        return sql, params

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
                    k: json.dumps(v) if isinstance(v, (dict, list)) else v
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
        result = self._conn.execute(text(sql), params)
        keys = list(result.keys())
        return [_RowMapping(dict(zip(keys, row))) for row in result.fetchall()]


def _create_sqlite_harness_schema(conn: sa.Connection) -> None:
    """Create post-Phase-4 schema in SQLite for fast, Docker-free integration tests."""
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
            modified TIMESTAMP,
            UNIQUE (account, agent_id, workflow_id, namespace)
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
            modified TIMESTAMP,
            UNIQUE (account, agent_id, skill_id)
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
            modified TIMESTAMP,
            UNIQUE (account, agent_id, identifier)
        );
        CREATE TABLE harness_conversations (
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            created_datetime TIMESTAMP NOT NULL,
            updated_datetime TIMESTAMP NOT NULL,
            PRIMARY KEY (account, agent_id, id)
        );
        CREATE TABLE harness_events (
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            id TEXT NOT NULL,
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL,
            turn_id TEXT,
            harness_agent_id TEXT,
            parent_agent_id TEXT,
            parent_call_id TEXT,
            category TEXT NOT NULL,
            tags TEXT NOT NULL DEFAULT '[]',
            metadata TEXT NOT NULL DEFAULT '{}',
            type TEXT NOT NULL,
            created_datetime TIMESTAMP NOT NULL,
            payload TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE harness_memories (
            account TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            id TEXT NOT NULL,
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            created_datetime TIMESTAMP NOT NULL
        );
        """
    )


# ---------------------------------------------------------------------------
# 1. Deterministic 4-stage-to-Harness Transpiler Tests
# ---------------------------------------------------------------------------


class TestWorkflowTranspiler:
    def test_pure_loop_smart_with_multi_mcp_and_citations(self):
        compiled = transpile_stages_to_harness(
            preprocess=[],
            context=[
                {
                    "module": "smart",
                    "executor_model": {
                        "_type": "llm_config",
                        "model_id": "claude-sonnet-4",
                    },
                    "planning_mode": "plan_execute",
                    "enable_user_feedback": True,
                    "max_iterations": 4,
                    "extra_prompt": "Prefer concise SQL queries.",
                    "registered_agents": [
                        {"module": "basic_ask", "id": "kb_docs", "sources": ["docs"]},
                        {
                            "module": "multi_mcp",
                            "configs": [
                                {"module": "mcp", "source": "github_mcp"},
                                {"module": "mcp", "source": "jira_mcp"},
                            ],
                        },
                    ],
                }
            ],
            generation=[
                {
                    "module": "summarize",
                    "system_prompt": "You are an enterprise assistant.",
                    "prompt": "Include concrete examples.",
                    "citations": True,
                }
            ],
            postprocess=[],
        )

        assert compiled["execution_mode"] == "loop"
        assert compiled["workflow_code"] is None
        hc = compiled["harness_config"]
        assert hc["model"] == "claude-sonnet-4"
        assert hc["feedback_enabled"] is True
        assert hc["usage_limits"] == {"max_turns": 8}
        assert hc["enabled_skills"] == ["plan_and_execute", "citation_grounded_answer"]
        assert "You are an enterprise assistant." in hc["system_prompt"]
        assert "Prefer concise SQL queries." in hc["system_prompt"]
        assert len(compiled["toolkits"]) == 3
        assert compiled["toolkits"][0]["toolkit"] == "nucliadb"
        assert compiled["toolkits"][0]["lazy_load"] is False
        assert compiled["toolkits"][1]["toolkit"] == "mcp"
        assert compiled["toolkits"][1]["lazy_load"] is True
        assert (
            compiled["toolkits"][2]["namespace"] != compiled["toolkits"][1]["namespace"]
        )

    def test_parallel_code_mode_for_unpruned_retrievers(self):
        compiled = transpile_stages_to_harness(
            preprocess=[{"module": "rephrase", "split_question": True}],
            context=[
                {"module": "basic_ask", "sources": ["kb1"], "prune_context": False},
                {"module": "google", "source": "gsearch", "prune_context": False},
            ],
            generation=[{"module": "summarize", "citations": True}],
            postprocess=[{"module": "remi"}, {"module": "related"}],
        )

        assert compiled["execution_mode"] == "code"
        code = compiled["workflow_code"]
        assert code is not None
        compile_restricted(code, filename="<test>", mode="exec")
        assert "parallel([" in code
        assert "call(" in code
        assert "evaluate_remi(" in code
        assert "suggest_related_questions(" in code
        assert "output({" in code

    def test_conditionals_fallback_next_agent_and_restart_loop_compile_cleanly(self):
        compiled = transpile_stages_to_harness(
            preprocess=[
                {
                    "module": "pre_conditional",
                    "prompt": "Is technical?",
                    "then": [{"module": "rephrase"}],
                    "else_": [],
                }
            ],
            context=[
                {
                    "module": "context_conditional",
                    "prompt": "Needs KB?",
                    "then": [
                        {
                            "module": "advanced_ask",
                            "sources": ["kb1"],
                            "top_k": 15,
                            "min_score": 0.4,
                            "fallback": {
                                "module": "perplexity_search",
                                "source": "pplx",
                                "prune_context": False,
                            },
                            "next_agent": {
                                "module": "http",
                                "url": "https://api.example.com/items",
                                "prune_context": False,
                            },
                        }
                    ],
                    "else_": [{"module": "static", "context": "Fallback text"}],
                }
            ],
            generation=[
                {
                    "module": "generation_conditional",
                    "prompt": "Raw contexts requested?",
                    "then": [{"module": "passthrough"}],
                    "else_": [{"module": "generate", "prompt": "Write report"}],
                }
            ],
            postprocess=[
                {"module": "restart", "retries": 2},
                {
                    "module": "post_conditional",
                    "on": "ANSWER",
                    "prompt": "Notify webhook?",
                    "then": [
                        {
                            "module": "external",
                            "url": "https://hooks.example.com/notify",
                            "context": False,
                        }
                    ],
                },
            ],
        )

        assert compiled["execution_mode"] == "code"
        code = compiled["workflow_code"]
        assert code is not None
        compile_restricted(code, filename="<complex_migrated>", mode="exec")
        assert "while not wf_answered and wf_attempt < 3:" in code
        assert "detect_missing_info(" in code
        assert "validate_contexts(" in code
        assert "rephrase_for_next(" in code
        assert "webhook_extract_and_send(" in code

    def test_external_plugin_generic_passthrough_and_migration_hook(self):
        # 1) Generic pass-through for unregistered third-party module (e.g. snowflake)
        generic = transpile_stages_to_harness(
            preprocess=[],
            context=[
                {
                    "module": "snowflake",
                    "warehouse": "ANALYTICS_WH",
                    "published_functions": ["run_snowflake_sql"],
                    "prune_context": False,
                }
            ],
            generation=[{"module": "summarize"}],
            postprocess=[],
        )
        sf_tk = next(t for t in generic["toolkits"] if t["toolkit"] == "snowflake")
        assert sf_tk["warehouse"] == "ANALYTICS_WH"
        assert "snowflake__run_snowflake_sql(" in generic["workflow_code"]

        # 2) Explicit `@migration("snowflake")` hook registered by external package
        def custom_sf_migration(raw: dict[str, Any]) -> dict[str, Any]:
            return {
                "toolkit": "progress_snowflake",
                "namespace": "sf_analytics",
                "warehouse": raw["warehouse"],
                "_primary_tool": "execute_sql",
            }

        hooked = transpile_stages_to_harness(
            preprocess=[],
            context=[
                {
                    "module": "snowflake",
                    "warehouse": "ANALYTICS_WH",
                    "prune_context": False,
                }
            ],
            generation=[{"module": "summarize"}],
            postprocess=[],
            external_migrations={"snowflake": custom_sf_migration},
        )
        sf_hooked = next(
            t for t in hooked["toolkits"] if t["toolkit"] == "progress_snowflake"
        )
        assert sf_hooked["namespace"] == "sf_analytics"
        assert "sf_analytics__execute_sql(" in hooked["workflow_code"]


# ---------------------------------------------------------------------------
# 2. Alembic PostgreSQL Offline SQL & Lossless Upgrade/Downgrade Data Migration
# ---------------------------------------------------------------------------


class TestAlembicAndDatabaseMigration:
    def test_alembic_offline_postgresql_upgrade_and_downgrade_ddl(self, capsys):
        ini_path = Path(hyperforge.__file__).parent / "alembic.ini"
        cfg = alembic.config.Config(str(ini_path))
        cfg.set_main_option(
            "sqlalchemy.url", "postgresql://postgres:postgres@localhost:5432/test_db"
        )

        alembic.command.upgrade(cfg, "8c7d2e4f91ab:c4e8a91b2d30", sql=True)
        up_sql = capsys.readouterr().out
        assert (
            "ALTER TABLE retrieval_agent_workflow ADD COLUMN execution_mode" in up_sql
        )
        assert "CREATE TABLE agent_workflow_toolkits" in up_sql
        assert "CREATE TABLE agent_skills" in up_sql
        assert "CREATE TABLE harness_conversations" in up_sql
        assert "CREATE TABLE harness_events" in up_sql
        assert "CREATE TABLE harness_memories" in up_sql
        assert "DROP TABLE retrieval_agent_preprocess" in up_sql
        assert "DROP TABLE retrieval_agent_context" in up_sql
        assert "DROP TABLE retrieval_agent_generation" in up_sql
        assert "DROP TABLE retrieval_agent_postprocess" in up_sql

        cfg_down = alembic.config.Config(str(ini_path))
        cfg_down.set_main_option(
            "sqlalchemy.url", "postgresql://postgres:postgres@localhost:5432/test_db"
        )
        alembic.command.downgrade(cfg_down, "c4e8a91b2d30:8c7d2e4f91ab", sql=True)
        down_sql = capsys.readouterr().out
        assert "CREATE TABLE retrieval_agent_preprocess" in down_sql
        assert "CREATE TABLE retrieval_agent_context" in down_sql
        assert "CREATE TABLE retrieval_agent_generation" in down_sql
        assert "CREATE TABLE retrieval_agent_postprocess" in down_sql
        assert "DROP TABLE agent_workflow_toolkits" in down_sql
        assert "DROP TABLE agent_skills" in down_sql
        assert "DROP TABLE harness_conversations" in down_sql

    def test_database_data_migration_and_lossless_snapshot_restore(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as conn:
            _create_sqlite_harness_schema(conn)
            for tbl, col in (
                ("retrieval_agent_preprocess", "preprocess"),
                ("retrieval_agent_context", "context"),
                ("retrieval_agent_generation", "generation"),
                ("retrieval_agent_postprocess", "postprocess"),
            ):
                conn.exec_driver_sql(
                    f"""
                    CREATE TABLE {tbl} (
                        id TEXT PRIMARY KEY,
                        account TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        workflow_id TEXT NOT NULL,
                        {col} TEXT NOT NULL,
                        created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        modified TIMESTAMP
                    )
                    """
                )

            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_config (account, agent_id, rules, memory) "
                    "VALUES ('acc1', 'ag1', '{\"rules\": []}', '{}')"
                )
            )
            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_workflow (account, agent_id, workflow_id, name) "
                    "VALUES ('acc1', 'ag1', 'default', 'Default'), "
                    "       ('acc1', 'ag1', 'loop_wf', 'Smart Loop')"
                )
            )

            prep_id = str(uuid4())
            ctx_id = str(uuid4())
            gen_id = str(uuid4())
            smart_id = str(uuid4())

            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_preprocess (id, account, agent_id, workflow_id, preprocess) "
                    "VALUES (:id, 'acc1', 'ag1', 'default', :cfg)"
                ),
                {"id": prep_id, "cfg": json.dumps({"module": "rephrase"})},
            )
            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_context (id, account, agent_id, workflow_id, context) "
                    "VALUES (:id1, 'acc1', 'ag1', 'default', :cfg1), "
                    "       (:id2, 'acc1', 'ag1', 'loop_wf', :cfg2)"
                ),
                {
                    "id1": ctx_id,
                    "cfg1": json.dumps(
                        {
                            "module": "static",
                            "context": "Doc body",
                            "prune_context": False,
                        }
                    ),
                    "id2": smart_id,
                    "cfg2": json.dumps(
                        {
                            "module": "smart",
                            "planning_mode": "plan_execute",
                            "registered_agents": [
                                {"module": "google", "source": "google_drv"}
                            ],
                        }
                    ),
                },
            )
            conn.execute(
                text(
                    "INSERT INTO retrieval_agent_generation (id, account, agent_id, workflow_id, generation) "
                    "VALUES (:id, 'acc1', 'ag1', 'default', :cfg)"
                ),
                {
                    "id": gen_id,
                    "cfg": json.dumps({"module": "summarize", "citations": True}),
                },
            )

            count = migrate_database_workflows_to_harness(conn)
            assert count == 2

            wf_default = conn.execute(
                text(
                    "SELECT execution_mode, workflow_code FROM retrieval_agent_workflow "
                    "WHERE workflow_id = 'default'"
                )
            ).fetchone()
            assert wf_default is not None
            assert wf_default[0] == "code"
            assert "get_static_context()" in wf_default[1]

            wf_loop = conn.execute(
                text(
                    "SELECT execution_mode, harness_config, workflow_code FROM retrieval_agent_workflow "
                    "WHERE workflow_id = 'loop_wf'"
                )
            ).fetchone()
            assert wf_loop is not None
            assert wf_loop[0] == "loop"
            assert wf_loop[2] is None
            assert "plan_and_execute" in json.loads(wf_loop[1])["enabled_skills"]

            tk_rows = conn.execute(
                text(
                    "SELECT workflow_id, toolkit FROM agent_workflow_toolkits ORDER BY workflow_id, toolkit"
                )
            ).fetchall()
            assert ("default", "rephrase") in tk_rows
            assert ("default", "static") in tk_rows
            assert ("default", "synthesis") in tk_rows
            assert ("loop_wf", "google") in tk_rows

            # Simulate dropping stage tables on upgrade, recreating them on downgrade, and restoring from snapshot
            for tbl in (
                "retrieval_agent_preprocess",
                "retrieval_agent_context",
                "retrieval_agent_generation",
                "retrieval_agent_postprocess",
            ):
                conn.exec_driver_sql(f"DELETE FROM {tbl}")

            restored = restore_database_workflows_from_snapshot(conn)
            assert restored == 2

            restored_prep = conn.execute(
                text("SELECT id, preprocess FROM retrieval_agent_preprocess")
            ).fetchall()
            assert len(restored_prep) == 1
            assert restored_prep[0][0] == prep_id
            assert json.loads(restored_prep[0][1])["module"] == "rephrase"

            restored_ctx = conn.execute(
                text("SELECT id FROM retrieval_agent_context")
            ).fetchall()
            assert {r[0] for r in restored_ctx} == {ctx_id, smart_id}


# ---------------------------------------------------------------------------
# 3. AgentManager, PostgresHarnessStorage, FastAPI v1, and V1/V2 Export-Import
# ---------------------------------------------------------------------------


class _TestAuthBackend:
    async def authenticate(self, conn):
        return AuthCredentials([StashRoles.OWNER]), User(username="tester")


@pytest.mark.asyncio
async def test_agent_manager_api_crud_storage_and_export_import():
    GLOBAL_REGISTRY.clear()
    load_entry_point_plugins()
    for mod in ("hyperforge_static", "hyperforge_summarize", "hyperforge_rephrase"):
        hyperforge.configure.scan(mod)
        hyperforge.configure.load_all_configurations(mod)

    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _create_sqlite_harness_schema(conn)

    db = _SQLiteAsyncDatabase(engine)
    manager = AgentManager(
        database=db,  # type: ignore[arg-type]
        settings=DataManagerSettings(postgresql_dsn="postgresql://localhost/unused"),
    )

    await manager.add_agent(
        account="acc",
        agent_id="agent-1",
        memory=MemoryConfig(),
        rules=Rules(rules=["Always be accurate"]),
    )

    # Mount FastAPI v1 router against our SQLite-backed AgentManager
    app = FastAPI()
    app.agent_manager = manager  # type: ignore[attr-defined]
    app.include_router(v1_router)
    app.add_middleware(AuthenticationMiddleware, backend=_TestAuthBackend())
    headers = {
        "X-STF-USER": "u1",
        "X-STF-ACCOUNT": "acc",
        "X-STF-ACCOUNT-TYPE": "enterprise",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Add skill via REST endpoint
        resp = await client.post(
            "/api/v1/agent/agent-1/skills",
            headers=headers,
            json={
                "id": "domain_sop",
                "name": "Domain SOP",
                "description": "Standard operating procedure",
                "instructions": "Follow SOP step by step.",
                "preload": True,
            },
        )
        assert resp.status_code == 200

        skills_resp = await client.get("/api/v1/agent/agent-1/skills", headers=headers)
        assert skills_resp.status_code == 200
        assert [s["id"] for s in skills_resp.json()] == ["domain_sop"]

        # Add toolkit via REST endpoint
        tk_resp = await client.post(
            "/api/v1/agent/agent-1/workflow/default/toolkits",
            headers=headers,
            json={
                "toolkit": "static",
                "namespace": "kb_static",
                "context": "Unified Harness rocks!",
            },
        )
        assert tk_resp.status_code == 200
        tk_id = tk_resp.json()["id"]

        list_tk = await client.get(
            "/api/v1/agent/agent-1/workflow/default/toolkits", headers=headers
        )
        assert list_tk.status_code == 200
        assert len(list_tk.json()) == 1
        assert list_tk.json()[0]["namespace"] == "kb_static"

        # Update workflow harness configuration to Code Mode via REST endpoint
        patch_h = await client.patch(
            "/api/v1/agent/agent-1/workflow/default/harness",
            headers=headers,
            json={
                "execution_mode": "code",
                "harness_config": {"model": ""},
                "workflow_code": "res = kb_static__get_static_context()\noutput(res['contexts'][0]['chunks'][0]['text'])\n",
            },
        )
        assert patch_h.status_code == 200

        get_h = await client.get(
            "/api/v1/agent/agent-1/workflow/default/harness", headers=headers
        )
        assert get_h.status_code == 200
        assert get_h.json()["execution_mode"] == "code"

        # Verify unified `get_agent_config()` returns executable `HarnessAgentConfig`
        cfg = await manager.get_agent_config("acc", "agent-1", workflow_id="default")
        assert isinstance(cfg, HarnessAgentConfig)
        assert cfg.execution_mode == "code"
        assert len(cfg.toolkits) == 1
        assert any(s.id == "domain_sop" for s in cfg.skills)

        # Delete toolkit via REST endpoint check
        await manager.add_workflow(
            "acc",
            "agent-1",
            WorkflowInput(id="legacy_api_wf", name="Legacy Stage API"),
        )
        stage_resp = await client.post(
            "/api/v1/agent/agent-1/workflow/legacy_api_wf/context",
            headers=headers,
            json={"module": "static", "context": "From legacy endpoint"},
        )
        assert stage_resp.status_code == 200
        legacy_compiled_cfg = await manager.get_agent_config(
            "acc", "agent-1", workflow_id="legacy_api_wf"
        )
        assert legacy_compiled_cfg.execution_mode == "code"
        assert any(t.toolkit == "static" for t in legacy_compiled_cfg.toolkits)

        del_tk = await client.delete(
            f"/api/v1/agent/agent-1/workflow/default/toolkits/{tk_id}",
            headers=headers,
        )
        assert del_tk.status_code == 200

    # V2 Export -> Import round-trip
    await manager.add_toolkit(
        account="acc",
        agent_id="agent-1",
        toolkit=hyperforge.configure.validate_toolkit(
            {"toolkit": "static", "namespace": "kb_static", "context": "Exported text"}
        ),
    )
    passphrase = "super-secret-passphrase-1234"
    exported_bytes, _ = await manager.export("acc", "agent-1", passphrase)

    await manager.add_agent(
        account="acc",
        agent_id="agent-imported-v2",
        memory=MemoryConfig(),
        rules=Rules(),
    )
    await manager.import_config(
        account="acc",
        agent_id="agent-imported-v2",
        import_file=UploadFile(filename="export.bin", file=io.BytesIO(exported_bytes)),
        passphrase=passphrase,
        overwrite=True,
    )
    imported_v2 = await manager.get_agent_config("acc", "agent-imported-v2")
    assert imported_v2.execution_mode == "code"
    assert any(t.namespace == "kb_static" for t in imported_v2.toolkits)
    assert any(s.id == "domain_sop" for s in imported_v2.skills)

    # V1 Export -> Import auto-transpilation to Harness
    from cryptography.fernet import Fernet

    from hyperforge.db.encryption import fernet_key_from_passphrase

    v1_payload = RetrievalAgentExportV1(
        agent_config=RetrievalAgentConfig.model_validate(
            {
                "drivers": [],
                "rules": {"rules": ["Rule V1"]},
                "memory": {},
                "workflow": {
                    "id": "default",
                    "name": "Default V1",
                    "description": "Legacy V1",
                    "parameters": {},
                },
                "preprocess": [],
                "context": [{"module": "static", "context": "Legacy V1 static"}],
                "generation": [{"module": "summarize"}],
                "postprocess": [],
            }
        ),
        prompts=[
            PromptConfig(
                name="welcome", description="Welcome prompt", prompt="Hello {{name}}"
            )
        ],
    )
    key, salt = fernet_key_from_passphrase(passphrase, None)
    v1_encrypted = salt + Fernet(key).encrypt(v1_payload.model_dump_json().encode())
    await manager.add_agent(
        account="acc",
        agent_id="agent-imported-v1",
        memory=MemoryConfig(),
        rules=Rules(),
    )
    await manager.import_config(
        account="acc",
        agent_id="agent-imported-v1",
        import_file=UploadFile(filename="v1.bin", file=io.BytesIO(v1_encrypted)),
        passphrase=passphrase,
        overwrite=True,
    )
    imported_v1 = await manager.get_agent_config("acc", "agent-imported-v1")
    assert imported_v1.execution_mode == "code"
    assert {t.toolkit for t in imported_v1.toolkits} >= {"static", "synthesis"}
    assert len(imported_v1.prompts) == 1

    # PostgresHarnessStorage round-trip on SQLite adapter
    storage = PostgresHarnessStorage(db, account="acc", agent_id="agent-1")  # type: ignore[arg-type]
    conv = HarnessConversation(id="conv-1", title="Test Chat")
    await storage.create_conversation(conv)
    loaded_conv = await storage.get_conversation("conv-1")
    assert loaded_conv is not None and loaded_conv.title == "Test Chat"
    ev = HarnessEvent(
        id="ev-1",
        conversation_id="conv-1",
        turn_id="t1",
        agent_id="h1",
        type=HarnessEventType.TURN_COMPLETED,
        payload={"text": "Hello from storage"},
    )
    await storage.append_event(ev)
    loaded_events = [item async for item in storage.iter_events("conv-1")]
    assert len(loaded_events) == 1
    assert loaded_events[0].id == "ev-1"
    await storage.remember(
        HarnessMemory(id="m1", scope="/agent", text="User likes Python")
    )
    recalled = await storage.recall(scope="/agent", query="Python")
    assert len(recalled) == 1
    await storage.forget("m1")
    assert await storage.recall(scope="/agent", query="Python") == []

    await db.disconnect()


# ---------------------------------------------------------------------------
# 4. Standalone & Engine Unified Cutover Execution Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_standalone_and_engine_execute_native_and_legacy_workflows():
    GLOBAL_REGISTRY.clear()
    load_entry_point_plugins()
    for mod in ("hyperforge_static", "hyperforge_passthrough"):
        hyperforge.configure.scan(mod)
        hyperforge.configure.load_all_configurations(mod)

    # 1) StaticAgentManager supports both native Harness and legacy 4-stage workflows
    sa_cfg = StandAloneAgentConfig(
        title="Unified Agent",
        instructions="Follow instructions.",
        workflows={
            "native_code": WorkflowConfig(
                name="Native Code",
                execution_mode="code",
                toolkits=[
                    {
                        "toolkit": "static",
                        "namespace": "docs",
                        "context": "Native Harness context payload",
                    }
                ],
                workflow_code=(
                    "res = docs__get_static_context()\n"
                    "text = res['contexts'][0]['chunks'][0]['text']\n"
                    "output({'answer': 'Answer: ' + text, 'contexts': res['contexts']})\n"
                ),
            ),
            "legacy_stages": WorkflowConfig(
                name="Legacy Stages",
                context=[
                    {
                        "module": "static",
                        "context": "Transpiled legacy context",
                        "prune_context": False,
                    }
                ],
                generation=[{"module": "passthrough"}],
            ),
        },
    )
    static_mgr = StaticAgentManager({"my_agent": sa_cfg})
    native_resolved = await static_mgr.get_agent_config(
        "standalone", "my_agent", workflow_id="native_code"
    )
    assert native_resolved.execution_mode == "code"

    legacy_resolved = await static_mgr.get_agent_config(
        "standalone", "my_agent", workflow_id="legacy_stages"
    )
    assert legacy_resolved.execution_mode == "code"
    assert legacy_resolved.workflow_code is not None

    # 2) Execute native Code Mode workflow end-to-end through `engine.main()` with zero LLM calls
    qmem_native = await engine_main(
        retrieval_config=native_resolved,
        question="What is the payload?",
    )
    assert qmem_native.final_answer == "Answer: Native Harness context payload"
    assert len(qmem_native.contexts) == 1

    # 3) Execute legacy 4-stage dict directly through `engine.main()` (transparently transpiled to Harness)
    qmem_legacy = await engine_main(
        config={
            "drivers": [],
            "preprocess": [],
            "context": [
                {
                    "module": "static",
                    "context": "Zero-LLM passthrough over Harness!",
                    "prune_context": False,
                }
            ],
            "generation": [{"module": "passthrough"}],
            "postprocess": [],
        },
        question="Show passthrough",
    )
    assert "Zero-LLM passthrough over Harness!" in qmem_legacy.final_answer
    assert len(qmem_legacy.contexts) == 1
