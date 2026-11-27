from __future__ import annotations

import importlib.metadata
from collections.abc import AsyncIterator, Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from hyperforge.configure import (
    GLOBAL_REGISTRY,
    Registry,
    get_toolkit_schemas,
    load_all_configurations,
    load_entry_point_plugins,
    migration,
    skill,
    toolkit,
)
from hyperforge.db.harness_storage import PostgresHarnessStorage
from hyperforge.harness import HarnessAgent, HarnessAgentConfig
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessConversation,
    HarnessEvent,
    HarnessEventType,
    HarnessMemory,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    tool,
)
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory
from hyperforge.models import Chunk, Context, MemoryConfig, Rules
from hyperforge.prompts import (
    PromptArgument,
    PromptConfig,
    prompt_config_to_skill,
    render_prompt_config,
)


class ConnectorQuery(BaseModel):
    question: str


class ConnectorResult(BaseModel):
    contexts: list[dict[str, Any]]


class ExternalToolkitConfig(ToolkitConfig):
    toolkit: str = "ext_connector"
    endpoint: str = "https://api.example.com"


class ExternalConnectorToolkit(BaseToolkit[ExternalToolkitConfig]):
    toolkit_id = "ext_connector"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        endpoint = self.config.endpoint

        @tool(name="fetch_docs", description="Fetch documents from external connector.")
        async def fetch_docs(
            _context: ToolCallContext, value: ConnectorQuery
        ) -> ConnectorResult:
            ctx = Context(
                agent_id="ext",
                original_question_uuid=None,
                actual_question_uuid=None,
                question=value.question,
                source=endpoint,
                agent="ext_connector",
                title="External Docs",
                chunks=[
                    Chunk(
                        chunk_id="c-1",
                        title="Architecture",
                        text=f"Doc for {value.question}",
                        origin_agent="ext_connector",
                    )
                ],
                image_urls=["https://cdn.example.com/diagram.png?eph-token=abc"],
            )
            return ConnectorResult(contexts=[ctx.model_dump(mode="json")])

        return [fetch_docs]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="connector_skill",
                name="External Connector Guide",
                description="Guidance for querying {namespace}.",
                instructions="Use `{namespace_prefix}fetch_docs` for factual lookups.",
                recommended_tools=("fetch_docs",),
                preload=True,
            )
        ]


def test_entry_point_plugin_expansion_registers_toolkits_skills_and_migrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_registry = Registry()

    @toolkit(
        id="ext_connector",
        title="External Connector",
        description="Third-party toolkit",
        config_schema=ExternalToolkitConfig,
    )
    class _RegisteredToolkit(ExternalConnectorToolkit):
        pass

    @skill(id="global_triage")
    def _build_skill() -> HarnessSkill:
        return HarnessSkill(
            id="global_triage",
            name="Global Triage",
            description="Triage requests",
            instructions="Classify before acting.",
        )

    @migration(id="ext_legacy")
    def _migrate_ext(raw: dict[str, Any]) -> dict[str, Any]:
        return {"toolkit": "ext_connector", **raw}

    load_all_configurations(__name__, _context=local_registry)
    assert "ext_connector" in local_registry.toolkits
    assert "global_triage" in local_registry.skills
    assert "ext_legacy" in local_registry.migrations

    ep_skill = HarnessSkill(
        id="ep_skill",
        name="Entry Point Skill",
        description="Loaded from entry_points",
        instructions="Follow entry point rules.",
    )
    fake_eps = {
        "hyperforge.skills": [
            MagicMock(
                name="ep_skill_entry",
                value=f"{__name__}:ep_skill",
                load=MagicMock(return_value=ep_skill),
            )
        ],
        "hyperforge.migrations": [
            MagicMock(
                name="ep_mig_entry",
                value=f"{__name__}:_migrate_ext",
                load=MagicMock(return_value=_migrate_ext),
            )
        ],
    }
    fake_eps["hyperforge.skills"][0].name = "ep_skill"
    fake_eps["hyperforge.migrations"][0].name = "ep_mig"

    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: fake_eps.get(group, []),
    )
    monkeypatch.setattr("hyperforge.configure.has_feature", lambda **kwargs: False)

    ep_registry = Registry()
    load_entry_point_plugins(_context=ep_registry)
    assert "ep_skill" in ep_registry.skills
    assert "ep_mig" in ep_registry.migrations
    schemas = get_toolkit_schemas(_context=local_registry)
    assert "ext_connector" in schemas


def test_prompt_config_rendering_and_skill_conversion() -> None:
    jinja_prompt = PromptConfig(
        prompt_id="p-1",
        name="domain_assistant",
        description="Domain instructions",
        prompt="Assist tenant {{ tenant }} with {{ topic }}.",
        arguments=[
            PromptArgument(name="tenant", required=True),
            PromptArgument(name="topic", required=True),
        ],
    )
    assert (
        render_prompt_config(jinja_prompt, {"tenant": "Acme", "topic": "billing"})
        == "Assist tenant Acme with billing."
    )
    with pytest.raises(ValueError, match="Missing required prompt argument"):
        render_prompt_config(jinja_prompt, {"tenant": "Acme"})

    converted = prompt_config_to_skill(
        jinja_prompt, arguments={"tenant": "Acme", "topic": "RAG"}, preload=True
    )
    assert converted.id == "p-1"
    assert converted.preload is True
    assert converted.instructions == "Assist tenant Acme with RAG."


@pytest.mark.asyncio
async def test_harness_agent_ingests_toolkit_skills_and_bridges_steps_and_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hyperforge.configure import ToolkitRegistry

    monkeypatch.setitem(
        GLOBAL_REGISTRY.toolkits,
        "ext_connector",
        ToolkitRegistry(
            id="ext_connector",
            title="External Connector",
            description="Connector toolkit",
            config_schema=ExternalToolkitConfig,
            klass=ExternalConnectorToolkit,
        ),
    )

    class FakeNua:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_completions_stream(
            self, payload: dict[str, Any], **kwargs: Any
        ) -> AsyncIterator[dict[str, Any]]:
            del kwargs
            self.calls += 1
            system_messages = [
                m["content"] for m in payload["messages"] if m["role"] == "system"
            ]
            assert any("`kb__fetch_docs`" in text for text in system_messages)
            assert any("Tenant: acme" in text for text in system_messages)
            if self.calls == 1:
                yield {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "tc-1",
                                        "function": {
                                            "name": "kb__fetch_docs",
                                            "arguments": '{"question":"hyperforge"}',
                                        },
                                    }
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            else:
                yield {
                    "choices": [{"index": 0, "delta": {"content": "Grounded answer"}}]
                }

    config = HarnessAgentConfig.model_validate(
        {
            "model": "test-model",
            "toolkits": [
                {
                    "toolkit": "ext_connector",
                    "namespace": "kb",
                    "endpoint": "https://kb.example.com",
                }
            ],
            "prompts": [
                {
                    "prompt_id": "sys-1",
                    "name": "tenant_prompt",
                    "description": "Tenant prompt",
                    "prompt": "Tenant: {{ tenant }}",
                }
            ],
            "prompt_id": "sys-1",
        }
    )
    agent = await HarnessAgent.from_config_class(config)
    manager = Manager()
    manager.nua = FakeNua()  # type: ignore[assignment]
    session = BaseSessionMemory.from_config(
        MemoryConfig(),
        agent_id="agent-1",
        workflow_id="default",
        rules=Rules(rules=["Cite sources"]),
    )
    session.init("sess-1")
    memory = session.start_question(
        "Explain hyperforge", arguments={"tenant": "acme"}, streaming=True
    )

    await agent(memory, manager)

    assert memory.final_answer == "Grounded answer"
    assert len(memory.contexts) == 1
    assert memory.contexts[0].chunks[0].text == "Doc for hyperforge"
    assert memory.final_answer_urls == [
        "https://cdn.example.com/diagram.png?eph-token=abc"
    ]
    assert any(step.title == "Tool: kb__fetch_docs" for step in memory.steps)


@pytest.mark.asyncio
async def test_postgres_harness_storage_enforces_tenant_scope_and_roundtrips() -> None:
    rows_conv: dict[tuple[str, str, str], dict[str, Any]] = {}
    rows_events: list[dict[str, Any]] = []
    rows_mem: list[dict[str, Any]] = []

    db = MagicMock()

    async def fake_fetch_one(stmt: Any) -> dict[str, Any] | None:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        for (acc, aid, cid), record in rows_conv.items():
            if (
                f"'{acc}'" in compiled
                and f"'{aid}'" in compiled
                and f"'{cid}'" in compiled
            ):
                return record
        return None

    async def fake_execute(stmt: Any) -> None:
        table_name = stmt.table.name
        compiled_params = stmt.compile().params
        if table_name == "harness_conversations":
            if "title" in compiled_params and "created_datetime" in compiled_params:
                key = (
                    compiled_params["account"],
                    compiled_params["agent_id"],
                    compiled_params["id"],
                )
                rows_conv[key] = dict(compiled_params)
            elif "title" in compiled_params:
                for key, existing in rows_conv.items():
                    if existing["id"] == compiled_params.get("id_1"):
                        existing.update(
                            title=compiled_params["title"],
                            updated_datetime=compiled_params["updated_datetime"],
                        )
        elif table_name == "harness_events":
            rows_events.append({**compiled_params, "seq": len(rows_events) + 1})
        elif table_name == "harness_memories":
            if "text" in compiled_params:
                rows_mem.append({**compiled_params, "seq": len(rows_mem) + 1})
            else:
                rows_mem[:] = [
                    m for m in rows_mem if m["id"] != compiled_params.get("id_1")
                ]

    async def fake_fetch_all(stmt: Any) -> list[dict[str, Any]]:
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        if "FROM harness_events" in sql:
            return [
                r
                for r in rows_events
                if f"'{r['account']}'" in sql
                and f"'{r['agent_id']}'" in sql
                and f"'{r['conversation_id']}'" in sql
            ]
        if "FROM harness_memories" in sql:
            return list(
                reversed(
                    [
                        r
                        for r in rows_mem
                        if f"'{r['account']}'" in sql and f"'{r['agent_id']}'" in sql
                    ]
                )
            )
        return []

    db.fetch_one = AsyncMock(side_effect=fake_fetch_one)
    db.execute = AsyncMock(side_effect=fake_execute)
    db.fetch_all = AsyncMock(side_effect=fake_fetch_all)

    storage_a = PostgresHarnessStorage(db, account="acct-a", agent_id="agent-1")
    storage_b = PostgresHarnessStorage(db, account="acct-b", agent_id="agent-1")

    conv = HarnessConversation(id="c-1", title="Thread")
    await storage_a.create_conversation(conv)
    assert await storage_a.get_conversation("c-1") is not None
    assert await storage_b.get_conversation("c-1") is None

    event = HarnessEvent(
        id="e-1",
        conversation_id="c-1",
        agent_id="h-1",
        type=HarnessEventType.TURN_STARTED,
        payload={"prompt": "hi"},
    )
    await storage_a.append_event(event)
    loaded_events = [item async for item in storage_a.iter_events("c-1")]
    assert len(loaded_events) == 1
    assert loaded_events[0].payload == {"prompt": "hi"}

    mem = HarnessMemory(id="m-1", scope="user", text="Prefers tables")
    await storage_a.remember(mem)
    assert len(await storage_a.recall(scope="user", query="tables")) == 1
    assert await storage_b.recall(scope="user", query="tables") == []
    await storage_a.forget("m-1")
    assert await storage_a.recall(scope="user", query="tables") == []
