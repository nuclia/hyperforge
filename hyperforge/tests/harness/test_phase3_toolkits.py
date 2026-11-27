from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as mcp_types
import pytest
from hyperforge_a2a import A2AToolkit, A2AToolkitConfig
from hyperforge_external import WebhookToolkit, WebhookToolkitConfig
from hyperforge_google import GoogleSearchToolkit, GoogleSearchToolkitConfig
from hyperforge_http import HTTPToolkit, HTTPToolkitConfig
from hyperforge_mcp import MCPToolkit, MCPToolkitConfig
from hyperforge_nucliadb import (
    NucliaDBSyncToolkit,
    NucliaDBSyncToolkitConfig,
    NucliaDBToolkit,
    NucliaDBToolkitConfig,
)
from hyperforge_perplexity import PerplexityToolkit, PerplexityToolkitConfig
from nuclia_models.predict.generative_responses import GenerativeFullResponse
from nuclia_models.predict.remi import AnswerRelevance, RemiResponse

from hyperforge.configure import (
    Registry,
    load_all_configurations,
    load_entry_point_plugins,
    scan,
)
from hyperforge.harness_sdk import (
    AgentHarness,
    HarnessEventType,
    ModelDelta,
    ToolCallContext,
    ToolkitRuntime,
    assemble_toolkits,
    create_codemode_tool,
)
from hyperforge.manager import Manager
from hyperforge.models import Chunk, Context
from hyperforge.toolkits import (
    ContextOpsToolkit,
    ContextOpsToolkitConfig,
    DecisionToolkit,
    DecisionToolkitConfig,
    EvaluationToolkit,
    EvaluationToolkitConfig,
    RephraseToolkit,
    RephraseToolkitConfig,
    StaticContextToolkit,
    StaticContextToolkitConfig,
    SynthesisToolkit,
    SynthesisToolkitConfig,
)


class NoopModelClient:
    async def stream(self, **kwargs: Any):
        del kwargs
        yield ModelDelta(text="unused")


def _make_harness_with_manager(
    manager: Manager, tools: list[Any] | None = None
) -> AgentHarness:
    return AgentHarness(
        model="test-model",
        model_client=NoopModelClient(),
        tools=tools or [],
        execution_context={"manager": manager, "drivers": manager.drivers},
    )


def test_all_builtin_and_connector_toolkits_register_with_bundled_skills() -> None:
    reg = Registry()
    load_entry_point_plugins(_context=reg)
    for mod in (
        "hyperforge_nucliadb.toolkit",
        "hyperforge_mcp.toolkit",
        "hyperforge_a2a.toolkit",
        "hyperforge_google.toolkit",
        "hyperforge_perplexity.toolkit",
        "hyperforge_http.toolkit",
        "hyperforge_external.toolkit",
    ):
        scan(mod)
        load_all_configurations(mod, _context=reg)

    expected_toolkits = {
        "rephrase",
        "context_ops",
        "static",
        "synthesis",
        "decision",
        "evaluation",
        "nucliadb",
        "nucliadb_sync",
        "mcp",
        "a2a",
        "google",
        "perplexity",
        "http",
        "external",
    }
    assert expected_toolkits.issubset(reg.toolkits.keys())
    assert {"plan_and_execute", "citation_grounded_answer"}.issubset(reg.skills.keys())

    # Verify every connector toolkit bundles at least one skill when configured
    connector_instances = [
        NucliaDBToolkit(NucliaDBToolkitConfig(namespace="kb")),
        NucliaDBSyncToolkit(NucliaDBSyncToolkitConfig(namespace="kbsync")),
        A2AToolkit(A2AToolkitConfig(source="remote-a2a", namespace="partner")),
        GoogleSearchToolkit(GoogleSearchToolkitConfig(namespace="gweb")),
        PerplexityToolkit(PerplexityToolkitConfig(namespace="pplx")),
        HTTPToolkit(
            HTTPToolkitConfig(url="https://example.com/api", namespace="ext_http")
        ),
        WebhookToolkit(
            WebhookToolkitConfig(url="https://example.com/hook", namespace="hook")
        ),
    ]
    for instance in connector_instances:
        skills = instance.get_skills()
        assert skills, f"{type(instance).__name__} should bundle skills"
        assert all(s.id.startswith(f"{instance.namespace}__") for s in skills)


@pytest.mark.asyncio
async def test_core_pipeline_toolkits_execute_end_to_end() -> None:
    manager = MagicMock(spec=Manager)
    manager.drivers = {}

    async def fake_execute_json(
        *, user_id: str, prompt: str, schema: dict[str, Any], **kwargs: Any
    ) -> tuple[dict[str, Any], float, float]:
        del prompt, schema, kwargs
        if user_id == "rephrase":
            return (
                {
                    "questions": ["What is Q1 revenue?", "What is Q1 margin?"],
                    "rules": ["USD"],
                },
                5.0,
                3.0,
            )
        if user_id == "context_ops_validate":
            return (
                {
                    "missing_info_query": "Q1 margin details",
                    "contexts": [
                        {
                            "context_id": "ctx-1",
                            "answer": "Q1 revenue was $10M.",
                            "citations": ["block-AA"],
                        }
                    ],
                },
                8.0,
                4.0,
            )
        if user_id == "context_ops_next_rephrase":
            return (
                {
                    "rephrased_question": "Q1 margin for $10M revenue",
                    "needed": True,
                    "reason": "Refine",
                },
                3.0,
                2.0,
            )
        if user_id == "decision_condition":
            return ({"yes": True, "reason": "Matches financial query"}, 2.0, 1.0)
        if user_id == "decision_choose":
            return ({"selected": "finance", "reason": "Best match"}, 2.0, 1.0)
        if user_id == "decision_plan":
            return (
                {
                    "status": "continue",
                    "reasoning": "Need margin",
                    "summary": "Have revenue",
                    "steps": [{"description": "Query margin KB", "reason": "Missing"}],
                },
                4.0,
                2.0,
            )
        if user_id == "evaluation_missing":
            return ({"answered": True, "questions": [], "reason": "Complete"}, 2.0, 1.0)
        if user_id == "evaluation_related":
            return ({"related": ["What was Q2 revenue?"]}, 2.0, 1.0)
        raise AssertionError(f"Unexpected user_id: {user_id}")

    manager.execute_json = AsyncMock(side_effect=fake_execute_json)
    manager.execute_raw = AsyncMock(
        return_value=(
            GenerativeFullResponse(
                answer="Q1 revenue was $10M [1].\n\n[1]: block-AA",
                code="0",
            ),
            10.0,
            6.0,
        )
    )
    manager.remi = AsyncMock(
        side_effect=[
            RemiResponse(
                answer_relevance=AnswerRelevance(score=5, reason="exact"), time=0.1
            ),
            RemiResponse(groundedness=[4], time=0.1),
        ]
    )

    toolkits = [
        RephraseToolkit(
            RephraseToolkitConfig(
                namespace="prep", provided_synonyms={"rev": ["revenue"]}
            )
        ),
        ContextOpsToolkit(ContextOpsToolkitConfig(namespace="ctxops")),
        StaticContextToolkit(
            StaticContextToolkitConfig(namespace="stat", context="Glossary: FY=2026")
        ),
        SynthesisToolkit(
            SynthesisToolkitConfig(
                namespace="synth",
                citations=True,
                force_chunk_level_citations=True,
            )
        ),
        DecisionToolkit(DecisionToolkitConfig(namespace="dec")),
        EvaluationToolkit(EvaluationToolkitConfig(namespace="eval")),
    ]
    tools, skills = await assemble_toolkits(toolkits, ToolkitRuntime())
    assert len(skills) >= 5

    harness = _make_harness_with_manager(manager, tools)
    tool_map = {t.name: t for t in tools}

    split_out = await tool_map["prep__split_query"].execute(
        ToolCallContext(harness=harness, name="prep__split_query"),
        {"question": "Compare Q1 rev and margin"},
    )
    assert split_out.questions == ["What is Q1 revenue?", "What is Q1 margin?"]

    raw_ctx = Context(
        id="ctx-1",
        original_question_uuid=None,
        actual_question_uuid=None,
        question="What is Q1 revenue?",
        source="kb-1",
        agent="nucliadb",
        chunks=[
            Chunk(
                chunk_id="ch-keep",
                title="Filing",
                text="Q1 revenue was $10M",
                origin_url="https://sec.example.com/q1",
            ),
            Chunk(
                chunk_id="ch-drop",
                title="Noise",
                text="Irrelevant footer",
            ),
        ],
    ).model_dump(mode="json")

    val_out = await tool_map["ctxops__validate_contexts"].execute(
        ToolCallContext(harness=harness, name="ctxops__validate_contexts"),
        {"question": "What is Q1 revenue?", "contexts": [raw_ctx], "prune": True},
    )
    assert val_out.useful is True
    assert val_out.missing_query == "Q1 margin details"
    assert [c["chunk_id"] for c in val_out.contexts[0]["chunks"]] == ["ch-keep"]

    synth_out = await tool_map["synth__synthesize_answer"].execute(
        ToolCallContext(harness=harness, name="synth__synthesize_answer"),
        {"question": "What is Q1 revenue?", "contexts": val_out.contexts},
    )
    assert "[1]: block-AA-0" in synth_out.answer
    assert synth_out.citations is not None
    assert "block-AA-0" in synth_out.citations["metadata"]

    cond_out = await tool_map["dec__check_condition"].execute(
        ToolCallContext(harness=harness, name="dec__check_condition"),
        {"text": "What is Q1 revenue?", "condition_prompt": "Is financial?"},
    )
    assert cond_out.matched is True

    plan_out = await tool_map["dec__generate_plan"].execute(
        ToolCallContext(harness=harness, name="dec__generate_plan"),
        {"question": "Compare Q1", "tools_description": "kb__ask"},
    )
    assert plan_out.status == "continue"
    assert len(plan_out.steps) == 1

    remi_out = await tool_map["eval__evaluate_remi"].execute(
        ToolCallContext(harness=harness, name="eval__evaluate_remi"),
        {
            "question": "What is Q1 revenue?",
            "answer": synth_out.answer,
            "contexts": val_out.contexts,
        },
    )
    assert remi_out.answer_relevance == 5.0
    assert remi_out.groundedness == 4.0

    stat_out = await tool_map["stat__get_static_context"].execute(
        ToolCallContext(harness=harness, name="stat__get_static_context"),
        {},
    )
    assert stat_out.contexts[0]["chunks"][0]["text"] == "Glossary: FY=2026"


@pytest.mark.asyncio
async def test_mcp_toolkit_separates_tools_and_ingests_dynamic_skills_in_codemode() -> (
    None
):
    fake_session = MagicMock()
    fake_session.list_tools = AsyncMock(
        return_value=mcp_types.ListToolsResult(
            tools=[
                mcp_types.Tool(
                    name="search_issues",
                    description="Search issue tracker by JQL",
                    inputSchema={
                        "type": "object",
                        "properties": {"jql": {"type": "string"}},
                        "required": ["jql"],
                    },
                )
            ]
        )
    )
    fake_session.list_prompts = AsyncMock(
        return_value=mcp_types.ListPromptsResult(
            prompts=[
                mcp_types.Prompt(
                    name="triage_bug",
                    description="Bug triage checklist",
                    arguments=[
                        mcp_types.PromptArgument(
                            name="severity", description="P1-P4", required=True
                        )
                    ],
                )
            ]
        )
    )
    fake_session.list_resources = AsyncMock(
        return_value=mcp_types.ListResourcesResult(resources=[])
    )
    fake_session.call_tool = AsyncMock(
        return_value=mcp_types.CallToolResult(
            content=[
                mcp_types.TextContent(
                    type="text", text="ISSUE-42: Memory leak fixed in 1.2"
                )
            ],
            isError=False,
        )
    )

    init_res = MagicMock(instructions="Always filter issues by project key.")

    @asynccontextmanager
    async def fake_open_session(*_args: Any, **_kwargs: Any):
        yield fake_session, init_res

    mcp_tk = MCPToolkit(MCPToolkitConfig(source="jira-mcp", namespace="jira"))
    with patch.object(mcp_tk, "_open_session", side_effect=fake_open_session):
        tools, skills = await assemble_toolkits([mcp_tk], ToolkitRuntime())

        tool_names = {t.name for t in tools}
        assert "jira__search_issues" in tool_names
        assert "jira__get_mcp_prompt" in tool_names

        skill_ids = {s.id for s in skills}
        assert "jira__mcp_connector_guide" in skill_ids
        assert "jira__prompt_triage_bug" in skill_ids
        connector_skill = next(s for s in skills if s.id == "jira__mcp_connector_guide")
        assert "Always filter issues by project key." in connector_skill.instructions

        # Execute the MCP tool directly inside Code Mode with zero inner LLM loop!
        cm = create_codemode_tool(remote_required=False)
        manager = Manager()
        harness = AgentHarness(
            model="unused",
            model_client=NoopModelClient(),
            tools=[*tools, cm],
            skills=skills,
            execution_context={"manager": manager, "drivers": {}},
        )
        code = """
res = jira__search_issues(jql="project = HYP")
chunk_text = res["contexts"][0]["chunks"][0]["text"]
output({"answer": chunk_text, "contexts": res["contexts"]})
"""
        events = [
            e
            async for e in harness.run_code_workflow(
                code, "Find HYP issues", codemode_tool=cm
            )
        ]

    assert events[-1].type == HarnessEventType.TURN_COMPLETED
    assert events[-1].payload["text"] == "ISSUE-42: Memory leak fixed in 1.2"
    fake_session.call_tool.assert_awaited_once_with(
        name="search_issues",
        arguments={"jql": "project = HYP"},
        read_timeout_seconds=fake_session.call_tool.await_args.kwargs[
            "read_timeout_seconds"
        ],
    )
