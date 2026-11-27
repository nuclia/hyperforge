from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from hyperforge.codemode import RestrictedPythonTask, WorkerExecutionRequest
from hyperforge.harness_sdk import (
    AgentHarness,
    BaseToolkit,
    CodeModeDispatch,
    HarnessEventType,
    HarnessSkill,
    HarnessTool,
    HarnessToolCall,
    InMemoryHarnessStorage,
    ModelDelta,
    ToolCallContext,
    ToolkitConfig,
    ToolkitRuntime,
    assemble_toolkits,
    create_codemode_tool,
    create_subagent_tool,
    namespace_skill,
    tool,
)
from hyperforge.result_payload import ResultPayloadBudget


class QueryInput(BaseModel):
    query: str


class QueryOutput(BaseModel):
    items: list[str]


class StaticModel:
    def __init__(self, deltas: Sequence[Sequence[ModelDelta]] = ()) -> None:
        self.turns = list(deltas)
        self.seen_requests: list[dict[str, Any]] = []

    async def stream(self, **kwargs: Any) -> AsyncIterator[ModelDelta]:
        self.seen_requests.append(kwargs)
        if not self.turns:
            yield ModelDelta(text="done", input_tokens=2, output_tokens=1)
            return
        for delta in self.turns.pop(0):
            yield delta


def test_namespace_skill_preserves_json_braces_and_scopes_tools() -> None:
    raw = HarnessSkill(
        id="nucliadb_guide",
        name="NucliaDB Guide",
        description="Search {namespace} documents.",
        instructions=(
            'Call `{namespace_prefix}search` with JSON like {"filters": ["/l/topic/ai"]} '
            "on connector `{namespace}`."
        ),
        recommended_tools=("search", "external__ping"),
        code_examples=(
            'res = {namespace_prefix}search(query="test")\noutput({"ok": res})',
        ),
    )

    scoped = namespace_skill(raw, "docs")

    assert scoped.id == "docs__nucliadb_guide"
    assert scoped.recommended_tools == ("docs__search", "external__ping")
    assert "`docs__search`" in scoped.instructions
    assert '{"filters": ["/l/topic/ai"]}' in scoped.instructions
    assert 'docs__search(query="test")' in scoped.code_examples[0]


def test_load_skill_tool_only_exposed_when_catalog_skills_exist() -> None:
    plain = AgentHarness(model="m", model_client=StaticModel())
    assert "load_skill" not in {item.name for item in plain.iter_tools()}

    preloaded_only = AgentHarness(
        model="m",
        model_client=StaticModel(),
        skills=[
            HarnessSkill(
                id="rules",
                name="Rules",
                description="Always-on rules",
                instructions="Follow policy.",
                preload=True,
            )
        ],
    )
    assert "load_skill" not in {item.name for item in preloaded_only.iter_tools()}
    assert "## Active Skills" in preloaded_only.system_prompt

    catalog_harness = AgentHarness(
        model="m",
        model_client=StaticModel(),
        skills=[
            HarnessSkill(
                id="guide",
                name="Guide",
                description="On-demand guide",
                instructions="Detailed steps.",
                preload=False,
            )
        ],
    )
    assert "load_skill" in {item.name for item in catalog_harness.iter_tools()}
    assert "<available_skills>" in catalog_harness.system_prompt


@pytest.mark.asyncio
async def test_load_skill_activates_lazy_recommended_tools_and_persists_across_reload() -> (
    None
):
    @tool(description="Lazy search tool", lazy_load=True)
    async def docs__search(_context: ToolCallContext, value: QueryInput) -> QueryOutput:
        return QueryOutput(items=[f"hit:{value.query}"])

    skill = HarnessSkill(
        id="docs__guide",
        name="Docs Guide",
        description="Use when searching docs.",
        instructions="Always filter carefully.",
        recommended_tools=("docs__search",),
        code_examples=('output(docs__search(query="hi"))',),
        preload=False,
    )
    storage = InMemoryHarnessStorage()
    model = StaticModel(
        [
            [
                ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="c1",
                            name="load_skill",
                            arguments={"skill_id": "docs__guide"},
                        )
                    ]
                )
            ],
            [
                ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="c2",
                            name="call_tool",
                            arguments={
                                "tool_name": "docs__search",
                                "arguments": {"query": "rag"},
                            },
                        )
                    ]
                )
            ],
            [ModelDelta(text="Found hit:rag")],
        ]
    )
    inline_codemode = create_codemode_tool(remote_required=False)
    harness = AgentHarness(
        model="m",
        model_client=model,
        tools=[docs__search, inline_codemode],
        skills=[skill],
        conversation_id="conv-skill-1",
        storage=storage,
    )

    events = [event async for event in harness.run("Search rag")]

    assert events[-1].payload["text"] == "Found hit:rag"
    assert harness.is_skill_loaded("docs__guide")
    assert harness.is_tool_active("docs__search")
    assert any(event.type == HarnessEventType.SKILLS_LOADED for event in events)

    resumed = AgentHarness(
        model="m",
        model_client=StaticModel(),
        tools=[docs__search, inline_codemode],
        skills=[skill],
        conversation_id="conv-skill-1",
        storage=storage,
    )
    await resumed.load(create=False)
    assert resumed.is_skill_loaded("docs__guide")
    assert resumed.is_tool_active("docs__search")


@pytest.mark.asyncio
async def test_toolkit_ingests_namespaced_tools_and_skills() -> None:
    class DemoToolkitConfig(ToolkitConfig):
        toolkit: str = "demo"
        prefix_tag: str = "tag"

    class DemoToolkit(BaseToolkit[DemoToolkitConfig]):
        toolkit_id = "demo"
        preloaded: bool = False

        async def preload(self, runtime: ToolkitRuntime) -> None:
            del runtime
            self.preloaded = True

        def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
            tag = self.config.prefix_tag

            @tool(name="lookup", description="Look up items.")
            async def lookup(
                _context: ToolCallContext, value: QueryInput
            ) -> QueryOutput:
                return QueryOutput(items=[f"{tag}:{value.query}"])

            return [lookup]

        def build_skills(self) -> Sequence[HarnessSkill]:
            return [
                HarnessSkill(
                    id="playbook",
                    name="Demo Playbook",
                    description="How to query {namespace}",
                    instructions="Prefer `{namespace_prefix}lookup`.",
                    recommended_tools=("lookup",),
                )
            ]

    tk = DemoToolkit(DemoToolkitConfig(namespace="crm", lazy_load=True))
    tools, skills = await assemble_toolkits([tk], ToolkitRuntime())

    assert tk.preloaded is True
    assert [item.name for item in tools] == ["crm__lookup"]
    assert tools[0].lazy_load is True
    assert [item.id for item in skills] == ["crm__playbook"]
    assert skills[0].recommended_tools == ("crm__lookup",)
    assert "`crm__lookup`" in skills[0].instructions

    tk_no_skills = DemoToolkit(DemoToolkitConfig(namespace="erp", ingest_skills=False))
    _, skipped_skills = await assemble_toolkits([tk_no_skills])
    assert skipped_skills == []


@pytest.mark.asyncio
async def test_tool_payload_budget_rejects_oversized_output_as_recoverable_error() -> (
    None
):
    @tool(
        description="Returns large items",
        payload_budget=ResultPayloadBudget(max_bytes=64, max_item_bytes=32),
    )
    async def big_tool(_context: ToolCallContext, value: QueryInput) -> QueryOutput:
        return QueryOutput(items=[value.query * 20])

    model = StaticModel(
        [
            [
                ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="b1", name="big_tool", arguments={"query": "abcdef"}
                        )
                    ]
                )
            ],
            [ModelDelta(text="Recovered with narrower query")],
        ]
    )
    harness = AgentHarness(model="m", model_client=model, tools=[big_tool])
    events = [event async for event in harness.run("Run big")]

    failed = [e for e in events if e.type == HarnessEventType.TOOL_FAILED]
    assert len(failed) == 1
    assert (
        "exceeds the configured safety budget" in failed[0].payload["result"]["error"]
    )
    assert events[-1].payload["text"] == "Recovered with narrower query"


@pytest.mark.asyncio
async def test_subagent_tool_works_inside_codemode() -> None:
    sub_model = StaticModel(
        [[ModelDelta(text="sub-agent synthesis", input_tokens=3, output_tokens=2)]]
    )
    researcher = create_subagent_tool(
        name="researcher",
        description="Delegate synthesis to a sub-agent",
        model="sub-m",
        model_client=sub_model,
    )

    class InlineRunner:
        async def run(
            self, request: WorkerExecutionRequest, dispatch: CodeModeDispatch
        ) -> Any:
            del request
            sub_res = await dispatch(
                RestrictedPythonTask(
                    function="researcher",
                    agent="harness",
                    args=(),
                    keyword_args={"prompt": "Summarize findings"},
                )
            )
            await dispatch(
                RestrictedPythonTask(
                    function="output",
                    agent="harness",
                    args=({"delegated": sub_res["text"]},),
                    keyword_args={},
                )
            )

    cm = create_codemode_tool(runner=InlineRunner(), remote_required=False)
    root_model = StaticModel(
        [
            [
                ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="cm1",
                            name="codemode",
                            arguments={"code": 'r = researcher(prompt="x")\noutput(r)'},
                        )
                    ]
                )
            ],
            [ModelDelta(text="Done via codemode")],
        ]
    )
    harness = AgentHarness(
        model="root-m",
        model_client=root_model,
        tools=[researcher, cm],
    )

    events = [event async for event in harness.run("Orchestrate")]

    assert events[-1].payload["text"] == "Done via codemode"
    completed = [
        e
        for e in events
        if e.type == HarnessEventType.TOOL_COMPLETED
        and e.payload.get("tool") == "codemode"
    ]
    assert completed[0].payload["result"]["value"] == {
        "delegated": "sub-agent synthesis"
    }
    assert harness.usage.turns >= 3
