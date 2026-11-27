from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from hyperforge.configure import GLOBAL_REGISTRY, ToolkitRegistry
from hyperforge.harness import HarnessAgent, HarnessAgentConfig
from hyperforge.harness_sdk import (
    AgentHarness,
    BaseToolkit,
    CodeModeLimits,
    HarnessEventType,
    HarnessTool,
    InMemoryHarnessStorage,
    ModelDelta,
    ToolCallContext,
    ToolkitConfig,
    create_codemode_tool,
    parse_workflow_output,
    tool,
)
from hyperforge.llm import NoopNuaClient
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory
from hyperforge.models import Chunk, Context, HistoryQuestionAnswer, MemoryConfig, Rules


class FailIfCalledModel:
    """Asserts that deterministic code workflows make zero solver LLM calls."""

    async def stream(self, **kwargs: Any) -> AsyncIterator[ModelDelta]:
        del kwargs
        raise AssertionError("Solver LLM must not be called in code workflow mode")
        yield ModelDelta(text="")


class QueryInput(BaseModel):
    query: str


class ItemsOutput(BaseModel):
    items: list[str]


class SynthInput(BaseModel):
    question: str
    contexts: list[dict[str, Any]]


class SynthOutput(BaseModel):
    answer: str
    citations: dict[str, Any]
    is_answered: bool


class GapInput(BaseModel):
    question: str
    attempt: int


class GapOutput(BaseModel):
    answered: bool
    questions: list[str]


@pytest.mark.asyncio
async def test_codemode_parallel_executes_concurrently_and_preserves_order() -> None:
    active = 0
    peak_active = 0

    @tool(description="Simulated async connector lookup")
    async def fetch_source(_ctx: ToolCallContext, value: QueryInput) -> ItemsOutput:
        nonlocal active, peak_active
        active += 1
        peak_active = max(peak_active, active)
        try:
            await asyncio.sleep(0.05)
            return ItemsOutput(items=[f"doc:{value.query}"])
        finally:
            active -= 1

    cm = create_codemode_tool(remote_required=False)
    harness = AgentHarness(
        model="unused",
        model_client=FailIfCalledModel(),
        tools=[fetch_source, cm],
    )

    code = """
empty = parallel([])
results = parallel([
    call(fetch_source, query="alpha"),
    call("fetch_source", query="beta"),
    call(fetch_source, "gamma"),
])
output({"empty": empty, "items": [r["items"][0] for r in results]})
"""
    res = await cm.execute(
        ToolCallContext(harness=harness, name=cm.name, id="p-outer"),
        {"code": code, "question": "run batch"},
    )

    assert res.value == {
        "empty": [],
        "items": ["doc:alpha", "doc:beta", "doc:gamma"],
    }
    assert peak_active >= 2
    assert harness.usage.tool_calls == 3


@pytest.mark.asyncio
async def test_codemode_parallel_enforces_nested_limits_terminally_even_when_caught() -> (
    None
):
    @tool(description="Echo")
    async def echo(_ctx: ToolCallContext, value: QueryInput) -> ItemsOutput:
        return ItemsOutput(items=[value.query])

    cm = create_codemode_tool(
        limits=CodeModeLimits(max_nested_calls=2),
        remote_required=False,
    )
    harness = AgentHarness(
        model="unused",
        model_client=FailIfCalledModel(),
        tools=[echo, cm],
    )

    code = """
try:
    parallel([
        call(echo, query="1"),
        call(echo, query="2"),
        call(echo, query="3"),
    ])
except Exception:
    pass
try:
    output("escaped")
except Exception:
    pass
"""
    with pytest.raises(RuntimeError, match="nested call limit exceeded: 3 > 2"):
        await cm.execute(
            ToolCallContext(harness=harness, name=cm.name, id="p-limit"),
            {"code": code},
        )
    assert harness.usage.tool_calls == 0


@pytest.mark.asyncio
async def test_codemode_rejects_output_inside_call_or_parallel() -> None:
    cm = create_codemode_tool(remote_required=False)
    harness = AgentHarness(
        model="unused",
        model_client=FailIfCalledModel(),
        tools=[cm],
    )

    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await cm.execute(
            ToolCallContext(harness=harness, name=cm.name),
            {"code": 'c = call("output", 1)\noutput(c)'},
        )

    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await cm.execute(
            ToolCallContext(harness=harness, name=cm.name),
            {
                "code": (
                    'parallel([{"__call__": "output", "args": [1], "kwargs": {}}])\n'
                    "output(2)"
                )
            },
        )


@pytest.mark.asyncio
async def test_codemode_emit_step_and_workflow_context_globals() -> None:
    cm = create_codemode_tool(remote_required=False)
    harness = AgentHarness(
        model="unused",
        model_client=FailIfCalledModel(),
        tools=[cm],
        execution_context={
            "workflow_context": {
                "arguments": {"department": "finance"},
                "history": [{"role": "user", "content": "prior q"}],
                "rules": ["Be accurate"],
                "user_info": {"region": "EU"},
            }
        },
    )

    code = """
emit_step("Preprocess", reason="Inspecting arguments", value=workflow_context["arguments"]["department"])
output({
    "q": workflow_context["question"],
    "args": workflow_context["arguments"],
    "history_len": len(workflow_context["history"]),
    "rules": workflow_context["rules"],
    "region": workflow_context["user_info"]["region"],
})
"""
    events = [
        event
        async for event in harness.run_code_workflow(
            code,
            "What is Q4 revenue?",
            codemode_tool=cm,
            arguments={"quarter": "Q4"},
        )
    ]

    step_events = [e for e in events if e.type == HarnessEventType.WORKFLOW_STEP]
    assert len(step_events) == 1
    assert step_events[0].payload["title"] == "Preprocess"
    assert step_events[0].payload["value"] == "finance"

    completed = events[-1]
    assert completed.type == HarnessEventType.TURN_COMPLETED
    assert '"quarter": "Q4"' in completed.payload["text"]
    assert '"department": "finance"' in completed.payload["text"]
    assert '"history_len": 1' in completed.payload["text"]


@pytest.mark.asyncio
async def test_full_four_stage_pipeline_with_restart_runs_via_harness_code_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PipelineToolkitConfig(ToolkitConfig):
        toolkit: str = "pipeline_demo"

    class PipelineToolkit(BaseToolkit[PipelineToolkitConfig]):
        toolkit_id = "pipeline_demo"

        def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
            @tool(name="split_query", description="Preprocess: split complex query")
            async def split_query(
                _ctx: ToolCallContext, value: QueryInput
            ) -> ItemsOutput:
                if "retry" in value.query:
                    return ItemsOutput(items=["revenue retry", "margin retry"])
                return ItemsOutput(items=["revenue", "margin"])

            @tool(name="search_kb", description="Context: retrieve chunks")
            async def search_kb(_ctx: ToolCallContext, value: QueryInput) -> SynthInput:
                ctx = Context(
                    id=f"ctx-{value.query.replace(' ', '-')}",
                    original_question_uuid=None,
                    actual_question_uuid=None,
                    question=value.query,
                    source="kb-main",
                    agent="nucliadb",
                    title=f"KB {value.query}",
                    chunks=[
                        Chunk(
                            chunk_id=f"ch-{value.query}",
                            title="Report",
                            text=f"Evidence for {value.query}",
                            origin_url="https://docs.example.com/report.pdf",
                        )
                    ],
                )
                return SynthInput(
                    question=value.query,
                    contexts=[ctx.model_dump(mode="json")],
                )

            @tool(
                name="synthesize", description="Generation: synthesize with citations"
            )
            async def synthesize(
                _ctx: ToolCallContext, value: SynthInput
            ) -> SynthOutput:
                first_ctx_id = value.contexts[0]["id"]
                return SynthOutput(
                    answer=f"Synthesized {len(value.contexts)} contexts for: {value.question} [1]\n\n[1]: block-AA",
                    citations={
                        "metadata": {
                            "block-AA": {
                                "context_id": first_ctx_id,
                                "origin_urls": ["https://docs.example.com/report.pdf"],
                                "chunk_index": 0,
                            }
                        }
                    },
                    is_answered=True,
                )

            @tool(name="check_gaps", description="Postprocess: restart gap check")
            async def check_gaps(_ctx: ToolCallContext, value: GapInput) -> GapOutput:
                if value.attempt == 0:
                    return GapOutput(answered=False, questions=["revenue retry"])
                return GapOutput(answered=True, questions=[])

            return [split_query, search_kb, synthesize, check_gaps]

    monkeypatch.setitem(
        GLOBAL_REGISTRY.toolkits,
        "pipeline_demo",
        ToolkitRegistry(
            id="pipeline_demo",
            title="Pipeline Demo",
            description="4-stage demo toolkit",
            config_schema=PipelineToolkitConfig,
            klass=PipelineToolkit,
        ),
    )

    workflow_code = """
attempt = 0
answered = False
queries = []
contexts = []
final_synth = None

while not answered and attempt < 2:
    if attempt == 0:
        prep = pipe__split_query(query=question)
        queries = prep["items"]
        emit_step("Preprocess", reason="Split query", value=", ".join(queries))

    batches = parallel([call(pipe__search_kb, query=q) for q in queries])
    for batch in batches:
        contexts.extend(batch["contexts"])

    final_synth = pipe__synthesize(question=question, contexts=contexts)

    gap = pipe__check_gaps(question=question, attempt=attempt)
    if gap["answered"]:
        answered = True
    else:
        queries = gap["questions"]
        emit_step("Restart", reason="Missing details", value=", ".join(queries))
    attempt = attempt + 1

output({
    "answer": final_synth["answer"],
    "citations": final_synth["citations"],
    "contexts": contexts,
    "related_questions": ["What about Q1 guidance?"],
    "is_answered": final_synth["is_answered"],
})
"""

    config = HarnessAgentConfig.model_validate(
        {
            "execution_mode": "code",
            "workflow_code": workflow_code,
            "toolkits": [
                {
                    "toolkit": "pipeline_demo",
                    "namespace": "pipe",
                    "lazy_load": True,
                }
            ],
            "usage_limits": {"max_tool_calls": 25},
        }
    )
    storage = InMemoryHarnessStorage()
    agent = await HarnessAgent.from_config_class(config, storage=storage)
    manager = Manager()
    manager.nua = NoopNuaClient()

    session = BaseSessionMemory.from_config(
        MemoryConfig(),
        agent_id="rag-agent",
        workflow_id="default",
        rules=Rules(rules=["Strict grounding"]),
    )
    session.init("conv-rag-1")
    streamed: list[str] = []

    async def on_event(ans: Any) -> None:
        if ans.streaming_response_chunk and ans.streaming_response_chunk.text:
            streamed.append(ans.streaming_response_chunk.text)

    memory = session.start_question(
        "Compare revenue and margin",
        arguments={"fy": "2026"},
        streaming=True,
        chat_history=[HistoryQuestionAnswer(question="Hello", answer="Hi!")],
    )
    memory.set_callback_fn(on_event)

    await agent(memory, manager)

    assert memory.final_answer is not None
    assert "Synthesized 3 contexts" in memory.final_answer
    assert streamed and "Synthesized 3 contexts" in "".join(streamed)
    assert memory.is_answered is True
    assert memory.final_answer_citations is not None
    assert "block-AA" in memory.final_answer_citations.metadata
    assert len(memory.contexts) == 3
    assert list(memory.future_questions.values()) == ["What about Q1 guidance?"]
    step_titles = [step.title for step in memory.steps]
    assert "Preprocess" in step_titles
    assert "Restart" in step_titles


def test_parse_workflow_output_variants() -> None:
    assert parse_workflow_output("plain text").answer == "plain text"
    none_out = parse_workflow_output(None)
    assert none_out.answer == ""
    assert none_out.is_answered is False

    structured = parse_workflow_output(
        {
            "answer": "42",
            "related_questions": ["Why 42?"],
            "extra_metric": 0.98,
        }
    )
    assert structured.answer == "42"
    assert structured.related_questions == ["Why 42?"]
    assert structured.metadata == {"extra_metric": 0.98}

    arbitrary = parse_workflow_output({"count": 5})
    assert arbitrary.answer == '{"count": 5}'
