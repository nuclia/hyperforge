import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from pydantic import BaseModel, ValidationError

from hyperforge.harness.config import HarnessAgentConfig
from hyperforge.harness_sdk import (
    AgentHarness,
    HarnessEvent,
    HarnessEventType,
    HarnessTool,
    HarnessToolCall,
    InMemoryHarnessStorage,
    ModelDelta,
    ToolCallContext,
    UsageLimitExceeded,
    UsageLimits,
)
from hyperforge.procedural.graph import ProceduralGraph, ProcedureEdge, ProcedureNode
from hyperforge.procedural.guidance import ProceduralGuidanceConfig


class Value(BaseModel):
    value: str


class ScriptedModel:
    def __init__(self, *responses: ModelDelta):
        self.responses = responses
        self.calls = []

    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        self.calls.append(
            {
                **kwargs,
                "messages": [m.model_copy(deep=True) for m in kwargs["messages"]],
                "tools": list(kwargs.get("tools") or ()),
            }
        )
        assert len(self.calls) <= len(self.responses), "Unexpected solver call"
        yield self.responses[len(self.calls) - 1]


class GuidanceModel:
    def __init__(self, *, mode="success", text=None):
        self.mode = mode
        self.text = text
        self.calls = []
        self.prompts = []
        self.cancelled = False

    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        assert not kwargs.get("tools"), "Guidance must not receive executable tools"
        self.calls.append(kwargs)
        user_messages = [m for m in kwargs["messages"] if m.role == "user"]
        assert len(user_messages) == 1
        prompt = json.loads(user_messages[0].content)
        assert isinstance(prompt["query"], str)
        assert isinstance(prompt["graph_context"], dict)
        assert isinstance(prompt["recent_trajectory"], list)
        self.prompts.append(prompt)
        # Token counts are cumulative within a stream, not additive deltas.
        yield ModelDelta(input_tokens=2, output_tokens=1)
        if self.mode == "error":
            raise RuntimeError("guidance provider failed")
        if self.mode == "timeout":
            try:
                await asyncio.sleep(60)
            finally:
                self.cancelled = True
        text = (
            self.text if self.text is not None else f"private-advice-{len(self.calls)}"
        )
        calls = []
        if self.mode == "empty":
            text = " \n\t"
        elif self.mode == "malformed":
            text = {"not": "text"}
        elif self.mode == "tools":
            calls = [
                HarnessToolCall(
                    id="advice-tool", name="search", arguments={"value": "unsafe"}
                )
            ]
        yield ModelDelta(
            text=text,
            tool_calls=calls,
            input_tokens=7,
            output_tokens=3,
            model_input_tokens=7,
            model_output_tokens=3,
            nuclia_input_tokens=0.007,
            nuclia_output_tokens=0.003,
        )


@pytest.fixture
def graph():
    return ProceduralGraph(
        nodes=(
            ProcedureNode(id="Start", type="STATUS"),
            ProcedureNode(id="Search", type="ACTION", procedure_name="search"),
            ProcedureNode(id="Inspect", type="ACTION", procedure_name="inspect"),
            ProcedureNode(id="End", type="STATUS"),
        ),
        edges=(
            ProcedureEdge(source="Start", target="Search", guidance="Search first"),
            ProcedureEdge(
                source="Search", target="Inspect", guidance="Inspect results"
            ),
            ProcedureEdge(
                source="Inspect", target="End", guidance="Summarize findings"
            ),
        ),
    )


@pytest.fixture
def tools():
    async def execute(_context: ToolCallContext, value: Value) -> Value:
        return Value(value=value.value.upper())

    return [HarnessTool("search", execute), HarnessTool("inspect", execute)]


def call(name="search", call_id="call-1", value="hello"):
    return HarnessToolCall(id=call_id, name=name, arguments={"value": value})


def guidance_messages(messages):
    return [
        message
        for message in messages
        if message.role == "system" and "Procedural Graph Guidance" in message.content
    ]


async def run(harness, query="Find evidence"):
    async with asyncio.timeout(5):
        return [event async for event in harness.run(query)]


def test_config_defaults_frozen_and_serialization(graph):
    config = ProceduralGuidanceConfig(graph=graph)
    assert config.hops == 2
    assert config.window == 3
    assert config.model is None
    assert config.timeout_seconds == 30
    assert config.failure_policy == "unguided"
    assert config.single_action is False
    assert config.max_guidance_chars == 8000
    assert config.max_observation_chars == 8000
    assert config.max_graph_chars == 100000
    with pytest.raises(ValidationError, match="frozen"):
        config.window = 5

    agent_config = HarnessAgentConfig(model="solver", procedural_guidance=config)
    serialized = agent_config.model_dump_json()
    assert json.loads(serialized)["procedural_guidance"] == config.model_dump(
        mode="json"
    )
    restored = HarnessAgentConfig.model_validate_json(serialized)
    assert restored.procedural_guidance == config


def test_full_graph_budget_is_checked_before_startup(graph, tools):
    local_size = len(json.dumps(graph.neighborhood("Start", hops=0)))
    full_size = len(json.dumps(graph.neighborhood("", hops=0)))
    assert local_size < full_size
    config = ProceduralGuidanceConfig(graph=graph, hops=0, max_graph_chars=full_size)
    with pytest.raises(ValueError, match="max_graph_chars"):
        ProceduralGuidanceConfig(graph=graph, hops=0, max_graph_chars=local_size)

    solver = ScriptedModel()
    guidance = GuidanceModel()
    # model_copy bypasses validation, so the harness must revalidate its snapshot.
    with pytest.raises(ValueError, match="max_graph_chars"):
        AgentHarness(
            model="solver",
            model_client=solver,
            tools=tools,
            procedural_guidance=config.model_copy(
                update={"max_graph_chars": local_size}
            ),
            guidance_client=guidance,
        )
    assert solver.calls == guidance.calls == []


def test_unknown_action_binding_is_rejected_at_constructor(graph, tools):
    solver = ScriptedModel()
    guidance = GuidanceModel()
    with pytest.raises(ValueError, match="Unknown ACTION.*inspect"):
        AgentHarness(
            model="solver",
            model_client=solver,
            tools=[tools[0]],
            procedural_guidance=ProceduralGuidanceConfig(graph=graph),
            guidance_client=guidance,
        )
    assert solver.calls == guidance.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("hops", [0, 2])
async def test_start_exact_localization_and_temporary_guidance(graph, tools, hops):
    solver = ScriptedModel(
        ModelDelta(tool_calls=[call()]), ModelDelta(text="Evidence found")
    )
    guidance = GuidanceModel()
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph, hops=hops),
        guidance_client=guidance,
    )
    expected_tools = list(harness.iter_tools())
    events = await run(harness)

    assert len(guidance.calls) == len(solver.calls) == 2
    assert all(request["model"] == "solver" for request in guidance.calls)
    assert guidance.prompts[0]["query"] == "Find evidence"
    assert guidance.prompts[0]["graph_context"] == graph.neighborhood(
        "Start", hops=hops
    )
    assert guidance.prompts[0]["recent_trajectory"] == []
    assert guidance.prompts[1]["graph_context"] == graph.neighborhood(
        "search", hops=hops
    )
    for index, request in enumerate(solver.calls, 1):
        messages = guidance_messages(request["messages"])
        assert len(messages) == 1
        assert f"private-advice-{index}" in messages[0].content
        assert f"private-advice-{3 - index}" not in messages[0].content
        assert request["tools"] == expected_tools
    assert guidance_messages(harness.messages) == []
    assert "private-advice" not in json.dumps(
        [m.model_dump(mode="json") for m in harness.messages]
    )
    assert [
        e.payload["text"] for e in events if e.type == HarnessEventType.TEXT_DELTA
    ] == ["Evidence found"]

    guidance_events = [
        e for e in events if e.type == HarnessEventType.PROCEDURAL_GUIDANCE
    ]
    assert len(guidance_events) == 2
    for index, event in enumerate(guidance_events):
        assert event.payload["graph_version"] == graph.fingerprint
        assert event.payload["active_node"] == ["Start", "Search"][index]
        assert event.payload["scope"] == "local"
        assert event.payload["status"] == "completed"
        assert event.payload["text"] == f"private-advice-{index + 1}"
        assert not event.payload.get("error")
        assert event.payload["input_tokens"] == 7
        assert event.payload["output_tokens"] == 3
        assert event.payload["latency_seconds"] >= 0
    steps = harness.procedural_trajectory
    assert isinstance(steps, tuple)
    assert len(steps) == 1
    step = steps[0]
    assert step.procedure_name == "search"
    assert step.call_id == "call-1"
    assert step.status == "completed"
    assert step.decision_id
    assert "hello" in step.arguments
    assert "HELLO" in step.observation
    recorded = guidance.prompts[1]["recent_trajectory"][0]
    for field in (
        "procedure_name",
        "call_id",
        "arguments",
        "observation",
        "status",
        "decision_id",
    ):
        assert recorded[field] == getattr(step, field)
    step_events = [e for e in events if e.type == HarnessEventType.PROCEDURAL_STEP]
    assert len(step_events) == 1
    assert step_events[0].payload["call_id"] == step.call_id
    assert step_events[0].payload["decision_id"] == step.decision_id


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["Search", "SEARCH", "missing"])
async def test_localization_miss_uses_full_graph(graph, tools, name):
    solver = ScriptedModel(ModelDelta(tool_calls=[call(name)]), ModelDelta(text="done"))
    guidance = GuidanceModel()
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
        guidance_client=guidance,
    )
    await run(harness)

    assert guidance.prompts[1]["graph_context"] == graph.neighborhood(name)
    assert guidance.prompts[1]["graph_context"]["active_node"] is None
    assert guidance.prompts[1]["graph_context"]["scope"] == "full"
    step = harness.procedural_trajectory[0]
    assert step.status == "failed"
    assert "Unknown tool" in step.observation


@pytest.mark.asyncio
async def test_tool_errors_recorded_and_recent_window_is_last_three(graph):
    async def execute(_context: ToolCallContext, value: Value) -> Value:
        if value.value == "2":
            raise ValueError("search unavailable")
        return value

    solver = ScriptedModel(
        *(
            ModelDelta(tool_calls=[call(call_id=f"call-{i}", value=str(i))])
            for i in range(5)
        ),
        ModelDelta(text="done"),
    )
    guidance = GuidanceModel()
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=[HarnessTool("search", execute), HarnessTool("inspect", execute)],
        procedural_guidance=ProceduralGuidanceConfig(graph=graph, window=3),
        guidance_client=guidance,
    )
    events = await run(harness)

    assert len(harness.procedural_trajectory) == 5
    assert [len(p["recent_trajectory"]) for p in guidance.prompts] == [0, 1, 2, 3, 3, 3]
    recent = guidance.prompts[-1]["recent_trajectory"]
    assert [step["call_id"] for step in recent] == ["call-2", "call-3", "call-4"]
    assert [step["status"] for step in recent] == ["failed", "completed", "completed"]
    assert "search unavailable" in recent[0]["observation"]
    assert len({step.decision_id for step in harness.procedural_trajectory}) == 5
    steps = [e.payload for e in events if e.type == HarnessEventType.PROCEDURAL_STEP]
    assert [step["call_id"] for step in steps] == [f"call-{i}" for i in range(5)]
    assert steps[2]["status"] == "failed"


@pytest.mark.asyncio
async def test_parallel_steps_and_anchor_follow_model_order_not_completion(graph):
    second_finished = asyncio.Event()
    completion_order = []

    async def execute(context: ToolCallContext, value: Value) -> Value:
        if context.id == "first":
            await second_finished.wait()
        completion_order.append(context.id)
        if context.id == "second":
            second_finished.set()
        return value

    solver = ScriptedModel(
        ModelDelta(tool_calls=[call("search", "first"), call("inspect", "second")]),
        ModelDelta(text="done"),
    )
    guidance = GuidanceModel()
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=[HarnessTool("search", execute), HarnessTool("inspect", execute)],
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
        guidance_client=guidance,
    )
    events = await run(harness)

    assert completion_order == ["second", "first"]
    assert [step.call_id for step in harness.procedural_trajectory] == [
        "first",
        "second",
    ]
    assert len({step.decision_id for step in harness.procedural_trajectory}) == 1
    assert [
        e.payload["call_id"]
        for e in events
        if e.type == HarnessEventType.PROCEDURAL_STEP
    ] == ["first", "second"]
    assert [s["call_id"] for s in guidance.prompts[1]["recent_trajectory"]] == [
        "first",
        "second",
    ]
    assert guidance.prompts[1]["graph_context"] == graph.neighborhood("inspect")


@pytest.mark.asyncio
async def test_disabled_guidance_preserves_normal_tool_loop(tools):
    solver = ScriptedModel(ModelDelta(tool_calls=[call()]), ModelDelta(text="done"))
    guidance = GuidanceModel()
    harness = AgentHarness(
        model="solver", model_client=solver, tools=tools, guidance_client=guidance
    )
    events = await run(harness)

    assert guidance.calls == []
    assert all(not guidance_messages(request["messages"]) for request in solver.calls)
    assert not any(
        e.type
        in {HarnessEventType.PROCEDURAL_GUIDANCE, HarnessEventType.PROCEDURAL_STEP}
        for e in events
    )
    assert harness.procedural_trajectory == ()
    assert harness.usage.tool_calls == 1
    assert events[-1].payload["text"] == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["unguided", "raise"])
@pytest.mark.parametrize("mode", ["error", "timeout", "empty", "malformed", "tools"])
async def test_guidance_failures_never_execute_tools(graph, policy, mode):
    side_effects = []

    async def execute(_context: ToolCallContext, value: Value) -> Value:
        side_effects.append(value.value)
        return value

    solver = ScriptedModel(ModelDelta(text="unguided answer"))
    guidance = GuidanceModel(mode=mode)
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=[HarnessTool("search", execute), HarnessTool("inspect", execute)],
        procedural_guidance=ProceduralGuidanceConfig(
            graph=graph,
            failure_policy=policy,
            timeout_seconds=0.01 if mode == "timeout" else 30,
        ),
        guidance_client=guidance,
    )
    if policy == "raise":
        with pytest.raises((RuntimeError, ValueError, TypeError, TimeoutError)):
            await run(harness)
        assert solver.calls == []
    else:
        events = await run(harness)
        assert events[-1].payload["text"] == "unguided answer"
        assert len(solver.calls) == 1
        assert guidance_messages(solver.calls[0]["messages"]) == []
        assert [
            e.payload["text"] for e in events if e.type == HarnessEventType.TEXT_DELTA
        ] == ["unguided answer"]
    assert side_effects == []
    assert harness.procedural_trajectory == ()
    assert harness.usage.tool_calls == 0
    if mode == "timeout":
        assert guidance.cancelled
    failures = [
        e
        async for e in harness.history()
        if e.type == HarnessEventType.PROCEDURAL_GUIDANCE
    ]
    assert len(failures) == 1
    payload = failures[0].payload
    assert payload["status"] == "failed"
    assert payload["error"]
    assert payload["graph_version"] == graph.fingerprint
    assert payload["active_node"] == "Start"
    assert payload["scope"] == "local"
    assert payload["latency_seconds"] >= 0
    assert payload["input_tokens"] >= 2


@pytest.mark.asyncio
async def test_failed_refresh_does_not_reuse_previous_guidance(graph, tools):
    guidance = GuidanceModel()

    class Solver(ScriptedModel):
        async def stream(self, **kwargs):
            async for delta in super().stream(**kwargs):
                yield delta
            guidance.mode = "error"

    solver = Solver(ModelDelta(tool_calls=[call()]), ModelDelta(text="done"))
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
        guidance_client=guidance,
    )
    await run(harness)
    assert len(guidance_messages(solver.calls[0]["messages"])) == 1
    assert guidance_messages(solver.calls[1]["messages"]) == []
    assert guidance_messages(harness.messages) == []


@pytest.mark.asyncio
async def test_usage_includes_guidance_without_double_counting_cumulative_tokens(
    graph, tools
):
    class Solver(ScriptedModel):
        async def stream(self, **kwargs):
            yield ModelDelta(input_tokens=1, output_tokens=1)
            async for delta in super().stream(**kwargs):
                yield delta

    solver = Solver(
        ModelDelta(
            text="done",
            input_tokens=11,
            output_tokens=5,
            model_input_tokens=11,
            model_output_tokens=5,
            nuclia_input_tokens=0.011,
            nuclia_output_tokens=0.005,
        )
    )
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
        guidance_client=GuidanceModel(),
    )
    await run(harness)
    assert harness.usage.input_tokens == 18
    assert harness.usage.output_tokens == 8
    assert harness.usage.model_input_tokens == 18
    assert harness.usage.model_output_tokens == 8
    assert harness.usage.nuclia_input_tokens == pytest.approx(0.018)
    assert harness.usage.nuclia_output_tokens == pytest.approx(0.008)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits", [UsageLimits(max_input_tokens=6), UsageLimits(max_output_tokens=2)]
)
async def test_guidance_token_limit_is_not_swallowed_by_unguided_policy(
    graph, tools, limits
):
    solver = ScriptedModel(ModelDelta(text="must not run"))
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        usage_limits=limits,
        procedural_guidance=ProceduralGuidanceConfig(
            graph=graph, failure_policy="unguided"
        ),
        guidance_client=GuidanceModel(),
    )
    with pytest.raises(UsageLimitExceeded):
        await run(harness)
    assert solver.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["tools", "oversized", "malformed"])
@pytest.mark.parametrize("limit_name", ["max_input_tokens", "max_output_tokens"])
async def test_token_limit_takes_precedence_over_invalid_guidance_delta(
    graph, tools, invalid, limit_name
):
    class StreamingGuidance(ScriptedModel):
        async def stream(self, **kwargs):
            assert not kwargs.get("tools")
            yield ModelDelta(text="partial advice", input_tokens=2, output_tokens=1)
            async for delta in super().stream(**kwargs):
                yield delta

    guidance = StreamingGuidance(
        ModelDelta(
            text={"invalid": "text"}
            if invalid == "malformed"
            else "x" * 81
            if invalid == "oversized"
            else "",
            tool_calls=[call(call_id="forbidden-guidance-call")]
            if invalid == "tools"
            else [],
            input_tokens=7,
            output_tokens=7,
        )
    )
    solver = ScriptedModel(ModelDelta(text="must not run"))
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        usage_limits=UsageLimits(**{limit_name: 6}),
        procedural_guidance=ProceduralGuidanceConfig(
            graph=graph, max_guidance_chars=80
        ),
        guidance_client=guidance,
    )
    with pytest.raises(UsageLimitExceeded, match=limit_name):
        await run(harness)

    assert len(guidance.calls) == 1
    assert solver.calls == []
    assert harness.usage.input_tokens == 7
    assert harness.usage.output_tokens == 7
    assert harness.usage.tool_calls == 0
    assert harness.procedural_trajectory == ()
    assert guidance_messages(harness.messages) == []
    history = [event async for event in harness.history()]
    assert not any(event.type == HarnessEventType.TOOL_REQUESTED for event in history)
    failures = [
        event for event in history if event.type == HarnessEventType.PROCEDURAL_GUIDANCE
    ]
    assert len(failures) == 1
    assert failures[0].payload["status"] == "failed"
    assert failures[0].payload["text"] == ""
    assert "UsageLimitExceeded" in failures[0].payload["error"]
    assert limit_name in failures[0].payload["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["interrupt", "global_timeout"])
async def test_cancelled_guidance_records_cumulative_usage(graph, tools, stop):
    waiting = asyncio.Event()
    cancelled = asyncio.Event()

    class StreamingGuidance:
        async def stream(self, **kwargs):
            assert not kwargs.get("tools")
            for count in (2, 7, 7):
                yield ModelDelta(
                    text="partial advice ",
                    input_tokens=count,
                    output_tokens=count,
                    model_input_tokens=count,
                    model_output_tokens=count,
                    nuclia_input_tokens=count / 1000,
                    nuclia_output_tokens=count / 1000,
                )
            waiting.set()
            try:
                await asyncio.sleep(60)
            finally:
                cancelled.set()

    solver = ScriptedModel(ModelDelta(text="must not run"))
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        usage_limits=UsageLimits(max_time=0.05 if stop == "global_timeout" else None),
        procedural_guidance=ProceduralGuidanceConfig(graph=graph, timeout_seconds=30),
        guidance_client=StreamingGuidance(),
    )
    task = asyncio.create_task(run(harness))
    try:
        await asyncio.wait_for(waiting.wait(), timeout=2)
        if stop == "interrupt":
            harness.interrupt()
            events = await task
            assert events[-1].type == HarnessEventType.TURN_INTERRUPTED
            assert not any(e.type == HarnessEventType.TEXT_DELTA for e in events)
        else:
            with pytest.raises(UsageLimitExceeded, match="max_time"):
                await task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert cancelled.is_set()
    assert solver.calls == []
    assert harness.usage.input_tokens == 7
    assert harness.usage.output_tokens == 7
    assert harness.usage.model_input_tokens == 7
    assert harness.usage.model_output_tokens == 7
    assert harness.usage.nuclia_input_tokens == pytest.approx(0.007)
    assert harness.usage.nuclia_output_tokens == pytest.approx(0.007)
    assert harness.procedural_trajectory == ()
    assert guidance_messages(harness.messages) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_single_action_rejects_entire_batch_before_side_effects(
    graph, persistent
):
    side_effects = []

    async def execute(context: ToolCallContext, value: Value) -> Value:
        side_effects.append(context.id)
        return value

    class Solver:
        def __init__(self):
            self.calls = []

        async def stream(self, **kwargs):
            self.calls.append([m.model_copy(deep=True) for m in kwargs["messages"]])
            # A runaway retry loop should fail the test rather than hang it.
            assert len(self.calls) < 20, "Single-action retries must be bounded"
            if persistent or len(self.calls) == 1:
                assert side_effects == []
                yield ModelDelta(
                    tool_calls=[
                        call("search", "rejected-a"),
                        call("inspect", "rejected-b"),
                    ]
                )
            elif len(self.calls) == 2:
                assert side_effects == []
                assert any(
                    m.role == "user" and m.content != "Find evidence"
                    for m in kwargs["messages"]
                )
                yield ModelDelta(tool_calls=[call("search", "accepted")])
            else:
                yield ModelDelta(text="done")

    solver = Solver()
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=[HarnessTool("search", execute), HarnessTool("inspect", execute)],
        procedural_guidance=ProceduralGuidanceConfig(graph=graph, single_action=True),
        guidance_client=GuidanceModel(),
    )
    if persistent:
        with pytest.raises(ValueError):
            await run(harness)
        assert side_effects == []
        assert harness.procedural_trajectory == ()
    else:
        await run(harness)
        assert side_effects == ["accepted"]
        assert [s.call_id for s in harness.procedural_trajectory] == ["accepted"]
    history = [e async for e in harness.history()]
    assert not any(
        e.type == HarnessEventType.TOOL_REQUESTED
        and e.payload["call"]["id"].startswith("rejected")
        for e in history
    )
    assert harness.usage.tool_calls == len(side_effects)


@pytest.mark.asyncio
async def test_observation_arguments_and_guidance_are_bounded(graph, tools):
    solver = ScriptedModel(
        ModelDelta(tool_calls=[call(value="x" * 1000)]), ModelDelta(text="done")
    )
    guidance = GuidanceModel(text="advice " * 1000)
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(
            graph=graph,
            max_observation_chars=64,
            max_guidance_chars=80,
        ),
        guidance_client=guidance,
    )
    events = await run(harness)
    step = harness.procedural_trajectory[0]
    for value in (step.arguments, step.observation):
        assert isinstance(value, str)
        assert 0 < len(value) <= 64
    recent = guidance.prompts[1]["recent_trajectory"][0]
    assert recent["arguments"] == step.arguments
    assert recent["observation"] == step.observation
    guidance_events = [
        event for event in events if event.type == HarnessEventType.PROCEDURAL_GUIDANCE
    ]
    assert len(guidance_events) == 2
    for event in guidance_events:
        assert event.payload["status"] == "failed"
        assert event.payload["text"] == ""
        assert "max_guidance_chars" in event.payload["error"]
    assert all(guidance_messages(request["messages"]) == [] for request in solver.calls)
    assert guidance_messages(harness.messages) == []
    assert [
        e.payload["text"] for e in events if e.type == HarnessEventType.TEXT_DELTA
    ] == ["done"]
    assert events[-1].payload["text"] == "done"


@pytest.mark.asyncio
async def test_latest_query_and_trajectory_survive_turns_with_pinned_graph(
    graph, tools
):
    config = ProceduralGuidanceConfig(graph=graph)
    guidance = GuidanceModel()
    solver = ScriptedModel(
        ModelDelta(tool_calls=[call()]),
        ModelDelta(text="first done"),
        ModelDelta(text="second done"),
    )
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=config,
        guidance_client=guidance,
    )
    await run(harness, "first query")
    original_steps = harness.procedural_trajectory
    revised = ProceduralGraph(
        nodes=graph.nodes,
        edges=tuple(
            edge.model_copy(update={"guidance": "Revised guidance"})
            for edge in graph.edges
        ),
    )
    other_guidance = GuidanceModel()
    other = AgentHarness(
        model="solver",
        model_client=ScriptedModel(ModelDelta(text="other done")),
        tools=tools,
        procedural_guidance=config.model_copy(update={"graph": revised}),
        guidance_client=other_guidance,
    )
    await run(other, "other query")
    events = await run(harness, "latest query")

    assert guidance.prompts[-1]["query"] == "latest query"
    assert guidance.prompts[-1]["graph_context"] == graph.neighborhood("search")
    assert guidance.prompts[-1]["recent_trajectory"][0]["call_id"] == "call-1"
    assert harness.procedural_trajectory == original_steps
    assert other.procedural_trajectory == ()
    assert other_guidance.prompts[0]["recent_trajectory"] == []
    assert other_guidance.prompts[0]["graph_context"] == revised.neighborhood("Start")
    assert {
        e.payload["graph_version"]
        for e in events
        if e.type == HarnessEventType.PROCEDURAL_GUIDANCE
    } == {graph.fingerprint}
    assert revised.fingerprint != graph.fingerprint


@pytest.mark.asyncio
@pytest.mark.parametrize("child_version", [None, "matching", "different"])
async def test_resume_restores_root_trajectory_and_anchor_without_duplicates(
    graph, tools, child_version
):
    storage = InMemoryHarnessStorage()
    config = ProceduralGuidanceConfig(graph=graph)
    original = AgentHarness(
        model="solver",
        model_client=ScriptedModel(
            ModelDelta(tool_calls=[call()]), ModelDelta(text="done")
        ),
        tools=tools,
        storage=storage,
        procedural_guidance=config,
        guidance_client=GuidanceModel(),
    )
    await run(original, "original query")
    conversation = await storage.get_conversation(original.conversation_id)
    assert conversation.metadata["procedural_graph_version"] == graph.fingerprint
    original_steps = original.procedural_trajectory
    assert len(original_steps) == 1
    if child_version is not None:
        await storage.append_event(
            HarnessEvent(
                id="child-step-event",
                conversation_id=original.conversation_id,
                agent_id="child-agent",
                parent_agent_id=original.agent_id,
                type=HarnessEventType.PROCEDURAL_STEP,
                payload={
                    **original_steps[0].model_dump(mode="json"),
                    "call_id": "child-call",
                    "procedure_name": "inspect",
                    "graph_version": graph.fingerprint
                    if child_version == "matching"
                    else "other-graph-version",
                },
            )
        )

    solver = ScriptedModel(ModelDelta(text="resumed answer"))
    guidance = GuidanceModel()
    resumed = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        storage=storage,
        conversation_id=original.conversation_id,
        procedural_guidance=config,
        guidance_client=guidance,
    )
    assert resumed.agent_id != original.agent_id
    assert resumed.procedural_trajectory == ()
    for _ in range(2):
        await resumed.load(create=False)
        assert resumed.procedural_trajectory == original_steps
        assert resumed.messages == original.messages
    assert solver.calls == guidance.calls == []

    await run(resumed, "resumed query")
    assert len(guidance.prompts) == 1
    assert guidance.prompts[0]["query"] == "resumed query"
    assert guidance.prompts[0]["graph_context"] == graph.neighborhood("search")
    assert guidance.prompts[0]["recent_trajectory"] == [
        original_steps[0].model_dump(mode="json")
    ]
    assert resumed.procedural_trajectory == original_steps
    assert guidance_messages(resumed.messages) == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing_guidance", [False, True], ids=["legacy", "different-graph"]
)
async def test_resume_rejects_missing_or_different_graph_metadata_before_loading(
    graph, tools, existing_guidance, mocker
):
    storage = InMemoryHarnessStorage()
    original = AgentHarness(
        model="solver",
        model_client=ScriptedModel(ModelDelta(text="original answer")),
        tools=tools,
        storage=storage,
        procedural_guidance=ProceduralGuidanceConfig(graph=ProceduralGraph.skeleton())
        if existing_guidance
        else None,
        guidance_client=GuidanceModel(),
    )
    await run(original)
    conversation = await storage.get_conversation(original.conversation_id)
    if not existing_guidance:
        assert "procedural_graph_version" not in conversation.metadata
    else:
        assert conversation.metadata["procedural_graph_version"] != graph.fingerprint

    solver = ScriptedModel(ModelDelta(text="must not run"))
    guidance = GuidanceModel()
    resumed = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        storage=storage,
        conversation_id=original.conversation_id,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
        guidance_client=guidance,
    )
    read_events = mocker.spy(storage, "iter_events")
    with pytest.raises(ValueError, match="procedural graph"):
        await resumed.load(create=False)
    with pytest.raises(ValueError, match="procedural graph"):
        await run(resumed)
    read_events.assert_not_called()
    assert solver.calls == guidance.calls == []
    assert resumed.messages == []
    assert resumed.procedural_trajectory == ()
    assert resumed.conversation is None
    assert await storage.get_conversation(original.conversation_id) == conversation


@pytest.mark.asyncio
async def test_resume_rejects_root_step_from_different_graph(graph, tools):
    storage = InMemoryHarnessStorage()
    config = ProceduralGuidanceConfig(graph=graph)
    original = AgentHarness(
        model="solver",
        model_client=ScriptedModel(
            ModelDelta(tool_calls=[call()]), ModelDelta(text="done")
        ),
        tools=tools,
        storage=storage,
        procedural_guidance=config,
        guidance_client=GuidanceModel(),
    )
    await run(original)
    await storage.append_event(
        HarnessEvent(
            id="mismatched-root-step",
            conversation_id=original.conversation_id,
            agent_id=original.agent_id,
            type=HarnessEventType.PROCEDURAL_STEP,
            payload={
                **original.procedural_trajectory[0].model_dump(mode="json"),
                "call_id": "mismatched-call",
                "graph_version": "other-graph-version",
            },
        )
    )
    solver = ScriptedModel(ModelDelta(text="must not run"))
    guidance = GuidanceModel()
    resumed = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        storage=storage,
        conversation_id=original.conversation_id,
        procedural_guidance=config,
        guidance_client=guidance,
    )
    with pytest.raises(ValueError, match="different graph version"):
        await resumed.load(create=False)
    assert solver.calls == guidance.calls == []
    assert all(
        step.graph_version == graph.fingerprint
        for step in resumed.procedural_trajectory
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("include_history", [False, True])
async def test_child_does_not_inherit_guidance_or_trajectory(
    graph, tools, include_history
):
    solver = ScriptedModel(
        ModelDelta(tool_calls=[call()]),
        ModelDelta(text="parent done"),
        ModelDelta(text="child done"),
    )
    guidance = GuidanceModel()
    parent = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
        guidance_client=guidance,
    )
    await run(parent)
    guidance_count = len(guidance.calls)
    child = parent._create_child("child", include_history=include_history)
    events = await run(child, "child query")

    assert len(guidance.calls) == guidance_count
    assert child.procedural_trajectory == ()
    assert len(parent.procedural_trajectory) == 1
    assert guidance_messages(solver.calls[-1]["messages"]) == []
    assert not any(e.type == HarnessEventType.PROCEDURAL_GUIDANCE for e in events)


@pytest.mark.asyncio
@pytest.mark.parametrize("guidance_model", [None, "advisor"])
async def test_guidance_client_defaults_to_solver_client(graph, tools, guidance_model):
    solver = ScriptedModel(ModelDelta(text="private advice"), ModelDelta(text="answer"))
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph, model=guidance_model),
    )
    events = await run(harness)
    assert len(solver.calls) == 2
    assert solver.calls[0]["model"] == (guidance_model or "solver")
    assert solver.calls[0]["tools"] == []
    assert solver.calls[1]["model"] == "solver"
    assert solver.calls[1]["tools"] == list(harness.iter_tools())
    assert "private advice" in guidance_messages(solver.calls[1]["messages"])[0].content
    assert [
        e.payload["text"] for e in events if e.type == HarnessEventType.TEXT_DELTA
    ] == ["answer"]


@pytest.mark.asyncio
async def test_guidance_accepts_async_iterator_without_aclose(graph, tools):
    class GuidanceIterator:
        def __init__(self):
            self.deltas = iter(
                [
                    ModelDelta(text="private advice", input_tokens=2, output_tokens=1),
                    ModelDelta(input_tokens=7, output_tokens=3),
                ]
            )

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.deltas)
            except StopIteration:
                raise StopAsyncIteration from None

    class GuidanceClient:
        def stream(self, **kwargs):
            assert not kwargs.get("tools")
            return GuidanceIterator()

    solver = ScriptedModel(ModelDelta(text="answer"))
    harness = AgentHarness(
        model="solver",
        model_client=solver,
        tools=tools,
        procedural_guidance=ProceduralGuidanceConfig(
            graph=graph, failure_policy="raise"
        ),
        guidance_client=GuidanceClient(),
    )
    events = await run(harness)
    guidance_events = [
        e for e in events if e.type == HarnessEventType.PROCEDURAL_GUIDANCE
    ]
    assert len(guidance_events) == 1
    assert guidance_events[0].payload["status"] == "completed"
    assert "private advice" in guidance_messages(solver.calls[0]["messages"])[0].content
    assert harness.usage.input_tokens == 7
    assert harness.usage.output_tokens == 3
    assert [
        e.payload["text"] for e in events if e.type == HarnessEventType.TEXT_DELTA
    ] == ["answer"]
