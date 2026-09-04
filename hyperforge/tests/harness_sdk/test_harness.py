import asyncio
from collections.abc import AsyncIterator
from typing import ClassVar

import pytest
from nuclia_models.predict.generative_responses import GenerativeFullResponse
from pydantic import BaseModel, Field

from hyperforge.definition import FunctionDefinition
from hyperforge.harness_sdk import (
    AgentHarness,
    HarnessEvent,
    HarnessEventType,
    HarnessMessage,
    HarnessTool,
    HarnessToolCall,
    InMemoryHarnessStorage,
    LLMCallError,
    ModelDelta,
    NucliaModelClient,
    UsageLimitExceeded,
    UsageLimits,
    codemode,
    tool,
)
from hyperforge.harness_sdk.harness import (
    EMPTY_RESPONSE_RETRY_PROMPT,
    AgentResult,
    TurnLoopState,
)
from hyperforge.harness_sdk.tools.core import (
    AgentIdInput,
    SearchToolsInput,
    SpawnAgentInput,
    search_tools,
    spawn_agent,
    wait_agent,
)
from hyperforge.manager import Manager


class ToolInput(BaseModel):
    value: str


class ToolOutput(BaseModel):
    value: str


class Model:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        del kwargs
        self.calls += 1
        if self.calls == 1:
            yield ModelDelta(
                tool_calls=[
                    HarnessToolCall(
                        id="call-1", name="upper", arguments={"value": "hi"}
                    )
                ]
            )
        else:
            yield ModelDelta(text="HI", input_tokens=4, output_tokens=1)


async def run_harness(harness: AgentHarness, prompt: str) -> str:
    result = "interrupted"
    async for event in harness.run(prompt):
        if (
            event.agent_id == harness.agent_id
            and event.type == HarnessEventType.TURN_COMPLETED
        ):
            result = str(event.payload.get("text", ""))
    return result


def test_codemode_is_not_available_by_default() -> None:
    harness = AgentHarness(model="test-model", model_client=Model())

    assert "codemode" not in {tool.name for tool in harness.iter_tools()}


def test_codemode_can_be_registered_explicitly() -> None:
    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[codemode],
    )

    assert "codemode" in {tool.name for tool in harness.iter_tools()}


def test_turn_loop_clears_pending_tool_result_after_non_empty_response() -> None:
    state = TurnLoopState()
    state.record_tool_calls()

    assert state.retry_prompt(AgentResult(text="Answer")) is None
    assert state.retry_prompt(AgentResult(text="")) == EMPTY_RESPONSE_RETRY_PROMPT


@pytest.mark.asyncio
async def test_llm_wraps_model_failure_with_call_context() -> None:
    error = ValueError("provider failed")

    class FailingModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            yield ModelDelta(trace_id="trace-1", model="resolved-model")
            raise error

    harness = AgentHarness(model="requested-model", model_client=FailingModel())

    with pytest.raises(LLMCallError) as captured:
        await harness.llm()

    assert captured.value.cause is error
    assert captured.value.__cause__ is error
    assert captured.value.call_id
    assert captured.value.trace_id == "trace-1"
    assert captured.value.model == "resolved-model"


@pytest.mark.asyncio
async def test_run_preserves_wrapped_model_failure_context(caplog) -> None:
    class FailingModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            yield ModelDelta(trace_id="trace-1", model="resolved-model")
            raise ValueError("provider failed")

    harness = AgentHarness(model="requested-model", model_client=FailingModel())

    with pytest.raises(LLMCallError):
        await run_harness(harness, "fail")

    record = next(
        record
        for record in caplog.records
        if record.message.startswith("Agent turn failed")
    )
    assert record.call_id
    assert record.trace_id == "trace-1"
    assert record.model == "resolved-model"
    assert record.error_type == "ValueError"
    assert record.error == "provider failed"


@pytest.mark.asyncio
async def test_compact_preserves_tool_call_pair_for_next_generation() -> None:
    class CompactingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="compact-1",
                            name="compact",
                            arguments={
                                "summary": "The user asked about quarterly filings."
                            },
                        )
                    ]
                )
                return

            messages = kwargs["messages"]
            assert [message.role for message in messages] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert messages[-2].tool_calls[0].id == "compact-1"
            assert messages[-1].tool_call_id == "compact-1"
            assert messages[-1].tool_name == "compact"
            yield ModelDelta(text="Compaction succeeded")

    harness = AgentHarness(model="test-model", model_client=CompactingModel())

    assert await run_harness(harness, "Find the filings") == "Compaction succeeded"


@pytest.mark.asyncio
async def test_compact_twice_preserves_only_current_tool_call_pair() -> None:
    class CompactingModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls <= 2:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id=f"compact-{self.calls}",
                            name="compact",
                            arguments={"summary": f"Summary {self.calls}"},
                        )
                    ]
                )
                return
            messages = kwargs["messages"]
            assert [message.role for message in messages] == [
                "system",
                "user",
                "assistant",
                "tool",
            ]
            assert messages[-2].tool_calls[0].id == "compact-2"
            assert messages[-1].tool_call_id == "compact-2"
            yield ModelDelta(text="Compaction succeeded")

    harness = AgentHarness(model="test-model", model_client=CompactingModel())

    assert await run_harness(harness, "Find the filings") == "Compaction succeeded"


@pytest.mark.asyncio
async def test_agent_loop_streams_direct_answer_and_records_usage() -> None:
    class StreamingModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            assert [message.role for message in kwargs["messages"]] == [
                "system",
                "user",
            ]
            yield ModelDelta(text="Hello")
            yield ModelDelta(
                text=" world",
                input_tokens=5,
                output_tokens=2,
                model="resolved-model",
            )

    harness = AgentHarness(model="requested-model", model_client=StreamingModel())
    events = []

    async for event in harness.run("Say hello"):
        events.append(event)

    assert [
        event.payload["text"]
        for event in events
        if event.type == HarnessEventType.TEXT_DELTA
    ] == [
        "Hello",
        " world",
    ]
    assert events[-1].type == HarnessEventType.TURN_COMPLETED
    assert harness.usage.turns == 1
    assert harness.usage.tool_calls == 0
    assert harness.usage.input_tokens == 5
    assert harness.usage.output_tokens == 2


@pytest.mark.asyncio
async def test_agent_reasoning_effort_is_passed_to_model_and_children() -> None:
    class ReasoningModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            assert kwargs["reasoning_effort"] == "high"
            yield ModelDelta(text="Answer")

    harness = AgentHarness(
        model="test-model",
        model_client=ReasoningModel(),
        reasoning_effort="high",
    )

    assert await run_harness(harness, "Think carefully") == "Answer"
    assert (
        harness._create_child("child", include_history=False).reasoning_effort == "high"
    )


@pytest.mark.asyncio
async def test_iter_events_discards_events_from_unconsumed_previous_turn() -> None:
    class DirectModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            yield ModelDelta(text=f"answer-{self.calls}")

    harness = AgentHarness(model="test-model", model_client=DirectModel())
    assert await run_harness(harness, "first") == "answer-1"
    events = []

    async for event in harness.run("second"):
        events.append(event)

    assert {event.turn_id for event in events} == {harness._turn_id}
    assert events[-1].payload == {"text": "answer-2"}


@pytest.mark.asyncio
async def test_unconsumed_turn_does_not_buffer_live_events() -> None:
    harness = AgentHarness(model="test-model", model_client=Model())

    assert await run_harness(harness, "first") == "HI"

    assert harness._event_queue is None


@pytest.mark.asyncio
async def test_slow_consumer_drops_deltas_without_blocking_producer() -> None:
    harness = AgentHarness(model="test-model", model_client=Model(), event_queue_size=1)
    harness._event_queue = asyncio.Queue()

    await harness.emit(HarnessEventType.TEXT_DELTA, {"text": "first"}, persist=False)
    await harness.emit(HarnessEventType.TEXT_DELTA, {"text": "dropped"}, persist=False)
    await harness.emit(HarnessEventType.TURN_COMPLETED, {"text": "complete"})

    assert [harness._event_queue.get_nowait().type for _ in range(2)] == [
        HarnessEventType.TEXT_DELTA,
        HarnessEventType.TURN_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_abandoning_run_stream_cancels_turn() -> None:
    cancelled = asyncio.Event()

    class SlowModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            yield ModelDelta(text="started")
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    harness = AgentHarness(model="test-model", model_client=SlowModel())

    async for event in harness.run("slow"):
        if event.type == HarnessEventType.TEXT_DELTA:
            break

    await harness.__aexit__(None, None, None)
    await asyncio.sleep(0)
    assert cancelled.is_set()
    assert harness._event_queue is None


@pytest.mark.asyncio
async def test_agent_loop_executes_parallel_tools_and_synthesizes_answer() -> None:
    calls = []

    async def upper(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        calls.append(value.value)
        await asyncio.sleep(0)
        return ToolOutput(value=value.value.upper())

    class ParallelToolModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls == 1:
                assert "upper" in {tool.name for tool in kwargs["tools"]}
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="call-a", name="upper", arguments={"value": "a"}
                        ),
                        HarnessToolCall(
                            id="call-b", name="upper", arguments={"value": "b"}
                        ),
                    ],
                    input_tokens=4,
                    output_tokens=2,
                )
                return

            tool_messages = [
                message for message in kwargs["messages"] if message.role == "tool"
            ]
            assert [message.tool_call_id for message in tool_messages] == [
                "call-a",
                "call-b",
            ]
            assert [message.context.content for message in tool_messages] == [
                {"value": "A"},
                {"value": "B"},
            ]
            yield ModelDelta(text="A and B", input_tokens=8, output_tokens=3)

    harness = AgentHarness(
        model="test-model",
        model_client=ParallelToolModel(),
        tools=[HarnessTool("upper", upper)],
    )

    assert await run_harness(harness, "Uppercase both values") == "A and B"
    assert calls == ["a", "b"]
    assert harness.usage.turns == 2
    assert harness.usage.tool_calls == 2
    assert harness.usage.input_tokens == 12
    assert harness.usage.output_tokens == 5
    assert [message.role for message in harness.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_current_tool_call_matches_requested_event() -> None:
    observed: list[HarnessToolCall] = []

    async def inspect(harness: AgentHarness, value: ToolInput) -> ToolOutput:
        current = harness.current_tool_call
        assert current is not None
        observed.append(current)
        return ToolOutput(value=value.value)

    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[HarnessTool("inspect", inspect)],
    )
    call = HarnessToolCall(id="call-current", name="inspect", arguments={"value": "x"})

    assert harness.current_tool_call is None
    await harness._execute_tool_call(call)
    requested = [
        event
        async for event in harness.history()
        if event.type == HarnessEventType.TOOL_REQUESTED
    ]

    assert observed == [call]
    assert requested[0].payload["call"]["id"] == observed[0].id
    assert harness.current_tool_call is None


@pytest.mark.asyncio
async def test_current_tool_call_is_task_local_for_parallel_tools() -> None:
    ready = asyncio.Event()
    started = 0
    observed: dict[str, tuple[str | None, str | None]] = {}

    async def inspect(harness: AgentHarness, value: ToolInput) -> ToolOutput:
        nonlocal started
        started += 1
        if started == 2:
            ready.set()
        before = harness.current_tool_call
        await ready.wait()
        await asyncio.sleep(0)
        after = harness.current_tool_call
        observed[value.value] = (
            before.id if before is not None else None,
            after.id if after is not None else None,
        )
        return ToolOutput(value=value.value)

    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[HarnessTool("inspect", inspect)],
    )
    await asyncio.gather(
        harness._execute_tool_call(
            HarnessToolCall(id="call-a", name="inspect", arguments={"value": "a"})
        ),
        harness._execute_tool_call(
            HarnessToolCall(id="call-b", name="inspect", arguments={"value": "b"})
        ),
    )

    assert observed == {
        "a": ("call-a", "call-a"),
        "b": ("call-b", "call-b"),
    }
    assert harness.current_tool_call is None


@pytest.mark.asyncio
async def test_current_tool_call_is_cleared_after_cancellation() -> None:
    async def slow(harness: AgentHarness, value: ToolInput) -> ToolOutput:
        assert harness.current_tool_call is not None
        await asyncio.sleep(60)
        return ToolOutput(value=value.value)

    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[HarnessTool("slow", slow)],
    )
    call = HarnessToolCall(id="call-slow", name="slow", arguments={"value": "x"})

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await harness._execute_tool_call(call)

    assert harness.current_tool_call is None


@pytest.mark.asyncio
async def test_current_tool_call_is_cleared_after_tool_failure() -> None:
    async def fail(harness: AgentHarness, _value: ToolInput) -> ToolOutput:
        assert harness.current_tool_call is not None
        raise ValueError("failed")

    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[HarnessTool("fail", fail)],
    )
    call = HarnessToolCall(id="call-fail", name="fail", arguments={"value": "x"})

    message = await harness._execute_tool_call(call)

    assert message.content == '{"error":"failed"}'
    assert harness.current_tool_call is None


@pytest.mark.asyncio
async def test_agent_loop_applies_live_steering_before_completion() -> None:
    first_call_started = asyncio.Event()
    release_first_call = asyncio.Event()

    class SteeringModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls == 1:
                first_call_started.set()
                await release_first_call.wait()
                yield ModelDelta(text="Draft")
                return

            assert any(
                "Prefer a concise answer" in message.content
                for message in kwargs["messages"]
            )
            yield ModelDelta(text="Concise answer")

    harness = AgentHarness(model="test-model", model_client=SteeringModel())
    run_task = asyncio.create_task(run_harness(harness, "Answer the question"))
    await first_call_started.wait()
    message_id = await harness.steer("Prefer a concise answer")
    release_first_call.set()

    assert await run_task == "Concise answer"
    assert message_id
    event_types = [event.type async for event in harness.history()]
    assert HarnessEventType.INBOX_ADDED in event_types
    assert HarnessEventType.INBOX_CONSUMED in event_types


@pytest.mark.asyncio
async def test_agent_loop_feedback_round_trip() -> None:
    class FeedbackModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="feedback-call",
                            name="feedback",
                            arguments={"question": "Which format?"},
                        )
                    ]
                )
                return
            assert any(
                message.role == "tool" and "markdown" in message.content
                for message in kwargs["messages"]
            )
            yield ModelDelta(text="I will use markdown.")

    harness = AgentHarness(
        model="test-model", model_client=FeedbackModel(), feedback_enabled=True
    )
    run_task = asyncio.create_task(run_harness(harness, "Prepare the report"))

    while not harness.feedback_requests:
        await asyncio.sleep(0)
    request_id = next(iter(harness.feedback_requests))
    await harness.respond_feedback(request_id, "markdown")

    assert await run_task == "I will use markdown."
    event_types = [event.type async for event in harness.history()]
    assert HarnessEventType.FEEDBACK_REQUESTED in event_types
    assert HarnessEventType.FEEDBACK_RESOLVED in event_types


@pytest.mark.asyncio
async def test_agent_loop_spawns_and_waits_for_sub_agent() -> None:
    class DelegationModel:
        def __init__(self) -> None:
            self.root_calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            messages = kwargs["messages"]
            if any(message.content == "research topic" for message in messages):
                yield ModelDelta(
                    text="research result", input_tokens=2, output_tokens=2
                )
                return

            self.root_calls += 1
            if self.root_calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="spawn-call",
                            name="spawn_agent",
                            arguments={"prompt": "research topic"},
                        )
                    ]
                )
                return
            if self.root_calls == 2:
                spawn_message = next(
                    message
                    for message in reversed(messages)
                    if message.tool_name == "spawn_agent"
                )
                child_id = spawn_message.context.content["value"]["agent_id"]
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="wait-call",
                            name="wait_agent",
                            arguments={"agent_id": child_id},
                        )
                    ]
                )
                return

            assert any(
                message.tool_name == "wait_agent"
                and "research result" in message.content
                for message in messages
            )
            yield ModelDelta(text="Final: research result")

    harness = AgentHarness(model="test-model", model_client=DelegationModel())

    assert await run_harness(harness, "Delegate research") == "Final: research result"
    assert harness.children == {}
    assert len(harness.child_results) == 1
    assert harness.usage.turns == 4
    assert harness.usage.tool_calls == 2


@pytest.mark.asyncio
async def test_parallel_spawn_with_history_excludes_pending_spawn_calls() -> None:
    class DelegationModel:
        def __init__(self) -> None:
            self.root_calls = 0
            self.child_histories: list[list[HarnessMessage]] = []

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            messages = kwargs["messages"]
            if any(
                message.content in {"first child", "second child"}
                for message in messages
            ):
                self.child_histories.append(list(messages))
                assert any(message.content == "delegate" for message in messages)
                yield ModelDelta(text="child result")
                return
            self.root_calls += 1
            if self.root_calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="spawn-1",
                            name="spawn_agent",
                            arguments={
                                "prompt": "first child",
                                "include_history": True,
                            },
                        ),
                        HarnessToolCall(
                            id="spawn-2",
                            name="spawn_agent",
                            arguments={
                                "prompt": "second child",
                                "include_history": True,
                            },
                        ),
                    ]
                )
                return
            yield ModelDelta(text="done")

    model = DelegationModel()
    harness = AgentHarness(model="test-model", model_client=model)

    assert await run_harness(harness, "delegate") == "done"
    assert len(model.child_histories) == 2
    for messages in model.child_histories:
        NucliaModelClient._validate_tool_history(NucliaModelClient._messages(messages))
        assert not any(
            call.name == "spawn_agent"
            for message in messages
            for call in message.tool_calls
        )


@pytest.mark.asyncio
async def test_spawn_agent_returns_wait_result_at_conversation_limit() -> None:
    release = asyncio.Event()

    class SlowChildModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            await release.wait()
            yield ModelDelta(text="done")

    harness = AgentHarness(
        model="test-model",
        model_client=SlowChildModel(),
        usage_limits=UsageLimits(max_concurrent_agents=2),
    )
    first = await spawn_agent(
        harness, SpawnAgentInput(prompt="first", include_history=False)
    )
    first_id = first.value["agent_id"]

    blocked = await spawn_agent(
        harness, SpawnAgentInput(prompt="second", include_history=False)
    )

    assert blocked.value == {
        "status": "concurrency_limit_reached",
        "max_concurrent_agents": 2,
        "active_agents": 2,
        "waitable_agent_ids": [first_id],
        "message": (
            "The conversation is at its concurrent agent limit. "
            "Call wait_agent for an existing child before trying to spawn another."
        ),
    }

    release.set()
    await wait_agent(harness, AgentIdInput(agent_id=first_id))
    next_spawn = await spawn_agent(
        harness, SpawnAgentInput(prompt="second", include_history=False)
    )
    assert "agent_id" in next_spawn.value
    await harness._stop_children()


@pytest.mark.asyncio
async def test_concurrent_agent_limit_is_shared_with_descendants() -> None:
    release = asyncio.Event()

    class SlowChildModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            await release.wait()
            yield ModelDelta(text="done")

    harness = AgentHarness(
        model="test-model",
        model_client=SlowChildModel(),
        usage_limits=UsageLimits(max_spawn_depth=2, max_concurrent_agents=2),
    )
    spawned = await spawn_agent(
        harness, SpawnAgentInput(prompt="child", include_history=False)
    )
    child, _ = harness.children[spawned.value["agent_id"]]

    blocked = await spawn_agent(
        child, SpawnAgentInput(prompt="grandchild", include_history=False)
    )

    assert blocked.value["status"] == "concurrency_limit_reached"
    assert blocked.value["active_agents"] == 2
    assert blocked.value["waitable_agent_ids"] == []
    assert "Finish the current work" in blocked.value["message"]
    release.set()
    await harness._stop_children()


@pytest.mark.asyncio
async def test_load_repairs_tool_output_returned_before_its_call() -> None:
    storage = InMemoryHarnessStorage()
    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        storage=storage,
        conversation_id="conversation-1",
    )
    await harness.load()
    tool_output = HarnessMessage(
        role="tool",
        content="result",
        tool_call_id="call-1",
        tool_name="lookup",
    )
    tool_call = HarnessMessage(
        role="assistant",
        content="",
        tool_calls=[HarnessToolCall(id="call-1", name="lookup", arguments={})],
    )
    await storage.append_event(
        HarnessEvent(
            id="event-output",
            conversation_id="conversation-1",
            type=HarnessEventType.MESSAGE_ADDED,
            payload={"message": tool_output.model_dump(mode="json")},
        )
    )
    await storage.append_event(
        HarnessEvent(
            id="event-call",
            conversation_id="conversation-1",
            type=HarnessEventType.MESSAGE_ADDED,
            payload={"message": tool_call.model_dump(mode="json")},
        )
    )

    await harness.load(create=False)

    assert harness.messages == [tool_call, tool_output]
    NucliaModelClient._validate_tool_history(
        NucliaModelClient._messages(harness.messages)
    )


@pytest.mark.asyncio
async def test_wait_agent_returns_stable_failure_result() -> None:
    class FailingModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            raise RuntimeError("child failed")
            yield

    harness = AgentHarness(model="test-model", model_client=FailingModel())
    spawned = await spawn_agent(
        harness,
        SpawnAgentInput(prompt="fail", include_history=False),
    )
    child_id = spawned.value["agent_id"]
    input_value = AgentIdInput(agent_id=child_id)

    first = await wait_agent(harness, input_value)
    second = await wait_agent(harness, input_value)

    assert first == second
    assert first.value == {
        "agent_id": child_id,
        "status": "failed",
        "error": "child failed",
    }


@pytest.mark.asyncio
async def test_wait_agent_returns_when_child_fails_before_terminal_event() -> None:
    class FailingStorage(InMemoryHarnessStorage):
        async def append_event(self, event: HarnessEvent) -> None:
            if (
                event.parent_agent_id is not None
                and event.type == HarnessEventType.TURN_STARTED
            ):
                raise RuntimeError("storage failed")
            await super().append_event(event)

    harness = AgentHarness(
        model="test-model", model_client=Model(), storage=FailingStorage()
    )
    spawned = await spawn_agent(
        harness,
        SpawnAgentInput(prompt="fail", include_history=False),
    )
    child_id = spawned.value["agent_id"]

    result = await asyncio.wait_for(
        wait_agent(harness, AgentIdInput(agent_id=child_id)), timeout=1
    )

    assert result.value == {
        "agent_id": child_id,
        "status": "failed",
        "error": "storage failed",
    }


@pytest.mark.asyncio
async def test_harness_runs_tool_loop_and_resumes_history() -> None:
    async def upper(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value.upper())

    storage = InMemoryHarnessStorage()
    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[HarnessTool("upper", upper)],
        conversation_id="conversation",
        storage=storage,
    )

    assert await run_harness(harness, "say hi") == "HI"
    event_types = [event.type async for event in harness.history()]
    assert HarnessEventType.TOOL_COMPLETED in event_types
    assert HarnessEventType.LLM_COMPLETED in event_types

    resumed = AgentHarness(
        model="test-model",
        model_client=Model(),
        conversation_id="conversation",
        storage=storage,
    )
    await resumed.load(create=False)
    assert [message.role for message in resumed.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.asyncio
async def test_default_storage_is_ephemeral() -> None:
    harness = AgentHarness(model="test-model", model_client=Model())
    await harness.load()
    assert await harness.storage.get_conversation(harness.conversation_id) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "model"),
    [
        (UsageLimits(max_turns=1), Model()),
        (UsageLimits(max_input_tokens=3), Model()),
    ],
)
async def test_usage_limits(limits: UsageLimits, model: Model) -> None:
    async def upper(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value.upper())

    harness = AgentHarness(
        model="test-model",
        model_client=model,
        tools=[HarnessTool("upper", upper)],
        usage_limits=limits,
    )

    with pytest.raises(UsageLimitExceeded):
        await run_harness(harness, "say hi")


@pytest.mark.asyncio
async def test_tool_call_and_output_token_limits() -> None:
    class LimitModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            del kwargs
            self.calls += 1
            if self.calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(name="upper", arguments={"value": "a"}),
                        HarnessToolCall(name="upper", arguments={"value": "b"}),
                    ]
                )
            else:
                yield ModelDelta(text="done", output_tokens=2)

    async def upper(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value.upper())

    tool = HarnessTool("upper", upper)
    tool_limited = AgentHarness(
        model="test-model",
        model_client=LimitModel(),
        tools=[tool],
        usage_limits=UsageLimits(max_tool_calls=1),
    )
    with pytest.raises(UsageLimitExceeded):
        await run_harness(tool_limited, "call twice")

    output_limited = AgentHarness(
        model="test-model",
        model_client=LimitModel(),
        tools=[tool],
        usage_limits=UsageLimits(max_output_tokens=1),
    )
    with pytest.raises(UsageLimitExceeded):
        await run_harness(output_limited, "answer")


@pytest.mark.asyncio
async def test_max_time_interrupts_active_model_work() -> None:
    cancelled = asyncio.Event()

    class SlowModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            del kwargs
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            yield ModelDelta(text="late")

    harness = AgentHarness(
        model="test-model",
        model_client=SlowModel(),
        usage_limits=UsageLimits(max_time=0.01),
    )

    with pytest.raises(UsageLimitExceeded):
        await run_harness(harness, "wait")
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_interrupt_cancels_active_tool_work() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def slow_tool(_harness: AgentHarness, _value: ToolInput) -> ToolOutput:
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return ToolOutput(value="late")

    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[HarnessTool("upper", slow_tool)],
    )
    task = asyncio.create_task(run_harness(harness, "wait"))
    await started.wait()
    assert harness._active_task is not task
    harness.interrupt()

    assert await task == "interrupted"
    assert cancelled.is_set()
    assert task.cancelling() == 0


@pytest.mark.asyncio
async def test_interrupt_mid_tool_can_run_again_and_reload() -> None:
    started = asyncio.Event()

    async def slow(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        started.set()
        await asyncio.sleep(60)
        return ToolOutput(value=value.value)

    class RecoveringModel:
        def __init__(self):
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="slow", name="slow", arguments={"value": "x"}
                        )
                    ]
                )
            else:
                NucliaModelClient._validate_tool_history(
                    NucliaModelClient._messages(kwargs["messages"])
                )
                yield ModelDelta(text="recovered")

    storage = InMemoryHarnessStorage()
    model = RecoveringModel()
    harness = AgentHarness(
        model="model",
        model_client=model,
        tools=[HarnessTool("slow", slow)],
        storage=storage,
        conversation_id="recover",
    )
    run = asyncio.create_task(run_harness(harness, "first"))
    await started.wait()
    harness.interrupt()
    assert await run == "interrupted"
    assert await run_harness(harness, "second") == "recovered"

    resumed = AgentHarness(
        model="model",
        model_client=RecoveringModel(),
        storage=storage,
        conversation_id="recover",
    )
    await resumed.load(create=False)
    NucliaModelClient._validate_tool_history(
        NucliaModelClient._messages(resumed.messages)
    )


@pytest.mark.asyncio
async def test_pre_start_interrupt_is_not_lost() -> None:
    class NeverModel:
        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            raise AssertionError("model should not run")
            yield

    harness = AgentHarness(model="model", model_client=NeverModel())

    async def run() -> str:
        return await run_harness(harness, "stop")

    task = asyncio.create_task(run())
    harness.interrupt()
    assert await task == "interrupted"


@pytest.mark.asyncio
async def test_spawn_depth_defaults_to_one() -> None:
    harness = AgentHarness(model="model", model_client=Model())
    current = harness
    for expected_depth in range(1, 2):
        spawned = await spawn_agent(
            current, SpawnAgentInput(prompt=f"depth {expected_depth}")
        )
        child, _ = current.children[spawned.value["agent_id"]]
        assert child.spawn_depth == expected_depth
        current = child
    assert "spawn_agent" not in current._tools
    with pytest.raises(ValueError, match="Maximum spawn depth"):
        await spawn_agent(current, SpawnAgentInput(prompt="too deep"))
    await harness._stop_children()


@pytest.mark.asyncio
async def test_interrupt_cancels_sub_agents() -> None:
    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()

    class SpawnModel:
        def __init__(self) -> None:
            self.root_calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            messages = kwargs["messages"]
            if any(message.content == "child work" for message in messages):
                child_started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    child_cancelled.set()
                    raise
                yield ModelDelta(text="late")
                return
            self.root_calls += 1
            if self.root_calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            name="spawn_agent",
                            arguments={"prompt": "child work"},
                        )
                    ]
                )
            else:
                await asyncio.sleep(60)

    harness = AgentHarness(model="test-model", model_client=SpawnModel())
    task = asyncio.create_task(run_harness(harness, "delegate"))
    await child_started.wait()
    harness.interrupt()

    assert await task == "interrupted"
    assert child_cancelled.is_set()
    assert harness.children == {}
    assert all(
        result.value["status"] == "cancelled"
        for result in harness.child_results.values()
    )


@pytest.mark.asyncio
async def test_unwaited_child_is_stopped_when_parent_completes() -> None:
    child_started = asyncio.Event()
    child_cancelled = asyncio.Event()

    class SpawnWithoutWaitModel:
        def __init__(self):
            self.root_calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            if any(message.content == "child" for message in kwargs["messages"]):
                child_started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    child_cancelled.set()
                    raise
            self.root_calls += 1
            if self.root_calls == 1:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id="spawn",
                            name="spawn_agent",
                            arguments={"prompt": "child"},
                        )
                    ]
                )
            else:
                await child_started.wait()
                yield ModelDelta(text="parent done")

    harness = AgentHarness(model="model", model_client=SpawnWithoutWaitModel())
    assert await run_harness(harness, "root") == "parent done"
    assert child_cancelled.is_set()
    assert harness.children == {}
    assert all(
        result.value["status"] == "cancelled"
        for result in harness.child_results.values()
    )


@pytest.mark.asyncio
async def test_llm_only_passes_registered_tools() -> None:
    class CapturingModel:
        def __init__(self) -> None:
            self.tools = []

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.tools = kwargs["tools"]
            yield ModelDelta(text="ok")

    async def execute(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value)

    registered = HarnessTool("registered", execute)
    unknown = HarnessTool("unknown", execute)
    model = CapturingModel()
    harness = AgentHarness(model="test-model", model_client=model, tools=[registered])

    await harness.llm([registered, unknown])
    assert model.tools == [registered]


def test_agent_rejects_tools_that_conflict_with_core_tools() -> None:
    async def execute(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value)

    with pytest.raises(ValueError, match="conflict with core tools: remember"):
        AgentHarness(
            model="test-model",
            model_client=Model(),
            tools=[HarnessTool("remember", execute)],
        )


def test_agent_allows_disabled_core_tool_name() -> None:
    async def execute(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value)

    tool = HarnessTool("remember", execute)
    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[tool],
        disabled_core_tools=["remember"],
    )

    assert harness._tools["remember"] is tool


@pytest.mark.asyncio
async def test_llm_does_not_expose_inactive_lazy_tool_explicitly() -> None:
    class CapturingModel:
        def __init__(self) -> None:
            self.tools = []

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.tools = kwargs["tools"]
            yield ModelDelta(text="ok")

    async def execute(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value)

    lazy = HarnessTool("lazy", execute, lazy_load=True)
    model = CapturingModel()
    harness = AgentHarness(model="test-model", model_client=model, tools=[lazy])

    await harness.llm([lazy])
    assert model.tools == []


@pytest.mark.asyncio
async def test_lazy_tool_can_be_searched_activated_and_used() -> None:
    class LazyModel:
        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            names = {tool.name for tool in kwargs["tools"]}
            self.calls += 1
            if self.calls == 1:
                assert "upper" not in names
                assert {"search_tools", "activate_tools"} <= names
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            name="search_tools",
                            arguments={"query": "uppercase"},
                        )
                    ]
                )
            elif self.calls == 2:
                assert "upper" not in names
                assert any(
                    message.tool_name == "search_tools"
                    and '"name":"upper"' in message.content
                    for message in kwargs["messages"]
                )
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            name="activate_tools", arguments={"names": ["upper"]}
                        )
                    ]
                )
            elif self.calls == 3:
                assert "upper" in names
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(name="upper", arguments={"value": "hi"})
                    ]
                )
            else:
                yield ModelDelta(text="HI")

    async def upper(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value.upper())

    harness = AgentHarness(
        model="test-model",
        model_client=LazyModel(),
        tools=[
            HarnessTool(
                "upper",
                upper,
                description="Convert text to uppercase.",
                lazy_load=True,
            )
        ],
    )

    assert await run_harness(harness, "Uppercase hi") == "HI"


@pytest.mark.asyncio
async def test_search_tools_matches_natural_language_query() -> None:
    async def execute(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value)

    harness = AgentHarness(
        model="test-model",
        model_client=Model(),
        tools=[
            HarnessTool(
                "create_dataset",
                execute,
                description="Create a dataset.",
                lazy_load=True,
            ),
            HarnessTool(
                "list_datasets",
                execute,
                description="List datasets.",
                lazy_load=True,
            ),
        ],
    )

    result = await search_tools(
        harness,
        SearchToolsInput(
            query="create a dataset, upload data, import files, or manage datasets"
        ),
    )

    assert [item["name"] for item in result.items] == [
        "create_dataset",
        "list_datasets",
    ]


@pytest.mark.asyncio
async def test_tool_decorator_configures_lazy_loading_and_preserves_calls() -> None:
    @tool(description="Convert text", lazy_load=True)
    async def upper(_harness: AgentHarness, value: ToolInput) -> ToolOutput:
        return ToolOutput(value=value.value.upper())

    assert isinstance(upper, HarnessTool)
    assert upper.name == "upper"
    assert upper.lazy_load is True
    harness = AgentHarness(model="test-model", model_client=Model())
    assert await upper(harness, ToolInput(value="hi")) == ToolOutput(value="HI")


def test_tool_schema_preserves_descriptions() -> None:
    class DescribedInput(BaseModel):
        value: str = Field(description="Value to transform")

    async def execute(_harness: AgentHarness, _value: DescribedInput) -> ToolOutput:
        return ToolOutput(value="ok")

    tool = HarnessTool("described", execute)
    assert tool.input_model is DescribedInput
    assert tool.output_model is ToolOutput
    assert tool.parameters["properties"]["value"]["description"] == "Value to transform"


def test_tool_schema_flattens_nested_references() -> None:
    class Nested(BaseModel):
        value: str

    class NestedInput(BaseModel):
        nested: Nested
        values: list[Nested]

    async def execute(_harness: AgentHarness, _value: NestedInput) -> ToolOutput:
        return ToolOutput(value="ok")

    schema = HarnessTool("nested", execute).parameters
    rendered = str(schema)
    assert "$ref" not in rendered
    assert "$defs" not in rendered
    assert schema["properties"]["nested"]["properties"]["value"]["type"] == "string"


def test_published_schema_preserves_constraints_and_namespaces() -> None:
    class PublishedAgent:
        __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
            "lookup": FunctionDefinition(
                name="lookup",
                description="Lookup",
                parameters={
                    "kind": {"type": "string", "enum": ["a", "b"]},
                    "limit": {"type": "integer", "minimum": 1},
                },
                lazy_load=True,
            )
        }

        async def lookup(self, kind: str, **kwargs):
            return {"kind": kind, **kwargs}

    tool = AgentHarness.to_tools(PublishedAgent(), namespace="source")[0]
    assert tool.name == "source__lookup"
    assert tool.parameters["properties"]["kind"]["enum"] == ["a", "b"]
    assert tool.parameters["properties"]["limit"]["minimum"] == 1
    assert tool.parameters["required"] == ["kind"]
    assert tool.lazy_load is True


@pytest.mark.asyncio
async def test_published_tool_validates_the_advertised_input_schema() -> None:
    called = False

    class PublishedAgent:
        __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
            "lookup": FunctionDefinition(
                name="lookup",
                description="Lookup",
                parameters={},
                input_schema={
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "enum": ["a", "b"]},
                        "limit": {"type": "integer", "minimum": 1},
                    },
                    "required": ["kind", "limit"],
                    "additionalProperties": False,
                },
            )
        }

        async def lookup(self, **kwargs):
            nonlocal called
            called = True
            return kwargs

    tool = AgentHarness.to_tools(PublishedAgent())[0]
    harness = AgentHarness(model="test-model", model_client=Model())

    with pytest.raises(ValueError, match="Invalid lookup arguments"):
        await tool.execute(harness, {"kind": "invalid", "limit": 1})
    with pytest.raises(ValueError, match="Invalid lookup arguments"):
        await tool.execute(harness, {"kind": "a"})

    assert called is False


@pytest.mark.asyncio
async def test_malformed_tool_arguments_do_not_run_handler() -> None:
    called = False

    class OptionalInput(BaseModel):
        value: str = "default"

    async def execute(_harness: AgentHarness, _value: OptionalInput) -> ToolOutput:
        nonlocal called
        called = True
        return ToolOutput(value="ok")

    tool = HarnessTool("optional", execute)
    harness = AgentHarness(model="test-model", model_client=Model())

    with pytest.raises(ValueError, match="Malformed tool arguments"):
        await tool.execute(
            harness, {"_tool_error": "Malformed tool arguments: invalid JSON"}
        )

    assert called is False


@pytest.mark.asyncio
async def test_tool_failure_log_includes_actionable_detail(mocker) -> None:
    async def fail(_harness: AgentHarness, _value: ToolInput) -> ToolOutput:
        raise ValueError("identity field is invalid")

    harness = AgentHarness(
        model="test-model", model_client=Model(), tools=[HarnessTool("validate", fail)]
    )
    call = HarnessToolCall(id="call-42", name="validate", arguments={"value": "x"})
    warning = mocker.patch("hyperforge.harness_sdk.harness.logger.warning")

    await harness._execute_tool_call(call)

    assert warning.call_count == 1
    assert warning.call_args.args[:4] == (
        "Agent tool execution failed: tool=%s call_id=%s error_type=%s error=%s",
        "validate",
        "call-42",
        "ValueError",
    )
    assert str(warning.call_args.args[4]) == "identity field is invalid"


def test_tool_requires_pydantic_handler_annotations() -> None:
    async def missing_annotations(_harness, _value):
        return ToolOutput(value="ok")

    with pytest.raises(TypeError, match="input must be annotated"):
        HarnessTool("invalid", missing_annotations)


@pytest.mark.asyncio
async def test_published_agent_functions_convert_to_tools() -> None:
    nua_client = object()
    seen = []

    class PublishedAgent:
        __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
            "lookup": FunctionDefinition(
                name="knowledge_lookup",
                description="Look up a question in the knowledge source.",
                parameters={
                    "question": {
                        "type": "string",
                        "description": "Question to look up.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results.",
                    },
                },
            )
        }

        async def lookup(
            self,
            question: str,
            memory,
            manager,
            limit: int = 3,
        ) -> dict[str, object]:
            seen.append(
                {
                    "question": question,
                    "limit": limit,
                    "conversation": memory.headers["conversation-id"],
                    "nua": manager.nua,
                }
            )
            return {"answer": question.upper(), "limit": limit}

    class PublishedModel:
        client = nua_client

        def __init__(self) -> None:
            self.calls = 0

        async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
            self.calls += 1
            if self.calls == 1:
                tool = next(
                    tool for tool in kwargs["tools"] if tool.name == "knowledge_lookup"
                )
                assert tool.description == "Look up a question in the knowledge source."
                assert (
                    tool.parameters["properties"]["question"]["description"]
                    == "Question to look up."
                )
                assert tool.parameters["properties"]["limit"]["default"] == 3
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            name="knowledge_lookup",
                            arguments={"question": "hyperforge"},
                        )
                    ]
                )
                return
            assert any(
                message.tool_name == "knowledge_lookup"
                and "HYPERFORGE" in message.content
                for message in kwargs["messages"]
            )
            yield ModelDelta(text="Found it")

    model = PublishedModel()
    manager = Manager()
    manager.nua = nua_client
    tools = AgentHarness.to_tools(PublishedAgent(), manager=manager)
    harness = AgentHarness(
        model="test-model",
        model_client=model,
        tools=tools,
        conversation_id="published-conversation",
    )

    assert await run_harness(harness, "Look it up") == "Found it"
    assert seen == [
        {
            "question": "hyperforge",
            "limit": 3,
            "conversation": "published-conversation",
            "nua": nua_client,
        }
    ]


@pytest.mark.asyncio
async def test_published_agent_manager_uses_existing_nua_api() -> None:
    class FakeNua:
        async def generate(self, **kwargs):
            return GenerativeFullResponse(answer="legacy answer")

    class LegacyAgent:
        __published_functions__: ClassVar[dict[str, FunctionDefinition]] = {
            "ask": FunctionDefinition(
                name="ask",
                description="Ask",
                parameters={"question": {"type": "string"}},
            )
        }

        async def ask(self, question, manager, **kwargs):
            from nuclia.lib.nua_responses import ChatModel

            response, _, _ = await manager.execute_raw(ChatModel(question=question))
            return response.answer

    manager = Manager()
    manager.nua = FakeNua()
    tool = AgentHarness.to_tools(LegacyAgent(), manager=manager)[0]
    harness = AgentHarness(model="model", model_client=Model(), tools=[tool])
    output = await tool.execute(harness, {"question": "hello"})
    assert output.value == "legacy answer"


def test_agent_without_published_functions_is_not_supported() -> None:
    class UnsupportedAgent:
        pass

    with pytest.raises(TypeError, match="does not define any published functions"):
        AgentHarness.to_tools(UnsupportedAgent())
