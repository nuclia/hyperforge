import asyncio
import sys
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

import pytest
from pydantic import BaseModel

from hyperforge.codemode import RestrictedPythonTask, WorkerExecutionRequest
from hyperforge.harness_sdk import (
    AgentHarness,
    CodeModeCapability,
    CodeModeExecutionLimiter,
    CodemodeInput,
    CodeModeLimits,
    HarnessContextReference,
    HarnessContextType,
    HarnessEventType,
    HarnessTool,
    HarnessToolCall,
    ModelDelta,
    UsageLimits,
    codemode,
    create_codemode_tool,
    raw_codemode_result_adapter,
    tool,
)
from hyperforge.harness_sdk.tools import codemode as codemode_module


class UpperInput(BaseModel):
    value: str


class UpperOutput(BaseModel):
    value: str


class SensitiveOutput(BaseModel):
    value: str
    secret: str


@tool(description="Uppercase a value")
async def upper(_: AgentHarness, input_value: UpperInput) -> UpperOutput:
    return UpperOutput(value=input_value.value.upper())


class UnusedModel:
    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        del kwargs
        yield ModelDelta(text="unused")


def local_codemode(
    *capabilities: CodeModeCapability,
    limits: CodeModeLimits | None = None,
    execution_limiter: CodeModeExecutionLimiter | None = None,
) -> HarnessTool:
    if execution_limiter is None:
        return create_codemode_tool(
            capabilities=capabilities,
            limits=limits or CodeModeLimits(),
            remote_required=False,
        )
    return create_codemode_tool(
        capabilities=capabilities,
        limits=limits or CodeModeLimits(),
        execution_limiter=execution_limiter,
        remote_required=False,
    )


def value_adapter(_tool: HarnessTool, output: BaseModel) -> dict[str, str]:
    return {"value": UpperOutput.model_validate(output).value}


@pytest.mark.asyncio
async def test_codemode_calls_registered_tools_and_returns_output() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[upper],
        disabled_core_tools=[
            "remember",
            "recall",
            "forget",
            "spawn_agent",
            "send_message",
            "wait_agent",
            "compact",
            "search_tools",
            "activate_tools",
        ],
    )

    result = await codemode.execute(
        harness,
        CodemodeInput(
            code="result = upper(value='hello')\noutput(result['value'])"
        ).model_dump(),
    )

    assert result.value == "HELLO"


@pytest.mark.asyncio
async def test_codemode_counts_nested_tool_calls() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[upper],
        usage_limits=UsageLimits(max_tool_calls=1),
    )

    with pytest.raises(RuntimeError, match="max_tool_calls"):
        await codemode.execute(
            harness,
            CodemodeInput(code="upper(value='one')\nupper(value='two')").model_dump(),
        )


@pytest.mark.asyncio
async def test_codemode_propagates_tool_validation_errors() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[upper],
    )

    with pytest.raises(RuntimeError, match="Invalid upper arguments"):
        await codemode.execute(
            harness,
            CodemodeInput(code="upper(missing='value')").model_dump(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("output(sum(i * i for i in range(1, 101)))", 338350),
        (
            "total = 0\nfor i in range(1, 101):\n    total += i * i\noutput(total)",
            338350,
        ),
    ],
)
async def test_codemode_supports_common_aggregation_patterns(
    code: str, expected: int
) -> None:
    harness = AgentHarness(model="test", model_client=UnusedModel())

    result = await codemode.execute(harness, CodemodeInput(code=code).model_dump())

    assert result.value == expected


@pytest.mark.asyncio
async def test_codemode_preserves_context_when_calling_tools() -> None:
    request_context = ContextVar("request_context", default="missing")

    @tool()
    async def read_context(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        return UpperOutput(value=request_context.get())

    harness = AgentHarness(
        model="test", model_client=UnusedModel(), tools=[read_context]
    )
    token = request_context.set("available")
    try:
        result = await codemode.execute(
            harness,
            CodemodeInput(
                code="result = read_context(value='unused')\noutput(result['value'])"
            ).model_dump(),
        )
    finally:
        request_context.reset(token)

    assert result.value == "available"


@pytest.mark.asyncio
async def test_codemode_enforces_runtime_limit() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        usage_limits=UsageLimits(max_codemode_runtime_seconds=0.1),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        await codemode.execute(
            harness,
            CodemodeInput(code="while True: pass").model_dump(),
        )


@pytest.mark.asyncio
async def test_codemode_blocks_process_control_exceptions() -> None:
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="SystemExit.*not defined"):
        await codemode.execute(
            harness,
            CodemodeInput(code="raise SystemExit(1)").model_dump(),
        )


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "darwin", reason="RLIMIT_AS is unreliable on macOS")
async def test_codemode_enforces_memory_limit() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        usage_limits=UsageLimits(max_codemode_memory_bytes=512 * 1024 * 1024),
    )

    with pytest.raises(RuntimeError):
        await codemode.execute(
            harness,
            CodemodeInput(code="output('x' * (1024 * 1024 * 1024))").model_dump(),
        )


@pytest.mark.asyncio
async def test_scoped_codemode_uses_only_explicit_hidden_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @tool()
    async def lower(_: AgentHarness, input_value: UpperInput) -> UpperOutput:
        return UpperOutput(value=input_value.value.lower())

    scoped = local_codemode(CodeModeCapability(upper, result_adapter=value_adapter))
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[scoped, lower],
    )
    monkeypatch.setattr(
        harness,
        "iter_tools",
        lambda *args, **kwargs: pytest.fail("scoped Code Mode discovered tools"),
    )

    result = await scoped.execute(
        harness,
        {"code": "result = upper(value='hello')\noutput(result['value'])"},
    )

    assert result.value == "HELLO"
    assert upper.name not in harness._tools
    for unavailable in ("lower", "remember"):
        with pytest.raises(RuntimeError, match=f"{unavailable}.*not defined"):
            await scoped.execute(harness, {"code": f"{unavailable}(value='x')"})


@pytest.mark.asyncio
async def test_scoped_codemode_default_projection_uses_model_context() -> None:
    @tool(
        context_factory=lambda output: HarnessContextReference(
            type=HarnessContextType.STRUCTURED,
            content={"value": output.value},
        )
    )
    async def lookup(_: AgentHarness, input_value: UpperInput) -> SensitiveOutput:
        return SensitiveOutput(value=input_value.value.upper(), secret="internal")

    scoped = local_codemode(CodeModeCapability(lookup))
    harness = AgentHarness(model="test", model_client=UnusedModel(), tools=[scoped])

    result = await scoped.execute(
        harness,
        {"code": "result = lookup(value='hello')\noutput(result)"},
    )

    assert result.value == '{"value":"HELLO"}'


@pytest.mark.asyncio
async def test_scoped_codemode_sanitizes_formatted_context_events() -> None:
    @tool()
    async def lookup(_: AgentHarness, input_value: UpperInput) -> SensitiveOutput:
        return SensitiveOutput(value=input_value.value, secret="internal-secret")

    scoped = local_codemode(CodeModeCapability(lookup))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(harness, {"code": "lookup(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"] == {
        "value": {"type": "string", "utf8_bytes": 4},
        "secret": "[REDACTED]",
    }


@pytest.mark.asyncio
async def test_scoped_codemode_does_not_persist_opaque_string_results() -> None:
    def project(_tool: HarnessTool, _output: BaseModel) -> str:
        return "authorization: Bearer projected-secret"

    scoped = local_codemode(CodeModeCapability(upper, result_adapter=project))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(harness, {"code": "upper(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"] == {"type": "string", "utf8_bytes": 38}
    assert "projected-secret" not in str(completed.payload)


@pytest.mark.asyncio
async def test_scoped_codemode_does_not_persist_nested_opaque_strings() -> None:
    def project(_tool: HarnessTool, _output: BaseModel) -> dict[str, object]:
        return {
            "value": "authorization: Bearer nested-secret",
            "items": ["another-secret"],
        }

    scoped = local_codemode(CodeModeCapability(upper, result_adapter=project))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(harness, {"code": "upper(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"] == {
        "value": {"type": "string", "utf8_bytes": 35},
        "items": [{"type": "string", "utf8_bytes": 14}],
    }
    assert "secret" not in str(completed.payload)


@pytest.mark.asyncio
async def test_scoped_codemode_does_not_persist_non_finite_json_numbers() -> None:
    def project(_tool: HarnessTool, _output: BaseModel) -> str:
        return '{"score":1e9999}'

    scoped = local_codemode(CodeModeCapability(upper, result_adapter=project))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(harness, {"code": "upper(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"] == {"score": "<non-finite number>"}


@pytest.mark.asyncio
async def test_scoped_codemode_raw_projection_must_be_explicit() -> None:
    @tool(
        context_factory=lambda output: HarnessContextReference(
            type=HarnessContextType.STRUCTURED,
            content={"value": output.value},
        )
    )
    async def lookup(_: AgentHarness, input_value: UpperInput) -> SensitiveOutput:
        return SensitiveOutput(value=input_value.value, secret="raw-secret")

    scoped = local_codemode(
        CodeModeCapability(lookup, result_adapter=raw_codemode_result_adapter)
    )
    harness = AgentHarness(model="test", model_client=UnusedModel(), tools=[scoped])

    result = await scoped.execute(
        harness,
        {"code": "result = lookup(value='hello')\noutput(result)"},
    )

    assert result.value == {"value": "hello", "secret": "raw-secret"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "projection",
    [
        lambda _tool, _output: object(),
        lambda _tool, output: output,
        lambda _tool, _output: {"__model__": "Context", "value": "unsafe"},
    ],
)
async def test_scoped_codemode_rejects_unsafe_projected_results(
    projection: Any,
) -> None:
    scoped = local_codemode(CodeModeCapability(upper, result_adapter=projection))
    harness = AgentHarness(model="test", model_client=UnusedModel(), tools=[scoped])

    with pytest.raises(RuntimeError, match="capability 'upper' failed"):
        await scoped.execute(harness, {"code": "upper(value='hello')"})


@pytest.mark.asyncio
async def test_scoped_codemode_enforces_source_result_and_output_byte_limits() -> None:
    harness = AgentHarness(model="test", model_client=UnusedModel())
    source_limited = local_codemode(limits=CodeModeLimits(max_source_bytes=1))
    with pytest.raises(ValueError, match="source exceeds maximum size"):
        await source_limited.execute(harness, {"code": "é"})

    result_limited = local_codemode(
        CodeModeCapability(upper, result_adapter=value_adapter),
        limits=CodeModeLimits(max_result_bytes=4),
    )
    with pytest.raises(RuntimeError, match="capability 'upper' failed"):
        await result_limited.execute(harness, {"code": "upper(value='hello')"})

    output_limited = local_codemode(limits=CodeModeLimits(max_output_bytes=4))
    with pytest.raises(RuntimeError, match="output exceeds maximum size"):
        await output_limited.execute(harness, {"code": "output('hello')"})


@pytest.mark.asyncio
async def test_scoped_codemode_counts_nested_calls_once_and_enforces_local_limit() -> (
    None
):
    scoped = local_codemode(
        CodeModeCapability(upper, result_adapter=value_adapter),
        limits=CodeModeLimits(max_nested_calls=1),
    )
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[scoped],
        usage_limits=UsageLimits(max_tool_calls=10),
    )

    await harness._execute_tool_calls(
        [
            HarnessToolCall(
                id="outer-call",
                name="codemode",
                arguments={"code": "upper(value='one')\nupper(value='two')"},
            )
        ]
    )

    assert harness.usage.tool_calls == 3
    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_emits_sanitized_nested_events_with_parent_id() -> None:
    @tool()
    async def inspect_value(
        _: AgentHarness, input_value: UpperInput
    ) -> SensitiveOutput:
        return SensitiveOutput(value=input_value.value, secret="result-secret")

    def project(_tool: HarnessTool, output: BaseModel) -> dict[str, str]:
        value = SensitiveOutput.model_validate(output)
        return {"value": value.value, "token": "projected-secret"}

    scoped = local_codemode(CodeModeCapability(inspect_value, result_adapter=project))
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[scoped],
        execution_context={"actor_id": "actor-1", "tenant_id": "tenant-1"},
    )

    await harness._execute_tool_calls(
        [
            HarnessToolCall(
                id="outer-call",
                name="codemode",
                arguments={
                    "code": "inspect_value(value='safe', token='argument-secret')"
                },
            )
        ]
    )

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
    ]
    nested_id = nested[0].payload["call"]["id"]
    assert nested_id == nested[1].payload["call_id"]
    assert all(event.parent_call_id == "outer-call" for event in nested)
    assert all(event.payload["parent_call_id"] == "outer-call" for event in nested)
    assert nested[0].payload["call"]["arguments"]["value"] == {
        "type": "string",
        "utf8_bytes": 4,
    }
    assert nested[0].payload["call"]["arguments"]["token"] == "[REDACTED]"
    assert nested[1].payload["result"] == {
        "value": {"type": "string", "utf8_bytes": 4},
        "token": "[REDACTED]",
    }
    assert nested[0].payload["execution_context"] == {
        "actor_id": "actor-1",
        "tenant_id": "tenant-1",
    }


@pytest.mark.asyncio
async def test_scoped_codemode_emits_projected_completion_event() -> None:
    scoped = local_codemode(CodeModeCapability(upper, result_adapter=value_adapter))
    harness = AgentHarness(model="test", model_client=UnusedModel(), tools=[scoped])

    await harness._execute_tool_calls(
        [
            HarnessToolCall(
                id="outer-call",
                name="codemode",
                arguments={"code": "upper(value='safe'); output(None)"},
            )
        ]
    )

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
    ]
    assert nested[1].payload["result"] == {"value": {"type": "string", "utf8_bytes": 4}}
    assert nested[1].payload["result_bytes"] == len('{"value":"SAFE"}')
    assert nested[1].payload["duration_ms"] >= 0
    assert nested[0].payload["call"]["id"] == nested[1].payload["call_id"]


@pytest.mark.asyncio
async def test_scoped_codemode_records_nested_failure_before_propagating() -> None:
    @tool()
    async def fail(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        raise ValueError("secret failure detail")

    scoped = local_codemode(CodeModeCapability(fail))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match=r"capability 'fail' failed \(ValueError\)"):
        await scoped.execute(harness, {"code": "fail(value='safe')"})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert nested[1].payload["result"] == {"error": "ValueError"}
    assert "secret failure detail" not in str(nested)


@pytest.mark.asyncio
async def test_scoped_codemode_cancels_pending_nested_callback_on_timeout() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @tool()
    async def wait(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    scoped = local_codemode(CodeModeCapability(wait))
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        usage_limits=UsageLimits(max_codemode_runtime_seconds=2),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        await scoped.execute(harness, {"code": "wait(value='x')"})

    assert started.is_set()
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_scoped_codemode_propagates_external_cancellation() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    @tool()
    async def wait(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    scoped = local_codemode(CodeModeCapability(wait))
    harness = AgentHarness(model="test", model_client=UnusedModel())
    execution = asyncio.create_task(
        scoped.execute(harness, {"code": "wait(value='x')"})
    )
    await started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await execution

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_scoped_codemode_fails_closed_without_remote_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codemode_module.settings, "sandbox_socket", None)
    scoped = create_codemode_tool(capabilities=())
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="SANDBOX_SOCKET is absent"):
        await scoped.execute(harness, {"code": "output(None)"})


@pytest.mark.asyncio
async def test_scoped_codemode_fails_closed_without_remote_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codemode_module.settings, "sandbox_socket", "/unused.sock")
    monkeypatch.setattr(codemode_module.settings, "sandbox_token", None)
    scoped = create_codemode_tool(capabilities=())
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="SANDBOX_TOKEN is absent"):
        await scoped.execute(harness, {"code": "output(None)"})


class RecordingRunner:
    def __init__(self, tasks: list[RestrictedPythonTask]) -> None:
        self.tasks = tasks
        self.requests: list[WorkerExecutionRequest] = []
        self.results: dict[str, Any] = {}

    async def run(self, request: WorkerExecutionRequest, dispatch: Any) -> None:
        self.requests.append(request)
        for task in self.tasks:
            self.results[task.function] = await dispatch(task)


def worker_task(function: str, *args: Any, **kwargs: Any) -> RestrictedPythonTask:
    return RestrictedPythonTask(
        function=function, agent="harness", args=args, keyword_args=kwargs
    )


@pytest.mark.asyncio
async def test_scoped_codemode_uses_injected_runner_without_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codemode_module.settings, "sandbox_socket", None)
    monkeypatch.setattr(codemode_module.settings, "sandbox_token", None)
    runner = RecordingRunner(
        [
            worker_task("upper", "hello"),
            worker_task("output", {"done": True}),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=value_adapter),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    result = await scoped.execute(
        harness, {"code": "upper('hello'); output({'done': True})", "question": "Q?"}
    )

    assert result.value == {"done": True}
    request = runner.requests[0]
    assert request.code == "upper('hello'); output({'done': True})"
    assert request.question == "Q?"
    assert set(request.function_names["harness"]) == {"upper", "output"}
    assert runner.results["upper"] == {"value": "HELLO"}
    assert harness.usage.tool_calls == 1


@pytest.mark.asyncio
async def test_scoped_codemode_output_must_be_called_exactly_once() -> None:
    runner = RecordingRunner(
        [worker_task("output", 1), worker_task("output", 2)],
    )
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="may only be called once"):
        await scoped.execute(harness, {"code": "output(1); output(2)"})


@pytest.mark.asyncio
async def test_scoped_codemode_output_requires_a_value() -> None:
    runner = RecordingRunner([worker_task("output")])
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="output requires a value"):
        await scoped.execute(harness, {"code": "output()"})


@pytest.mark.asyncio
async def test_scoped_codemode_requires_output_call() -> None:
    runner = RecordingRunner([])
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="must call output\\(value\\) exactly once"):
        await scoped.execute(harness, {"code": "1 + 1"})


@pytest.mark.asyncio
async def test_scoped_codemode_enforces_cumulative_result_limit() -> None:
    runner = RecordingRunner([worker_task("upper", "hi"), worker_task("upper", "ok")])
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=value_adapter),),
        runner=runner,
        limits=CodeModeLimits(max_result_bytes=20, max_cumulative_result_bytes=20),
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="capability 'upper' failed"):
        await scoped.execute(harness, {"code": "upper('hi'); upper('ok')"})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert nested[-1].payload["result"] == {"error": "RuntimeError"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_cumulative_result_bytes": 0},
        {"max_result_bytes": 1024, "max_cumulative_result_bytes": 512},
    ],
)
def test_scoped_codemode_rejects_inconsistent_result_limits(kwargs: dict):
    with pytest.raises(ValueError, match="max_cumulative_result_bytes"):
        CodeModeLimits(**kwargs)


@pytest.mark.parametrize(
    "name",
    [
        "codemode",
        "output",
        "print",
        "printed",
        "save",
        "question",
        "agent_id",
        "dataclass",
        "Chunk",
        "Context",
        "List",
        "Any",
        "Dict",
    ],
)
def test_scoped_codemode_rejects_reserved_capability_names(name: str) -> None:
    reserved = HarnessTool(name, upper.handler)

    with pytest.raises(ValueError, match="reserved"):
        local_codemode(CodeModeCapability(reserved))


@pytest.mark.parametrize(
    "name", ["not-valid", "two words", "class", "1tool", "_private"]
)
def test_scoped_codemode_rejects_invalid_capability_names(name: str) -> None:
    invalid = HarnessTool(name, upper.handler)

    with pytest.raises(ValueError, match="valid Python identifier"):
        local_codemode(CodeModeCapability(invalid))


def test_scoped_codemode_rejects_agent_id_input_field() -> None:
    class AgentInput(BaseModel):
        agent_id: str

    @tool()
    async def routed(_: AgentHarness, input_value: AgentInput) -> UpperOutput:
        return UpperOutput(value=input_value.agent_id)

    with pytest.raises(ValueError, match="reserved 'agent_id' input field"):
        local_codemode(CodeModeCapability(routed))


def test_scoped_codemode_requires_immutable_capabilities() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        create_codemode_tool(capabilities=[])  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan")])
def test_scoped_codemode_rejects_invalid_limits(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CodeModeLimits(max_source_bytes=value)  # type: ignore[arg-type]


def test_scoped_codemode_description_includes_capability_contract() -> None:
    scoped = local_codemode(CodeModeCapability(upper))
    custom = create_codemode_tool(
        capabilities=(CodeModeCapability(upper),),
        description="Custom orchestration instructions.",
    )

    assert "Uppercase a value" in scoped.description
    assert '"required":["value"]' in scoped.description
    assert "Custom orchestration instructions." in custom.description
    assert '"required":["value"]' in custom.description


def test_codemode_execution_limiter_rejects_invalid_limit() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CodeModeExecutionLimiter(0)


@pytest.mark.asyncio
async def test_scoped_codemode_concurrency_saturation() -> None:
    started = asyncio.Event()

    @tool()
    async def wait(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        started.set()
        await asyncio.Future()

    scoped = local_codemode(
        CodeModeCapability(wait),
        execution_limiter=CodeModeExecutionLimiter(1),
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    first = asyncio.create_task(scoped.execute(harness, {"code": "wait(value='x')"}))
    await started.wait()

    with pytest.raises(RuntimeError, match="concurrency limit reached"):
        await scoped.execute(harness, {"code": "output(None)"})

    first.cancel()
    await asyncio.gather(first, return_exceptions=True)


@pytest.mark.asyncio
async def test_scoped_codemode_holds_slot_for_non_cooperative_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()
    callback_done = asyncio.Event()

    @tool()
    async def wait(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
            raise
        finally:
            callback_done.set()

    monkeypatch.setattr("hyperforge.codemode.sandbox.CALLBACK_CANCEL_TIMEOUT", 0.01)
    limiter = CodeModeExecutionLimiter(1)
    scoped = local_codemode(CodeModeCapability(wait), execution_limiter=limiter)
    harness = AgentHarness(model="test", model_client=UnusedModel())
    execution = asyncio.create_task(
        scoped.execute(harness, {"code": "wait(value='x')"})
    )
    await started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=0.5)
    await cancellation_seen.wait()
    with pytest.raises(RuntimeError, match="concurrency limit reached"):
        await scoped.execute(harness, {"code": "output(None)"})

    release.set()
    await asyncio.wait_for(callback_done.wait(), timeout=0.5)
    await asyncio.sleep(0)
    result = await scoped.execute(harness, {"code": "output('released')"})
    assert result.value == "released"
