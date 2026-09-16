import asyncio
import logging
import sys
import tempfile
import tracemalloc
from collections.abc import AsyncIterator
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, Field

from hyperforge.codemode import RestrictedPythonTask, WorkerExecutionRequest
from hyperforge.codemode import sandbox as sandbox_module
from hyperforge.harness_sdk import (
    AgentHarness,
    CodeModeCapability,
    CodeModeDispatch,
    CodeModeExecutionLimiter,
    CodemodeInput,
    CodeModeLimits,
    HarnessContextReference,
    HarnessContextType,
    HarnessEventType,
    HarnessTool,
    HarnessToolCall,
    ModelDelta,
    ToolCallContext,
    ToolInheritancePolicy,
    UsageLimits,
    codemode,
    create_codemode_tool,
    raw_codemode_result_adapter,
    tool,
)
from hyperforge.harness_sdk.tools import codemode as codemode_module


def _ctx(harness: Any) -> ToolCallContext:
    return ToolCallContext(harness=harness, name="test")


class UpperInput(BaseModel):
    value: str


class UpperOutput(BaseModel):
    value: str


class SensitiveOutput(BaseModel):
    value: str
    secret: str


@tool(description="Uppercase a value")
async def upper(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
    return UpperOutput(value=input_value.value.upper())


@tool(
    description="Look up a public value",
    context_factory=lambda output: HarnessContextReference(
        type=HarnessContextType.STRUCTURED,
        content={"value": output.value},
    ),
)
async def lookup(_: ToolCallContext, input_value: UpperInput) -> SensitiveOutput:
    return SensitiveOutput(value=input_value.value.upper(), secret="internal")


class UnusedModel:
    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        del kwargs
        yield ModelDelta(text="unused")


class RecordingRunner:
    def __init__(self, tasks: list[RestrictedPythonTask]) -> None:
        self.tasks = tasks
        self.requests: list[WorkerExecutionRequest] = []
        self.results: list[tuple[str, Any]] = []

    async def run(
        self, request: WorkerExecutionRequest, dispatch: CodeModeDispatch
    ) -> None:
        self.requests.append(request)
        for task in self.tasks:
            self.results.append((task.function, await dispatch(task)))


class CatchingRunner:
    def __init__(self, tasks: list[RestrictedPythonTask]) -> None:
        self.tasks = tasks
        self.errors: list[str] = []
        self.results: list[tuple[str, Any]] = []

    async def run(
        self, _request: WorkerExecutionRequest, dispatch: CodeModeDispatch
    ) -> None:
        for task in self.tasks:
            try:
                result = await dispatch(task)
            except Exception as exc:
                self.errors.append(str(exc))
            else:
                self.results.append((task.function, result))


def worker_task(function: str, *args: Any, **kwargs: Any) -> RestrictedPythonTask:
    return RestrictedPythonTask(
        function=function,
        agent="harness",
        args=args,
        keyword_args=kwargs,
    )


def local_codemode(
    *capabilities: CodeModeCapability[Any],
    limits: CodeModeLimits | None = None,
    execution_limiter: CodeModeExecutionLimiter | None = None,
) -> HarnessTool[Any, Any]:
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


def value_adapter(
    _tool: HarnessTool[Any, UpperOutput], output: UpperOutput
) -> dict[str, str]:
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
        ToolCallContext(harness=harness, name=codemode.name),
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
            ToolCallContext(harness=harness, name=codemode.name),
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
            ToolCallContext(harness=harness, name=codemode.name),
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

    result = await codemode.execute(
        ToolCallContext(harness=harness, name=codemode.name),
        CodemodeInput(code=code).model_dump(),
    )

    assert result.value == expected


@pytest.mark.asyncio
async def test_codemode_preserves_context_when_calling_tools() -> None:
    request_context = ContextVar("request_context", default="missing")
    call_ids: list[str | None] = []

    @tool()
    async def read_context(
        context: ToolCallContext, _input_value: UpperInput
    ) -> UpperOutput:
        call_ids.append(context.id)
        return UpperOutput(value=request_context.get())

    harness = AgentHarness(
        model="test", model_client=UnusedModel(), tools=[read_context]
    )
    token = request_context.set("available")
    try:
        result = await codemode.execute(
            ToolCallContext(harness=harness, name=codemode.name, id="outer-call"),
            CodemodeInput(
                code="result = read_context(value='unused')\noutput(result['value'])"
            ).model_dump(),
        )
    finally:
        request_context.reset(token)

    assert result.value == "available"
    assert call_ids == [None]


@pytest.mark.asyncio
async def test_codemode_enforces_runtime_limit() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        usage_limits=UsageLimits(max_codemode_runtime_seconds=0.1),
    )

    with pytest.raises(RuntimeError, match="timed out"):
        await codemode.execute(
            ToolCallContext(harness=harness, name=codemode.name),
            CodemodeInput(code="while True: pass").model_dump(),
        )


@pytest.mark.asyncio
async def test_codemode_blocks_process_control_exceptions() -> None:
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="SystemExit.*not defined"):
        await codemode.execute(
            ToolCallContext(harness=harness, name=codemode.name),
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
            ToolCallContext(harness=harness, name=codemode.name),
            CodemodeInput(code="output('x' * (1024 * 1024 * 1024))").model_dump(),
        )


@pytest.mark.asyncio
async def test_scoped_codemode_uses_only_explicit_hidden_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_call_ids: list[str | None] = []

    @tool(name="hidden_upper")
    async def hidden_upper(
        context: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        seen_call_ids.append(context.id)
        return UpperOutput(value=input_value.value.upper())

    @tool()
    async def lower(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        return UpperOutput(value=input_value.value.lower())

    runner = RecordingRunner(
        [worker_task("hidden_upper", "hello"), worker_task("output", "done")]
    )
    scoped = create_codemode_tool(
        capabilities=(
            CodeModeCapability(
                hidden_upper,
                result_adapter=raw_codemode_result_adapter,
            ),
        ),
        runner=runner,
    )
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
        ToolCallContext(harness=harness, name=scoped.name, id="outer-call"),
        {"code": "hidden_upper('hello'); output('done')", "question": "Why?"},
    )

    assert result.value == "done"
    assert len(seen_call_ids) == 1
    assert seen_call_ids[0] is not None
    assert seen_call_ids[0] != "outer-call"
    assert hidden_upper.name not in harness._tools
    assert set(runner.requests[0].function_names["harness"]) == {
        "hidden_upper",
        "output",
    }
    assert runner.requests[0].code == "hidden_upper('hello'); output('done')"
    assert runner.requests[0].question == "Why?"
    assert runner.requests[0].redact_errors is True
    assert runner.results[0] == ("hidden_upper", {"value": "HELLO"})


@pytest.mark.asyncio
async def test_remote_required_false_forces_isolated_process_with_ambient_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        codemode_module.settings, "sandbox_socket", "/ambient-sandbox.sock"
    )
    monkeypatch.setattr(codemode_module.settings, "sandbox_token", "ambient-token")

    @tool()
    async def lower(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        return UpperOutput(value=input_value.value.lower())

    scoped = create_codemode_tool(
        capabilities=(
            CodeModeCapability(
                upper,
                result_adapter=raw_codemode_result_adapter,
            ),
        ),
        remote_required=False,
    )
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[scoped, lower],
    )

    result = await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name),
        {
            "code": "result = upper(value='hello')\noutput(result['value'])",
        },
    )

    assert result.value == "HELLO"
    assert upper.name not in harness._tools
    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "lower(value='HELLO')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_default_projection_uses_formatted_context() -> None:
    runner = RecordingRunner(
        [worker_task("lookup", value="hello"), worker_task("output", None)]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(lookup),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name),
        {"code": "lookup(value='hello'); output('done')"},
    )

    assert runner.results == [("lookup", '{"value":"HELLO"}'), ("output", None)]


@pytest.mark.asyncio
async def test_scoped_codemode_raw_projection_must_be_explicit() -> None:
    runner = RecordingRunner(
        [worker_task("lookup", value="hello"), worker_task("output", None)]
    )
    scoped = create_codemode_tool(
        capabilities=(
            CodeModeCapability(
                lookup,
                result_adapter=raw_codemode_result_adapter,
            ),
        ),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name),
        {"code": "lookup(value='hello'); output('done')"},
    )

    assert runner.results == [
        ("lookup", {"value": "HELLO", "secret": "internal"}),
        ("output", None),
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_rejects_nested_unprojected_models() -> None:
    def wrap_model(
        _tool: HarnessTool[Any, UpperOutput], output: UpperOutput
    ) -> dict[str, BaseModel]:
        return {"wrapped": output}

    runner = RecordingRunner(
        [worker_task("upper", value="hello"), worker_task("output", None)]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=wrap_model),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="Code Mode capability 'upper' failed"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "upper(value='hello')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_rejects_nested_models_in_output() -> None:
    runner = RecordingRunner(
        [worker_task("output", {"items": [UpperOutput(value="unsafe")]})]
    )
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(TypeError, match="must not contain Pydantic model values"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "output({'items': []})"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_normalizes_adapter_results_as_strict_json() -> None:
    def project(
        _tool: HarnessTool[Any, UpperOutput], output: UpperOutput
    ) -> dict[str, Any]:
        return {"value": output.value, "positions": (1, 2)}

    runner = RecordingRunner(
        [worker_task("upper", value="hello"), worker_task("output", None)]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=project),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name),
        {"code": "upper(value='hello'); output('done')"},
    )

    assert runner.results == [
        ("upper", {"value": "HELLO", "positions": [1, 2]}),
        ("output", None),
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_rejects_reserved_model_marker() -> None:
    def project(
        _tool: HarnessTool[Any, UpperOutput], _output: UpperOutput
    ) -> dict[str, Any]:
        return {"wrapped": {"__model__": "Context"}}

    runner = RecordingRunner([worker_task("upper", value="hello")])
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=project),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="Code Mode capability 'upper' failed"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "upper(value='hello')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_rejects_non_finite_adapter_result() -> None:
    def project(
        _tool: HarnessTool[Any, UpperOutput], _output: UpperOutput
    ) -> dict[str, float]:
        return {"score": float("nan")}

    runner = RecordingRunner([worker_task("upper", value="hello")])
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=project),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="Code Mode capability 'upper' failed"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "upper(value='hello')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_normalizes_output_as_strict_json() -> None:
    runner = RecordingRunner(
        [worker_task("output", {"positions": (1, 2), "complete": True})]
    )
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    result = await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name),
        {"code": "output({'positions': (1, 2), 'complete': True})"},
    )

    assert result.value == {"positions": [1, 2], "complete": True}


@pytest.mark.asyncio
async def test_scoped_codemode_requires_output() -> None:
    scoped = create_codemode_tool(capabilities=(), runner=RecordingRunner([]))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match=r"must call output\(value\) exactly once"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "pass"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_rejects_repeated_output() -> None:
    runner = RecordingRunner(
        [worker_task("output", "first"), worker_task("output", "second")]
    )
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match=r"output\(value\) may only be called once"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "output('first'); output('second')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_rejects_reserved_model_marker_in_output() -> None:
    runner = RecordingRunner(
        [worker_task("output", {"wrapped": {"__model__": "Context"}})]
    )
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(
        ValueError,
        match="Code Mode output cannot contain the reserved '__model__' key",
    ):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "output({'wrapped': {'__model__': 'Context'}})"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_uses_normal_tool_validation() -> None:
    invalid_input_runner = RecordingRunner([worker_task("upper", missing="value")])
    input_scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper),),
        runner=invalid_input_runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match=r"capability 'upper' failed \(ValueError\)"):
        await input_scoped.execute(
            ToolCallContext(harness=harness, name=input_scoped.name),
            {"code": "upper(missing='value')"},
        )

    @tool()
    async def invalid_output(
        _context: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        return UpperInput(value=input_value.value)  # type: ignore[return-value]

    invalid_output_runner = RecordingRunner(
        [worker_task("invalid_output", value="hello")]
    )
    output_scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(invalid_output),),
        runner=invalid_output_runner,
    )
    with pytest.raises(
        RuntimeError, match=r"capability 'invalid_output' failed \(ValidationError\)"
    ):
        await output_scoped.execute(
            ToolCallContext(harness=harness, name=output_scoped.name),
            {"code": "invalid_output(value='hello')"},
        )


@pytest.mark.asyncio
async def test_scoped_positional_arguments_follow_advertised_alias_order() -> None:
    class AliasedInput(BaseModel):
        value: str = Field(alias="public")

    @tool(description="Use an aliased public argument")
    async def aliased(
        _context: ToolCallContext, input_value: AliasedInput
    ) -> UpperOutput:
        return UpperOutput(value=input_value.value.upper())

    runner = RecordingRunner(
        [worker_task("aliased", "hello"), worker_task("output", None)]
    )
    scoped = create_codemode_tool(
        capabilities=(
            CodeModeCapability(
                aliased,
                result_adapter=raw_codemode_result_adapter,
            ),
        ),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name),
        {"code": "aliased('hello'); output('done')"},
    )

    definition = runner.requests[0].function_names["harness"]["aliased"]
    assert list(definition.parameters) == ["public"]
    assert '"public"' in scoped.description
    assert runner.results == [
        ("aliased", {"value": "HELLO"}),
        ("output", None),
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_counts_nested_calls_as_ordinary_usage() -> None:
    runner = RecordingRunner(
        [
            worker_task("upper", value="one"),
            worker_task("upper", value="two"),
            worker_task("output", None),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper),),
        runner=runner,
    )
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        tools=[scoped],
        usage_limits=UsageLimits(max_tool_calls=3),
    )

    await harness._execute_tool_calls(
        [
            HarnessToolCall(
                id="outer-call",
                name=scoped.name,
                arguments={
                    "code": "upper(value='one'); upper(value='two'); output('done')"
                },
            )
        ]
    )

    assert harness.usage.tool_calls == 3


@pytest.mark.asyncio
async def test_scoped_codemode_assigns_unique_nested_call_ids() -> None:
    call_ids: list[str | None] = []

    @tool()
    async def record_id(
        context: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        call_ids.append(context.id)
        return UpperOutput(value=input_value.value)

    runner = RecordingRunner(
        [
            worker_task("record_id", value="one"),
            worker_task("record_id", value="two"),
            worker_task("output", None),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(record_id),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(
        ToolCallContext(harness=harness, name=scoped.name, id="outer-call"),
        {"code": "record_id(value='one'); record_id(value='two'); output('done')"},
    )

    assert len(call_ids) == 2
    assert all(call_id is not None for call_id in call_ids)
    assert len(set(call_ids)) == 2
    assert "outer-call" not in call_ids


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
        create_codemode_tool(capabilities=(CodeModeCapability(reserved),))


@pytest.mark.parametrize(
    "name",
    [
        "not-valid",
        "two words",
        "class",
        "1tool",
        "_private",
        "café",
        "ｏｕｔｐｕｔ",
    ],
)
def test_scoped_codemode_rejects_invalid_capability_names(name: str) -> None:
    invalid = HarnessTool(name, upper.handler)

    with pytest.raises(ValueError, match="ASCII public Python identifier"):
        create_codemode_tool(capabilities=(CodeModeCapability(invalid),))


def test_scoped_codemode_rejects_duplicate_capability_names() -> None:
    duplicate = HarnessTool(upper.name, upper.handler)

    with pytest.raises(ValueError, match="must be unique"):
        create_codemode_tool(
            capabilities=(CodeModeCapability(upper), CodeModeCapability(duplicate))
        )


def test_scoped_codemode_rejects_agent_id_input_field() -> None:
    class AgentInput(BaseModel):
        agent_id: str

    @tool()
    async def routed(_context: ToolCallContext, input_value: AgentInput) -> UpperOutput:
        return UpperOutput(value=input_value.agent_id)

    with pytest.raises(ValueError, match="reserved 'agent_id' input field"):
        create_codemode_tool(capabilities=(CodeModeCapability(routed),))


def test_scoped_codemode_requires_immutable_capability_tuple() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        create_codemode_tool(capabilities=[])  # type: ignore[arg-type]


def test_inherited_codemode_rejects_non_inherited_capability() -> None:
    @tool(inheritance=ToolInheritancePolicy.DO_NOT_INHERIT)
    async def parent_only(
        _context: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        return UpperOutput(value=input_value.value)

    with pytest.raises(ValueError, match="cannot include.*DO_NOT_INHERIT"):
        create_codemode_tool(
            capabilities=(CodeModeCapability(parent_only),),
            inheritance=ToolInheritancePolicy.INHERIT,
        )


def test_scoped_codemode_description_includes_capability_schema() -> None:
    scoped = create_codemode_tool(capabilities=(CodeModeCapability(upper),))
    custom = create_codemode_tool(
        capabilities=(CodeModeCapability(upper),),
        description="Custom orchestration instructions.",
    )

    assert "Uppercase a value" in scoped.description
    assert '"required":["value"]' in scoped.description
    assert "Custom orchestration instructions." in custom.description
    assert '"required":["value"]' in custom.description


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
        _ctx(harness),
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

    await scoped.execute(_ctx(harness), {"code": "lookup(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"]["type"] == "string"
    assert completed.payload["result"]["utf8_bytes"] > 0
    assert "internal-secret" not in str(completed.payload)


@pytest.mark.asyncio
async def test_scoped_codemode_does_not_persist_opaque_string_results() -> None:
    def project(_tool: HarnessTool, _output: BaseModel) -> str:
        return "authorization: Bearer projected-secret"

    scoped = local_codemode(CodeModeCapability(upper, result_adapter=project))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(_ctx(harness), {"code": "upper(value='safe'); output(None)"})

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

    await scoped.execute(_ctx(harness), {"code": "upper(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"] == {
        "type": "object",
        "entries": [
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": {"type": "string", "utf8_bytes": 35},
            },
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": {
                    "type": "array",
                    "items": [{"type": "string", "utf8_bytes": 14}],
                },
            },
        ],
    }
    assert "secret" not in str(completed.payload)


@pytest.mark.asyncio
async def test_scoped_codemode_treats_json_looking_strings_as_opaque() -> None:
    def project(_tool: HarnessTool, _output: BaseModel) -> str:
        return '{"score":1e9999}'

    scoped = local_codemode(CodeModeCapability(upper, result_adapter=project))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(_ctx(harness), {"code": "upper(value='safe'); output(None)"})

    events = [event async for event in harness.history()]
    completed = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_COMPLETED and event.payload.get("nested")
    )
    assert completed.payload["result"] == {
        "type": "string",
        "utf8_bytes": len('{"score":1e9999}'.encode()),
    }


def test_event_sanitizer_bounds_large_list() -> None:
    sanitized = codemode_module._sanitize_event_value(list(range(100_000)))

    assert sanitized["type"] == "array"
    assert len(sanitized["items"]) == codemode_module._EVENT_SANITIZE_MAX_ITEMS
    assert sanitized["truncated"] == {
        "reason": "item_limit",
        "omitted_items": 100_000 - codemode_module._EVENT_SANITIZE_MAX_ITEMS,
    }


def test_event_sanitizer_bounds_large_dictionary() -> None:
    sanitized = codemode_module._sanitize_event_value(
        {f"untrusted-key-{index}": index for index in range(100_000)}
    )

    assert sanitized["type"] == "object"
    assert len(sanitized["entries"]) == codemode_module._EVENT_SANITIZE_MAX_ITEMS
    assert sanitized["truncated"] == {
        "reason": "item_limit",
        "omitted_entries": 100_000 - codemode_module._EVENT_SANITIZE_MAX_ITEMS,
    }
    assert "untrusted-key" not in str(sanitized)


def test_event_sanitizer_enforces_global_node_budget() -> None:
    sanitized = codemode_module._sanitize_event_value(
        [[0] * codemode_module._EVENT_SANITIZE_MAX_ITEMS for _ in range(10_000)]
    )

    assert sanitized["truncated"]["reason"] == "node_budget"
    assert len(sanitized["items"]) < codemode_module._EVENT_SANITIZE_MAX_ITEMS
    assert sanitized["items"][-1]["truncated"]["reason"] == "node_budget"


def test_event_sanitizer_enforces_depth_limit() -> None:
    value: Any = 0
    for _ in range(100):
        value = [value]

    sanitized = codemode_module._sanitize_event_value(value)
    for _ in range(codemode_module._EVENT_SANITIZE_MAX_DEPTH):
        sanitized = sanitized["items"][0]

    assert sanitized == {
        "type": "array",
        "items": [],
        "truncated": {"reason": "depth_limit", "omitted_items": 1},
    }


def test_scoped_worker_encoding_stops_before_traversing_wide_value() -> None:
    class TrackingList(list[int]):
        visited = 0

        def __iter__(self):
            for item in super().__iter__():
                self.visited += 1
                if self.visited >= 100:
                    pytest.fail("strict worker encoding traversed the full container")
                yield item

    value = TrackingList(range(100_000))

    with pytest.raises(ValueError, match="exceeds maximum size"):
        codemode_module._normalize_worker_value(value, 32, "Wide value")

    assert value.visited < 100


def test_scoped_worker_encoding_stops_before_traversing_wide_dictionary() -> None:
    class TrackingDict(dict[str, int]):
        visited = 0

        def items(self):
            for item in super().items():
                self.visited += 1
                if self.visited >= 100:
                    pytest.fail("strict worker encoding traversed the full dictionary")
                yield item

    value = TrackingDict({str(index): index for index in range(100_000)})

    with pytest.raises(ValueError, match="exceeds maximum size"):
        codemode_module._normalize_worker_value(value, 32, "Wide value")

    assert value.visited < 100


def test_scoped_source_sizing_stops_after_crossing_limit() -> None:
    class GuardedSource(str):
        def encode(self, *_args, **_kwargs):
            pytest.fail("source sizing encoded the complete source")

    source = GuardedSource("é" * 1_000_000)
    tracemalloc.start()
    try:
        assert codemode_module._utf8_exceeds_limit(source, 32)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 100_000


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

    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await scoped.execute(_ctx(harness), {"code": "upper(value='hello')"})


@pytest.mark.asyncio
async def test_scoped_codemode_enforces_source_result_and_output_byte_limits() -> None:
    harness = AgentHarness(model="test", model_client=UnusedModel())
    source_limited = local_codemode(limits=CodeModeLimits(max_source_bytes=1))
    with pytest.raises(ValueError, match="source exceeds maximum size"):
        await source_limited.execute(_ctx(harness), {"code": "é"})

    result_limited = local_codemode(
        CodeModeCapability(upper, result_adapter=value_adapter),
        limits=CodeModeLimits(max_result_bytes=4),
    )
    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await result_limited.execute(_ctx(harness), {"code": "upper(value='hello')"})

    output_limited = local_codemode(limits=CodeModeLimits(max_output_bytes=4))
    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await output_limited.execute(_ctx(harness), {"code": "output('hello')"})


@pytest.mark.asyncio
async def test_scoped_codemode_result_policy_failure_remains_terminal_when_caught() -> (
    None
):
    calls: list[str] = []

    @tool()
    async def count(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    scoped = local_codemode(
        CodeModeCapability(count, result_adapter=value_adapter),
        limits=CodeModeLimits(max_result_bytes=8),
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    code = """
try:
    count(value='one')
except Exception:
    pass
try:
    count(value='retry')
except Exception:
    pass
try:
    output('apparently-successful')
except Exception:
    pass
"""

    with pytest.raises(ValueError, match="result from count exceeds maximum size"):
        await scoped.execute(_ctx(harness), {"code": code})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert calls == ["one"]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_preflights_isolated_result_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    @tool()
    async def transport_result(
        _: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    scoped = local_codemode(
        CodeModeCapability(
            transport_result,
            result_adapter=lambda _tool, _output: "x" * 127,
        ),
        limits=CodeModeLimits(max_result_bytes=256),
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    monkeypatch.setattr(codemode_module, "MAX_PACKET_BYTES", 128)
    code = """
try:
    transport_result(value='one')
except Exception:
    pass
try:
    transport_result(value='retry')
except Exception:
    pass
try:
    output('apparently-successful')
except Exception:
    pass
"""

    with pytest.raises(ValueError, match="result transport.*exceeds maximum size"):
        await scoped.execute(_ctx(harness), {"code": code})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert calls == ["one"]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert all(event.type != HarnessEventType.TOOL_COMPLETED for event in nested)


@pytest.mark.asyncio
async def test_scoped_codemode_preflights_remote_result_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    @tool()
    async def transport_result(
        _: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    runner = CatchingRunner(
        [
            worker_task("transport_result", value="one"),
            worker_task("transport_result", value="retry"),
            worker_task("output", "apparently-successful"),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(
            CodeModeCapability(
                transport_result,
                result_adapter=lambda _tool, _output: "x" * 110,
            ),
        ),
        limits=CodeModeLimits(max_result_bytes=128),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    monkeypatch.setattr(codemode_module, "MAX_PACKET_BYTES", 128)

    with pytest.raises(ValueError, match="result transport.*exceeds maximum size"):
        await scoped.execute(_ctx(harness), {"code": "unused"})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert calls == ["one"]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert all(event.type != HarnessEventType.TOOL_COMPLETED for event in nested)


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

    assert harness.usage.tool_calls == 2
    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_nested_limit_remains_terminal_when_caught() -> None:
    calls: list[str] = []

    @tool()
    async def count(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    scoped = local_codemode(
        CodeModeCapability(count, result_adapter=value_adapter),
        limits=CodeModeLimits(max_nested_calls=1),
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    code = """
count(value='one')
try:
    count(value='two')
except Exception:
    pass
try:
    count(value='three')
except Exception:
    pass
try:
    output('apparently-successful')
except Exception:
    pass
"""

    with pytest.raises(RuntimeError, match="nested call limit exceeded: 2 > 1"):
        await scoped.execute(_ctx(harness), {"code": code})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert calls == ["one"]
    assert harness.usage.tool_calls == 1
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
    ]


@pytest.mark.asyncio
async def test_scoped_codemode_global_tool_limit_remains_terminal_when_caught() -> None:
    calls: list[str] = []

    @tool()
    async def count(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    scoped = local_codemode(CodeModeCapability(count, result_adapter=value_adapter))
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        usage_limits=UsageLimits(max_tool_calls=1),
    )
    code = """
count(value='one')
try:
    count(value='two')
except Exception:
    pass
try:
    count(value='three')
except Exception:
    pass
try:
    output('apparently-successful')
except Exception:
    pass
"""

    with pytest.raises(RuntimeError, match="max_tool_calls limit exceeded: 2 > 1"):
        await scoped.execute(_ctx(harness), {"code": code})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert calls == ["one"]
    assert harness.usage.tool_calls == 2
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_point",
    [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
        HarnessEventType.TOOL_FAILED,
    ],
)
async def test_scoped_codemode_event_failures_are_redacted_and_terminal(
    monkeypatch: pytest.MonkeyPatch, failure_point: HarnessEventType
) -> None:
    backend_secret = f"{failure_point.value}-backend-secret"
    fixed_failure = "Code Mode capability failed"
    calls: list[str] = []

    @tool()
    async def event_target(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        if failure_point == HarnessEventType.TOOL_FAILED:
            raise RuntimeError("handler-secret")
        return UpperOutput(value=input_value.value)

    runner = CatchingRunner(
        [
            worker_task("event_target", value="one"),
            worker_task("event_target", value="two"),
            worker_task("output", "apparently-successful"),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(event_target, result_adapter=value_adapter),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    original_emit = harness.emit

    async def failing_emit(
        event_type: HarnessEventType, payload: dict[str, Any], **kwargs: Any
    ):
        if event_type == failure_point:
            raise RuntimeError(backend_secret)
        return await original_emit(event_type, payload, **kwargs)

    monkeypatch.setattr(harness, "emit", failing_emit)

    with pytest.raises(RuntimeError, match=f"^{fixed_failure}$") as exc:
        await scoped.execute(_ctx(harness), {"code": "unused"})

    events = [event async for event in harness.history()]
    assert runner.errors == [fixed_failure] * 3
    assert calls == (
        [] if failure_point == HarnessEventType.TOOL_REQUESTED else ["one"]
    )
    assert [event.type for event in events] == (
        []
        if failure_point == HarnessEventType.TOOL_REQUESTED
        else [HarnessEventType.TOOL_REQUESTED]
    )
    exposed = str([exc.value, runner.errors, events])
    assert backend_secret not in exposed
    assert "handler-secret" not in exposed


@pytest.mark.asyncio
async def test_scoped_codemode_tool_authored_event_failure_is_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_secret = "tool-authored-event-backend-secret"
    calls: list[str] = []

    @tool()
    async def event_target(
        context: ToolCallContext, input_value: UpperInput
    ) -> UpperOutput:
        calls.append(input_value.value)
        try:
            await context.emit(HarnessEventType.MESSAGE_ADDED, {"authored": True})
        except RuntimeError:
            pass
        return UpperOutput(value=input_value.value)

    runner = CatchingRunner(
        [
            worker_task("event_target", value="one"),
            worker_task("event_target", value="two"),
            worker_task("output", "apparently-successful"),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(event_target, result_adapter=value_adapter),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    original_emit = harness.emit

    async def failing_emit(
        event_type: HarnessEventType, payload: dict[str, Any], **kwargs: Any
    ):
        if event_type == HarnessEventType.MESSAGE_ADDED:
            raise RuntimeError(backend_secret)
        return await original_emit(event_type, payload, **kwargs)

    monkeypatch.setattr(harness, "emit", failing_emit)

    with pytest.raises(RuntimeError, match="^Code Mode capability failed$") as exc:
        await scoped.execute(_ctx(harness), {"code": "unused"})

    events = [event async for event in harness.history()]
    assert calls == ["one"]
    assert runner.errors == ["Code Mode capability failed"] * 3
    assert [event.type for event in events] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert backend_secret not in str([exc.value, runner.errors, events])


@pytest.mark.asyncio
async def test_scoped_codemode_can_recover_from_ordinary_handler_failure() -> None:
    calls: list[str] = []

    @tool()
    async def recoverable(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        if input_value.value == "fail":
            raise RuntimeError("handler-secret")
        return UpperOutput(value=input_value.value)

    runner = CatchingRunner(
        [
            worker_task("recoverable", value="fail"),
            worker_task("recoverable", value="ok"),
            worker_task("output", "done"),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(recoverable, result_adapter=value_adapter),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    result = await scoped.execute(_ctx(harness), {"code": "unused"})

    events = [event async for event in harness.history()]
    assert result.value == "done"
    assert calls == ["fail", "ok"]
    assert runner.errors == ["Code Mode capability 'recoverable' failed (RuntimeError)"]
    assert [event.type for event in events] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_COMPLETED,
    ]
    assert "handler-secret" not in str([runner.errors, events])


@pytest.mark.asyncio
async def test_scoped_codemode_unknown_capability_is_terminal() -> None:
    calls: list[str] = []

    @tool()
    async def allowed(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    runner = CatchingRunner(
        [
            worker_task("unknown"),
            worker_task("allowed", value="retry"),
            worker_task("output", "apparently-successful"),
        ]
    )
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(allowed, result_adapter=value_adapter),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="Unknown Code Mode capability: unknown"):
        await scoped.execute(_ctx(harness), {"code": "unused"})

    assert calls == []
    assert runner.errors == ["Unknown Code Mode capability: unknown"] * 3
    assert [event async for event in harness.history()] == []


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
    assert nested[0].payload["call"]["arguments"] == {
        "type": "object",
        "entries": [
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": {"type": "string", "utf8_bytes": 4},
            },
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": "[REDACTED]",
            },
        ],
    }
    assert nested[1].payload["result"] == {
        "type": "object",
        "entries": [
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": {"type": "string", "utf8_bytes": 4},
            },
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": "[REDACTED]",
            },
        ],
    }
    assert nested[0].payload["execution_context"] == {
        "actor_id": "actor-1",
        "tenant_id": "tenant-1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("marker", "expected"),
    [
        (
            "x" * (codemode_module._EVENT_CONTEXT_MAX_STRING_BYTES + 1),
            "exceeds maximum size",
        ),
        (float("nan"), "must contain finite numbers"),
        (float("inf"), "must contain finite numbers"),
        (float("-inf"), "must contain finite numbers"),
        (
            1 << (codemode_module._EVENT_CONTEXT_MAX_INTEGER_BYTES * 4 + 1),
            "exceeds maximum size",
        ),
        (
            -(1 << (codemode_module._EVENT_CONTEXT_MAX_INTEGER_BYTES * 4 + 1)),
            "exceeds maximum size",
        ),
    ],
)
async def test_scoped_codemode_rejects_unsafe_event_context_before_execution(
    marker: Any, expected: str
) -> None:
    runner = RecordingRunner([worker_task("output", "not-run")])
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        execution_context={"actor_id": marker},
    )

    with pytest.raises(ValueError, match=expected):
        await scoped.execute(_ctx(harness), {"code": "unused"})

    assert runner.requests == []
    assert [event async for event in harness.history()] == []


@pytest.mark.asyncio
async def test_scoped_codemode_enforces_total_event_context_budget() -> None:
    runner = RecordingRunner([worker_task("output", "not-run")])
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        execution_context={
            "actor": "x" * codemode_module._EVENT_CONTEXT_MAX_STRING_BYTES,
            "actor_id": "x" * codemode_module._EVENT_CONTEXT_MAX_STRING_BYTES,
            "tenant": "x" * codemode_module._EVENT_CONTEXT_MAX_STRING_BYTES,
            "tenant_id": "x" * codemode_module._EVENT_CONTEXT_MAX_STRING_BYTES,
            "user_id": 1,
        },
    )

    with pytest.raises(ValueError, match="exceed maximum total size"):
        await scoped.execute(_ctx(harness), {"code": "unused"})

    assert runner.requests == []
    assert [event async for event in harness.history()] == []


def test_scoped_codemode_event_context_keeps_booleans_distinct() -> None:
    harness = AgentHarness(
        model="test",
        model_client=UnusedModel(),
        execution_context={"actor": True, "tenant": False},
    )

    assert codemode_module._nested_event_context(harness) == {
        "execution_context": {"actor": True, "tenant": False}
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
    assert nested[1].payload["result"] == {
        "type": "object",
        "entries": [
            {
                "key": {"type": "string", "utf8_bytes": 5},
                "value": {"type": "string", "utf8_bytes": 4},
            }
        ],
    }
    assert nested[1].payload["result_bytes"] == len('{"value":"SAFE"}')
    assert nested[1].payload["duration_ms"] >= 0
    assert nested[0].payload["call"]["id"] == nested[1].payload["call_id"]


@pytest.mark.asyncio
async def test_scoped_codemode_never_persists_raw_event_dictionary_keys() -> None:
    class MappingInput(BaseModel):
        payload: dict[str, str]

    @tool()
    async def inspect_mapping(
        _: AgentHarness, _input_value: MappingInput
    ) -> UpperOutput:
        return UpperOutput(value="safe")

    def project(_tool: HarnessTool, _output: BaseModel) -> dict[str, str]:
        return {
            "result-secret-in-key": "visible",
            "password": "sensitive-result-value",
        }

    scoped = local_codemode(CodeModeCapability(inspect_mapping, result_adapter=project))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    await scoped.execute(
        _ctx(harness),
        {
            "code": (
                "inspect_mapping(payload={'argument-secret-in-key': 'visible', "
                "'token': 'sensitive-argument-value'}); output(None)"
            )
        },
    )

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    requested_arguments = nested[0].payload["call"]["arguments"]
    completed_result = nested[1].payload["result"]
    persisted = str([requested_arguments, completed_result])

    assert (
        requested_arguments["entries"][0]["value"]["entries"][1]["value"]
        == "[REDACTED]"
    )
    assert completed_result["entries"][1]["value"] == "[REDACTED]"
    assert "argument-secret-in-key" not in persisted
    assert "result-secret-in-key" not in persisted
    assert "token" not in persisted
    assert "password" not in persisted
    assert "sensitive-argument-value" not in persisted
    assert "sensitive-result-value" not in persisted


@pytest.mark.asyncio
async def test_scoped_codemode_records_nested_failure_before_propagating() -> None:
    @tool()
    async def fail(_: AgentHarness, _input_value: UpperInput) -> UpperOutput:
        raise ValueError("secret failure detail")

    scoped = local_codemode(CodeModeCapability(fail))
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await scoped.execute(_ctx(harness), {"code": "fail(value='safe')"})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert nested[1].payload["result"] == {"error": "ValueError"}
    assert "secret failure detail" not in str(nested)


def _secret_capability(secret: str) -> CodeModeCapability[Any]:
    @tool(name="secret_value")
    async def secret_value(
        _context: ToolCallContext, _input_value: UpperInput
    ) -> UpperOutput:
        return UpperOutput(value="internal")

    def project(_tool: HarnessTool, _output: BaseModel) -> str:
        return secret

    return CodeModeCapability(secret_value, result_adapter=project)


async def _assert_reraised_secret_is_redacted(
    scoped: HarnessTool[Any, Any], harness: AgentHarness, secret: str, caplog
) -> None:
    message = await harness._execute_tool_call(
        HarnessToolCall(
            id="outer-call",
            name=scoped.name,
            arguments={
                "code": (
                    "projected = secret_value(value='x')\nraise RuntimeError(projected)"
                )
            },
        )
    )
    events = [event async for event in harness.history()]
    persisted = str([event.payload for event in events])
    outer_failure = next(
        event
        for event in events
        if event.type == HarnessEventType.TOOL_FAILED
        and not event.payload.get("nested")
    )

    assert "Generated code execution failed" in message.content
    assert "Generated code execution failed" in str(outer_failure.payload)
    assert secret not in message.content
    assert secret not in persisted
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_isolated_scoped_codemode_redacts_reraised_projected_secret(
    caplog,
) -> None:
    secret = "isolated-projected-secret"
    scoped = local_codemode(_secret_capability(secret))
    harness = AgentHarness(model="test", model_client=UnusedModel(), tools=[scoped])
    caplog.set_level(logging.INFO)

    await _assert_reraised_secret_is_redacted(scoped, harness, secret, caplog)


@pytest.mark.asyncio
async def test_remote_scoped_codemode_redacts_reraised_projected_secret(
    monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    secret = "remote-projected-secret"
    caplog.set_level(logging.INFO)
    with tempfile.TemporaryDirectory(prefix="sbx", dir="/tmp") as directory:
        socket = str(Path(directory) / "sandbox.sock")
        monkeypatch.setenv("SANDBOX_SOCKET", socket)
        monkeypatch.setenv("SANDBOX_VERIFY", "false")
        monkeypatch.setenv("SANDBOX_TOKEN", "server-token")
        monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")
        monkeypatch.setattr(codemode_module.settings, "sandbox_socket", socket)
        monkeypatch.setattr(codemode_module.settings, "sandbox_token", "server-token")
        server_task = asyncio.create_task(sandbox_module.run_sandbox_server())
        for _ in range(200):
            if Path(socket).exists():
                break
            await asyncio.sleep(0.01)

        scoped = create_codemode_tool(capabilities=(_secret_capability(secret),))
        harness = AgentHarness(model="test", model_client=UnusedModel(), tools=[scoped])
        try:
            await _assert_reraised_secret_is_redacted(scoped, harness, secret, caplog)
            assert any(
                record.name == "hyperforge_codemode_sandbox"
                and "Generated code execution failed" in record.getMessage()
                for record in caplog.records
            )
        finally:
            server_task.cancel()
            await asyncio.gather(server_task, return_exceptions=True)


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
        await scoped.execute(_ctx(harness), {"code": "wait(value='x')"})

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
        scoped.execute(_ctx(harness), {"code": "wait(value='x')"})
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
        await scoped.execute(_ctx(harness), {"code": "output(None)"})


@pytest.mark.asyncio
async def test_scoped_codemode_fails_closed_without_remote_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codemode_module.settings, "sandbox_socket", "/unused.sock")
    monkeypatch.setattr(codemode_module.settings, "sandbox_token", None)
    scoped = create_codemode_tool(capabilities=())
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="SANDBOX_TOKEN is absent"):
        await scoped.execute(_ctx(harness), {"code": "output(None)"})


@pytest.mark.asyncio
async def test_scoped_codemode_passes_captured_token_when_settings_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str | None] = {}

    class RemoteRunner:
        def __init__(self, dispatch: CodeModeDispatch) -> None:
            self.dispatch = dispatch

        async def run(self, _request: WorkerExecutionRequest) -> None:
            monkeypatch.setattr(
                codemode_module.settings, "sandbox_token", "replacement-token"
            )
            await self.dispatch(worker_task("output", "ok"))

        def run_when_callbacks_complete(self, _callback: Any) -> bool:
            return False

    def remote(
        _socket: str,
        dispatch: CodeModeDispatch,
        _debug: bool = False,
        *,
        token: str | None = None,
    ) -> RemoteRunner:
        captured["token"] = token
        return RemoteRunner(dispatch)

    monkeypatch.setattr(codemode_module.settings, "sandbox_socket", "/sandbox.sock")
    monkeypatch.setattr(codemode_module.settings, "sandbox_token", "checked-token")
    monkeypatch.setattr(codemode_module.SandboxRunner, "remote", remote)
    scoped = create_codemode_tool(capabilities=())
    harness = AgentHarness(model="test", model_client=UnusedModel())

    result = await scoped.execute(_ctx(harness), {"code": "output('ok')"})

    assert result.value == "ok"
    assert captured == {"token": "checked-token"}
    assert codemode_module.settings.sandbox_token == "replacement-token"


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
        _ctx(harness),
        {"code": "upper('hello'); output({'done': True})", "question": "Q?"},
    )

    assert result.value == {"done": True}
    request = runner.requests[0]
    assert request.code == "upper('hello'); output({'done': True})"
    assert request.question == "Q?"
    assert set(request.function_names["harness"]) == {"upper", "output"}
    assert ("upper", {"value": "HELLO"}) in runner.results
    assert harness.usage.tool_calls == 1


@pytest.mark.asyncio
async def test_scoped_codemode_output_must_be_called_exactly_once() -> None:
    runner = RecordingRunner(
        [worker_task("output", 1), worker_task("output", 2)],
    )
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="may only be called once"):
        await scoped.execute(_ctx(harness), {"code": "output(1); output(2)"})


@pytest.mark.asyncio
async def test_scoped_codemode_repeated_output_remains_terminal_when_caught() -> None:
    scoped = local_codemode()
    harness = AgentHarness(model="test", model_client=UnusedModel())
    code = """
output(1)
try:
    output(2)
except Exception:
    pass
"""

    with pytest.raises(ValueError, match="may only be called once"):
        await scoped.execute(_ctx(harness), {"code": code})


@pytest.mark.asyncio
async def test_scoped_worker_output_serialization_failure_is_terminal() -> None:
    scoped = local_codemode()
    harness = AgentHarness(model="test", model_client=UnusedModel())
    code = """
try:
    output({1, 2})
except Exception:
    pass
output('apparently-successful')
"""

    with pytest.raises(RuntimeError, match="Generated code execution failed"):
        await scoped.execute(_ctx(harness), {"code": code})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_attempt", "expected"),
    [
        ("output()", "output requires a value"),
        ("output('too-large')", "exceeds maximum size"),
        (
            "output({'__model__': 'attacker-controlled'})",
            "reserved '__model__' key",
        ),
    ],
)
async def test_scoped_codemode_invalid_first_output_attempt_remains_terminal(
    first_attempt: str, expected: str
) -> None:
    scoped = local_codemode(limits=CodeModeLimits(max_output_bytes=8))
    harness = AgentHarness(model="test", model_client=UnusedModel())
    code = f"""
try:
    {first_attempt}
except Exception:
    pass
try:
    output(None)
except Exception:
    pass
"""

    with pytest.raises((TypeError, ValueError), match=expected):
        await scoped.execute(_ctx(harness), {"code": code})


@pytest.mark.asyncio
async def test_scoped_codemode_output_requires_a_value() -> None:
    runner = RecordingRunner([worker_task("output")])
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="output requires a value"):
        await scoped.execute(_ctx(harness), {"code": "output()"})


@pytest.mark.asyncio
async def test_scoped_codemode_requires_output_call() -> None:
    runner = RecordingRunner([])
    scoped = create_codemode_tool(capabilities=(), runner=runner)
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(ValueError, match="must call output\\(value\\) exactly once"):
        await scoped.execute(_ctx(harness), {"code": "1 + 1"})


@pytest.mark.asyncio
async def test_scoped_codemode_enforces_cumulative_result_limit() -> None:
    calls: list[str] = []

    @tool()
    async def count(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        calls.append(input_value.value)
        return UpperOutput(value=input_value.value)

    scoped = local_codemode(
        CodeModeCapability(count, result_adapter=value_adapter),
        limits=CodeModeLimits(max_result_bytes=20, max_cumulative_result_bytes=20),
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())
    code = """
count(value='hi')
try:
    count(value='ok')
except Exception:
    pass
try:
    count(value='retry')
except Exception:
    pass
try:
    output('apparently-successful')
except Exception:
    pass
"""

    with pytest.raises(RuntimeError, match="cumulative result limit exceeded"):
        await scoped.execute(_ctx(harness), {"code": code})

    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    assert calls == ["hi", "ok"]
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


@pytest.mark.parametrize("value", [0, -1, True, 1.5, float("nan")])
def test_scoped_codemode_rejects_invalid_limits(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        CodeModeLimits(max_source_bytes=value)  # type: ignore[arg-type]


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
    first = asyncio.create_task(
        scoped.execute(_ctx(harness), {"code": "wait(value='x')"})
    )
    await started.wait()

    with pytest.raises(RuntimeError, match="concurrency limit reached"):
        await scoped.execute(_ctx(harness), {"code": "output(None)"})

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
    async def wait(context: ToolCallContext, _input_value: UpperInput) -> UpperOutput:
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()
            await context.emit(HarnessEventType.MESSAGE_ADDED, {"tool_authored": True})
            raise
        finally:
            callback_done.set()

    monkeypatch.setattr("hyperforge.codemode.sandbox.CALLBACK_CANCEL_TIMEOUT", 0.01)
    limiter = CodeModeExecutionLimiter(1)
    scoped = local_codemode(CodeModeCapability(wait), execution_limiter=limiter)
    harness = AgentHarness(model="test", model_client=UnusedModel())
    harness._turn_id = "originating-turn"
    execution = asyncio.create_task(
        scoped.execute(_ctx(harness), {"code": "wait(value='x')"})
    )
    await started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=0.5)
    await cancellation_seen.wait()
    with pytest.raises(RuntimeError, match="concurrency limit reached"):
        await scoped.execute(_ctx(harness), {"code": "output(None)"})

    harness._turn_id = "later-turn"
    release.set()
    await asyncio.wait_for(callback_done.wait(), timeout=0.5)
    await asyncio.sleep(0)
    result = await scoped.execute(_ctx(harness), {"code": "output('released')"})
    events = [event async for event in harness.history()]
    nested = [event for event in events if event.payload.get("nested")]
    tool_authored = [event for event in events if event.payload.get("tool_authored")]

    assert result.value == "released"
    assert [event.type for event in nested] == [
        HarnessEventType.TOOL_REQUESTED,
        HarnessEventType.TOOL_FAILED,
    ]
    assert len(tool_authored) == 1
    assert tool_authored[0].parent_call_id == nested[0].payload["call"]["id"]
    assert {event.turn_id for event in [*nested, *tool_authored]} == {
        "originating-turn"
    }
