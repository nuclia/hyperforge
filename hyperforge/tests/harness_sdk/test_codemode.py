import sys
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

import pytest
from pydantic import BaseModel, Field, ValidationError

from hyperforge.codemode import RestrictedPythonTask, WorkerExecutionRequest
from hyperforge.harness_sdk import (
    AgentHarness,
    CodeModeCapability,
    CodeModeDispatch,
    CodemodeInput,
    HarnessContextReference,
    HarnessContextType,
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


def worker_task(function: str, *args: Any, **kwargs: Any) -> RestrictedPythonTask:
    return RestrictedPythonTask(
        function=function,
        agent="harness",
        args=args,
        keyword_args=kwargs,
    )


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
            "call_tool",
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
    assert runner.results[0] == ("hidden_upper", {"value": "HELLO"})


@pytest.mark.asyncio
async def test_scoped_codemode_runs_hidden_capability_in_isolated_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(codemode_module.settings, "sandbox_socket", None)

    @tool()
    async def lower(_: ToolCallContext, input_value: UpperInput) -> UpperOutput:
        return UpperOutput(value=input_value.value.lower())

    scoped = create_codemode_tool(
        capabilities=(
            CodeModeCapability(
                upper,
                result_adapter=raw_codemode_result_adapter,
            ),
        )
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
    with pytest.raises(RuntimeError, match="lower.*not defined"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "lower(value='HELLO')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_default_projection_uses_formatted_context() -> None:
    runner = RecordingRunner(
        [worker_task("lookup", value="hello"), worker_task("output", "done")]
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
        [worker_task("lookup", value="hello"), worker_task("output", "done")]
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

    runner = RecordingRunner([worker_task("upper", value="hello")])
    scoped = create_codemode_tool(
        capabilities=(CodeModeCapability(upper, result_adapter=wrap_model),),
        runner=runner,
    )
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(TypeError, match="must not contain Pydantic model values"):
        await scoped.execute(
            ToolCallContext(harness=harness, name=scoped.name),
            {"code": "upper(value='hello')"},
        )


@pytest.mark.asyncio
async def test_scoped_codemode_normalizes_adapter_results_as_strict_json() -> None:
    def project(
        _tool: HarnessTool[Any, UpperOutput], output: UpperOutput
    ) -> dict[str, Any]:
        return {"value": output.value, "positions": (1, 2)}

    runner = RecordingRunner(
        [worker_task("upper", value="hello"), worker_task("output", "done")]
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

    with pytest.raises(ValueError, match="reserved '__model__' key"):
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

    with pytest.raises(ValueError, match="strict JSON value"):
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

    with pytest.raises(ValueError, match="Invalid upper arguments"):
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
    with pytest.raises(ValidationError):
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
        [worker_task("aliased", "hello"), worker_task("output", "done")]
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
            worker_task("output", "done"),
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
            worker_task("output", "done"),
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


@pytest.mark.parametrize("name", ["codemode", "output", "question", "agent_id"])
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
