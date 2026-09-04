import sys
from collections.abc import AsyncIterator
from contextvars import ContextVar

import pytest
from pydantic import BaseModel

from hyperforge.harness_sdk import (
    AgentContext,
    AgentHarness,
    CodemodeInput,
    ModelDelta,
    UsageLimits,
    codemode,
    tool,
)


class UpperInput(BaseModel):
    value: str


class UpperOutput(BaseModel):
    value: str


@tool(description="Uppercase a value")
async def upper(_: AgentContext, input_value: UpperInput) -> UpperOutput:
    return UpperOutput(value=input_value.value.upper())


class UnusedModel:
    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        del kwargs
        yield ModelDelta(text="unused")


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
        AgentContext(harness=harness, name=codemode.name),
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
            AgentContext(harness=harness, name=codemode.name),
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
            AgentContext(harness=harness, name=codemode.name),
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
        AgentContext(harness=harness, name=codemode.name),
        CodemodeInput(code=code).model_dump(),
    )

    assert result.value == expected


@pytest.mark.asyncio
async def test_codemode_preserves_context_when_calling_tools() -> None:
    request_context = ContextVar("request_context", default="missing")

    @tool()
    async def read_context(_: AgentContext, _input_value: UpperInput) -> UpperOutput:
        return UpperOutput(value=request_context.get())

    harness = AgentHarness(
        model="test", model_client=UnusedModel(), tools=[read_context]
    )
    token = request_context.set("available")
    try:
        result = await codemode.execute(
            AgentContext(harness=harness, name=codemode.name),
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
            AgentContext(harness=harness, name=codemode.name),
            CodemodeInput(code="while True: pass").model_dump(),
        )


@pytest.mark.asyncio
async def test_codemode_blocks_process_control_exceptions() -> None:
    harness = AgentHarness(model="test", model_client=UnusedModel())

    with pytest.raises(RuntimeError, match="SystemExit.*not defined"):
        await codemode.execute(
            AgentContext(harness=harness, name=codemode.name),
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
            AgentContext(harness=harness, name=codemode.name),
            CodemodeInput(code="output('x' * (1024 * 1024 * 1024))").model_dump(),
        )
