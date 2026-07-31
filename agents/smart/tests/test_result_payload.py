from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.models import Chunk, Context

from hyperforge_smart.agent import RegisteredAgent, SmartAgent, ToolError
from hyperforge_smart.config import SmartAgentConfig


def test_process_results_replaces_an_oversized_context_with_retry_guidance():
    agent = SmartAgent(
        SmartAgentConfig(
            max_tool_result_bytes=32,
            max_tool_result_item_bytes=32,
        )
    )
    result = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        source="test",
        agent="test",
        chunks=[Chunk(chunk_id="large", text="x" * 100)],
    )
    collected_contexts: list[Context] = []

    texts = agent._process_results(
        [("tool of test", result)], collected_contexts=collected_contexts
    )

    assert len(texts) == 1
    assert "safety budget" in texts[0]
    assert "x" * 100 not in texts[0]
    assert len(collected_contexts) == 1
    assert "safety budget" in collected_contexts[0].chunks[0].text
    assert collected_contexts[0].chunks[0].text != result.chunks[0].text


def test_process_results_budgets_a_context_summary_before_raw_chunks():
    agent = SmartAgent(
        SmartAgentConfig(
            max_tool_result_bytes=32,
            max_tool_result_item_bytes=32,
        )
    )
    result = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        source="test",
        agent="test",
        summary="The answer is Haverhill.",
        chunks=[Chunk(chunk_id="large", text="x" * 100)],
    )
    collected_contexts: list[Context] = []

    texts = agent._process_results(
        [("search of test", result)], collected_contexts=collected_contexts
    )

    assert texts == ["[search of test]:\nThe answer is Haverhill."]
    assert collected_contexts == [result]


def test_process_results_replaces_an_oversized_tool_error():
    agent = SmartAgent(
        SmartAgentConfig(
            max_tool_result_bytes=32,
            max_tool_result_item_bytes=32,
        )
    )
    error = ToolError(tool_name="search", tool_arguments={}, error="x" * 100)

    texts = agent._process_results([("search", error)])

    assert len(texts) == 1
    assert "safety budget" in texts[0]
    assert "x" * 100 not in texts[0]


@pytest.mark.asyncio
async def test_report_tool_error_records_a_safe_rejection_step():
    agent = SmartAgent(
        SmartAgentConfig(
            max_tool_result_bytes=32,
            max_tool_result_item_bytes=32,
        )
    )
    memory = MagicMock(add_step=AsyncMock())

    _, error = await agent._report_tool_error(
        memory=memory,
        title="LLM Execution error",
        error="x" * 100,
        tool_name="search",
        tool_arguments={},
    )

    assert "safety budget" in error.error
    assert "x" * 100 not in error.error
    assert (
        memory.add_step.await_args.kwargs["step_title"]
        == "Smart agent: Tool result rejected"
    )
    assert "observed_bytes=100" in memory.add_step.await_args.kwargs["step_value"]


class OversizedResultAgent(ContextAgent):
    agent_id = "oversized-result-agent"

    async def get_large_result(self, memory, manager):
        return Context(
            original_question_uuid=None,
            actual_question_uuid=None,
            question="question",
            source="test",
            agent="test",
            chunks=[Chunk(chunk_id="large", text="x" * 100)],
        )


class BoundedResultAgent(ContextAgent):
    agent_id = "bounded-result-agent"

    async def get_bounded_result(self, memory, manager):
        return Context(
            original_question_uuid=None,
            actual_question_uuid=None,
            question="question",
            source="test",
            agent="test",
            chunks=[
                Chunk(chunk_id="one", text="x" * 20),
                Chunk(chunk_id="two", text="y" * 10),
            ],
        )


@pytest.mark.asyncio
async def test_execute_tool_call_records_an_oversized_result_step():
    agent = SmartAgent(
        SmartAgentConfig(
            max_tool_result_bytes=32,
            max_tool_result_item_bytes=32,
        )
    )
    agent.registered_agents = [
        RegisteredAgent(
            agent=OversizedResultAgent(),
            available_functions={
                "get_large_result": FunctionDefinition(
                    name="get_large_result",
                    description="Returns a large result",
                    parameters={},
                )
            },
        )
    ]
    memory = MagicMock(add_step=AsyncMock())

    action_info, result = await agent.execute_tool_call(
        memory=memory,
        manager=MagicMock(),
        tool_name="get_large_result__oversized-result-agent",
        tool_arguments={},
    )

    assert action_info == "get_large_result of oversized-result-agent"
    assert isinstance(result, Context)
    memory.add_step.assert_awaited_once()
    assert (
        memory.add_step.await_args.kwargs["step_title"]
        == "Smart agent: Tool result rejected"
    )
    assert "observed_bytes=" in memory.add_step.await_args.kwargs["step_value"]
    assert "byte_limit=32" in memory.add_step.await_args.kwargs["step_value"]


@pytest.mark.asyncio
async def test_execute_tool_call_does_not_reject_a_bounded_context():
    agent = SmartAgent(
        SmartAgentConfig(
            max_tool_result_bytes=100,
            max_tool_result_item_bytes=64,
        )
    )
    agent.registered_agents = [
        RegisteredAgent(
            agent=BoundedResultAgent(),
            available_functions={
                "get_bounded_result": FunctionDefinition(
                    name="get_bounded_result",
                    description="Returns a bounded result",
                    parameters={},
                )
            },
        )
    ]
    memory = MagicMock(add_step=AsyncMock())

    await agent.execute_tool_call(
        memory=memory,
        manager=MagicMock(),
        tool_name="get_bounded_result__bounded-result-agent",
        tool_arguments={},
    )

    memory.add_step.assert_not_awaited()
