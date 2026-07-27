from unittest.mock import AsyncMock, MagicMock

import mcp.types as types
import pytest
from hyperforge.models import Context

from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig


async def run_tool_result(result, resource_result=None):
    agent = MCPAgent(
        MCPAgentConfig(
            source="test-source",
            max_tool_result_bytes=32,
            max_tool_result_item_bytes=32,
        )
    )
    context = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        source="test-source",
        agent="mcp",
    )
    session = AsyncMock()
    session.call_tool.return_value = result
    if resource_result is not None:
        session.read_resource.return_value = resource_result
    memory = MagicMock(add_step=AsyncMock())
    messages = []

    await agent.process_tool(
        memory=memory,
        tool_name="search",
        tool_arguments={},
        context=context,
        messages=messages,
        images=[],
        session=session,
    )
    return context, messages, memory


def rejection_steps(memory):
    return [
        call.kwargs
        for call in memory.add_step.await_args_list
        if call.kwargs["step_title"] == "MCP: Tool result rejected"
    ]


@pytest.mark.asyncio
async def test_rejects_an_oversized_text_result():
    context, messages, memory = await run_tool_result(
        types.CallToolResult(
            content=[types.TextContent(type="text", text="x" * 100)],
            isError=False,
        )
    )

    assert len(context.chunks) == 1
    assert "safety budget" in context.chunks[0].text
    assert "x" * 100 not in context.chunks[0].text
    assert "safety budget" in messages[0].text
    assert len(rejection_steps(memory)) == 1
    assert "observed_bytes=100" in rejection_steps(memory)[0]["step_value"]


@pytest.mark.asyncio
async def test_rejects_oversized_structured_content():
    oversized = "x" * 100
    context, messages, memory = await run_tool_result(
        types.CallToolResult(
            content=[types.TextContent(type="text", text="small")],
            structuredContent={"data": oversized},
            isError=False,
        )
    )

    assert context.structured == []
    assert len(context.chunks) == 1
    assert all(oversized not in message.text for message in messages)
    assert len(rejection_steps(memory)) == 1


@pytest.mark.asyncio
async def test_rejects_multiple_text_blocks_over_total_limit():
    context, messages, _ = await run_tool_result(
        types.CallToolResult(
            content=[
                types.TextContent(type="text", text="x" * 20),
                types.TextContent(type="text", text="y" * 20),
            ],
            isError=False,
        )
    )

    assert len(context.chunks) == 1
    assert "safety budget" in context.chunks[0].text
    assert all("x" * 20 not in message.text for message in messages)
    assert all("y" * 20 not in message.text for message in messages)


@pytest.mark.asyncio
async def test_rejects_linked_resources_over_total_limit():
    context, messages, _ = await run_tool_result(
        types.CallToolResult(
            content=[
                types.ResourceLink(
                    type="resource_link", name="report", uri="resource://report"
                )
            ],
            isError=False,
        ),
        types.ReadResourceResult(
            contents=[
                types.TextResourceContents(uri="resource://one", text="x" * 20),
                types.TextResourceContents(uri="resource://two", text="y" * 20),
            ]
        ),
    )

    assert len(context.chunks) == 1
    assert "safety budget" in context.chunks[0].text
    assert all("x" * 20 not in message.text for message in messages)
    assert all("y" * 20 not in message.text for message in messages)


@pytest.mark.asyncio
async def test_adds_a_safe_error_to_history_for_retry():
    _, messages, memory = await run_tool_result(
        types.CallToolResult(
            content=[types.TextContent(type="text", text="x" * 100)],
            isError=True,
        )
    )

    assert len(messages) == 1
    assert "safety budget" in messages[0].text
    assert "x" * 100 not in messages[0].text
    assert len(rejection_steps(memory)) == 1
