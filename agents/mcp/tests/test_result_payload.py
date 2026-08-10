from unittest.mock import AsyncMock, MagicMock

import mcp.types as types
import pytest
from hyperforge.models import Context

from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig


def test_payload_limits_are_visible_in_kb_with_friendly_defaults():
    config = MCPAgentConfig(source="test-source")
    schema = MCPAgentConfig.model_json_schema()["properties"]

    assert config.max_tool_result_kb == 64
    assert config.max_tool_result_item_kb == 16
    assert schema["max_tool_result_kb"]["title"] == "Maximum tool result (KB)"
    assert "widget" not in schema["max_tool_result_kb"]
    assert "max_tool_result_bytes" not in schema


def test_legacy_byte_limits_are_migrated_to_kb():
    config = MCPAgentConfig.model_validate(
        {
            "source": "test-source",
            "max_tool_result_bytes": 65 * 1024,
            "max_tool_result_item_bytes": 17 * 1024,
        }
    )

    assert config.max_tool_result_kb == 65
    assert config.max_tool_result_item_kb == 17
    assert "max_tool_result_bytes" not in config.model_dump()


async def run_tool_result(result, resource_result=None):
    agent = MCPAgent(
        MCPAgentConfig(
            source="test-source",
            max_tool_result_kb=1,
            max_tool_result_item_kb=1,
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
            content=[types.TextContent(type="text", text="x" * 2048)],
            isError=False,
        )
    )

    assert len(context.chunks) == 1
    assert "safety budget" in context.chunks[0].text
    assert "x" * 2048 not in context.chunks[0].text
    assert "safety budget" in messages[0].text
    assert len(rejection_steps(memory)) == 1
    assert "observed_bytes=2048" in rejection_steps(memory)[0]["step_value"]


@pytest.mark.asyncio
async def test_rejects_oversized_structured_content():
    oversized = "x" * 2048
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
                types.TextContent(type="text", text="x" * 600),
                types.TextContent(type="text", text="y" * 600),
            ],
            isError=False,
        )
    )

    assert len(context.chunks) == 1
    assert "safety budget" in context.chunks[0].text
    assert all("x" * 600 not in message.text for message in messages)
    assert all("y" * 600 not in message.text for message in messages)


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
                types.TextResourceContents(uri="resource://one", text="x" * 600),
                types.TextResourceContents(uri="resource://two", text="y" * 600),
            ]
        ),
    )

    assert len(context.chunks) == 1
    assert "safety budget" in context.chunks[0].text
    assert all("x" * 600 not in message.text for message in messages)
    assert all("y" * 600 not in message.text for message in messages)


@pytest.mark.asyncio
async def test_adds_a_safe_error_to_history_for_retry():
    _, messages, memory = await run_tool_result(
        types.CallToolResult(
            content=[types.TextContent(type="text", text="x" * 2048)],
            isError=True,
        )
    )

    assert len(messages) == 1
    assert "safety budget" in messages[0].text
    assert "x" * 2048 not in messages[0].text
    assert len(rejection_steps(memory)) == 1
