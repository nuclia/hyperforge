import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer
from hyperforge.minimal_fixtures import cassette_nua_key
from nuclia.lib.nua_responses import Tool, ToolChoiceAuto

from hyperforge_mcp.agent import MCPAgent, _tool_parameters
from hyperforge_mcp.config import MCPAgentConfig

from .mcp_server import run
from .mcp_server_no_prompt import run_mcp_server_no_prompt
from .mcp_server_prompts import run_mcp_server_prompt
from .mcp_server_token import (
    TEST_TOKEN_FOR_UNIT_TESTS,
    run_mcp_server_with_token_auth,
)

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")
pytestmark = [pytest.mark.vcr(ignore_localhost=True), pytest.mark.asyncio]


async def test_tool_parameters_preserve_array_item_schema():
    parameters = _tool_parameters(
        {
            "properties": {
                "repo_types": {
                    "type": "array",
                    "description": "Repository types to search",
                    "items": {"type": "string", "enum": ["model", "dataset"]},
                }
            }
        }
    )

    assert parameters["repo_types"] == {
        "type": "array",
        "description": "Repository types to search",
        "items": {"type": "string", "enum": ["model", "dataset"]},
    }


async def test_choose_tool_uses_auto_tool_choice():
    mcp_agent = MCPAgent(
        MCPAgentConfig.model_validate(
            {
                "id": "mcp-test",
                "module": "mcp",
                "source": "mcphttp-01",
            }
        )
    )
    captured_items = []

    async def execute_raw(item, tracking=None):
        captured_items.append(item)
        return SimpleNamespace(tools=None), 0.0, 0.0

    await mcp_agent.choose_tool(
        manager=SimpleNamespace(execute_raw=execute_raw),
        images=[],
        messages=[],
        extra_tools=[
            Tool(name="task_complete", description="", parameters={"type": "object"})
        ],
    )

    assert isinstance(captured_items[0].tool_choice, ToolChoiceAuto)


async def test_mcp_interaction_stops_when_no_tool_is_selected():
    mcp_agent = MCPAgent(
        MCPAgentConfig.model_validate(
            {
                "id": "mcp-test",
                "module": "mcp",
                "source": "mcphttp-01",
                "work_chain": True,
                "max_turns": 5,
            }
        )
    )
    choose_tool = AsyncMock(return_value=(SimpleNamespace(tools=None), 1.0, 2.0))

    with (
        patch.object(
            MCPAgent,
            "get_tool_selection_prompt",
            AsyncMock(return_value=([], [])),
        ),
        patch.object(MCPAgent, "choose_tool", choose_tool),
    ):
        tokens = await mcp_agent.mcp_interaction(
            memory=SimpleNamespace(get_tracking_info=lambda: None),
            manager=SimpleNamespace(),
            question="No tool is needed",
            context=SimpleNamespace(),
        )

    assert tokens == (1.0, 2.0)
    choose_tool.assert_awaited_once()


CONFIG = {
    "drivers": [
        {
            "provider": "mcphttp",
            "identifier": "mcphttp-01",
            "name": "mcphttp",
            "config": {
                "uri": "http://localhost:3001",
            },
        },
    ],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [],
    "context": [
        {
            "module": "mcp",
            "transport": "HTTP",
            "description": "mcp server with info about SPORTS",
            "source": "mcphttp-01",
            "valid_headers": ["AUTHORIZATION"],
        }
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}


@pytest.fixture
async def disable_safe_transport():
    with patch(
        "hyperforge.utils.http._resolve_public_addresses",
        return_value=["127.0.0.1"],
    ):
        yield


async def test_mcp_client(disable_safe_transport):
    async def get_answer(question, expected_answers):
        answers = []

        async def callback(obj: AragAnswer):
            answers.append(obj)

        question_memory = await arag_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question=question,
            config=CONFIG,
            callback=callback,
            loaded_modules=["hyperforge_mcp", "hyperforge_summarize"],
        )

        assert all(["Error" not in step.value for step in question_memory.steps])
        step_titles = [step.title for step in question_memory.steps]
        assert "MCP: Tool result" in step_titles
        assert "MCP: Interaction finished" in step_titles
        assert "Summarize: Summarize" in step_titles
        assert all(": " in title for title in step_titles)
        for expected in expected_answers:
            assert (
                question_memory.final_answer
                and expected in question_memory.final_answer
            )

    async with run() as url:
        CONFIG["drivers"][0]["config"]["uri"] = f"{url}/mcp"

        # Test simple question
        await get_answer("How much is 2 + 2?", ["4.0"])

        # Test multiple mcp tools
        await get_answer("How much is 2 + 2? and also 6/3?", ["4.0", "2.0"])


async def test_mcp_client_headers(disable_safe_transport):
    async with run_mcp_server_with_token_auth() as url:
        config = deepcopy(CONFIG)
        config["drivers"][0]["config"]["uri"] = f"{url}/mcp"
        answers = []

        async def callback(obj: AragAnswer):
            answers.append(obj)

        question_memory = await arag_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question="How much is 2 + 2?",
            config=config,
            callback=callback,
            headers={"AUTHORIZATION": f"Bearer {TEST_TOKEN_FOR_UNIT_TESTS}"},
            loaded_modules=["hyperforge_mcp", "hyperforge_summarize"],
        )

        assert question_memory.final_answer and "4.0" in question_memory.final_answer


async def test_mcp_client_prompt(disable_safe_transport):
    async with run_mcp_server_prompt() as url:
        config = deepcopy(CONFIG)
        config["drivers"][0]["config"]["uri"] = f"{url}/mcp"
        config["context"][0]["include_mcp_prompts"] = True
        config["generation"][0]["include_mcp_prompts"] = True

        answers = []

        async def callback(obj: AragAnswer):
            answers.append(obj)

        question_memory = await arag_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question="How much is 2 * 5?",
            config=config,
            callback=callback,
            headers={"AUTHORIZATION": f"Bearer {TEST_TOKEN_FOR_UNIT_TESTS}"},
            loaded_modules=["hyperforge_mcp", "hyperforge_summarize"],
        )

        assert question_memory.final_answer and "10" in question_memory.final_answer
        assert (
            "apples" in question_memory.final_answer.lower()
            or "banana" in question_memory.final_answer.lower()
        )


async def test_mcp_client_no_prompt(disable_safe_transport):
    async with run_mcp_server_no_prompt() as url:
        config = deepcopy(CONFIG)
        config["drivers"][0]["config"]["uri"] = f"{url}/mcp"
        config["context"][0]["include_mcp_prompts"] = True
        config["generation"][0]["include_mcp_prompts"] = True
        answers = []

        async def callback(obj: AragAnswer):
            answers.append(obj)

        question_memory = await arag_main(
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            question="How much is 2 * 5?",
            config=config,
            callback=callback,
            headers={"AUTHORIZATION": f"Bearer {TEST_TOKEN_FOR_UNIT_TESTS}"},
            loaded_modules=["hyperforge_mcp", "hyperforge_summarize"],
        )

        assert question_memory.final_answer and "10" in question_memory.final_answer
