import json
import socket
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from hyperforge.api.models import (
    InteractionOperation,
    InteractionRequest,
)
from hyperforge.context.agent import ContextAgent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.engine import State
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
)
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.trace import trace_agent
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.config import StandaloneConfig
from hyperforge.standalone.settings import StandaloneSettings
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent
from websockets.asyncio.client import connect

pytestmark = pytest.mark.asyncio

AGENT_ID = "test-agent"

# Minimal agent config — no drivers, no real LLM calls needed since we mock get_state.
# The "ask" workflow is exposed as an MCP tool so we can test list_tools / call_tool.


@pytest.fixture
def local_agents_config(load_agents):
    return StandaloneConfig.validate_python(
        {
            AGENT_ID: {
                "title": "Test Agent",
                "instructions": "A test agent.",
                "workflows": {
                    "default": {
                        "name": "Default",
                        "generation": [{"module": "summarize"}],
                    },
                    "ask": {
                        "name": "ask",
                        "description": "Ask a question and get an answer.",
                        "parameters": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask.",
                            }
                        },
                        "required": ["question"],
                        "generation": [{"module": "summarize"}],
                    },
                },
            }
        }
    )


STANDALONE_SETTINGS = StandaloneSettings(
    agents_config=Path("/dev/null"),  # not read at runtime; config is passed directly
    external_nua_api_key="dummy",
)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class MockContextAgent(ContextAgent):
    """Context agent that echoes the question back as its answer."""

    config: ContextAgentConfig = ContextAgentConfig(module="mock")
    agent_id: str = "mock"

    @classmethod
    def config_class(cls):
        return ContextAgentConfig

    @trace_agent
    async def get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: Optional[str] = None,
        question: Optional[str] = None,
        flow_id: Optional[str] = None,
        extra_context: Optional[dict[str, Any]] = None,
    ):
        await memory.add_step(
            step_module="mock",
            step_title="mock step",
            step_agent_path="path",
            step_value="context retrieved",
            step_reason=None,
            timeit=0.01,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
            error=None,
        )
        await memory.add_answer(f"The answer to '{question}' is 42.", "mock", "path")


_mock_retrieval_agent = RetrievalAgent(context=[MockContextAgent()])
_mock_state = State(
    manager=None,  # type: ignore[arg-type]
    agent=_mock_retrieval_agent,
)


@pytest.fixture
def standalone_app(local_agents_config):
    """StandaloneApplication with get_state patched to avoid real NUA calls."""
    app = StandaloneApplication(local_agents_config, STANDALONE_SETTINGS)
    with patch(
        "hyperforge.server.session.get_state",
        return_value=_mock_state,
    ):
        yield app


@pytest.fixture
async def standalone_client(standalone_app: StandaloneApplication):
    """In-process ASGI client — no real network, no real port."""
    async with (
        standalone_app.router.lifespan_context(standalone_app),
        AsyncClient(
            transport=ASGITransport(app=standalone_app),
            base_url="http://test",
        ) as client,
    ):
        yield client


@pytest.fixture
async def standalone_http(standalone_app: StandaloneApplication):
    """Real uvicorn server on a free port — needed for WebSocket tests."""
    port = _free_port()
    config = uvicorn.Config(
        standalone_app, host="127.0.0.1", port=port, log_level="warning"
    )
    server = uvicorn.Server(config)
    if not config.loaded:
        config.load()
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=None)
    yield f"127.0.0.1:{port}"
    await server.shutdown()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_ndjson(raw: str) -> list[AragAnswer]:
    return [
        AragAnswer.model_validate_json(line)
        for line in raw.splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_standalone_http_post(standalone_client: AsyncClient):
    """POST /api/v1/agent/{agent_id}/session/ephemeral returns a valid ndjson stream."""
    async with standalone_client.stream(
        "POST",
        f"/api/v1/agent/{AGENT_ID}/session/ephemeral",
        json=InteractionRequest(question="What is the answer?").model_dump(),
        timeout=30,
    ) as response:
        assert response.status_code == 200
        assert "ndjson" in response.headers["content-type"]
        lines = [line async for line in response.aiter_lines() if line.strip()]

    messages = [AragAnswer.model_validate_json(line) for line in lines]
    operations = [m.operation for m in messages]

    assert AnswerOperation.START in operations
    assert AnswerOperation.DONE in operations
    assert AnswerOperation.ERROR not in operations

    # The final answer message should contain the mock response.
    answer_msgs = [m for m in messages if m.answer]
    assert answer_msgs, "Expected at least one message with an answer"
    assert answer_msgs[-1].answer is not None
    assert "42" in answer_msgs[-1].answer


async def test_standalone_http_post_unknown_agent(standalone_client: AsyncClient):
    """Asking a non-existent agent_id should return an error operation, not a 500."""
    async with standalone_client.stream(
        "POST",
        "/api/v1/agent/does-not-exist/session/ephemeral",
        json=InteractionRequest(question="hello").model_dump(),
        timeout=30,
    ) as response:
        assert response.status_code == 200
        lines = [line async for line in response.aiter_lines() if line.strip()]

    messages = [AragAnswer.model_validate_json(line) for line in lines]
    operations = [m.operation for m in messages]
    assert AnswerOperation.ERROR in operations


async def test_standalone_websocket(standalone_http: str):
    """WebSocket endpoint delivers START → answer chunks → DONE."""
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "local",
        "X-STF-ACCOUNT-TYPE": "v3starter",
    }

    received: list[AragAnswer] = []

    async with connect(
        f"ws://{standalone_http}/api/v1/agent/{AGENT_ID}/session/ephemeral/ws",
        additional_headers=headers,
    ) as ws:
        await ws.send(
            json.dumps(
                {
                    "question": "What is the answer?",
                    "operation": InteractionOperation.QUESTION,
                }
            )
        )

        async for raw in ws:
            msg = AragAnswer.model_validate_json(raw)
            received.append(msg)
            if msg.operation in (AnswerOperation.DONE, AnswerOperation.ERROR):
                break

    operations = [m.operation for m in received]
    assert AnswerOperation.START in operations
    assert AnswerOperation.DONE in operations
    assert AnswerOperation.ERROR not in operations

    answer_msgs = [m for m in received if m.answer]
    assert answer_msgs, "Expected at least one message with an answer"
    assert answer_msgs[-1].answer is not None
    assert "42" in answer_msgs[-1].answer


async def test_standalone_health(standalone_client: AsyncClient):
    resp = await standalone_client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# MCP tests — require a real HTTP server because the MCP client uses httpx
# ---------------------------------------------------------------------------


async def test_standalone_mcp_list_tools(standalone_http: str):
    """MCP list_tools returns the workflows defined in the agent config."""
    mcp_url = f"http://{standalone_http}/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"

    http_client = httpx.AsyncClient(timeout=30)
    async with streamable_http_client(mcp_url, http_client=http_client) as (
        read_stream,
        write_stream,
        _,
    ):
        assert read_stream is not None and write_stream is not None, (
            "Expected both read and write streams from streamable_http_client"
        )
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

    tool_names = [t.name for t in result.tools]
    assert "ask" in tool_names
    assert "Default" in tool_names

    ask_tool = next(t for t in result.tools if t.name == "ask")
    assert ask_tool.description == "Ask a question and get an answer."
    assert "question" in ask_tool.inputSchema.get("properties", {})


async def test_standalone_mcp_call_tool(standalone_http: str):
    """MCP call_tool invokes the mock agent and returns its answer."""
    mcp_url = f"http://{standalone_http}/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"

    http_client = httpx.AsyncClient(timeout=30)
    async with streamable_http_client(mcp_url, http_client=http_client) as (
        read_stream,
        write_stream,
        _,
    ):
        assert read_stream is not None and write_stream is not None, (
            "Expected both read and write streams from streamable_http_client"
        )
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(
                "ask", arguments={"question": "What is the answer?"}
            )

    assert not result.isError
    contents = result.content
    assert contents, "Expected at least one content item in tool result"
    # The mock agent answers with "The answer to '...' is 42."
    full_text = " ".join(
        c.text for c in contents if isinstance(c, TextContent) and c.text
    )
    assert "42" in full_text
