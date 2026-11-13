import asyncio
import json
import socket
from base64 import b64encode
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional
from unittest.mock import patch

import httpx
import pytest
import uvicorn
from anyio import fail_after
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import ElicitResult, TextContent
from pydantic import SecretStr
from websockets.asyncio.client import connect

from hyperforge.api.mcp_transport import MCPTransport
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
    Feedback,
    OAuthAuthenticateURL,
)
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.config import StandaloneConfig
from hyperforge.standalone.settings import StandaloneSettings
from hyperforge.trace import trace_agent

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


async def test_standalone_ui_is_disabled_without_password(
    standalone_client: AsyncClient,
):
    response = await standalone_client.get("/api/v1/ui/config")
    assert response.status_code == 404


async def test_standalone_ui_requires_admin_credentials(local_agents_config):
    settings = STANDALONE_SETTINGS.model_copy(
        update={"ui_admin_password": SecretStr("secret-password")}
    )
    app = StandaloneApplication(local_agents_config, settings)
    with patch("hyperforge.server.session.get_state", return_value=_mock_state):
        async with (
            app.router.lifespan_context(app),
            AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client,
        ):
            response = await client.get("/api/v1/ui/config")
            assert response.status_code == 401
            assert response.headers["www-authenticate"].startswith("Basic ")

            response = await client.get(
                "/api/v1/ui/config",
                headers={
                    "Authorization": "Basic "
                    + b64encode(b"admin:secret-password").decode()
                },
            )
            assert response.status_code == 200


async def test_standalone_security_defaults():
    settings = StandaloneSettings()
    assert settings.host == "0.0.0.0"
    assert settings.cors_allow_origin == []


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
    async with (
        asyncio.timeout(20),
        streamable_http_client(mcp_url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ),
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
    async with (
        asyncio.timeout(20),
        streamable_http_client(mcp_url, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ),
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


async def test_standalone_mcp_unknown_tool_returns_protocol_error(
    standalone_http: str,
):
    mcp_url = f"http://{standalone_http}/api/v1/agent/{AGENT_ID}/session/unknown/mcp"

    async with (
        streamable_http_client(mcp_url) as (
            read_stream,
            write_stream,
            _,
        ),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        result = await session.call_tool("missing", {})

    assert result.isError is True


async def test_standalone_mcp_rejects_oversized_request(
    standalone_client: AsyncClient,
):
    mcp_url = f"/api/v1/agent/{AGENT_ID}/session/oversized/mcp"
    response = await standalone_client.post(
        mcp_url,
        content=b"x" * (STANDALONE_SETTINGS.mcp_max_request_bytes + 1),
    )

    assert response.status_code == 413


async def test_standalone_mcp_delete_requires_session_id(
    standalone_client: AsyncClient,
):
    mcp_url = f"/api/v1/agent/{AGENT_ID}/session/delete/mcp"

    first = await standalone_client.delete(mcp_url)
    second = await standalone_client.delete(mcp_url)

    assert first.status_code == 400
    assert second.status_code == 400
    assert first.json()["detail"] == "Missing Mcp-Session-Id"


async def initialize_mcp(client: AsyncClient, mcp_url: str) -> str:
    response = await client.post(
        mcp_url,
        headers={"Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200, response.text
    session_id = response.headers["mcp-session-id"]
    initialized = await client.post(
        mcp_url,
        headers=mcp_headers(session_id),
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert initialized.status_code == 202, initialized.text
    return session_id


def mcp_headers(session_id: str) -> dict[str, str]:
    return {
        "Accept": "application/json, text/event-stream",
        "Mcp-Session-Id": session_id,
        "MCP-Protocol-Version": "2025-03-26",
    }


async def mcp_tools(client: AsyncClient, url: str, session_id: str) -> httpx.Response:
    return await client.post(
        url,
        headers=mcp_headers(session_id),
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )


async def test_mcp_concurrent_initializations_share_a_manager(
    standalone_client: AsyncClient, standalone_app: StandaloneApplication
):
    url = f"/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    first, second = await asyncio.gather(
        initialize_mcp(standalone_client, url),
        initialize_mcp(standalone_client, url),
    )
    assert first != second
    assert len(standalone_app.mcp_server_pool._servers) == 1
    managed = next(iter(standalone_app.mcp_server_pool._servers.values()))
    assert standalone_app.mcp_sessions is None
    assert isinstance(managed.manager, StreamableHTTPSessionManager)
    for session_id in (first, second):
        response = await mcp_tools(standalone_client, url, session_id)
        assert response.status_code == 200, response.text
        message = json.loads(
            next(
                line[6:]
                for line in response.text.splitlines()
                if line.startswith("data: ")
            )
        )
        assert "ask" in [tool["name"] for tool in message["result"]["tools"]]


async def test_standalone_mcp_uses_shared_transport_only_with_redis(
    standalone_app: StandaloneApplication, valkey_url, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        standalone_app._standalone_settings, "broker_redis_dsn", valkey_url
    )
    async with (
        standalone_app.router.lifespan_context(standalone_app),
        AsyncClient(
            transport=ASGITransport(app=standalone_app), base_url="http://test"
        ) as client,
    ):
        url = f"/api/v1/agent/{AGENT_ID}/session/redis/mcp"
        session_id = await initialize_mcp(client, url)
        managed = next(iter(standalone_app.mcp_server_pool._servers.values()))
        assert isinstance(managed.manager, MCPTransport)
        assert managed.manager.store is standalone_app.mcp_sessions
        response = await mcp_tools(client, url, session_id)
        assert response.headers["content-type"] == "application/json"
        assert (
            await client.delete(url, headers=mcp_headers(session_id))
        ).status_code == 200
        assert await managed.session_count() == 0


async def test_mcp_delete_only_terminates_the_requested_session(
    standalone_client: AsyncClient,
):
    url = f"/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    first = await initialize_mcp(standalone_client, url)
    second = await initialize_mcp(standalone_client, url)
    deleted = await standalone_client.delete(url, headers=mcp_headers(first))
    assert deleted.status_code == 200
    assert (await mcp_tools(standalone_client, url, first)).status_code == 404
    assert (await mcp_tools(standalone_client, url, second)).status_code == 200
    stale_delete = await standalone_client.delete(url, headers=mcp_headers(first))
    assert stale_delete.status_code == 404
    assert (await mcp_tools(standalone_client, url, second)).status_code == 200


async def test_mcp_session_capacity_is_reclaimed_after_delete(
    standalone_client: AsyncClient,
    standalone_app: StandaloneApplication,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        standalone_app._standalone_settings, "mcp_max_sessions_per_server", 2
    )
    url = f"/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    first = await initialize_mcp(standalone_client, url)
    second = await initialize_mcp(standalone_client, url)
    full = await standalone_client.post(
        url,
        headers={"Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert full.status_code == 503
    assert full.json()["detail"] == "MCP session capacity is exhausted"
    assert (await mcp_tools(standalone_client, url, second)).status_code == 200
    deleted = await standalone_client.delete(url, headers=mcp_headers(first))
    assert deleted.status_code == 200
    third = await initialize_mcp(standalone_client, url)
    assert third not in (first, second)
    assert (await mcp_tools(standalone_client, url, second)).status_code == 200


async def test_mcp_idle_sessions_expire_independently(
    standalone_client: AsyncClient,
    standalone_app: StandaloneApplication,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        standalone_app._standalone_settings, "mcp_session_idle_ttl_seconds", 1
    )
    url = f"/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    first = await initialize_mcp(standalone_client, url)
    second = await initialize_mcp(standalone_client, url)
    for _ in range(6):
        await asyncio.sleep(0.2)
        assert (await mcp_tools(standalone_client, url, second)).status_code == 200
    assert (await mcp_tools(standalone_client, url, first)).status_code == 404
    assert (await mcp_tools(standalone_client, url, second)).status_code == 200
    managed = next(iter(standalone_app.mcp_server_pool._servers.values()))
    assert await managed.session_count() == 1


async def test_mcp_pool_reclaims_capacity_after_last_connection_closes(
    standalone_client: AsyncClient,
    standalone_app: StandaloneApplication,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(standalone_app.mcp_server_pool, "_max_servers", 1)
    first_url = f"/api/v1/agent/{AGENT_ID}/session/first/mcp"
    first = await initialize_mcp(standalone_client, first_url)
    deleted = await standalone_client.delete(first_url, headers=mcp_headers(first))
    assert deleted.status_code == 200
    second_url = f"/api/v1/agent/{AGENT_ID}/session/second/mcp"
    second = await initialize_mcp(standalone_client, second_url)
    assert second != first
    assert (await mcp_tools(standalone_client, second_url, second)).status_code == 200
    assert len(standalone_app.mcp_server_pool._servers) == 1


async def test_mcp_connection_churn_does_not_retain_deleted_transports(
    standalone_client: AsyncClient, standalone_app: StandaloneApplication
):
    url = f"/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    survivor = await initialize_mcp(standalone_client, url)
    managed = next(iter(standalone_app.mcp_server_pool._servers.values()))
    for _ in range(10):
        session_id = await initialize_mcp(standalone_client, url)
        assert (
            await standalone_client.delete(url, headers=mcp_headers(session_id))
        ).status_code == 200
        assert await managed.session_count() == 1
    assert (await mcp_tools(standalone_client, url, survivor)).status_code == 200


async def test_mcp_get_requires_session_id_without_allocating_a_transport(
    standalone_client: AsyncClient, standalone_app: StandaloneApplication
):
    url = f"/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    await initialize_mcp(standalone_client, url)
    response = await standalone_client.get(url, headers={"Accept": "text/event-stream"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Missing Mcp-Session-Id"
    managed = next(iter(standalone_app.mcp_server_pool._servers.values()))
    assert await managed.session_count() == 1


async def test_mcp_sdk_clients_can_share_url_and_close_independently(
    standalone_http: str, standalone_app: StandaloneApplication
):
    url = f"http://{standalone_http}/api/v1/agent/{AGENT_ID}/session/ephemeral/mcp"
    with fail_after(10):
        async with (
            streamable_http_client(url) as (first_read, first_write, first_id),
            ClientSession(first_read, first_write) as first,
        ):
            await first.initialize()
            managed = next(iter(standalone_app.mcp_server_pool._servers.values()))
            async with (
                streamable_http_client(url) as (second_read, second_write, second_id),
                ClientSession(second_read, second_write) as second,
            ):
                await second.initialize()
                assert first_id() != second_id()
                for session in (first, second):
                    tools = await session.list_tools()
                    assert "ask" in [tool.name for tool in tools.tools]
                    result = await session.call_tool("ask", {"question": "Hello"})
                    assert not result.isError
            tools = await first.list_tools()
            assert "ask" in [tool.name for tool in tools.tools]
            assert await managed.session_count() == 1


async def test_standalone_mcp_sessions_are_independent(standalone_http: str):
    async def list_tool_names(session_name: str) -> list[str]:
        mcp_url = (
            f"http://{standalone_http}/api/v1/agent/{AGENT_ID}"
            f"/session/{session_name}/mcp"
        )
        async with (
            streamable_http_client(mcp_url) as (
                read_stream,
                write_stream,
                _,
            ),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.list_tools()
        return [tool.name for tool in result.tools]

    first_tools = await list_tool_names("first")
    second_tools = await list_tool_names("second")

    assert first_tools == second_tools
    assert "ask" in first_tools


async def test_standalone_mcp_url_elicitation(
    standalone_http: str, monkeypatch: pytest.MonkeyPatch
):
    from hyperforge.api.v1 import mcp_interaction

    async def oauth_response(*args, **kwargs):
        yield AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            oauth=OAuthAuthenticateURL(oauth_url="https://example.com/authorize"),
        )
        yield AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=Feedback(
                request_id="workflow-question",
                question="Choose",
                module="test",
                agent_id=AGENT_ID,
                data=None,
                response_schema={
                    "type": "object",
                    "properties": {"choice": {"type": "string"}},
                    "required": ["choice"],
                },
            ),
        )
        feedback = await args[1].queue.get()
        assert feedback.request_id == "workflow-question"
        assert json.loads(feedback.response) == {"choice": "accepted"}
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(mcp_interaction, "stream_response", oauth_response)
    elicitation_requests = []

    async def handle_elicitation(context, params):
        elicitation_requests.append(params)
        return ElicitResult(
            action="accept",
            content={"choice": "accepted"} if params.mode == "form" else None,
        )

    mcp_url = (
        f"http://{standalone_http}/api/v1/agent/{AGENT_ID}/session/elicitation/mcp"
    )
    with fail_after(3):
        async with streamable_http_client(mcp_url) as (read_stream, write_stream, _):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=timedelta(seconds=2),
                elicitation_callback=handle_elicitation,
            ) as session:
                await session.initialize()
                result = await session.call_tool("ask", {"question": "Authenticate"})

    assert result.isError is False
    assert [request.mode for request in elicitation_requests] == ["url", "form"]
