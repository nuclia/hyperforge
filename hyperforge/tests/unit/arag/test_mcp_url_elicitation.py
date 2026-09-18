import asyncio
import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import anyio
import pytest
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import ElicitResult
from starlette.datastructures import Headers
from starlette.requests import Request

from hyperforge.api.v1 import mcp_interaction
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
    ARAGException,
    Feedback,
    OAuthAuthenticateURL,
    OAuthFeedbackReturnSchema,
    Provider,
)


def feedback(**kwargs) -> Feedback:
    return Feedback(
        request_id="request",
        question="credentials",
        module="oauth",
        agent_id="agent",
        data=None,
        response_schema=OAuthFeedbackReturnSchema.model_json_schema(),
        **kwargs,
    )


def setup_call(monkeypatch, events):
    async def stream_response(*args, **kwargs):
        for event in events:
            yield event

    monkeypatch.setattr(mcp_interaction, "stream_response", stream_response)
    monkeypatch.setattr(
        mcp_interaction,
        "WebsocketReceiver",
        lambda websocket: SimpleNamespace(queue=asyncio.Queue()),
    )
    agent_manager = SimpleNamespace(
        get_sync_oauth_credentials=AsyncMock(return_value=None),
        upsert_sync_oauth_credentials=AsyncMock(),
    )
    app = SimpleNamespace(agent_manager=agent_manager, settings=SimpleNamespace())
    session = SimpleNamespace(
        elicit_url=AsyncMock(return_value=ElicitResult(action="accept")),
        elicit_form=AsyncMock(),
    )
    server = SimpleNamespace(request_context=SimpleNamespace(session=session))
    workflow = SimpleNamespace(
        name="ask", description="Ask", required=[], id="workflow"
    )
    headers = Headers(
        {
            "x-stf-user": "user",
            "x-stf-account": "account",
            "x-stf-account-type": "basic",
        }
    )
    return app, server, workflow, headers, session, agent_manager


@pytest.mark.asyncio
async def test_mcp_propagates_agent_error(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.ERROR,
            exception=ARAGException(detail="Unable to start agent"),
        )
    ]
    app, server, workflow, headers, _, _ = setup_call(monkeypatch, events)

    with pytest.raises(ResourceError, match="Unable to start agent"):
        await mcp_interaction.call_tool(
            app,
            server,
            "account",
            "agent",
            "session",
            [workflow],
            headers,
            "ask",
            {},
        )


@pytest.mark.asyncio
async def test_mcp_elicits_url_and_persists_sharefile_credentials(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                get_credentials={"sync-config": Provider.SHAREFILE_OAUTH}
            ),
        ),
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            oauth=OAuthAuthenticateURL(oauth_url="https://sharefile.example/authorize"),
        ),
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                credentials={"sync-config": {"external-connection": "secret"}}
            ),
        ),
    ]
    app, server, workflow, headers, session, agent_manager = setup_call(
        monkeypatch, events
    )

    await mcp_interaction.call_tool(
        app,
        server,
        "account",
        "agent",
        "session",
        [workflow],
        headers,
        "ask",
        {},
    )

    session.elicit_url.assert_awaited_once()
    assert session.elicit_url.await_args.kwargs["url"] == (
        "https://sharefile.example/authorize"
    )
    agent_manager.upsert_sync_oauth_credentials.assert_awaited_once_with(
        account="account",
        user_id="user",
        agent_id="agent",
        provider="sharefile_oauth",
        sync_config_id="sync-config",
        credentials={"external-connection": "secret"},
    )


@pytest.mark.asyncio
async def test_mcp_returns_stored_credentials_without_elicitation(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                get_credentials={"sync-config": Provider.SHAREFILE_OAUTH}
            ),
        )
    ]
    app, server, workflow, headers, session, agent_manager = setup_call(
        monkeypatch, events
    )
    agent_manager.get_sync_oauth_credentials.return_value = {
        "external-connection": "secret"
    }

    await mcp_interaction.call_tool(
        app,
        server,
        "account",
        "agent",
        "session",
        [workflow],
        headers,
        "ask",
        {},
    )

    session.elicit_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_discards_partial_stored_credentials(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                get_credentials={
                    "stored": Provider.SHAREFILE_OAUTH,
                    "missing": Provider.SHAREFILE_OAUTH,
                }
            ),
        )
    ]
    app, server, workflow, headers, _, agent_manager = setup_call(monkeypatch, events)
    agent_manager.get_sync_oauth_credentials.side_effect = [
        {"external-connection": "secret"},
        None,
    ]
    websocket = SimpleNamespace(queue=asyncio.Queue())
    monkeypatch.setattr(
        mcp_interaction, "WebsocketReceiver", lambda websocket: websocket_receiver
    )
    websocket_receiver = websocket

    await mcp_interaction.call_tool(
        app,
        server,
        "account",
        "agent",
        "session",
        [workflow],
        headers,
        "ask",
        {},
    )

    response = await websocket.queue.get()
    assert json.loads(response.response) == {"existing_credentials": {}}


@pytest.mark.asyncio
async def test_mcp_production_uses_current_request_headers(monkeypatch):
    app, server, workflow, headers, _, _ = setup_call(monkeypatch, [])
    api_settings_type = type("ApiSettings", (), {})
    monkeypatch.setattr(mcp_interaction, "ApiSettings", api_settings_type)
    app.settings = api_settings_type()
    server.request_context.request = Request(
        {
            "type": "http",
            "headers": [
                (b"authorization", b"Bearer refreshed"),
                (b"x-stf-user", b"user"),
                (b"x-stf-account", b"account"),
                (b"x-stf-account-type", b"basic"),
            ],
        }
    )
    captured_headers = None

    async def stream_response(*args, **kwargs):
        nonlocal captured_headers
        captured_headers = kwargs["interaction"].headers
        if False:
            yield

    monkeypatch.setattr(mcp_interaction, "stream_response", stream_response)

    await mcp_interaction.call_tool(
        app,
        server,
        "account",
        "agent",
        "session",
        [workflow],
        headers,
        "ask",
        {},
    )

    assert captured_headers["authorization"] == "Bearer refreshed"


@pytest.mark.asyncio
async def test_mcp_server_limit_evicts_and_awaits_oldest_manager():
    oldest_task = asyncio.create_task(asyncio.Event().wait())
    oldest = SimpleNamespace(task=oldest_task, close=oldest_task.cancel)
    newest = SimpleNamespace(task=None, close=AsyncMock())
    app = SimpleNamespace(mcp_servers={"oldest": oldest, "newest": newest})

    await mcp_interaction._evict_mcp_servers(app, max_servers=2)

    assert oldest_task.done()
    assert app.mcp_servers == {"newest": newest}
    newest.close.assert_not_called()


def test_mcp_manager_accepts_only_one_request_without_session_id():
    manager = cast(StreamableHTTPSessionManager, SimpleNamespace())
    managed_server = mcp_interaction._ManagedMCPServer(manager)
    initial_request = Request({"type": "http", "method": "POST", "headers": []})
    established_request = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"mcp-session-id", b"session-id")],
        }
    )
    sessionless_get = Request({"type": "http", "method": "GET", "headers": []})

    assert managed_server.accept_request(initial_request) is True
    assert managed_server.accept_request(initial_request) is False
    assert managed_server.accept_request(sessionless_get) is False
    assert managed_server.accept_request(established_request) is True


def test_mcp_manager_identifies_sessionless_reinitialization():
    manager = cast(StreamableHTTPSessionManager, SimpleNamespace())
    managed_server = mcp_interaction._ManagedMCPServer(manager)
    initial_request = Request({"type": "http", "method": "POST", "headers": []})
    initialize_body = b'{"jsonrpc":"2.0","method":"initialize","id":1}'
    established_request = Request(
        {
            "type": "http",
            "method": "POST",
            "headers": [(b"mcp-session-id", b"session-id")],
        }
    )

    assert managed_server.is_reinitialization(initial_request, initialize_body) is False
    assert managed_server.accept_request(initial_request) is True
    assert managed_server.is_reinitialization(initial_request, initialize_body) is True
    assert managed_server.is_reinitialization(initial_request, b"not json") is False
    assert (
        managed_server.is_reinitialization(
            initial_request, b'{"jsonrpc":"2.0","method":"tools/list","id":2}'
        )
        is False
    )
    assert (
        managed_server.is_reinitialization(established_request, initialize_body)
        is False
    )


@pytest.mark.asyncio
async def test_mcp_delete_waits_for_concurrent_server_creation():
    key = ("account", "user", "type", "agent", "session")
    server_task = asyncio.create_task(asyncio.Event().wait())
    managed_server = SimpleNamespace(task=server_task, close=server_task.cancel)
    app = SimpleNamespace(mcp_servers={}, mcp_server_lock=anyio.Lock())

    async with app.mcp_server_lock:
        delete_task = asyncio.create_task(mcp_interaction._remove_mcp_server(app, key))
        await asyncio.sleep(0)
        assert not delete_task.done()
        app.mcp_servers[key] = managed_server

    await delete_task

    assert key not in app.mcp_servers
    assert server_task.done()


@pytest.mark.asyncio
async def test_mcp_rejects_provider_not_in_allowlist(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(get_credentials={"server": Provider.MCP_OAUTH}),
        )
    ]
    app, server, workflow, headers, session, agent_manager = setup_call(
        monkeypatch, events
    )

    with pytest.raises(ResourceError, match="does not support"):
        await mcp_interaction.call_tool(
            app,
            server,
            "account",
            "agent",
            "session",
            [workflow],
            headers,
            "ask",
            {},
        )

    agent_manager.get_sync_oauth_credentials.assert_not_awaited()
    session.elicit_url.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_mcp_stops_when_url_elicitation_is_rejected(monkeypatch, action):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            oauth=OAuthAuthenticateURL(oauth_url="https://sharefile.example/authorize"),
        )
    ]
    app, server, workflow, headers, session, agent_manager = setup_call(
        monkeypatch, events
    )
    session.elicit_url.return_value = ElicitResult(action=action)

    with pytest.raises(ResourceError, match=f"authentication was {action}"):
        await mcp_interaction.call_tool(
            app,
            server,
            "account",
            "agent",
            "session",
            [workflow],
            headers,
            "ask",
            {},
        )

    agent_manager.upsert_sync_oauth_credentials.assert_not_awaited()
