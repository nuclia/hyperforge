import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.types import ElicitResult
from starlette.datastructures import Headers
from starlette.requests import Request

from hyperforge.api.mcp_server_pool import MCPRequestLease
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
        upsert_sync_oauth_credentials_batch=AsyncMock(),
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
    agent_manager.upsert_sync_oauth_credentials_batch.assert_awaited_once_with(
        account="account",
        user_id="user",
        agent_id="agent",
        credentials_by_config={
            "sync-config": (
                "sharefile_oauth",
                {"external-connection": "secret"},
            )
        },
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
async def test_mcp_rejects_partial_received_credentials(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                get_credentials={
                    "first": Provider.SHAREFILE_OAUTH,
                    "second": Provider.SHAREFILE_OAUTH,
                }
            ),
        ),
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(credentials={"first": {"external-connection": "secret"}}),
        ),
    ]
    app, server, workflow, headers, _, agent_manager = setup_call(monkeypatch, events)

    with pytest.raises(ResourceError, match="do not match"):
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

    agent_manager.upsert_sync_oauth_credentials_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_validates_all_received_credentials_before_persisting(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                get_credentials={
                    "first": Provider.SHAREFILE_OAUTH,
                    "second": Provider.SHAREFILE_OAUTH,
                }
            ),
        ),
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                credentials={
                    "first": {"external-connection": "secret"},
                    "second": {"external-connection": 42},
                }
            ),
        ),
    ]
    app, server, workflow, headers, _, agent_manager = setup_call(monkeypatch, events)

    with pytest.raises(ResourceError, match="invalid Sync OAuth credentials"):
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

    agent_manager.upsert_sync_oauth_credentials_batch.assert_not_awaited()


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
async def test_mcp_transport_releases_lease_when_handling_fails():
    managed_server = SimpleNamespace(
        manager=SimpleNamespace(
            handle_request=AsyncMock(side_effect=RuntimeError("transport failed"))
        ),
        active_requests=1,
    )

    def release_request():
        managed_server.active_requests -= 1

    managed_server.release_request = release_request
    response = mcp_interaction._MCPTransportResponse(
        Request({"type": "http", "method": "POST", "headers": []}),
        MCPRequestLease(managed_server),
        b"",
        1024,
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        await response(
            {"type": "http", "method": "POST"},
            AsyncMock(),
            AsyncMock(),
        )

    assert managed_server.active_requests == 0


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

    agent_manager.upsert_sync_oauth_credentials_batch.assert_not_awaited()
