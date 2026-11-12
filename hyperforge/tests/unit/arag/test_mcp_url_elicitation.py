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

OAUTH_PROVIDERS = (
    Provider.SHAREFILE_OAUTH,
    Provider.GOOGLE_OAUTH,
    Provider.AZURE_OAUTH,
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
    server = SimpleNamespace(
        request_context=SimpleNamespace(
            request_id="client-request",
            session=session,
            request=Request(
                {
                    "type": "http",
                    "headers": headers.raw,
                }
            ),
        )
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
            "ask",
            {},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
async def test_mcp_elicits_url_and_persists_credentials(monkeypatch, provider):
    oauth_url = f"https://{provider.value}.example/authorize"
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(get_credentials={"sync-config": provider}),
        ),
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            oauth=OAuthAuthenticateURL(oauth_url=oauth_url),
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
        "ask",
        {},
    )

    session.elicit_url.assert_awaited_once()
    assert session.elicit_url.await_args.kwargs["url"] == oauth_url
    assert session.elicit_url.await_args.kwargs["message"] == (
        "Authenticate with the data source to access the requested content."
    )
    agent_manager.upsert_sync_oauth_credentials_batch.assert_awaited_once_with(
        account="account",
        user_id="user",
        agent_id="agent",
        credentials_by_config={
            "sync-config": (
                provider.value,
                {"external-connection": "secret"},
            )
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["accept", "decline", "cancel"])
async def test_mcp_form_elicitation_only_enqueues_accepted_content(monkeypatch, action):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(),
        )
    ]
    app, server, workflow, _, session, _ = setup_call(monkeypatch, events)
    receiver = SimpleNamespace(queue=asyncio.Queue())
    monkeypatch.setattr(
        mcp_interaction, "WebsocketReceiver", lambda websocket: receiver
    )
    content = {"choice": "confirmed"} if action == "accept" else None
    session.elicit_form.return_value = ElicitResult(action=action, content=content)

    if action == "accept":
        await mcp_interaction.call_tool(
            app, server, "account", "agent", "session", [workflow], "ask", {}
        )
        response = receiver.queue.get_nowait()
        assert response.request_id == "request"
        assert json.loads(response.response) == content
    else:
        with pytest.raises(ResourceError, match=f"Form elicitation was {action}"):
            await mcp_interaction.call_tool(
                app, server, "account", "agent", "session", [workflow], "ask", {}
            )
    assert receiver.queue.empty()
    session.elicit_form.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
async def test_mcp_returns_stored_credentials_without_elicitation(
    monkeypatch, provider
):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(get_credentials={"sync-config": provider}),
        )
    ]
    app, server, workflow, headers, session, agent_manager = setup_call(
        monkeypatch, events
    )
    agent_manager.get_sync_oauth_credentials.return_value = {
        "external-connection": "secret"
    }
    receiver = SimpleNamespace(queue=asyncio.Queue())
    monkeypatch.setattr(
        mcp_interaction, "WebsocketReceiver", lambda websocket: receiver
    )

    await mcp_interaction.call_tool(
        app,
        server,
        "account",
        "agent",
        "session",
        [workflow],
        "ask",
        {},
    )

    session.elicit_url.assert_not_awaited()
    session.elicit_form.assert_not_awaited()
    agent_manager.get_sync_oauth_credentials.assert_awaited_once_with(
        account="account",
        user_id="user",
        agent_id="agent",
        provider=provider.value,
        sync_config_id="sync-config",
    )
    response = receiver.queue.get_nowait()
    assert response.request_id == "request"
    assert json.loads(response.response) == {
        "existing_credentials": {"sync-config": {"external-connection": "secret"}}
    }
    agent_manager.upsert_sync_oauth_credentials_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_authenticates_mixed_providers(monkeypatch):
    providers = {provider.value: provider for provider in OAUTH_PROVIDERS}
    credentials = {
        config: {f"{config}-connection": f"{config}-secret"} for config in providers
    }
    oauth_urls = [f"https://{config}.example/authorize" for config in providers]
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(get_credentials=providers),
        ),
        *[
            AragAnswer(
                operation=AnswerOperation.AGENT_REQUEST,
                oauth=OAuthAuthenticateURL(oauth_url=url),
            )
            for url in oauth_urls
        ],
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(credentials=credentials),
        ),
    ]
    app, server, workflow, _, session, agent_manager = setup_call(monkeypatch, events)
    receiver = SimpleNamespace(queue=asyncio.Queue())
    monkeypatch.setattr(
        mcp_interaction, "WebsocketReceiver", lambda websocket: receiver
    )

    await mcp_interaction.call_tool(
        app, server, "account", "agent", "session", [workflow], "ask", {}
    )

    assert [
        call.kwargs["url"] for call in session.elicit_url.await_args_list
    ] == oauth_urls
    assert [
        call.kwargs["provider"]
        for call in agent_manager.get_sync_oauth_credentials.await_args_list
    ] == list(providers)
    agent_manager.upsert_sync_oauth_credentials_batch.assert_awaited_once_with(
        account="account",
        user_id="user",
        agent_id="agent",
        credentials_by_config={
            config: (provider.value, credentials[config])
            for config, provider in providers.items()
        },
    )
    assert json.loads(receiver.queue.get_nowait().response) == {
        "existing_credentials": {}
    }
    assert json.loads(receiver.queue.get_nowait().response) == {
        "existing_credentials": credentials
    }
    assert receiver.queue.empty()
    session.elicit_form.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_discards_partial_stored_credentials(monkeypatch):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(
                get_credentials={
                    "stored": Provider.SHAREFILE_OAUTH,
                    "missing": Provider.GOOGLE_OAUTH,
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
            "ask",
            {},
        )

    agent_manager.upsert_sync_oauth_credentials_batch.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_standalone_uses_current_request_headers(monkeypatch):
    app, server, workflow, headers, _, _ = setup_call(monkeypatch, [])
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
        "ask",
        {},
    )

    assert captured_headers["authorization"] == "Bearer refreshed"


@pytest.mark.asyncio
async def test_mcp_standalone_forwards_refreshed_token_in_same_session(monkeypatch):
    app, server, workflow, _, _, _ = setup_call(monkeypatch, [])
    captured_authorization = []

    async def stream_response(*args, **kwargs):
        captured_authorization.append(
            kwargs["interaction"].headers.get("authorization")
        )
        if False:
            yield

    monkeypatch.setattr(mcp_interaction, "stream_response", stream_response)

    for token in ("initial-token", "refreshed-token"):
        server.request_context.request = Request(
            {
                "type": "http",
                "headers": [
                    (b"authorization", f"Bearer {token}".encode()),
                    (b"x-stf-user", b"user"),
                    (b"x-stf-account", b"account"),
                    (b"x-stf-account-type", b"basic"),
                ],
            }
        )
        await mcp_interaction.call_tool(
            app,
            server,
            "account",
            "agent",
            "session",
            [workflow],
            "ask",
            {},
        )

    assert captured_authorization == [
        "Bearer initial-token",
        "Bearer refreshed-token",
    ]


@pytest.mark.asyncio
async def test_mcp_transport_releases_lease_when_handling_fails():
    managed_server = SimpleNamespace(
        handle_request=AsyncMock(side_effect=RuntimeError("transport failed")),
        active_requests=1,
    )

    def release_request():
        managed_server.active_requests -= 1

    managed_server.release_request = release_request
    response = mcp_interaction._MCPTransportResponse(
        MCPRequestLease(managed_server),
        b"",
    )

    with pytest.raises(RuntimeError, match="transport failed"):
        await response(
            {"type": "http", "method": "POST"},
            AsyncMock(),
            AsyncMock(),
        )

    assert managed_server.active_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider",
    (
        Provider.MCP_OAUTH,
        Provider.AZURE_CERTIFICATE_CREDENTIALS,
        Provider.AWS_S3_ACCESS_KEYS,
    ),
)
async def test_mcp_rejects_provider_not_in_allowlist(monkeypatch, provider):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(get_credentials={"server": provider}),
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
            "ask",
            {},
        )

    agent_manager.get_sync_oauth_credentials.assert_not_awaited()
    session.elicit_url.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", OAUTH_PROVIDERS)
@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_mcp_stops_when_url_elicitation_is_rejected(
    monkeypatch, action, provider
):
    events = [
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            feedback=feedback(get_credentials={"sync-config": provider}),
        ),
        AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST,
            oauth=OAuthAuthenticateURL(
                oauth_url=f"https://{provider.value}.example/authorize"
            ),
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
    session.elicit_url.return_value = ElicitResult(action=action)

    with pytest.raises(
        ResourceError, match=f"^Data source authentication was {action} by the user$"
    ):
        await mcp_interaction.call_tool(
            app,
            server,
            "account",
            "agent",
            "session",
            [workflow],
            "ask",
            {},
        )

    agent_manager.upsert_sync_oauth_credentials_batch.assert_not_awaited()
