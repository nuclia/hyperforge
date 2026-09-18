import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.types import ElicitResult
from starlette.datastructures import Headers

from hyperforge.api.v1 import mcp_interaction
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
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
    app = SimpleNamespace(agent_manager=agent_manager)
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
