import asyncio
import base64
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from hyperforge_mcp.http import (
    _MCP_OAUTH_SINGLE_FLIGHT_LOCKS,
    MCPOAuthRoutingParams,
    _generate_pkce_parameters,
    _RoutedOAuthClientProvider,
    handle_callback,
)
from mcp.client.auth import PKCEParameters
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata
from pydantic import AnyUrl


def _make_provider(**kwargs) -> _RoutedOAuthClientProvider:
    redirect_handler = kwargs.pop("redirect_handler", AsyncMock())
    callback_handler = kwargs.pop("callback_handler_for_oauth_uuid", AsyncMock())
    provider = _RoutedOAuthClientProvider(
        server_url="https://auth.example.com",
        client_metadata=OAuthClientMetadata(
            client_name="client",
            redirect_uris=[AnyUrl("https://app.example.com/callback")],
            grant_types=["authorization_code"],
            response_types=["code"],
        ),
        storage=SimpleNamespace(),
        routing=MCPOAuthRoutingParams(
            account_id="account",
            agent_id="agent",
            workflow_id="workflow",
            session_id="session",
            question_id="question",
            oauth_uuid="base-oauth",
        ),
        redirect_handler=redirect_handler,
        callback_handler_for_oauth_uuid=callback_handler,
        **kwargs,
    )
    provider.context.client_info = OAuthClientInformationFull(
        client_id="client-id",
        redirect_uris=[AnyUrl("https://app.example.com/callback")],
    )
    return provider


def _make_pkce(verifier: str) -> PKCEParameters:
    digest = hashlib.sha256(verifier.encode()).digest()
    return PKCEParameters(
        code_verifier=verifier,
        code_challenge=base64.urlsafe_b64encode(digest).decode().rstrip("="),
    )


def test_generate_pkce_parameters_uses_salesforce_safe_verifier():
    pkce = _generate_pkce_parameters()

    assert len(pkce.code_verifier) == 128
    assert pkce.code_verifier.isalnum()
    assert pkce.code_challenge == _make_pkce(pkce.code_verifier).code_challenge


@pytest.mark.asyncio
async def test_handle_callback_uses_separate_oauth_uuid():
    memory = SimpleNamespace(
        original_question_uuid="question-id",
        oauth_callback_fn=object(),
        recv_oauth_callback=AsyncMock(return_value="code=auth-code&state=sdk-state"),
        get_session_id=lambda: "session-id",
    )

    code, state = await handle_callback(
        memory=memory,
        module="module",
        agent_id="agent-id",
        oauth_uuid="oauth-attempt-id",
    )

    assert code == "auth-code"
    assert state == "sdk-state"
    memory.recv_oauth_callback.assert_awaited_once_with(
        question_id="question-id",
        oauth_uuid="oauth-attempt-id",
    )


@pytest.mark.asyncio
async def test_oauth_uuid_is_unique_per_authorization_grant():
    sdk_states_by_oauth_uuid: dict[str, str] = {}
    callback_oauth_uuids: list[str] = []

    def encrypt_state(routing: MCPOAuthRoutingParams) -> str:
        sdk_states_by_oauth_uuid[routing.oauth_uuid] = routing.sdk_state
        return f"encrypted-{routing.oauth_uuid}"

    async def callback_handler(oauth_uuid: str) -> tuple[str, str | None]:
        callback_oauth_uuids.append(oauth_uuid)
        return "auth-code", sdk_states_by_oauth_uuid[oauth_uuid]

    provider = _make_provider(
        authorization_endpoint="https://auth.example.com/authorize",
        callback_handler_for_oauth_uuid=callback_handler,
    )

    with (
        patch(
            "hyperforge_mcp.http.uuid4",
            side_effect=[
                SimpleNamespace(hex="oauth-grant-1"),
                SimpleNamespace(hex="oauth-grant-2"),
            ],
        ),
        patch(
            "hyperforge_mcp.http.encrypt_mcp_oauth_state",
            side_effect=encrypt_state,
        ),
    ):
        await provider._perform_authorization_code_grant()
        await provider._perform_authorization_code_grant()

    assert callback_oauth_uuids == ["oauth-grant-1", "oauth-grant-2"]


@pytest.mark.asyncio
async def test_authorization_grant_keeps_matching_pkce_pair():
    sdk_states_by_oauth_uuid: dict[str, str] = {}

    def encrypt_state(routing: MCPOAuthRoutingParams) -> str:
        sdk_states_by_oauth_uuid[routing.oauth_uuid] = routing.sdk_state
        return f"encrypted-{routing.oauth_uuid}"

    async def callback_handler(oauth_uuid: str) -> tuple[str, str | None]:
        return f"auth-code-{oauth_uuid}", sdk_states_by_oauth_uuid[oauth_uuid]

    redirect_handler = AsyncMock()
    provider = _make_provider(
        authorization_endpoint="https://auth.example.com/authorize",
        token_endpoint="https://auth.example.com/token",
        redirect_handler=redirect_handler,
        callback_handler_for_oauth_uuid=callback_handler,
    )

    verifier = "A" * 128
    pkce = _make_pkce(verifier)
    with (
        patch(
            "hyperforge_mcp.http.uuid4",
            return_value=SimpleNamespace(hex="grant-id"),
        ),
        patch(
            "hyperforge_mcp.http._generate_pkce_parameters",
            return_value=pkce,
        ),
        patch(
            "hyperforge_mcp.http.encrypt_mcp_oauth_state",
            side_effect=encrypt_state,
        ),
    ):
        auth_code, code_verifier = await provider._perform_authorization_code_grant()

    redirect_url = redirect_handler.await_args.args[0]
    auth_params = parse_qs(urlparse(redirect_url).query)
    assert auth_params["code_challenge"] == [pkce.code_challenge]
    assert auth_params["code_challenge_method"] == ["S256"]
    assert auth_code == "auth-code-grant-id"
    assert code_verifier == verifier

    token_request = await provider._exchange_token_authorization_code(
        auth_code, code_verifier
    )
    token_params = parse_qs(token_request.content.decode())
    assert token_params["code"] == ["auth-code-grant-id"]
    assert token_params["code_verifier"] == [verifier]


@pytest.mark.asyncio
async def test_token_exchange_uses_fresh_token_data_per_grant():
    provider = _make_provider(
        token_endpoint="https://auth.example.com/token",
    )

    first_request = await provider._exchange_token_authorization_code(
        "auth-code-1", "verifier-1"
    )
    second_request = await provider._exchange_token_authorization_code(
        "auth-code-2", "verifier-2"
    )

    first_params = parse_qs(first_request.content.decode())
    second_params = parse_qs(second_request.content.decode())

    assert first_params["code"] == ["auth-code-1"]
    assert first_params["code_verifier"] == ["verifier-1"]
    assert second_params["code"] == ["auth-code-2"]
    assert second_params["code_verifier"] == ["verifier-2"]


@pytest.mark.asyncio
async def test_oauth_auth_flow_is_single_flight_per_question_and_uri():
    _MCP_OAUTH_SINGLE_FLIGHT_LOCKS.clear()
    started: list[int] = []
    release_first_flow = asyncio.Event()

    async def fake_auth_flow(self, request):
        started.append(id(self))
        auth_request = httpx.Request("GET", "https://auth.example.com/authorize")
        yield auth_request
        await release_first_flow.wait()

    async def drive_auth_flow(provider):
        flow = provider.async_auth_flow(httpx.Request("GET", "https://mcp.example.com"))
        auth_request = await flow.__anext__()
        with pytest.raises(StopAsyncIteration):
            await flow.asend(httpx.Response(200, request=auth_request))

    provider_1 = _make_provider(single_flight_key="same-question-uri")
    provider_2 = _make_provider(single_flight_key="same-question-uri")

    try:
        with patch(
            "mcp.client.auth.OAuthClientProvider.async_auth_flow",
            fake_auth_flow,
        ):
            task_1 = asyncio.create_task(drive_auth_flow(provider_1))
            while len(started) < 1:
                await asyncio.sleep(0)

            task_2 = asyncio.create_task(drive_auth_flow(provider_2))
            await asyncio.sleep(0)
            assert len(started) == 1

            release_first_flow.set()
            await asyncio.gather(task_1, task_2)
    finally:
        _MCP_OAUTH_SINGLE_FLIGHT_LOCKS.clear()

    assert len(started) == 2
