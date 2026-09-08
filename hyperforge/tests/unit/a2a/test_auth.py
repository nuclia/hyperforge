from unittest.mock import AsyncMock

import grpc
import httpx
import pytest
from a2a.types import a2a_pb2, a2a_pb2_grpc

from hyperforge.a2a.auth import (
    A2AAuthInterceptor,
    A2AAuthorizerClient,
    AuthorizationError,
    AuthorizationFailure,
    _bearer_from_metadata,
)


class _A2AService(a2a_pb2_grpc.A2AServiceServicer):
    async def GetExtendedAgentCard(self, request, context):
        return a2a_pb2.AgentCard(name="public")

    async def SendMessage(self, request, context):
        return a2a_pb2.SendMessageResponse()


def test_bearer_metadata_requires_exactly_one_non_empty_token():
    assert _bearer_from_metadata((("authorization", "Bearer token"),)) == (
        "Bearer token"
    )

    for metadata in (
        (),
        (("authorization", "Basic token"),),
        (("authorization", "Bearer "),),
        (("authorization", "Bearer one"), ("authorization", "Bearer two")),
    ):
        with pytest.raises(AuthorizationError) as exc:
            _bearer_from_metadata(metadata)
        assert exc.value.failure is AuthorizationFailure.UNAUTHENTICATED


@pytest.mark.parametrize(
    ("status_code", "failure"),
    [
        (401, AuthorizationFailure.UNAUTHENTICATED),
        (403, AuthorizationFailure.FORBIDDEN),
        (500, AuthorizationFailure.UNAVAILABLE),
    ],
)
async def test_authorizer_maps_rejections(status_code, failure):
    client = AsyncMock()
    client.post.return_value = httpx.Response(status_code)
    authorizer = A2AAuthorizerClient("http://authorizer", "agent-1", 2, client=client)

    with pytest.raises(AuthorizationError) as exc:
        await authorizer.authorize("Bearer token")

    assert exc.value.failure is failure
    client.post.assert_awaited_once_with(
        "http://authorizer/authorize/api/v1/agent/agent-1/a2a",
        headers={"authorization": "Bearer token"},
        timeout=2,
    )


async def test_authorizer_accepts_success():
    client = AsyncMock()
    client.post.return_value = httpx.Response(200)
    authorizer = A2AAuthorizerClient("http://authorizer", "agent-1", 2, client=client)

    await authorizer.authorize("Bearer token")


async def test_authorizer_fails_closed_on_transport_error():
    client = AsyncMock()
    client.post.side_effect = httpx.ConnectError("offline")
    authorizer = A2AAuthorizerClient("http://authorizer", "agent-1", 2, client=client)

    with pytest.raises(AuthorizationError) as exc:
        await authorizer.authorize("Bearer token")

    assert exc.value.failure is AuthorizationFailure.UNAVAILABLE


def test_authenticated_settings_require_authorizer_url():
    from hyperforge.a2a.settings import A2ASettings

    with pytest.raises(ValueError, match="A2A_AUTHORIZER_URL"):
        A2ASettings(a2a_auth_enabled=True)


async def test_grpc_interceptor_keeps_card_public_and_protects_tasks():
    authorizer = AsyncMock()
    server = grpc.aio.server(interceptors=[A2AAuthInterceptor(authorizer)])
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(_A2AService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"127.0.0.1:{port}")
    stub = a2a_pb2_grpc.A2AServiceStub(channel)
    try:
        card = await stub.GetExtendedAgentCard(a2a_pb2.GetExtendedAgentCardRequest())
        assert card.name == "public"

        with pytest.raises(grpc.aio.AioRpcError) as exc:
            await stub.SendMessage(a2a_pb2.SendMessageRequest())
        assert exc.value.code() is grpc.StatusCode.UNAUTHENTICATED

        await stub.SendMessage(
            a2a_pb2.SendMessageRequest(),
            metadata=(("authorization", "Bearer token"),),
        )
        authorizer.authorize.assert_awaited_once_with("Bearer token")
    finally:
        await channel.close()
        await server.stop(grace=None)
