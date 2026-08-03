from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from enum import Enum
from typing import Any, Protocol, cast

import grpc
import httpx

AGENT_CARD_METHOD = "/lf.a2a.v1.A2AService/GetExtendedAgentCard"


class _HandlerCallDetails(Protocol):
    method: str


class _RpcMethodHandler(Protocol):
    unary_unary: Any
    unary_stream: Any
    request_deserializer: Any
    response_serializer: Any


class AuthorizationFailure(Enum):
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    UNAVAILABLE = "unavailable"


class AuthorizationError(Exception):
    def __init__(self, failure: AuthorizationFailure) -> None:
        self.failure = failure
        super().__init__(failure.value)


class A2AAuthorizerClient:
    def __init__(
        self,
        base_url: str,
        agent_id: str,
        timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/authorize/api/v1/agent/{agent_id}/a2a"
        self._timeout_seconds = timeout_seconds
        self._client = client

    async def authorize(self, authorization: str) -> None:
        try:
            if self._client is not None:
                response = await self._client.post(
                    self._url,
                    headers={"authorization": authorization},
                    timeout=self._timeout_seconds,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self._url,
                        headers={"authorization": authorization},
                        timeout=self._timeout_seconds,
                    )
        except httpx.HTTPError as exc:
            raise AuthorizationError(AuthorizationFailure.UNAVAILABLE) from exc

        if 200 <= response.status_code < 300:
            return
        if response.status_code == 401:
            raise AuthorizationError(AuthorizationFailure.UNAUTHENTICATED)
        if response.status_code == 403:
            raise AuthorizationError(AuthorizationFailure.FORBIDDEN)
        raise AuthorizationError(AuthorizationFailure.UNAVAILABLE)


def _bearer_from_metadata(
    metadata: Iterable[tuple[str, str | bytes]] | None,
) -> str:
    values = [value for key, value in metadata or () if key.lower() == "authorization"]
    if len(values) != 1 or not isinstance(values[0], str):
        raise AuthorizationError(AuthorizationFailure.UNAUTHENTICATED)
    scheme, separator, token = values[0].partition(" ")
    if scheme.lower() != "bearer" or not separator or not token.strip():
        raise AuthorizationError(AuthorizationFailure.UNAUTHENTICATED)
    return values[0]


class A2AAuthInterceptor(grpc.aio.ServerInterceptor):
    def __init__(self, authorizer: A2AAuthorizerClient) -> None:
        self._authorizer = authorizer

    async def _authorize(self, context: grpc.aio.ServicerContext) -> None:
        authorization = _bearer_from_metadata(context.invocation_metadata())
        await self._authorizer.authorize(authorization)

    async def _abort(
        self, context: grpc.aio.ServicerContext, error: AuthorizationError
    ) -> None:
        code = {
            AuthorizationFailure.UNAUTHENTICATED: grpc.StatusCode.UNAUTHENTICATED,
            AuthorizationFailure.FORBIDDEN: grpc.StatusCode.PERMISSION_DENIED,
            AuthorizationFailure.UNAVAILABLE: grpc.StatusCode.UNAVAILABLE,
        }[error.failure]
        await context.abort(code, error.failure.value)

    async def intercept_service(
        self,
        continuation: Callable[
            [grpc.HandlerCallDetails], Awaitable[grpc.RpcMethodHandler]
        ],
        handler_call_details: grpc.HandlerCallDetails,
    ) -> grpc.RpcMethodHandler:
        handler = await continuation(handler_call_details)
        method = cast(_HandlerCallDetails, handler_call_details).method
        if method == AGENT_CARD_METHOD:
            return handler

        typed_handler = cast(_RpcMethodHandler, handler)
        if typed_handler.unary_unary:

            async def unary_unary(request: Any, context: grpc.aio.ServicerContext):
                try:
                    await self._authorize(context)
                except AuthorizationError as error:
                    await self._abort(context, error)
                return await typed_handler.unary_unary(request, context)

            return grpc.unary_unary_rpc_method_handler(
                unary_unary,
                request_deserializer=typed_handler.request_deserializer,
                response_serializer=typed_handler.response_serializer,
            )

        if typed_handler.unary_stream:

            async def unary_stream(
                request: Any, context: grpc.aio.ServicerContext
            ) -> AsyncIterator[Any]:
                try:
                    await self._authorize(context)
                except AuthorizationError as error:
                    await self._abort(context, error)
                async for response in typed_handler.unary_stream(request, context):
                    yield response

            return grpc.unary_stream_rpc_method_handler(
                unary_stream,
                request_deserializer=typed_handler.request_deserializer,
                response_serializer=typed_handler.response_serializer,
            )

        return handler
