import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import anyio
from fastapi import HTTPException
from mcp import types
from mcp.server.lowlevel import Server
from mcp.shared.message import ServerMessageMetadata, SessionMessage
from mcp.shared.version import SUPPORTED_PROTOCOL_VERSIONS
from pydantic import ValidationError
from sse_starlette import EventSourceResponse
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import Receive, Scope, Send

from hyperforge.api import logger
from hyperforge.api.mcp_sessions import (
    MAX_ACTIVE_CALLS,
    MCPServerKey,
    MCPSession,
    MCPSessionStore,
    MCPUpdates,
)


async def control_response(
    store: MCPSessionStore,
    key: MCPServerKey,
    request: Request,
    body: bytes,
    ttl: int,
) -> Response | None:
    session_id = request.headers.get("mcp-session-id")
    if session_id is not None and not re.fullmatch("[0-9a-f]{32}", session_id):
        raise HTTPException(404, "MCP session not found")
    if request.method == "POST":
        accept = request.headers.get("accept", "")
        if "application/json" not in accept or "text/event-stream" not in accept:
            raise HTTPException(406, "Client must accept JSON and event-stream")
        if request.headers.get("content-type", "").split(";")[0] != "application/json":
            raise HTTPException(415, "Content-Type must be application/json")
        try:
            rpc = types.JSONRPCMessage.model_validate_json(body).root
        except ValidationError as exc:
            raise HTTPException(400, "Invalid JSON-RPC message") from exc
        if isinstance(rpc, types.JSONRPCRequest) and rpc.method == "initialize":
            if session_id:
                raise HTTPException(400, "Initialize must create a new MCP session")
            try:
                types.InitializeRequest.model_validate(rpc.model_dump())
            except ValidationError as exc:
                raise HTTPException(400, "Invalid MCP initialization") from exc
            return None
    elif request.method not in {"GET", "DELETE"}:
        raise HTTPException(405, "Unsupported MCP HTTP method")
    if session_id is None:
        raise HTTPException(400, "Missing Mcp-Session-Id")
    session = await store.get(key, session_id, ttl)
    if request.method == "DELETE":
        await store.delete(key, session_id, ttl)
        return Response(status_code=200)
    protocol = request.headers.get("mcp-protocol-version", "2025-03-26")
    if protocol not in SUPPORTED_PROTOCOL_VERSIONS:
        raise HTTPException(400, "Unsupported MCP protocol version")
    if request.method == "GET":
        return Response(status_code=405, headers={"Allow": "POST, DELETE"})
    if isinstance(rpc, types.JSONRPCNotification):
        try:
            notification = types.ClientNotification.model_validate(
                rpc.model_dump()
            ).root
        except ValidationError as exc:
            raise HTTPException(400, "Invalid MCP notification") from exc
        if isinstance(notification, types.InitializedNotification):
            await store.ready(key, session_id, ttl)
        elif isinstance(notification, types.CancelledNotification):
            if notification.params.requestId is None:
                raise HTTPException(400, "Cancellation requires a requestId")
            await store.cancel(key, session_id, notification.params.requestId, ttl)
        else:
            logger.debug("MCP notification %s has no registered handler", rpc.method)
        return Response(status_code=202)
    if isinstance(rpc, types.JSONRPCResponse | types.JSONRPCError):
        matched = await store.reply(
            key,
            session_id,
            str(rpc.id),
            json.dumps(
                rpc.model_dump(by_alias=True, exclude_none=True), sort_keys=True
            ),
            ttl,
        )
        if not matched:
            logger.info("MCP response has no matching active request")
        return Response(status_code=202)
    if not session.ready:
        raise HTTPException(400, "MCP session initialization is not complete")
    return None


class MCPTransport:
    """Run each request locally, delivering only protocol replies through the store."""

    def __init__(
        self,
        app: Server,
        store: MCPSessionStore,
        key: MCPServerKey,
        *,
        session_ttl: int,
        max_sessions: int,
        max_response_bytes: int,
    ):
        self.app = app
        self.store = store
        self.key = key
        self.session_ttl = session_ttl
        self.max_sessions = max_sessions
        self.max_response_bytes = max_response_bytes
        self._closed = True
        self._tasks: set[asyncio.Task[Any]] = set()

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        self._closed = False
        try:
            yield
        finally:
            self._closed = True
            tasks = list(self._tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def session_count(self) -> int:
        return await self.store.session_count(self.key, self.session_ttl)

    async def handle_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        task = asyncio.current_task()
        if self._closed or task is None:
            raise HTTPException(503, "MCP transport is shutting down")
        if len(self.store.active_requests) >= MAX_ACTIVE_CALLS:
            raise HTTPException(503, "MCP request capacity is exhausted")
        self._tasks.add(task)
        self.store.active_requests.add(task)
        try:
            await self._handle_request(scope, receive, send)
        finally:
            self._tasks.discard(task)
            self.store.active_requests.discard(task)

    async def _handle_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        request = Request(scope, receive)
        body = await request.body()
        rpc = types.JSONRPCMessage.model_validate_json(body).root
        if not isinstance(rpc, types.JSONRPCRequest):
            raise HTTPException(400, "Expected an MCP request")
        initializing = rpc.method == "initialize"
        if initializing:
            params = types.InitializeRequest.model_validate(rpc.model_dump()).params
            session_id = await self.store.create(
                self.key, params, self.session_ttl, self.max_sessions
            )
            session = MCPSession(params=params, ready=False)
        else:
            session_id = request.headers["mcp-session-id"]
            session = await self.store.get(self.key, session_id, self.session_ttl)
        initialized = False
        token = None
        pending: dict[str, types.RequestId] = {}
        try:
            token = await self.store.start(
                self.key, session_id, rpc.id, self.session_ttl
            )
            initialized = await self._serve(
                request, rpc, session, session_id, token, pending, send
            )
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    if initializing and not initialized:
                        await self.store.delete(self.key, session_id, self.session_ttl)
                    if token is not None:
                        await self.store.finish(
                            self.key,
                            session_id,
                            rpc.id,
                            token,
                            list(pending),
                            self.session_ttl,
                        )
                except HTTPException as exc:
                    if exc.status_code != 404:
                        logger.warning("MCP request cleanup failed: %s", exc.detail)

    async def _serve(
        self,
        request: Request,
        rpc: types.JSONRPCRequest,
        session: MCPSession,
        session_id: str,
        token: str,
        pending: dict[str, types.RequestId],
        send: Send,
    ) -> bool:
        bootstrap = uuid.uuid4().hex
        initialized = False

        def wrapped(value: dict[str, Any]) -> SessionMessage:
            return SessionMessage(types.JSONRPCMessage.model_validate(value))

        sdk_input, sdk_read = anyio.create_memory_object_stream[
            SessionMessage | Exception
        ](16)
        sdk_write, sdk_output = anyio.create_memory_object_stream[SessionMessage](16)
        async with (
            sdk_input,
            sdk_read,
            sdk_write,
            sdk_output,
            anyio.create_task_group() as group,
        ):

            async def messages() -> AsyncIterator[dict[str, str]]:
                nonlocal initialized
                async for item in sdk_output:
                    value = item.message.root
                    if isinstance(value, types.JSONRPCRequest):
                        if len(pending) >= MAX_ACTIVE_CALLS:
                            raise HTTPException(
                                503, "MCP pending request capacity is exhausted"
                            )
                        wire_id = uuid.uuid4().hex
                        await self.store.pending(
                            self.key, session_id, wire_id, self.session_ttl
                        )
                        pending[wire_id] = value.id
                        value = value.model_copy(update={"id": wire_id})
                    data = value.model_dump_json(by_alias=True, exclude_none=True)
                    if len(data.encode()) > self.max_response_bytes:
                        logger.warning("MCP response exceeds configured size limit")
                        value = types.JSONRPCError(
                            jsonrpc="2.0",
                            id=rpc.id,
                            error=types.ErrorData(
                                code=-32603, message="MCP response is too large"
                            ),
                        )
                        data = value.model_dump_json(by_alias=True)
                    final = (
                        isinstance(value, types.JSONRPCResponse | types.JSONRPCError)
                        and value.id == rpc.id
                    )
                    if final:
                        initialized = isinstance(value, types.JSONRPCResponse)
                    if final or rpc.method == "tools/call":
                        yield {"event": "message", "data": data}
                    if final:
                        return

            async def poll() -> None:
                await anyio.sleep(0)
                while True:
                    try:
                        update = await self.store.poll(
                            self.key,
                            session_id,
                            rpc.id,
                            token,
                            list(pending),
                            self.session_ttl,
                        )
                    except HTTPException as exc:
                        if exc.status_code not in {404, 409}:
                            raise
                        update = MCPUpdates(cancelled=True)
                    if update.cancelled:
                        await sdk_input.send(
                            wrapped(
                                {
                                    "jsonrpc": "2.0",
                                    "method": "notifications/cancelled",
                                    "params": {"requestId": rpc.id},
                                }
                            )
                        )
                        return
                    for wire_id, reply in update.replies.items():
                        local_id = pending.pop(wire_id, None)
                        if local_id is not None:
                            value = types.JSONRPCMessage.model_validate_json(reply)
                            if not isinstance(
                                value.root, types.JSONRPCResponse | types.JSONRPCError
                            ):
                                raise RuntimeError("Stored MCP reply is not a response")
                            value.root.id = local_id
                            await sdk_input.send(SessionMessage(value))
                    await self.store.acknowledge(
                        self.key, session_id, list(update.replies), self.session_ttl
                    )
                    await anyio.sleep(min(0.25, self.session_ttl / 3))

            group.start_soon(
                self.app.run,
                sdk_read,
                sdk_write,
                self.app.create_initialization_options(),
            )
            if rpc.method != "initialize":
                await sdk_input.send(
                    wrapped(
                        {
                            "jsonrpc": "2.0",
                            "id": bootstrap,
                            "method": "initialize",
                            "params": session.params.model_dump(
                                by_alias=True, exclude_none=True
                            ),
                        }
                    )
                )
                result = (await sdk_output.receive()).message.root
                if (
                    not isinstance(result, types.JSONRPCResponse)
                    or result.id != bootstrap
                ):
                    raise RuntimeError("Failed to restore MCP initialization")
                await sdk_input.send(
                    wrapped({"jsonrpc": "2.0", "method": "notifications/initialized"})
                )
            await sdk_input.send(
                SessionMessage(
                    types.JSONRPCMessage(rpc),
                    ServerMessageMetadata(request_context=request),
                )
            )
            group.start_soon(poll)
            headers = {"Mcp-Session-Id": session_id}
            try:
                if rpc.method == "tools/call":
                    await EventSourceResponse(messages(), headers=headers)(
                        request.scope, request.receive, send
                    )
                else:
                    async for event in messages():
                        await Response(
                            event["data"],
                            media_type="application/json",
                            headers=headers,
                        )(request.scope, request.receive, send)
            finally:
                group.cancel_scope.cancel()
        return initialized
