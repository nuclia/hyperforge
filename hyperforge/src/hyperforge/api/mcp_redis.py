import asyncio
import base64
import hashlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import Message, Receive, Scope, Send

from hyperforge.api import logger
from hyperforge.api.mcp_server_pool import MCPServerKey

LocalHandler = Callable[[MCPServerKey, Request, bytes], Awaitable[Response]]
RelayKind = Literal["request", "stream", "control"]
_RELAY_KINDS: tuple[RelayKind, ...] = ("request", "stream", "control")

_RENEW = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('EXPIRE', KEYS[1], ARGV[2])
"""
_REMOVE = """
if redis.call('GET', KEYS[1]) ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""
_APPEND = """
if redis.call('XLEN', KEYS[1]) >= tonumber(ARGV[1]) then return 0 end
redis.call('XADD', KEYS[1], '*', 'data', ARGV[2])
redis.call('EXPIRE', KEYS[1], ARGV[3])
return 1
"""


class _RelayConsumerUnavailable(Exception):
    pass


class _RelayRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    owner: str
    key: MCPServerKey
    method: Literal["GET", "POST", "DELETE"]
    path: str
    query: str
    scheme: str
    headers: list[tuple[str, str]]
    body: str


def _relay_kind(method: str, body: bytes) -> RelayKind:
    if method == "GET":
        return "stream"
    if method == "DELETE":
        return "control"
    try:
        message = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        # Leave invalid messages to the SDK's normal validation.
        return "request"
    if isinstance(message, dict) and message.get("jsonrpc") == "2.0":
        rpc_method = message.get("method")
        if (
            isinstance(rpc_method, str)
            and rpc_method.startswith("notifications/")
            and "id" not in message
        ) or (
            "method" not in message
            and "id" in message
            and ("result" in message or "error" in message)
        ):
            return "control"
    return "request"


async def _until_disconnected(
    serving: Coroutine[Any, Any, None], disconnected: Coroutine[Any, Any, None]
) -> None:
    serve_task = asyncio.create_task(serving)
    disconnect_task = asyncio.create_task(disconnected)
    try:
        done, _ = await asyncio.wait(
            (serve_task, disconnect_task), return_when=asyncio.FIRST_COMPLETED
        )
        await (serve_task if serve_task in done else disconnect_task)
    finally:
        serve_task.cancel()
        disconnect_task.cancel()
        await asyncio.gather(serve_task, disconnect_task, return_exceptions=True)


class RedisMCPSessionRouter:
    """Keep SDK transports local and relay authenticated requests over Redis."""

    def __init__(
        self,
        client: Redis,
        local_handler: LocalHandler,
        *,
        invalidate_sessions: Callable[[], Coroutine[Any, Any, None]],
        session_ttl: int,
        max_request_bytes: int,
        max_response_bytes: int,
        max_pending_requests: int = 100,
        lease_seconds: int = 30,
    ):
        self.client = client
        self.local_handler = local_handler
        self.invalidate_sessions = invalidate_sessions
        self.session_ttl = session_ttl
        self.max_request_bytes = max_request_bytes
        self.max_response_bytes = max_response_bytes
        self.max_pending_requests = max_pending_requests
        self.lease_seconds = lease_seconds
        self.owner = uuid.uuid4().hex
        self._tasks: dict[asyncio.Task[None], RelayKind] = {}
        self._local_tasks: set[asyncio.Task[None]] = set()
        self._workers: list[asyncio.Task[None]] = []
        self._supervisor: asyncio.Task[None] | None = None
        self._failed = asyncio.Event()
        self._lease_deadline = 0.0
        self._healthy = False
        self._closed = False

    @staticmethod
    def _owner_key(owner: str) -> str:
        return f"mcp.owner.{owner}"

    @staticmethod
    def _inbox(owner: str, kind: RelayKind = "request") -> str:
        return f"mcp.inbox.{owner}.{kind}"

    @staticmethod
    def _session_key(key: MCPServerKey, session_id: str) -> str:
        digest = hashlib.sha256(
            json.dumps([*key, session_id], separators=(",", ":")).encode()
        ).hexdigest()
        return f"mcp.session.{digest}"

    @staticmethod
    def _request_key(request_id: str) -> str:
        return f"mcp.request.{request_id}"

    @staticmethod
    def _reply_key(request_id: str) -> str:
        return f"mcp.reply.{request_id}"

    @staticmethod
    def _live_key(request_id: str) -> str:
        return f"mcp.live.{request_id}"

    async def start(self) -> None:
        await self._register_owner()
        self._healthy = True
        self._start_workers()
        self._supervisor = asyncio.create_task(
            self._supervise(), name="mcp-owner-supervisor"
        )
        self._supervisor.add_done_callback(self._completed)

    async def _register_owner(self) -> None:
        deadline = time.monotonic() + self.lease_seconds
        try:
            async with asyncio.timeout(self.lease_seconds):
                await self.client.set(
                    self._owner_key(self.owner), self.owner, ex=self.lease_seconds
                )
            if time.monotonic() >= deadline:
                raise TimeoutError("Owner lease acknowledgement arrived too late")
        except TimeoutError as exc:
            raise RedisError("MCP owner lease registration timed out") from exc
        self._lease_deadline = deadline

    def _start_workers(self) -> None:
        owner = self.owner
        self._workers = [
            asyncio.create_task(self._heartbeat(owner), name="mcp-heartbeat"),
            *(
                asyncio.create_task(self._listen(owner, kind), name=f"mcp-inbox-{kind}")
                for kind in _RELAY_KINDS
            ),
        ]

    async def shutdown(self) -> None:
        self._closed = True
        self._fail_owner()
        if self._supervisor is not None:
            await asyncio.shield(self._supervisor)
        else:
            await self._retire_owner()

    def _fail_owner(self) -> None:
        self._healthy = False
        self._failed.set()

    def _ensure_healthy(self) -> None:
        if self._healthy and time.monotonic() >= self._lease_deadline:
            self._fail_owner()
        if not self._healthy:
            raise HTTPException(503, "MCP session routing is unavailable")

    async def _retire_owner(self) -> None:
        self._healthy = False
        tasks = [*self._workers, *self._tasks, *self._local_tasks]
        for task in tasks:
            task.cancel()
        try:
            async with asyncio.TaskGroup() as cleanup:
                cleanup.create_task(self._revoke_owner(self.owner))
                cleanup.create_task(self.invalidate_sessions())
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self._workers.clear()

    async def _revoke_owner(self, owner: str) -> None:
        try:
            async with asyncio.timeout(self.lease_seconds):
                await self.client.eval(_REMOVE, 1, self._owner_key(owner), owner)
        except (RedisError, TimeoutError):
            logger.warning("MCP owner revocation failed; its lease will expire")

    async def _supervise(self) -> None:
        while True:
            failed = asyncio.create_task(self._failed.wait())
            try:
                while True:
                    done, _ = await asyncio.wait(
                        [*self._workers, failed],
                        timeout=max(0, self._lease_deadline - time.monotonic()),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if done or time.monotonic() >= self._lease_deadline:
                        break
                if not self._closed:
                    if not done:
                        logger.warning("MCP owner lease renewal deadline expired")
                    for worker in done - {failed}:
                        error = None if worker.cancelled() else worker.exception()
                        logger.error(
                            "MCP owner worker %s exited (%s); retiring its sessions",
                            worker.get_name(),
                            type(error).__name__ if error is not None else "stopped",
                        )
            finally:
                self._healthy = False
                failed.cancel()
                try:
                    await self._retire_owner()
                finally:
                    await asyncio.gather(failed, return_exceptions=True)

            if self._closed:
                return
            self.owner = uuid.uuid4().hex
            self._failed.clear()
            while not self._closed:
                try:
                    await self._register_owner()
                except RedisError:
                    logger.warning("Redis unavailable; MCP owner recovery will retry")
                    await asyncio.sleep(0.5)
                else:
                    if not self._closed and not self._failed.is_set():
                        self._healthy = True
                        self._start_workers()
                    break

    async def _heartbeat(self, owner: str) -> None:
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            deadline = time.monotonic() + self.lease_seconds
            renewed = await self.client.eval(
                _RENEW, 1, self._owner_key(owner), owner, self.lease_seconds
            )
            if (
                not renewed
                or not self._healthy
                or time.monotonic() >= self._lease_deadline
            ):
                return
            # Include the Redis round trip; a late acknowledgement cannot revive us.
            self._lease_deadline = deadline

    async def _listen(self, owner: str, kind: RelayKind) -> None:
        inbox = self._inbox(owner, kind)
        while True:
            messages = await self.client.xread({inbox: "0-0"}, block=1000, count=1)
            for _, entries in messages:
                for entry_id, fields in entries:
                    await self.client.xdel(inbox, entry_id)
                    request_id = fields["data"]
                    value = await self.client.getdel(self._request_key(request_id))
                    if value is None:
                        continue
                    if (
                        sum(active == kind for active in self._tasks.values())
                        >= self.max_pending_requests
                    ):
                        try:
                            await self._publish(
                                request_id,
                                {"type": "error", "status": 503},
                                sequence=1,
                            )
                        except _RelayConsumerUnavailable:
                            logger.info("MCP relay caller disconnected at capacity")
                        continue
                    task = asyncio.create_task(
                        self._execute(request_id, value), name="mcp-relay"
                    )
                    self._tasks[task] = kind
                    task.add_done_callback(self._completed)

    def _completed(self, task: asyncio.Task[None]) -> None:
        self._tasks.pop(task, None)
        if task.cancelled():
            if self._healthy:
                logger.error(
                    "MCP routing task %s was unexpectedly cancelled", task.get_name()
                )
                self._fail_owner()
        elif task.exception() is not None:
            logger.error(
                "MCP routing task %s failed (%s)",
                task.get_name(),
                type(task.exception()).__name__,
            )
            self._fail_owner()

    async def _append(self, key: str, data: str, limit: int) -> bool:
        return bool(
            await self.client.eval(_APPEND, 1, key, limit, data, self.lease_seconds * 2)
        )

    async def _publish(
        self, request_id: str, message: dict[str, Any], *, sequence: int
    ) -> None:
        payload = json.dumps({**message, "sequence": sequence})
        deadline = time.monotonic() + self.lease_seconds
        if not await self.client.exists(self._live_key(request_id)):
            raise _RelayConsumerUnavailable
        while not await self._append(self._reply_key(request_id), payload, 16):
            if time.monotonic() >= deadline or not await self.client.exists(
                self._live_key(request_id)
            ):
                raise _RelayConsumerUnavailable
            await asyncio.sleep(0.01)

    def response(self, key: MCPServerKey, request: Request, body: bytes) -> Response:
        return _RedisMCPResponse(self, key, request, body)

    async def _owner(self, key: MCPServerKey, session_id: str) -> str:
        owner = await self.client.get(self._session_key(key, session_id))
        if owner is None or not await self.client.exists(self._owner_key(owner)):
            raise HTTPException(404, "MCP session not found; initialize a new session")
        return owner

    async def _local(
        self,
        key: MCPServerKey,
        request: Request,
        body: bytes,
        receive: Receive,
        send: Send,
    ) -> None:
        self._ensure_healthy()
        owner = self.owner
        task = asyncio.create_task(self._serve_local(key, request, body, receive, send))
        self._local_tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            if (not self._healthy or self.owner != owner) and not self._closed:
                raise HTTPException(503, "MCP owner lease was lost") from None
            raise
        finally:
            self._local_tasks.discard(task)

    async def _serve_local(
        self,
        key: MCPServerKey,
        request: Request,
        body: bytes,
        receive: Receive,
        send: Send,
    ) -> None:
        owner = self.owner
        if await self.client.get(self._owner_key(owner)) != owner:
            self._fail_owner()
            raise HTTPException(503, "MCP owner lease was lost")
        session_id = request.headers.get("mcp-session-id")
        if session_id and not await self.client.eval(
            _RENEW, 1, self._session_key(key, session_id), owner, self.session_ttl
        ):
            raise HTTPException(404, "MCP session not found")
        self._ensure_healthy()
        response = await self.local_handler(key, request, body)

        async def registered_send(message: Message) -> None:
            if self.owner != owner:
                raise HTTPException(503, "MCP owner lease was lost")
            if message["type"] == "http.response.start":
                if message["status"] == 200 and not session_id:
                    for name, value in message.get("headers", []):
                        if name.lower() == b"mcp-session-id":
                            session_key = self._session_key(key, value.decode())
                            registered = await self.client.set(
                                session_key,
                                owner,
                                ex=self.session_ttl,
                                nx=True,
                            )
                            if (
                                not registered
                                and await self.client.get(session_key) != owner
                            ):
                                raise HTTPException(
                                    503, "MCP session registration failed"
                                )
                if session_id:
                    session_key = self._session_key(key, session_id)
                    if message["status"] == 404 or (
                        request.method == "DELETE" and message["status"] == 200
                    ):
                        await self.client.eval(_REMOVE, 1, session_key, owner)
            self._ensure_healthy()
            await send(message)

        await response(request.scope, receive, registered_send)

    async def _execute(self, request_id: str, value: str | bytes) -> None:
        sequence = 0

        async def publish(message: dict[str, Any]) -> None:
            nonlocal sequence
            sequence += 1
            await self._publish(request_id, message, sequence=sequence)

        try:
            envelope = _RelayRequest.model_validate_json(value)
            if envelope.owner != self.owner:
                raise HTTPException(404, "MCP owner is no longer available")
            self._ensure_healthy()
            scope: Scope = {
                "type": "http",
                "asgi": {"version": "3.0", "spec_version": "2.4"},
                "http_version": "1.1",
                "method": envelope.method,
                "path": envelope.path,
                "root_path": "",
                "query_string": base64.b64decode(envelope.query),
                "scheme": envelope.scheme,
                "headers": [
                    (name.encode("latin-1"), value.encode("latin-1"))
                    for name, value in envelope.headers
                ],
            }
            request = Request(scope)
            session_id = request.headers.get("mcp-session-id")
            if (
                not session_id
                or await self._owner(envelope.key, session_id) != self.owner
            ):
                raise HTTPException(404, "MCP session not found")
            await publish({"type": "accepted"})
            body = base64.b64decode(envelope.body)
            if len(body) > self.max_request_bytes:
                raise HTTPException(413, "MCP request is too large")

            async def disconnected() -> None:
                while await self.client.exists(self._live_key(request_id)):
                    await asyncio.sleep(0.1)

            async def receive() -> Message:
                await disconnected()
                return {"type": "http.disconnect"}

            send_lock = asyncio.Lock()

            async def send(message: Message) -> None:
                async with send_lock:
                    if message["type"] == "http.response.start":
                        await publish(
                            {
                                "type": "start",
                                "status": message["status"],
                                "headers": [
                                    (name.decode("latin-1"), value.decode("latin-1"))
                                    for name, value in message.get("headers", [])
                                ],
                            },
                        )
                    elif message["type"] == "http.response.body":
                        chunk = message.get("body", b"")
                        for offset in range(0, len(chunk), 65536):
                            await publish(
                                {
                                    "type": "body",
                                    "body": base64.b64encode(
                                        chunk[offset : offset + 65536]
                                    ).decode(),
                                    "more": True,
                                },
                            )
                        if not message.get("more_body", False):
                            await publish({"type": "body", "body": "", "more": False})

            await _until_disconnected(
                self._local(envelope.key, request, body, receive, send), disconnected()
            )
        except _RelayConsumerUnavailable:
            logger.info("MCP relay caller disconnected")
            return
        except HTTPException as exc:
            status = exc.status_code
        except (asyncio.CancelledError, RedisError):
            raise
        except Exception:
            logger.error("MCP owner could not process a relayed request")
            status = 503
        else:
            return
        try:
            await publish({"type": "error", "status": status})
        except _RelayConsumerUnavailable:
            logger.info("MCP relay caller disconnected")

    async def serve(
        self,
        key: MCPServerKey,
        request: Request,
        body: bytes,
        receive: Receive,
        send: Send,
    ) -> None:
        started = False

        async def tracked_send(message: Message) -> None:
            nonlocal started
            await send(message)
            if message["type"] == "http.response.start":
                started = True

        try:
            self._ensure_healthy()
            if (
                sum(len(name) + len(value) for name, value in request.scope["headers"])
                > 65536
            ):
                raise HTTPException(431, "MCP request headers are too large")
            session_id = request.headers.get("mcp-session-id")
            owner = await self._owner(key, session_id) if session_id else self.owner
            self._ensure_healthy()
            if owner == self.owner:
                await self._local(key, request, body, receive, tracked_send)
            else:
                await self._remote(owner, key, request, body, receive, tracked_send)
        except (HTTPException, RedisError) as exc:
            if isinstance(exc, RedisError):
                logger.warning("Redis unavailable while routing an MCP request")
                self._fail_owner()
            if started:
                raise
            status = exc.status_code if isinstance(exc, HTTPException) else 503
            response = JSONResponse(
                {
                    "detail": "MCP session not found"
                    if status == 404
                    else "MCP relay unavailable"
                },
                status_code=status,
            )
            await response(request.scope, receive, send)

    async def _remote(
        self,
        owner: str,
        key: MCPServerKey,
        request: Request,
        body: bytes,
        receive: Receive,
        send: Send,
    ) -> None:
        request_id = uuid.uuid4().hex
        envelope = _RelayRequest(
            owner=owner,
            key=key,
            method=request.scope["method"],
            path=request.scope["path"],
            query=base64.b64encode(request.scope.get("query_string", b"")).decode(),
            scheme=request.scope.get("scheme", "http"),
            headers=[
                (name.decode("latin-1"), value.decode("latin-1"))
                for name, value in request.scope["headers"]
            ],
            body=base64.b64encode(body).decode(),
        )
        payload = envelope.model_dump_json()
        if len(payload) > self.max_request_bytes * 2 + 65536:
            raise HTTPException(413, "MCP relay request is too large")
        await self.client.set(self._live_key(request_id), "1", ex=self.lease_seconds)
        try:
            await self.client.set(
                self._request_key(request_id),
                payload,
                ex=self.lease_seconds,
            )
            if not await self._append(
                self._inbox(owner, _relay_kind(request.method, body)),
                request_id,
                self.max_pending_requests,
            ):
                raise HTTPException(503, "MCP relay capacity is exhausted")

            async def pump() -> None:
                last_id = "0-0"
                received = 0
                started = False
                accepted = False
                last_sequence = 0
                deadline = time.monotonic() + self.lease_seconds
                while True:
                    if not await self.client.exists(self._owner_key(owner)):
                        raise HTTPException(404, "MCP owner is no longer available")
                    if not started and not accepted and time.monotonic() > deadline:
                        raise HTTPException(503, "MCP owner did not respond")
                    if not await self.client.expire(
                        self._live_key(request_id), self.lease_seconds
                    ):
                        raise HTTPException(503, "MCP relay consumer lease was lost")
                    stream = self._reply_key(request_id)
                    replies = await self.client.xread(
                        {stream: last_id}, block=1000, count=16
                    )
                    for _, entries in replies:
                        for entry_id, fields in entries:
                            last_id = entry_id
                            message = json.loads(fields["data"])
                            sequence = message["sequence"]
                            if sequence <= last_sequence:
                                await self.client.xdel(stream, entry_id)
                                continue
                            if sequence != last_sequence + 1:
                                raise HTTPException(
                                    503, "MCP relay response is incomplete"
                                )
                            last_sequence = sequence
                            if message["type"] == "accepted":
                                accepted = True
                                await self.client.xdel(stream, entry_id)
                                continue
                            if message["type"] == "error":
                                raise HTTPException(
                                    message["status"], "MCP relay failed"
                                )
                            if message["type"] == "start":
                                started = True
                                await send(
                                    {
                                        "type": "http.response.start",
                                        "status": message["status"],
                                        "headers": [
                                            (
                                                name.encode("latin-1"),
                                                value.encode("latin-1"),
                                            )
                                            for name, value in message["headers"]
                                        ],
                                    }
                                )
                            else:
                                chunk = base64.b64decode(message["body"])
                                received += len(chunk)
                                if (
                                    request.method != "GET"
                                    and received > self.max_response_bytes
                                ):
                                    raise HTTPException(
                                        502, "MCP response is too large"
                                    )
                                await send(
                                    {
                                        "type": "http.response.body",
                                        "body": chunk,
                                        "more_body": message["more"],
                                    }
                                )
                                if not message["more"]:
                                    return
                            await self.client.xdel(stream, entry_id)

            async def disconnected() -> None:
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return

            await _until_disconnected(pump(), disconnected())
        finally:
            for redis_key in (
                self._live_key(request_id),
                self._request_key(request_id),
                self._reply_key(request_id),
            ):
                try:
                    await self.client.delete(redis_key)
                except RedisError:
                    logger.warning("Redis unavailable during MCP relay cleanup")


class _RedisMCPResponse(Response):
    def __init__(
        self,
        router: RedisMCPSessionRouter,
        key: MCPServerKey,
        request: Request,
        body: bytes,
    ):
        super().__init__()
        self.router = router
        self.key = key
        self.request = request
        self.request_body = body

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.router.serve(
            self.key, self.request, self.request_body, receive, send
        )
