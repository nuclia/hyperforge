import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
from fastapi import HTTPException
from mcp.server.streamable_http import MCP_SESSION_ID_HEADER
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.requests import Request

MCPServerKey = tuple[str, str, str, str, str]
MCPServerFactory = Callable[[], Awaitable["ManagedMCPServer"]]


class ManagedMCPServer:
    def __init__(self, manager: StreamableHTTPSessionManager):
        self.manager = manager
        self.started = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.session_created = False
        self.active_requests = 0
        self.last_used = time.monotonic()

    async def run(self) -> None:
        async with self.manager.run():
            self.started.set()
            await anyio.sleep_forever()

    async def start(self) -> None:
        self.task = asyncio.create_task(self.run())
        started_task = asyncio.create_task(self.started.wait())
        try:
            done, _ = await asyncio.wait(
                (self.task, started_task), return_when=asyncio.FIRST_COMPLETED
            )
            if self.task in done:
                await self.task
        except asyncio.CancelledError:
            await self.stop()
            raise
        finally:
            started_task.cancel()
            await asyncio.gather(started_task, return_exceptions=True)
            if self.task.done():
                await asyncio.gather(self.task, return_exceptions=True)

    def close(self) -> None:
        if self.task is not None:
            self.task.cancel()

    async def stop(self) -> None:
        self.close()
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)

    def reserve_request(self) -> None:
        self.active_requests += 1
        self.last_used = time.monotonic()

    def release_request(self) -> None:
        if self.active_requests <= 0:
            raise RuntimeError("MCP request lease released more than once")
        self.active_requests -= 1
        self.last_used = time.monotonic()

    def accept_request(self, request: Request) -> bool:
        if request.headers.get(MCP_SESSION_ID_HEADER):
            return True
        if self.session_created:
            return False
        self.session_created = True
        return True

    def is_reinitialization(self, request: Request, body: bytes) -> bool:
        if (
            not self.session_created
            or request.method != "POST"
            or request.headers.get(MCP_SESSION_ID_HEADER)
        ):
            return False
        try:
            message = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        return isinstance(message, dict) and message.get("method") == "initialize"


class MCPRequestLease:
    def __init__(self, server: ManagedMCPServer):
        self.server = server
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self.server.release_request()


class MCPServerPool:
    def __init__(
        self,
        max_servers: int,
        idle_ttl_seconds: int,
        startup_timeout_seconds: float,
    ):
        self._max_servers = max_servers
        self._idle_ttl_seconds = idle_ttl_seconds
        self._startup_timeout_seconds = startup_timeout_seconds
        self._servers: dict[MCPServerKey, ManagedMCPServer] = {}
        self._creating: dict[MCPServerKey, asyncio.Task[Any]] = {}
        self._retired_tasks: set[asyncio.Task[Any]] = set()
        self._lock = anyio.Lock()
        self._shutting_down = False
        self._reaper_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None

    async def acquire(
        self,
        key: MCPServerKey,
        request: Request,
        body: bytes,
        factory: MCPServerFactory,
    ) -> MCPRequestLease:
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("MCP request task is unavailable")

        stale_server = None
        async with self._lock:
            self._ensure_running()
            if self._reaper_task is None:
                self._reaper_task = asyncio.create_task(self._reap_idle_servers())
            server = self._servers.get(key)
            if (
                server is not None
                and server.active_requests == 0
                and time.monotonic() - server.last_used >= self._idle_ttl_seconds
            ):
                stale_server = self._servers.pop(key)
                stale_server.close()
                server = None
            if server is not None and server.is_reinitialization(request, body):
                if server.active_requests:
                    raise HTTPException(
                        status_code=409,
                        detail="MCP session is still active for this path",
                    )
                stale_server = self._servers.pop(key)
                stale_server.close()
                server = None

            if server is not None:
                return self._lease(server, request)
            if key in self._creating:
                raise HTTPException(
                    status_code=409,
                    detail="MCP session initialization is already in progress",
                )
            self._creating[key] = current_task

        candidate = None
        registered = False
        try:
            if stale_server is not None:
                await stale_server.stop()
            async with asyncio.timeout(self._startup_timeout_seconds):
                candidate = await factory()

            async with self._lock:
                self._ensure_running()
                self._evict_for_capacity()
                lease = self._lease(candidate, request)
                self._servers[key] = candidate
                self._creating.pop(key, None)
                registered = True
                return lease
        except BaseException:
            if candidate is not None and not registered:
                await candidate.stop()
            async with self._lock:
                if self._creating.get(key) is current_task:
                    self._creating.pop(key, None)
            raise

    async def remove(self, key: MCPServerKey) -> None:
        async with self._lock:
            server = self._servers.pop(key, None)
            creating_task = self._creating.pop(key, None)
        if creating_task is not None:
            creating_task.cancel()
        if server is not None:
            await server.stop()
        if creating_task is not None:
            await asyncio.gather(creating_task, return_exceptions=True)

    async def shutdown(self) -> None:
        async with self._lock:
            if self._shutdown_task is None:
                self._shutting_down = True
                servers = list(self._servers.values())
                self._servers.clear()
                creating_tasks = list(self._creating.values())
                self._creating.clear()
                retired_tasks = list(self._retired_tasks)
                reaper_task = self._reaper_task
                for server in servers:
                    server.close()
                for task in creating_tasks:
                    task.cancel()
                if reaper_task is not None:
                    reaper_task.cancel()
                self._shutdown_task = asyncio.create_task(
                    self._finish_shutdown(
                        servers, creating_tasks, retired_tasks, reaper_task
                    )
                )
            shutdown_task = self._shutdown_task
        await asyncio.shield(shutdown_task)

    async def _finish_shutdown(
        self,
        servers: list[ManagedMCPServer],
        creating_tasks: list[asyncio.Task[Any]],
        retired_tasks: list[asyncio.Task[Any]],
        reaper_task: asyncio.Task[None] | None,
    ) -> None:
        await asyncio.gather(
            *(server.task for server in servers if server.task is not None),
            *creating_tasks,
            *retired_tasks,
            *(task for task in (reaper_task,) if task is not None),
            return_exceptions=True,
        )

    async def _reap_idle_servers(self) -> None:
        while True:
            await asyncio.sleep(self._idle_ttl_seconds)
            now = time.monotonic()
            async with self._lock:
                expired = [
                    (key, server)
                    for key, server in self._servers.items()
                    if server.active_requests == 0
                    and now - server.last_used >= self._idle_ttl_seconds
                ]
                for key, server in expired:
                    self._servers.pop(key)
                    server.close()
            await asyncio.gather(
                *(server.stop() for _, server in expired), return_exceptions=True
            )

    def _ensure_running(self) -> None:
        if self._shutting_down:
            raise HTTPException(status_code=503, detail="MCP service is shutting down")

    def _lease(self, server: ManagedMCPServer, request: Request) -> MCPRequestLease:
        if not server.accept_request(request):
            raise HTTPException(
                status_code=409,
                detail="MCP session already initialized for this path",
            )
        server.reserve_request()
        return MCPRequestLease(server)

    def _evict_for_capacity(self) -> None:
        while len(self._servers) >= self._max_servers:
            now = time.monotonic()
            candidates = [
                (key, server)
                for key, server in self._servers.items()
                if server.active_requests == 0
                and (
                    not server.session_created
                    or now - server.last_used >= self._idle_ttl_seconds
                )
            ]
            if not candidates:
                raise HTTPException(
                    status_code=503,
                    detail="MCP server capacity is occupied by established sessions",
                )
            key, server = min(candidates, key=lambda item: item[1].last_used)
            self._servers.pop(key)
            server.close()
            if server.task is not None:
                self._retired_tasks.add(server.task)
                server.task.add_done_callback(self._retired_tasks.discard)
