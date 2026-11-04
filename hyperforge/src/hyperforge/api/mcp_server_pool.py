import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import anyio
from fastapi import HTTPException
from starlette.types import Receive, Scope, Send

from hyperforge.api import logger
from hyperforge.api.mcp_sessions import MCPServerKey
from hyperforge.api.mcp_transport import MCPTransport

MCPServerFactory = Callable[[], Awaitable["ManagedMCPServer"]]


class ManagedMCPServer:
    def __init__(self, manager: MCPTransport):
        self.manager = manager
        self.started = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.has_been_leased = False
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
        self.has_been_leased = True
        self.active_requests += 1
        self.last_used = time.monotonic()

    def release_request(self) -> None:
        if self.active_requests <= 0:
            raise RuntimeError("MCP request lease released more than once")
        self.active_requests -= 1
        self.last_used = time.monotonic()

    async def handle_request(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.manager.handle_request(scope, receive, send)


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
        self._creating: dict[MCPServerKey, asyncio.Task[ManagedMCPServer]] = {}
        self._retired_tasks: set[asyncio.Task[Any]] = set()
        self._lock = anyio.Lock()
        self._shutting_down = False
        self._reaper_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None

    async def acquire(
        self,
        key: MCPServerKey,
        factory: MCPServerFactory,
    ) -> MCPRequestLease:
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
                self._retire_server(stale_server)
                server = None
            if server is not None:
                return self._lease(server)
            creating_task = self._creating.get(key)
            if creating_task is None:
                await self._reserve_capacity()
                creating_task = asyncio.create_task(
                    self._create_server(key, factory, stale_server)
                )
                self._creating[key] = creating_task
                creating_task.add_done_callback(self._log_creation_failure)

        created = await asyncio.shield(creating_task)
        async with self._lock:
            self._ensure_running()
            if self._servers.get(key) is not created:
                raise HTTPException(
                    status_code=503, detail="MCP server initialization was interrupted"
                )
            return self._lease(created)

    async def _create_server(
        self,
        key: MCPServerKey,
        factory: MCPServerFactory,
        stale_server: ManagedMCPServer | None,
    ) -> ManagedMCPServer:
        candidate = None
        registered = False
        try:
            if stale_server is not None:
                await stale_server.stop()
            async with asyncio.timeout(self._startup_timeout_seconds):
                candidate = await factory()

            async with self._lock:
                self._ensure_running()
                await self._evict_for_capacity()
                self._servers[key] = candidate
                self._creating.pop(key, None)
                registered = True
                return candidate
        except BaseException:
            if candidate is not None and not registered:
                await candidate.stop()
            async with self._lock:
                if self._creating.get(key) is asyncio.current_task():
                    self._creating.pop(key, None)
            raise

    @staticmethod
    def _log_creation_failure(task: asyncio.Task[ManagedMCPServer]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error(
                "MCP server initialization failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def remove(self, key: MCPServerKey) -> None:
        async with self._lock:
            server = self._servers.pop(key, None)
            creating_task = self._creating.get(key)
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

    def _lease(self, server: ManagedMCPServer) -> MCPRequestLease:
        server.reserve_request()
        return MCPRequestLease(server)

    async def _evict_for_capacity(self) -> None:
        while len(self._servers) >= self._max_servers:
            await self._evict_one()

    async def _reserve_capacity(self) -> None:
        while len(self._servers) + len(self._creating) >= self._max_servers:
            await self._evict_one()

    async def _evict_one(self) -> None:
        now = time.monotonic()
        candidates = [
            (key, server)
            for key, server in self._servers.items()
            if server.active_requests == 0
            and (
                (server.has_been_leased and await server.manager.session_count() == 0)
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

        self._retire_server(server)

    def _retire_server(self, server: ManagedMCPServer) -> None:
        server.close()
        if server.task is not None:
            self._retired_tasks.add(server.task)
            server.task.add_done_callback(self._retired_tasks.discard)
