import asyncio
import logging
import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

import mcp.types as mcp_types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from hyperforge.mcp_proxy.config import UpstreamServerConfig

logger = logging.getLogger("hyperforge.mcp_proxy")


class UpstreamError(Exception):
    pass


class UpstreamServer:
    """A live client connection to one upstream MCP server.

    The connection is owned by a dedicated task so that anyio cancel scopes
    created by the MCP client are entered and exited in the same task.
    """

    def __init__(self, config: UpstreamServerConfig):
        self.config = config
        self.session: ClientSession | None = None
        self.tools: list[mcp_types.Tool] = []
        self.error: str | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self.config.name

    async def _open(self, stack: AsyncExitStack) -> ClientSession:
        cfg = self.config
        if cfg.transport == "stdio":
            assert cfg.command
            params = StdioServerParameters(
                command=cfg.command,
                args=cfg.args,
                env={**os.environ, **cfg.env} if cfg.env else None,
                cwd=cfg.cwd,
            )
            read, write = await stack.enter_async_context(stdio_client(params))
        else:
            assert cfg.url
            read, write, _ = await stack.enter_async_context(
                streamablehttp_client(
                    cfg.url,
                    headers=cfg.headers or None,
                    timeout=cfg.timeout_seconds,
                )
            )
        session = await stack.enter_async_context(
            ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=cfg.timeout_seconds),
            )
        )
        await session.initialize()
        return session

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                self.session = await asyncio.wait_for(
                    self._open(stack), self.config.timeout_seconds
                )
                self.tools = await self._list_tools()
                self._ready.set()
                await self._stop.wait()
        except Exception as exc:
            while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
            self.error = f"{type(exc).__name__}: {exc}"
            logger.warning("MCP upstream %s failed: %s", self.name, self.error)
        finally:
            self.session = None
            self._ready.set()

    async def _list_tools(self) -> list[mcp_types.Tool]:
        assert self.session is not None
        tools: list[mcp_types.Tool] = []
        cursor: str | None = None
        while True:
            result = await self.session.list_tools(cursor=cursor)
            tools.extend(t for t in result.tools if self.config.exposes(t.name))
            cursor = result.nextCursor
            if not cursor:
                return tools

    async def start(self) -> None:
        self._task = asyncio.create_task(self._run(), name=f"mcp-upstream-{self.name}")
        await self._ready.wait()

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10)
            except (asyncio.TimeoutError, Exception):
                self._task.cancel()
                await asyncio.gather(self._task, return_exceptions=True)

    async def call_tool(
        self, tool: str, arguments: dict[str, Any]
    ) -> mcp_types.CallToolResult:
        if self.session is None:
            raise UpstreamError(
                f"server {self.name!r} is not connected"
                + (f": {self.error}" if self.error else "")
            )
        return await self.session.call_tool(tool, arguments)
