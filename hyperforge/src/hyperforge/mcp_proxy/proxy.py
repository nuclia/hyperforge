import asyncio
import logging
from typing import Any

import mcp.types as mcp_types
from pydantic import BaseModel

from hyperforge.mcp_proxy.config import ProxyConfig
from hyperforge.mcp_proxy.index import IndexedTool, ToolIndex
from hyperforge.mcp_proxy.provider import ConfigProvider
from hyperforge.mcp_proxy.upstream import UpstreamServer

logger = logging.getLogger("hyperforge.mcp_proxy")


class ToolMatch(BaseModel):
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]
    score: float


class SearchResult(BaseModel):
    query: str
    total_tools: int
    results: list[ToolMatch]
    unavailable_servers: dict[str, str] = {}


class ToolNotFound(Exception):
    def __init__(self, tool_name: str, suggestions: list[str]):
        self.tool_name = tool_name
        self.suggestions = suggestions
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        super().__init__(
            f"Unknown tool {tool_name!r}. Use the search tool to discover tools.{hint}"
        )


class MCPProxy:
    """Progressive-discovery proxy over a set of upstream MCP servers.

    Transport agnostic: it can be served over MCP (see `server.py`) or
    embedded in another application which supplies its own ConfigProvider.
    """

    def __init__(self, provider: ConfigProvider):
        self.provider = provider
        self.config = ProxyConfig()
        self.servers: dict[str, UpstreamServer] = {}
        self.index = ToolIndex()
        self._tools: dict[str, IndexedTool] = {}
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        async with self._lock:
            self.config = await self.provider.get_config()
            servers = [UpstreamServer(cfg) for cfg in self.config.enabled_servers]
            await asyncio.gather(*(s.start() for s in servers))
            self.servers = {s.name: s for s in servers}
            self._build_index()

    async def stop(self) -> None:
        async with self._lock:
            await asyncio.gather(*(s.stop() for s in self.servers.values()))
            self.servers = {}
            self._build_index()

    async def reload(self) -> None:
        await self.stop()
        await self.start()

    async def __aenter__(self) -> "MCPProxy":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    def qualified_name(self, server: str, tool: str) -> str:
        return f"{server}{self.config.tool_name_separator}{tool}"

    def _build_index(self) -> None:
        tools: list[IndexedTool] = []
        for server in self.servers.values():
            for tool in server.tools:
                tools.append(
                    IndexedTool(
                        name=self.qualified_name(server.name, tool.name),
                        server=server.name,
                        tool=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                        server_description=server.config.description,
                        tags=server.config.tags,
                    )
                )
        self.index.build(tools)
        self._tools = {t.name: t for t in tools}
        logger.info(
            "Indexed %d tools from %d MCP servers", len(tools), len(self.servers)
        )

    @property
    def unavailable_servers(self) -> dict[str, str]:
        return {
            name: s.error for name, s in self.servers.items() if s.error is not None
        }

    def search(self, query: str, limit: int | None = None) -> SearchResult:
        limit = min(
            limit or self.config.default_search_limit, self.config.max_search_limit
        )
        matches = [
            ToolMatch(
                name=tool.name,
                server=tool.server,
                description=tool.description,
                input_schema=tool.input_schema,
                score=round(score, 4),
            )
            for tool, score in self.index.search(query, max(limit, 1))
        ]
        return SearchResult(
            query=query,
            total_tools=len(self.index),
            results=matches,
            unavailable_servers=self.unavailable_servers,
        )

    async def execute(
        self, tool_name: str, params: dict[str, Any] | None = None
    ) -> mcp_types.CallToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            suggestions = [t.name for t, _ in self.index.search(tool_name, 3)]
            raise ToolNotFound(tool_name, suggestions)
        return await self.servers[tool.server].call_tool(tool.tool, params or {})
