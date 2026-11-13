from typing import Any

import mcp.types as mcp_types
from mcp.server.lowlevel import Server

from hyperforge.mcp_proxy.proxy import MCPProxy, ToolNotFound
from hyperforge.mcp_proxy.upstream import UpstreamError

DEFAULT_INSTRUCTIONS = (
    "This server proxies many tools from other MCP servers. They are not listed "
    "directly. Use `search` with a short description of what you want to do to "
    "discover tools and their input schemas, then call `execute` with the exact "
    "tool `name` from the search results and `params` matching its input schema."
)

SEARCH_TOOL = "search"
EXECUTE_TOOL = "execute"


def proxy_tools(proxy: MCPProxy) -> list[mcp_types.Tool]:
    cfg = proxy.config
    servers = ", ".join(sorted(proxy.servers)) or "none"
    return [
        mcp_types.Tool(
            name=SEARCH_TOOL,
            description=(
                "Full text search over the tools available through this proxy "
                f"(servers: {servers}). Returns matching tool names, descriptions "
                "and input schemas. An empty query lists tools."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keywords describing the capability needed.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": cfg.max_search_limit,
                        "default": cfg.default_search_limit,
                        "description": "Maximum number of tools to return.",
                    },
                },
                "required": ["query"],
            },
        ),
        mcp_types.Tool(
            name=EXECUTE_TOOL,
            description=(
                "Execute a tool discovered via `search`. `tool_name` must be the "
                "exact `name` from the search results; `params` must match its "
                "input schema."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "Fully qualified tool name from search.",
                    },
                    "params": {
                        "type": "object",
                        "description": "Arguments for the tool.",
                        "additionalProperties": True,
                    },
                },
                "required": ["tool_name"],
            },
        ),
    ]


def _error(message: str) -> mcp_types.CallToolResult:
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(type="text", text=message)], isError=True
    )


async def handle_call(
    proxy: MCPProxy, name: str, arguments: dict[str, Any]
) -> mcp_types.CallToolResult:
    if name == SEARCH_TOOL:
        result = proxy.search(
            str(arguments.get("query", "")), arguments.get("limit") or None
        )
        return mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text=result.model_dump_json())],
            structuredContent=result.model_dump(),
        )
    if name == EXECUTE_TOOL:
        params = arguments.get("params") or {}
        if not isinstance(params, dict):
            return _error("`params` must be an object")
        try:
            return await proxy.execute(str(arguments["tool_name"]), params)
        except (ToolNotFound, UpstreamError) as exc:
            return _error(str(exc))
    return _error(f"Unknown tool {name!r}")


def build_mcp_server(proxy: MCPProxy, name: str | None = None) -> Server:
    """Expose an MCPProxy as an MCP server with `search` and `execute` tools."""
    server: Server = Server(
        name or proxy.config.name,
        instructions=proxy.config.instructions or DEFAULT_INSTRUCTIONS,
    )

    @server.list_tools()
    async def _list_tools() -> list[mcp_types.Tool]:
        return proxy_tools(proxy)

    # Upstream results may not match an outputSchema; validation happens upstream.
    @server.call_tool(validate_input=True)
    async def _call_tool(name: str, arguments: dict[str, Any]):
        return await handle_call(proxy, name, arguments)

    return server
