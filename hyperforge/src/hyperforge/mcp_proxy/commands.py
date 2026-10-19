import argparse
import logging

import anyio
import uvicorn
from mcp.server.stdio import stdio_server

from hyperforge.mcp_proxy.app import create_app
from hyperforge.mcp_proxy.provider import FileConfigProvider
from hyperforge.mcp_proxy.proxy import MCPProxy
from hyperforge.mcp_proxy.server import build_mcp_server
from hyperforge.mcp_proxy.settings import Settings


async def serve_stdio(provider: FileConfigProvider) -> None:
    async with MCPProxy(provider) as proxy:
        server = build_mcp_server(proxy)
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())


def run(argv: list[str] | None = None) -> None:  # pragma: no cover
    settings = Settings()
    parser = argparse.ArgumentParser(description="Progressive discovery MCP proxy")
    parser.add_argument("--config", default=settings.config_file)
    parser.add_argument(
        "--transport", choices=["stdio", "http"], default=settings.transport
    )
    parser.add_argument("--host", default=settings.http_host)
    parser.add_argument("--port", type=int, default=settings.http_port)
    args = parser.parse_args(argv)
    if not args.config:
        parser.error("--config or MCP_PROXY_CONFIG_FILE is required")

    # stdout is the protocol channel for stdio; logging goes to stderr.
    logging.basicConfig(level=settings.log_level)
    provider = FileConfigProvider(args.config)
    if args.transport == "stdio":
        anyio.run(serve_stdio, provider)
    else:
        uvicorn.run(
            create_app(provider, settings.mcp_path), host=args.host, port=args.port
        )
