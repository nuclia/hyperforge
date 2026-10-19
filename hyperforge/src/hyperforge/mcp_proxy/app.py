from contextlib import asynccontextmanager

from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

from hyperforge.mcp_proxy.provider import ConfigProvider
from hyperforge.mcp_proxy.proxy import MCPProxy
from hyperforge.mcp_proxy.server import build_mcp_server


def create_app(provider: ConfigProvider, mcp_path: str = "/mcp") -> Starlette:
    """Streamable HTTP app serving a single proxy instance."""
    proxy = MCPProxy(provider)
    manager = StreamableHTTPSessionManager(
        app=build_mcp_server(proxy), stateless=True, json_response=True
    )

    async def handle_mcp(scope, receive, send):
        await manager.handle_request(scope, receive, send)

    async def health(_request):
        return JSONResponse(
            {
                "status": "ok",
                "tools": len(proxy.index),
                "unavailable_servers": proxy.unavailable_servers,
            }
        )

    @asynccontextmanager
    async def lifespan(_app):
        async with proxy, manager.run():
            yield

    app = Starlette(
        routes=[
            Route("/health/ready", health),
            Mount(mcp_path, app=handle_mcp),
        ],
        lifespan=lifespan,
    )
    app.state.proxy = proxy
    return app
