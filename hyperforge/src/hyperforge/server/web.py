import asyncio
import os

import prometheus_client
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.routing import Route

from hyperforge.server import logger

PORT = os.environ.get("HEALTH_CHECK_PORT", "8000")


async def metrics(request: Request) -> Response:
    output = prometheus_client.exposition.generate_latest()
    return PlainTextResponse(output.decode("utf8"))


async def health(request: Request) -> Response:
    return PlainTextResponse("OK")


async def not_found(request: Request, exc: Exception) -> Response:
    return PlainTextResponse("OK", status_code=404)


app = Starlette(
    routes=[
        Route("/metrics", metrics),
        Route("/health/alive", health),
        Route("/health/ready", health),
    ],
    exception_handlers={404: not_found},
)


class WebServer:
    """Wraps a uvicorn server running as an asyncio task in the current event loop."""

    def __init__(self, server: uvicorn.Server) -> None:
        self._server = server
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.get_event_loop().create_task(self._server.serve())

    async def shutdown(self) -> None:
        self._server.should_exit = True
        if self._task is not None:
            await self._task
            self._task = None


async def start_web_server() -> WebServer:
    config = uvicorn.Config(app, host="0.0.0.0", port=int(PORT), log_level="warning")
    server = uvicorn.Server(config)
    web = WebServer(server)
    web.start()
    logger.info(f"======= Serving on http://0.0.0.0:{PORT}/ ======")
    return web


async def start_health_check() -> WebServer:
    return await start_web_server()
