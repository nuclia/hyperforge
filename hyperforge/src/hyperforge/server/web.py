import prometheus_client  # type: ignore
from aiohttp import web

from hyperforge.server import logger


async def http_handler(request: web.Request):
    if request.path == "/metrics":
        output = prometheus_client.exposition.generate_latest()
        return web.Response(text=output.decode("utf8"))
    elif request.path in ("/health/alive", "/health/ready"):
        # implement health check here
        return web.Response(text="OK")
    else:
        return web.Response(text="OK", status=404)


async def start_web_server() -> web.Server:
    server = web.Server(http_handler)  # type: ignore
    runner = web.ServerRunner(server)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8000)
    await site.start()

    logger.info("======= Serving on http://0.0.0.0:8000/ ======")
    return server


async def start_health_check():
    server = await start_web_server()
    return server
