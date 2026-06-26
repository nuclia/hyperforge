import asyncio
from importlib.metadata import version

import sentry_sdk
from nucliadb_telemetry.fastapi import application_metrics
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings
from sentry_sdk.integrations.excepthook import ExcepthookIntegration

from hyperforge_restricted import sandbox
from hyperforge.server.settings import Settings


def set_sentry(zone: str, environment: str, sentry_url: str):
    sentry_exception = ExcepthookIntegration(always_run=True)
    sentry_sdk.init(
        release=version("hyperforge"),
        environment=environment,
        dsn=sentry_url,
        integrations=[sentry_exception],
    )
    sentry_sdk.set_tag("zone", zone)


async def run_metrics_server(port: int):
    import uvicorn

    config = uvicorn.Config(
        application_metrics, host="0.0.0.0", port=port, log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


def run():  # pragma: no cover
    settings = Settings()
    sandbox_settings = sandbox.SandboxSettings()
    setup_logging(
        settings=LogSettings(
            debug=settings.debug,
            log_level=LogLevel(settings.log_level),
            logger_levels={
                "uvicorn.error": LogLevel.ERROR,
                "nucliadb_telemetry": LogLevel.ERROR,
                "mcp.client.streamable_http": LogLevel.WARNING,
                "mcp.server.lowlevel.server": LogLevel.WARNING,
                "hyperforge.configure": LogLevel.WARNING,
            },
        )
    )

    loop = asyncio.get_event_loop()

    loop.create_task(run_metrics_server(sandbox_settings.sandbox_metrics_port))
    loop.create_task(sandbox.run_sandbox_server())
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
