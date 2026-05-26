import asyncio
from importlib.metadata import version
from typing import Optional

import sentry_sdk
from nucliadb_telemetry.fastapi import application_metrics
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings
from nucliadb_telemetry.tracerprovider import AsyncTracerProvider
from nucliadb_telemetry.utils import get_telemetry, setup_telemetry
from sentry_sdk.integrations.excepthook import ExcepthookIntegration

from hyperforge.broker.redis import RedisBroker
from hyperforge.db.agents import AgentManager
from hyperforge.db.settings import DataManagerSettings
from hyperforge.server import SERVICE_NAME
from hyperforge.server.cache import ValkeyCache
from hyperforge.server.session import SessionManager
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


async def run_server(
    settings: Settings,
    tracer: Optional[AsyncTracerProvider],
    data_manager_settings: DataManagerSettings,
) -> SessionManager:
    if tracer:
        await setup_telemetry(SERVICE_NAME)
    # Connect to Valkey
    broker = RedisBroker.from_url(
        url=settings.valkey_url,
        activate_subject=settings.activate_subject,
        keepalive_ms=int(settings.pubsub_keepalive_seconds * 1000),
        cluster_mode=settings.valkey_cluster_mode,
    )

    agent_manager = await AgentManager.from_settings(
        settings=data_manager_settings,
    )
    await agent_manager.initialize()

    session = SessionManager(
        settings=settings,
        broker=broker,
        agent_manager=agent_manager,
        cache=ValkeyCache(broker._client),
    )

    return session


def run():  # pragma: no cover
    settings = Settings()
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
    data_manager_settings = DataManagerSettings()
    tracer = get_telemetry("nuclia-arag-server")
    if settings.sentry_url is not None:
        set_sentry(
            settings.zone,
            settings.running_environment,
            settings.sentry_url,
        )
    loop = asyncio.get_event_loop()

    loop.create_task(run_metrics_server(settings.metrics_port))

    session = loop.run_until_complete(
        run_server(settings, tracer, data_manager_settings)
    )
    loop.run_until_complete(session.initialize())
    try:
        loop.run_forever()
    finally:
        loop.run_until_complete(session.finalize())
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
