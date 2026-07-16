from contextlib import asynccontextmanager
from typing import Any, Optional, Tuple

import prometheus_client
from fastapi import APIRouter, FastAPI
from lru import LRU
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import (
    StreamableHTTPServerTransport,
)
from nucliadb_sdk.v2.sdk import NucliaDBAsync
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings
from nucliadb_telemetry.utils import clean_telemetry, setup_telemetry
from prometheus_client import CONTENT_TYPE_LATEST
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse

from hyperforge.api import SERVICE_NAME, internal, logger, v1
from hyperforge.api.authentication import RaoAuthenticationBackend
from hyperforge.api.logging import set_sentry
from hyperforge.api.settings import Settings
from hyperforge.broker import Broker
from hyperforge.broker.redis import RedisBroker
from hyperforge.configure import GLOBAL_REGISTRY, load_all_configurations, scan
from hyperforge.db.agents import AgentManager
from hyperforge.db.settings import DataManagerSettings
from hyperforge.feature_flag import get_flag_service

router = APIRouter()


@router.get("/metrics")
async def serve_metrics():  # pragma: no cover
    output = prometheus_client.exposition.generate_latest()
    return PlainTextResponse(
        output.decode("utf8"), headers={"Content-Type": CONTENT_TYPE_LATEST}
    )


@router.get("/health/ready")
async def health_ready():
    return {"status": "ok"}


@router.get("/health/alive")
async def health_alive():
    return {"status": "ok"}


class HTTPApplication(FastAPI):
    agent_manager: AgentManager
    arag_search: NucliaDBAsync
    arag_writer: NucliaDBAsync
    arag_reader: NucliaDBAsync
    broker: Broker
    _agents_cfg: dict[str, Any]
    extra_middlewares: Optional[list[Any]] = None

    def __init__(
        self,
        settings: Settings,
        data_manager_settings: DataManagerSettings,
        *args,
        **kwargs,
    ):
        @asynccontextmanager
        async def lifespan(app: "HTTPApplication"):
            await app.startup()
            yield
            await app.shutdown()

        super().__init__(*args, lifespan=lifespan, **kwargs)
        self.settings = settings
        self.data_manager_settings = data_manager_settings
        self._agents_cfg = {}
        self.include_router(internal.router)
        self.include_router(v1.router)
        self.include_router(router)
        self.add_middleware(
            AuthenticationMiddleware,
            backend=RaoAuthenticationBackend(),
        )

        if self.extra_middlewares is not None:
            for extra_middleware in self.extra_middlewares:
                self.add_middleware(extra_middleware)

    async def startup(self) -> None:
        GLOBAL_REGISTRY.clear()
        await setup_telemetry(SERVICE_NAME)
        setup_logging(
            settings=LogSettings(
                debug=self.settings.debug,
                log_level=LogLevel(self.settings.log_level),
                logger_levels={
                    "uvicorn.error": LogLevel.ERROR,
                    "nucliadb_telemetry": LogLevel.ERROR,
                    "mcp.client.streamable_http": LogLevel.WARNING,
                    "mcp.server.lowlevel.server": LogLevel.WARNING,
                    "hyperforge.configure": LogLevel.WARNING,
                },
            )
        )
        if self.settings.sentry_url is not None:
            set_sentry(
                self.settings.zone,
                self.settings.running_environment,
                self.settings.sentry_url,
            )

        get_flag_service()  # precache the flag service

        if self.settings.memory_apikey_nucliadb is None:
            api_key = None
            headers = {"X-NUCLIADB-ROLES": "WRITER;READER"}
        else:
            api_key = self.settings.memory_apikey_nucliadb
            headers = None

        self.arag_writer = NucliaDBAsync(
            url=self.settings.memory_writer_nucliadb,
            api_key=api_key,
            headers=headers,
        )
        self.arag_reader = NucliaDBAsync(
            url=self.settings.memory_reader_nucliadb,
            api_key=api_key,
            headers=headers,
        )
        self.arag_search = NucliaDBAsync(
            url=self.settings.memory_search_nucliadb,
            api_key=api_key,
            headers=headers,
        )

        self.broker = RedisBroker.from_url(
            url=self.settings.valkey_url,
            activate_subject=self.settings.activate_subject,
            keepalive_ms=int(self.settings.pubsub_keepalive_seconds * 1000),
            cluster_mode=self.settings.valkey_cluster_mode,
        )

        self.sses: LRU[Tuple[str, str], StreamableHTTPServerTransport] = LRU(size=100)
        self.mcp_servers: LRU[str, MCPServer] = LRU(size=100)

        self.agent_manager = await AgentManager.from_settings(
            settings=self.data_manager_settings
        )
        await self.agent_manager.initialize()

        for load_module in self.settings.load_modules:
            try:
                scan(load_module)
                load_all_configurations(load_module)
            except ImportError:
                logger.error(f"Module {load_module} could not be loaded")

    async def shutdown(self) -> None:
        await self.agent_manager.finalize()
        await self.broker.finalize()
        await clean_telemetry(SERVICE_NAME)
        GLOBAL_REGISTRY.clear()
