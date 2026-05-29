from typing import Tuple

import prometheus_client  # type: ignore
from fastapi import APIRouter, FastAPI
from grpc import aio  # type: ignore
from hyperforge.api import internal, logger
from hyperforge.api.authentication import RaoAuthenticationBackend
from hyperforge.api.logging import set_sentry
from hyperforge.broker import Broker
from hyperforge.broker.redis import RedisBroker
from hyperforge.configure import GLOBAL_REGISTRY, load_all_configurations, scan
from hyperforge.db.agents import AgentManager
from hyperforge.db.settings import DataManagerSettings
from hyperforge.feature_flag import get_flag_service
from lru import LRU
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import (
    StreamableHTTPServerTransport,
)
from nucliadb_sdk.v2.sdk import NucliaDBAsync
from nucliadb_telemetry.grpc import OpenTelemetryGRPC
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings
from nucliadb_telemetry.utils import clean_telemetry, setup_telemetry
from prometheus_client import CONTENT_TYPE_LATEST  # type: ignore
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.responses import PlainTextResponse

from nucliadb_agentic_api import SERVICE_NAME, v1
from nucliadb_agentic_api.ask.audit import (
    AuditMiddleware,
    start_audit_utility,
    stop_audit_utility,
)
from nucliadb_agentic_api.ask.predict import (
    start_predict_engine,
    stop_predict_engine,
)
from nucliadb_agentic_api.settings import Settings

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
    server: aio.Server
    broker: Broker

    def __init__(
        self,
        settings: Settings,
        data_manager_settings: DataManagerSettings,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.settings = settings
        self.data_manager_settings = data_manager_settings
        self.include_router(internal.router)
        self.include_router(v1.router)
        self.include_router(router)
        self.add_middleware(
            AuthenticationMiddleware,
            backend=RaoAuthenticationBackend(),
        )
        self.add_middleware(AuditMiddleware)
        self.add_event_handler("startup", self.startup)
        self.add_event_handler("shutdown", self.shutdown)

    async def startup(self) -> None:
        GLOBAL_REGISTRY.clear()
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

        tracer_provider = await setup_telemetry(SERVICE_NAME)

        if tracer_provider is not None:
            telemetry_grpc = OpenTelemetryGRPC(SERVICE_NAME, tracer_provider)

            self.idp_regional_channel = telemetry_grpc.init_client(
                self.settings.idp_regional_grpc
            )
        else:
            self.idp_regional_channel = aio.insecure_channel(
                self.settings.idp_regional_grpc
            )

        if self.settings.memory_apikey_nucliadb is None:
            api_key = None
            headers = {"X-NUCLIADB-ROLES": "WRITER;READER"}
        else:
            api_key = self.settings.memory_apikey_nucliadb
            headers = None

        await start_predict_engine()
        await start_audit_utility(SERVICE_NAME)

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
        await stop_audit_utility()
        await stop_predict_engine()
        await clean_telemetry(SERVICE_NAME)
        GLOBAL_REGISTRY.clear()
