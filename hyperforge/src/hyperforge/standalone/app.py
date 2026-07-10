"""
Standalone application.

Runs the API (HTTP + MCP) and the agent runner (SessionManager) in the same
process, connected via a LocalBroker.  No Redis, no gRPC, no PostgreSQL, no
NucliaDB required.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Tuple

import prometheus_client  # type: ignore
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from lru import LRU
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.streamable_http import StreamableHTTPServerTransport
from nucliadb_telemetry.logs import setup_logging
from nucliadb_telemetry.settings import LogLevel, LogSettings
from prometheus_client import CONTENT_TYPE_LATEST  # type: ignore
from redis.asyncio import Redis
from starlette.authentication import (
    AuthCredentials,
    AuthenticationBackend,
    AuthenticationError,
    BaseUser,
)
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse

from hyperforge.api import v1
from hyperforge.api.authentication import User
from hyperforge.api.models import AgentRole, StashRoles
from hyperforge.api.v1 import oauth as v1_oauth
from hyperforge.broker import Broker
from hyperforge.broker.local import LocalBroker
from hyperforge.broker.redis import RedisBroker
from hyperforge.configure import resolve_dotted_name
from hyperforge.server.cache import InMemoryCache, ValkeyCache
from hyperforge.server.session import SessionManager
from hyperforge.server.settings import Settings as ServerSettings
from hyperforge.standalone.oauth import (
    JWKSCache,
    get_enabled_mcp_auth,
    validate_mcp_bearer,
)
from hyperforge.standalone.settings import StandaloneSettings
from hyperforge.standalone.ui_router import router as ui_router

from .const import STANDALONE_ACCOUNT

# ---------------------------------------------------------------------------
# SessionManager subclass that skips the aiohttp health-check server started
# by the base class — the FastAPI app already exposes /health/ready and
# /health/alive.
# ---------------------------------------------------------------------------


class StandaloneSessionManager(SessionManager):
    """SessionManager without the embedded aiohttp metrics/health server."""

    async def initialize(self, health_check: bool = False) -> None:
        await super().initialize(health_check=health_check)


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


# ---------------------------------------------------------------------------
# Open authentication backend — grants every request all roles so that the
# existing @requires_one decorators on interaction and MCP endpoints pass
# without an external authoriser.
# ---------------------------------------------------------------------------

_ALL_ROLES = AuthCredentials(
    [
        AgentRole.MEMBER,
        StashRoles.OWNER,
        StashRoles.MEMBER,
        StashRoles.CONTRIBUTOR,
        "READER",
        "WRITER",
        "MANAGER",
    ]
)


class StandaloneAuthBackend(AuthenticationBackend):
    """Open standalone auth, with optional OAuth protection for MCP endpoints."""

    def __init__(self, agents_cfg: dict[str, Any]) -> None:
        self._agents_cfg = agents_cfg
        self._jwks_cache = JWKSCache()

    async def authenticate(
        self, conn: HTTPConnection
    ) -> tuple[AuthCredentials, BaseUser] | None:
        agent_id = self._mcp_agent_id(conn)
        if agent_id is not None:
            auth_config = get_enabled_mcp_auth(self._agents_cfg, agent_id)
            if auth_config is not None:
                conn.scope["hyperforge.mcp_auth_config"] = auth_config
                await validate_mcp_bearer(
                    conn.headers.get("authorization"),
                    auth_config,
                    self._jwks_cache,
                )

        # Inject the headers that interaction endpoints declare as required
        # FastAPI Header() dependencies, so they don't get rejected.
        if "x-stf-account" not in conn.headers:
            conn.scope["headers"] = [
                (b"x-stf-account", STANDALONE_ACCOUNT.encode()),
                (b"x-stf-user", b"standalone"),
                (b"x-stf-account-type", b"v3starter"),
                *[
                    (k, v)
                    for k, v in conn.scope["headers"]
                    if k
                    not in (
                        b"x-stf-account",
                        b"x-stf-user",
                        b"x-stf-account-type",
                    )
                ],
            ]
        return _ALL_ROLES, User(username="standalone")

    def _mcp_agent_id(self, conn: HTTPConnection) -> str | None:
        parts = conn.url.path.strip("/").split("/")
        if (
            len(parts) == 7
            and parts[:3] == ["api", "v1", "agent"]
            and parts[4] == "session"
            and parts[6] == "mcp"
        ):
            return parts[3]
        return None


def standalone_auth_error(
    conn: HTTPConnection, exc: AuthenticationError
) -> PlainTextResponse:
    authenticate = "Bearer"
    metadata_url = _mcp_resource_metadata_url(conn)
    if metadata_url is not None:
        authenticate = f'Bearer resource_metadata="{metadata_url}"'
    return PlainTextResponse(
        str(exc) or "Unauthorized",
        status_code=401,
        headers={"WWW-Authenticate": authenticate},
    )


def _mcp_resource_metadata_url(conn: HTTPConnection) -> str | None:
    auth_config = conn.scope.get("hyperforge.mcp_auth_config")
    if (
        auth_config is not None
        and auth_config.protected_resource_metadata_url is not None
    ):
        return auth_config.protected_resource_metadata_url

    parts = conn.url.path.strip("/").split("/")
    if (
        len(parts) == 7
        and parts[:3] == ["api", "v1", "agent"]
        and parts[4] == "session"
        and parts[6] == "mcp"
    ):
        agent_id = parts[3]
        session = parts[5]
        return str(
            conn.url.replace(
                scheme="https",
                path=f"/.well-known/oauth-protected-resource/api/v1/agent/{agent_id}/session/{session}/mcp",
                query="",
                fragment="",
            )
        )
    return None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


class StandaloneApplication(FastAPI):
    """Single-process arag: API + SessionManager sharing one LocalBroker."""

    _agents_cfg: dict[str, Any]

    def __init__(
        self,
        agents_cfg: dict[str, Any],
        settings: StandaloneSettings,
        **kwargs: Any,
    ) -> None:

        @asynccontextmanager
        async def lifespan(app: "StandaloneApplication"):
            await app._startup()
            yield
            await app._shutdown()

        super().__init__(
            title="arag standalone",
            description="Single-process agent RAG — interaction and MCP only",
            lifespan=lifespan,
            **kwargs,
        )
        self._agents_cfg = agents_cfg
        self._standalone_settings = settings

        # Only the interaction and MCP routes — no management endpoints.
        self.include_router(v1.interaction.router)
        self.include_router(v1.mcp_interaction.router)
        self.include_router(v1_oauth.router)
        self.include_router(router)
        self.include_router(ui_router)

        # Serve the built frontend SPA if the dist directory exists.
        # In development the Vite dev server runs separately (proxied to :8080).
        # app.py lives at: arag/src/hyperforge.standalone/app.py
        # three .parent steps → arag/
        _frontend_dist = Path(__file__).parent / "static"
        if _frontend_dist.is_dir():
            self.mount(
                "/",
                StaticFiles(directory=str(_frontend_dist), html=True),
                name="frontend",
            )

        self.add_middleware(
            AuthenticationMiddleware,
            backend=StandaloneAuthBackend(agents_cfg),
            on_error=standalone_auth_error,
        )
        self.add_middleware(
            CORSMiddleware,
            allow_credentials=True,
            allow_origins=settings.cors_allow_origin,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # ------------------------------------------------------------------
    # Property used by interaction.py / oauth.py to read answers_subject
    # and oauth_subject without knowing about StandaloneSettings.
    # ------------------------------------------------------------------

    @property
    def settings(self) -> ServerSettings:
        return self._server_settings

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _startup(self) -> None:
        s = self._standalone_settings

        setup_logging(
            settings=LogSettings(
                debug=s.debug,
                log_level=LogLevel(s.log_level),
                logger_levels={
                    "uvicorn.error": LogLevel.ERROR,
                    "mcp.client.streamable_http": LogLevel.WARNING,
                    "mcp.server.lowlevel.server": LogLevel.WARNING,
                    "hyperforge.configure": LogLevel.WARNING,
                },
            )
        )

        if s.broker_redis_dsn is None:
            # Shared in-process broker — no Redis needed.
            self.broker: Broker = LocalBroker(
                keepalive_ms=int(s.pubsub_keepalive_seconds * 1000)
            )
        else:
            self.broker = RedisBroker.from_url(
                s.broker_redis_dsn,
                s.broker_redis_activate_subject,
                int(s.pubsub_keepalive_seconds * 1000),
                cluster_mode=s.broker_redis_cluster_mode,
            )

        # LRU caches for MCP server instances (mirrors HTTPApplication).
        self.sses: LRU[Tuple[str, str], StreamableHTTPServerTransport] = LRU(size=100)
        self.mcp_servers: LRU[str, MCPServer] = LRU(size=100)

        # Agent manager backed by the JSON config — no PostgreSQL.
        agent_manager_class = resolve_dotted_name(s.agent_manager_class)
        self.agent_manager: Any = agent_manager_class(self._agents_cfg)

        # Build a ServerSettings instance so SessionManager and the subject
        # format strings work unchanged.  Redis settings are present but
        # never used (LocalBroker is injected directly).
        self._server_settings = ServerSettings(
            valkey_url="redis://localhost",
            question_timeout_seconds=s.question_timeout_seconds,
            pubsub_keepalive_seconds=s.pubsub_keepalive_seconds,
            internal_nua=s.internal_nua,
            internal_nua_api=s.internal_nua_api,
            external_nua_api_key=s.external_nua_api_key,
            local_openai=s.local_openai,
            internal_nucliadb=False,
            internal_nucliadb_url=None,
            standalone=True,
        )

        # use redis as cache backend if provided
        cache: ValkeyCache | InMemoryCache
        if s.session_cache_class is not None:
            cache_class = resolve_dotted_name(s.session_cache_class)
            cache = cache_class(s.session_cache_size)
        elif s.broker_redis_dsn is not None:
            redis_client: Redis = Redis.from_url(s.broker_redis_dsn)  # type: ignore[arg-type]
            cache = ValkeyCache(redis_client)
        else:
            cache = InMemoryCache(s.in_memory_cache_size)
        self.session_manager: SessionManager = StandaloneSessionManager(
            settings=self._server_settings,
            broker=self.broker,
            agent_manager=self.agent_manager,
            cache=cache,
        )
        await self.session_manager.initialize()

    async def _shutdown(self) -> None:
        await self.session_manager.finalize()
