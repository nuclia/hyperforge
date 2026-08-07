import socket
from unittest.mock import AsyncMock

import pytest

from hyperforge.a2a.server import build_grpc_server_from_runtime
from hyperforge.broker.redis import RedisBroker
from hyperforge.server.settings import Settings as ServerSettings
from hyperforge.standalone.agent import StaticAgentManager
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.config import StandAloneAgentConfig, WorkflowConfig
from hyperforge.standalone.settings import StandaloneSettings


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_standalone_a2a_requires_redis():
    settings = StandaloneSettings(a2a_enabled=True)

    with pytest.raises(ValueError, match="A2A_ENABLED requires BROKER_REDIS_DSN"):
        settings.a2a_settings()


def test_private_network_endpoint_defaults_are_environment_safe():
    assert ServerSettings().allow_private_network_endpoints is False
    assert StandaloneSettings().allow_private_network_endpoints is True
    assert (
        StandaloneSettings(
            allow_private_network_endpoints=False
        ).allow_private_network_endpoints
        is False
    )


def test_standalone_a2a_settings_reuse_shared_validation():
    settings = StandaloneSettings(
        a2a_enabled=True,
        broker_redis_dsn="redis://localhost",
        a2a_account="festival",
        a2a_agent_id="venue",
        a2a_grpc_host="127.0.0.1",
    )

    a2a_settings = settings.a2a_settings()

    assert a2a_settings.valkey_url == "redis://localhost"
    assert a2a_settings.a2a_account == "festival"
    assert a2a_settings.a2a_agent_id == "venue"


@pytest.mark.asyncio
async def test_standalone_shutdown_delegates_resource_cleanup_to_session_manager():
    app = StandaloneApplication.__new__(StandaloneApplication)
    app.a2a_server = AsyncMock()
    app.session_manager = AsyncMock()

    await app._shutdown()

    app.a2a_server.stop.assert_awaited_once_with(grace=5)
    app.session_manager.finalize.assert_awaited_once()


@pytest.mark.asyncio
async def test_static_agent_manager_starts_shared_a2a_server():
    port = _free_port()
    settings = StandaloneSettings(
        a2a_enabled=True,
        broker_redis_dsn="redis://127.0.0.1:1",
        a2a_account="local",
        a2a_agent_id="venue",
        a2a_grpc_host="127.0.0.1",
        a2a_grpc_port=port,
    )
    manager = StaticAgentManager(
        {
            "venue": StandAloneAgentConfig(
                workflows={"default": WorkflowConfig(name="Venue operations")}
            )
        }
    )
    broker = RedisBroker.from_url(
        settings.broker_redis_dsn,
        settings.broker_redis_activate_subject,
        int(settings.pubsub_keepalive_seconds * 1000),
    )

    server = await build_grpc_server_from_runtime(
        settings.a2a_settings(), manager, broker
    )
    await server.start()
    try:
        assert server
    finally:
        await server.stop(grace=1)
        await broker.finalize()
