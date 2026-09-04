from unittest.mock import AsyncMock, Mock

import pytest
from a2a.types import a2a_pb2

from hyperforge.a2a.server import build_grpc_server, build_grpc_server_from_runtime
from hyperforge.a2a.settings import A2ASettings


async def test_build_grpc_server_finalizes_broker_when_manager_creation_fails(
    monkeypatch,
):
    broker = AsyncMock()
    monkeypatch.setattr(
        "hyperforge.a2a.server.RedisBroker.from_url", Mock(return_value=broker)
    )
    monkeypatch.setattr(
        "hyperforge.a2a.server.AgentManager.from_settings",
        AsyncMock(side_effect=RuntimeError("manager unavailable")),
    )

    with pytest.raises(RuntimeError, match="manager unavailable"):
        await build_grpc_server(
            A2ASettings(a2a_account="account", a2a_agent_id="agent"), Mock()
        )

    broker.finalize.assert_awaited_once()


async def test_authenticated_server_registers_auth_interceptor(monkeypatch):
    grpc_server = Mock()
    grpc_server.add_insecure_port.return_value = 8034
    server_factory = Mock(return_value=grpc_server)
    monkeypatch.setattr("hyperforge.a2a.server.grpc.aio.server", server_factory)
    monkeypatch.setattr(
        "hyperforge.a2a.server.build_agent_skills",
        AsyncMock(
            return_value=[
                a2a_pb2.AgentSkill(
                    id="workflow",
                    name="Workflow",
                    description="Workflow",
                    tags=["test"],
                )
            ]
        ),
    )
    monkeypatch.setattr("hyperforge.a2a.server.GrpcHandler", Mock())
    monkeypatch.setattr(
        "hyperforge.a2a.server.a2a_pb2_grpc.add_A2AServiceServicer_to_server",
        Mock(),
    )
    broker = Mock(client=Mock())

    result = await build_grpc_server_from_runtime(
        A2ASettings(
            a2a_account="account",
            a2a_agent_id="agent",
            a2a_grpc_host="127.0.0.1",
            a2a_auth_enabled=True,
            a2a_authorizer_url="http://authorizer",
        ),
        Mock(),
        broker,
    )

    assert result is grpc_server
    assert len(server_factory.call_args.kwargs["interceptors"]) == 1
