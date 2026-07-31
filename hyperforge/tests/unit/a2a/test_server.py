from unittest.mock import AsyncMock, Mock

import pytest

from hyperforge.a2a.server import build_grpc_server
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
