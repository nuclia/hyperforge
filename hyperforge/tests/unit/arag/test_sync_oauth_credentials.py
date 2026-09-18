import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from hyperforge.db.agents import AgentManager


@pytest.mark.asyncio
async def test_upsert_sync_oauth_credentials_encrypts_payload(monkeypatch):
    database = AsyncMock()
    manager = AgentManager(database=database, settings=SimpleNamespace())  # type: ignore[arg-type]
    encrypt = Mock(return_value="ciphertext")
    monkeypatch.setattr("hyperforge.db.agents.encrypt_data", encrypt)

    await manager.upsert_sync_oauth_credentials(
        account="account",
        user_id="user",
        agent_id="agent",
        provider="sharefile_oauth",
        sync_config_id="sync-config",
        credentials={"external-connection": "secret"},
    )

    encrypt.assert_called_once_with('{"external-connection":"secret"}')
    statement = database.execute.await_args.args[0]
    assert statement.compile().params["param_1"] == "ciphertext"
    assert "secret" not in str(statement.compile().params)


@pytest.mark.asyncio
async def test_get_sync_oauth_credentials_is_scoped_and_decrypts(monkeypatch):
    database = AsyncMock()
    database.fetch_one.return_value = {"encrypted_credentials": "ciphertext"}
    manager = AgentManager(database=database, settings=SimpleNamespace())  # type: ignore[arg-type]
    monkeypatch.setattr(
        "hyperforge.db.agents.decrypt_data",
        lambda value: json.dumps({"external-connection": "secret"}),
    )

    result = await manager.get_sync_oauth_credentials(
        account="account",
        user_id="user",
        agent_id="agent",
        provider="sharefile_oauth",
        sync_config_id="sync-config",
    )

    assert result == {"external-connection": "secret"}
    params = database.fetch_one.await_args.args[0].compile().params
    assert set(params.values()) == {
        "account",
        "user",
        "agent",
        "sharefile_oauth",
        "sync-config",
    }


@pytest.mark.asyncio
async def test_get_sync_oauth_credentials_rejects_invalid_payload(monkeypatch):
    database = AsyncMock()
    database.fetch_one.return_value = {"encrypted_credentials": "ciphertext"}
    manager = AgentManager(database=database, settings=SimpleNamespace())  # type: ignore[arg-type]
    monkeypatch.setattr("hyperforge.db.agents.decrypt_data", lambda value: "secret")

    with pytest.raises(ValueError, match="Invalid stored Sync OAuth credentials"):
        await manager.get_sync_oauth_credentials(
            account="account",
            user_id="user",
            agent_id="agent",
            provider="sharefile_oauth",
            sync_config_id="sync-config",
        )
