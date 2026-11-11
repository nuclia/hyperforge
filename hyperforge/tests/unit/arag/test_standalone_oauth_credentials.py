import pytest

from hyperforge.standalone.agent import StaticAgentManager


@pytest.mark.parametrize("provider", ["sharefile_oauth", "google_oauth", "azure_oauth"])
async def test_standalone_oauth_credentials_are_encrypted_and_scoped(monkeypatch, provider):
    monkeypatch.delenv("ENCRYPTION_SECRET_KEY", raising=False)
    manager = StaticAgentManager({})

    await manager.upsert_sync_oauth_credentials(
        account="local",
        user_id="user",
        agent_id="agent",
        provider=provider,
        sync_config_id="sync-config",
        credentials={"connection": "secret"},
    )

    encrypted = next(iter(manager._oauth_credentials.values()))
    assert "secret" not in encrypted
    assert await manager.get_sync_oauth_credentials(
        account="local",
        user_id="user",
        agent_id="agent",
        provider=provider,
        sync_config_id="sync-config",
    ) == {"connection": "secret"}
    assert (
        await manager.get_sync_oauth_credentials(
            account="local",
            user_id="another-user",
            agent_id="agent",
            provider=provider,
            sync_config_id="sync-config",
        )
        is None
    )
