from httpx import ASGITransport, AsyncClient
from nucliadb_agentic_api.app import HTTPApplication
from nucliadb_agentic_api.db.agentic_configs import AgenticConfigs
from nucliadb_agentic_api.db.settings import DataManagerSettings
from nucliadb_agentic_api.settings import Settings
from nucliadb_models.resource import NucliaDBRoles


class FakeAgenticConfigs:
    def __init__(self):
        self.configs = {}

    async def initialize(self):
        pass

    async def finalize(self):
        pass

    async def create_agentic_config(self, account, kbid, agentic_id, config):
        key = (account, kbid, agentic_id)
        if key in self.configs:
            from nucliadb_agentic_api import exceptions

            raise exceptions.Conflict("Agentic configuration already exists")
        self.configs[key] = config

    async def get_agentic_config(self, account, kbid, agentic_id):
        key = (account, kbid, agentic_id)
        if key not in self.configs:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Agentic configuration not found")
        return self.configs[key]

    async def list_agentic_configs(self, account, kbid):
        return {
            agentic_id: config
            for (
                stored_account,
                stored_kbid,
                agentic_id,
            ), config in self.configs.items()
            if stored_account == account and stored_kbid == kbid
        }

    async def patch_agentic_config(self, account, kbid, agentic_id, config):
        key = (account, kbid, agentic_id)
        if key not in self.configs:
            from nucliadb_agentic_api import exceptions

            raise exceptions.NotFound("Agentic configuration not found")
        self.configs[key] = config


def create_api_client(application, roles: list[NucliaDBRoles]) -> AsyncClient:
    client = AsyncClient(
        transport=ASGITransport(app=application), base_url="http://test/api/v1"
    )
    client.headers["X-NUCLIADB-ROLES"] = ";".join([role.value for role in roles])
    client.headers["X-NUCLIADB-ACCOUNT"] = "account"
    return client


async def create_app(monkeypatch) -> HTTPApplication:
    fake_configs = FakeAgenticConfigs()

    async def from_settings(cls, settings):
        return fake_configs

    async def startup(self):
        self.agent_manager = await AgenticConfigs.from_settings(
            settings=self.data_manager_settings
        )
        await self.agent_manager.initialize()

    async def shutdown(self):
        await self.agent_manager.finalize()

    monkeypatch.setattr(AgenticConfigs, "from_settings", classmethod(from_settings))
    monkeypatch.setattr(HTTPApplication, "startup", startup)
    monkeypatch.setattr(HTTPApplication, "shutdown", shutdown)

    app = HTTPApplication(
        settings=Settings(running_environment="test", valkey_cluster_mode=False),
        data_manager_settings=DataManagerSettings(postgresql_dsn="postgresql://test"),
    )
    await app.startup()
    return app


async def test_agentic_config_crud(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(
        app, [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    ) as client:
        payload = {
            "title": "Support agent",
            "config": {
                "smart_agent": {
                    "mode": "reactive",
                    "sources": [{"type": "nucliadb", "description": "Current KB"}],
                },
                "summarize": {"conversational": True},
            },
        }

        resp = await client.post("/kb/kb/agentic_configs/support", json=payload)
        assert resp.status_code == 201, resp.text

        resp = await client.get("/kb/kb/agentic_configs/support")
        assert resp.status_code == 200, resp.text
        assert resp.json() == payload

        resp = await client.get("/kb/kb/agentic_configs")
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"support": payload}

        updated_payload = {
            "title": "Updated support agent",
            "config": {"summarize": {"conversational": False}},
        }
        resp = await client.patch(
            "/kb/kb/agentic_configs/support", json=updated_payload
        )
        assert resp.status_code == 204, resp.text

        resp = await client.get("/kb/kb/agentic_configs/support")
        assert resp.status_code == 200, resp.text
        assert resp.json() == updated_payload


async def test_agentic_config_not_found(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.get("/kb/kb/agentic_configs/missing")
        assert resp.status_code == 404


async def test_agentic_config_requires_manager(monkeypatch):
    app = await create_app(monkeypatch)
    async with create_api_client(app, [NucliaDBRoles.READER]) as client:
        resp = await client.post(
            "/kb/kb/agentic_configs/support",
            json={"title": "Support agent", "config": {"summarize": {}}},
        )
        assert resp.status_code == 403
