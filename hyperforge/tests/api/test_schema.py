from httpx import AsyncClient
from nucliadb_models.resource import KnowledgeBoxObj

from hyperforge.llm_config import llm_defaults

HEADERS = {
    "X-STF-USER": "user1",
    "X-STF-ACCOUNT": "nuclia",
    "X-STF-ACCOUNT-TYPE": "basic",
    "X-STF-ROLES": "SOWNER",
}


async def test_arag_schema(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/schema",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    assert "agents" in resp.json()
    assert "drivers" in resp.json()


async def test_arag_new_schema(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/schema",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    agents = resp.json().get("agents", {})
    drivers = resp.json().get("drivers", {})
    for key in ["preprocess", "context", "generation", "postprocess"]:
        assert key in agents
    for key in ["nucliadb", "sync"]:
        assert key in drivers

    summarize_model_schema = agents["generation"]["summarize"]["config_schema"][
        "properties"
    ]["model"]
    assert summarize_model_schema["default"]["model_id"] == llm_defaults.default
