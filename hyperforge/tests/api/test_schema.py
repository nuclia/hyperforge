from httpx import AsyncClient
from nucliadb_models.resource import KnowledgeBoxObj

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
    for key in ["alinia", "brave", "cypher", "google", "nucliadb", "perplexity", "sql"]:
        assert key in drivers
