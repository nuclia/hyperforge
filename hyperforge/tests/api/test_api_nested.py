import pytest
from httpx import AsyncClient
from nucliadb_models.resource import KnowledgeBoxObj

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["test"]),
    pytest.mark.asyncio,
]


async def test_arag_nested_context(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/context",
        json={
            "module": "context_conditional",
            "prompt": "If the question is about Robin Hobb",
            "then": [
                {
                    "module": "sql",
                    "description": "Check Robin Hobb books in the DB",
                    "source": "bf34f9e8-a320-4baa-8bdb-b79147a0fc1c",
                    "retries": 3,
                }
            ],
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )

    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/context",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert len(resp.json()[0]["then"]) == 1
    assert resp.json()[0]["then"][0]["source"] == "bf34f9e8-a320-4baa-8bdb-b79147a0fc1c"
