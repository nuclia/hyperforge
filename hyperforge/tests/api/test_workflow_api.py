import datetime

import pytest
from httpx import AsyncClient
from hyperforge.db.agents import (
    WORKFLOW_PURGE_RETENTION,
    retrieval_agent_generation,
    retrieval_agent_workflow,
)
from hyperforge.db.workflow_cleanup import cleanup_deleted_workflows
from nucliadb_models.resource import KnowledgeBoxObj

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["test"]),
    pytest.mark.asyncio,
]

WORKFLOW_ID = "default"
HEADERS = {
    "X-STF-USER": "user1",
    "X-STF-ACCOUNT": "nuclia",
    "X-STF-ACCOUNT-TYPE": "basic",
    "X-STF-ROLES": "SOWNER",
}


async def create_workflow(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient, workflow_id: str
):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflows",
        json={
            "id": workflow_id,
            "name": workflow_id,
            "description": "Workflow created for tests",
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200


async def test_arag_workflow_delete(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient, arag_api_app
):
    workflow_id = "delete-test-workflow"

    await create_workflow(arag_kb, arag_api, workflow_id)

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflows",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert workflow_id not in {workflow["id"] for workflow in resp.json()}

    row = await arag_api_app.agent_manager.database.fetch_one(
        retrieval_agent_workflow.select()
        .where(retrieval_agent_workflow.c.account == HEADERS["X-STF-ACCOUNT"])
        .where(retrieval_agent_workflow.c.agent_id == arag_kb.uuid)
        .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
    )
    assert row is not None
    assert row["is_deleted"] is True
    assert row["deleted_by"] == HEADERS["X-STF-USER"]
    assert row["deleted_at"] is not None


async def test_arag_workflow_deleted_is_not_actionable(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    workflow_id = "deleted-action-test-workflow"
    await create_workflow(arag_kb, arag_api, workflow_id)

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}",
        json={
            "name": "Updated deleted workflow",
            "description": "Should not update",
            "parameters": {},
            "required": [],
        },
        headers=HEADERS,
    )
    assert resp.status_code == 404

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}/rules",
        headers=HEADERS,
    )
    assert resp.status_code == 404

    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}/generation",
        json={"module": "summarize", "prompt": "Summary"},
        headers=HEADERS,
    )
    assert resp.status_code == 404


async def test_arag_workflow_cleanup_purges_expired_deleted_workflows(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient, arag_api_app
):
    workflow_id = "cleanup-test-workflow"
    await create_workflow(arag_kb, arag_api, workflow_id)

    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}/generation",
        json={"module": "summarize", "prompt": "Summary"},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    deleted_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=16
    )
    await arag_api_app.agent_manager.database.execute(
        retrieval_agent_workflow.update()
        .where(retrieval_agent_workflow.c.account == HEADERS["X-STF-ACCOUNT"])
        .where(retrieval_agent_workflow.c.agent_id == arag_kb.uuid)
        .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
        .values(deleted_at=deleted_at)
    )

    await cleanup_deleted_workflows(
        arag_api_app.agent_manager, older_than=WORKFLOW_PURGE_RETENTION
    )

    workflow = await arag_api_app.agent_manager.database.fetch_one(
        retrieval_agent_workflow.select()
        .where(retrieval_agent_workflow.c.account == HEADERS["X-STF-ACCOUNT"])
        .where(retrieval_agent_workflow.c.agent_id == arag_kb.uuid)
        .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
    )
    assert workflow is None

    generation = await arag_api_app.agent_manager.database.fetch_one(
        retrieval_agent_generation.select()
        .where(retrieval_agent_generation.c.account == HEADERS["X-STF-ACCOUNT"])
        .where(retrieval_agent_generation.c.agent_id == arag_kb.uuid)
        .where(retrieval_agent_generation.c.workflow_id == workflow_id)
    )
    assert generation is None


async def test_arag_workflow_cleanup_keeps_recent_deleted_workflows(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient, arag_api_app
):
    workflow_id = "cleanup-recent-test-workflow"
    await create_workflow(arag_kb, arag_api, workflow_id)

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    deleted_at = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        days=14
    )
    await arag_api_app.agent_manager.database.execute(
        retrieval_agent_workflow.update()
        .where(retrieval_agent_workflow.c.account == HEADERS["X-STF-ACCOUNT"])
        .where(retrieval_agent_workflow.c.agent_id == arag_kb.uuid)
        .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
        .values(deleted_at=deleted_at)
    )

    await cleanup_deleted_workflows(
        arag_api_app.agent_manager, older_than=WORKFLOW_PURGE_RETENTION
    )

    workflow = await arag_api_app.agent_manager.database.fetch_one(
        retrieval_agent_workflow.select()
        .where(retrieval_agent_workflow.c.account == HEADERS["X-STF-ACCOUNT"])
        .where(retrieval_agent_workflow.c.agent_id == arag_kb.uuid)
        .where(retrieval_agent_workflow.c.workflow_id == workflow_id)
    )
    assert workflow is not None
    assert workflow["is_deleted"] is True


async def test_arag_workflow_delete_default_is_rejected(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}",
        headers=HEADERS,
    )
    assert resp.status_code == 409

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflows",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert WORKFLOW_ID in {workflow["id"] for workflow in resp.json()}


async def test_arag_workflow_delete_not_found(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/missing-workflow",
        headers=HEADERS,
    )
    assert resp.status_code == 404


async def test_arag_workflow_rules(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/rules",
        json={"rules": ["rule1", "rule2", {"prompt": "rule3"}]},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/rules",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert {"rules": ["rule1", "rule2", {"prompt": "rule3"}]} == resp.json()


async def test_arag_workflow_preprocess(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/preprocess",
        json={"module": "rephrase", "title": "brave", "rules": []},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/preprocess",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["title"] == "brave"
    assert resp.json()[0]["all"] is False

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/preprocess/{uuid}",
        json={"module": "rephrase", "title": "brave2", "rules": []},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/preprocess",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["title"] == "brave2"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/preprocess/{uuid}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/preprocess",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_arag_workflow_generation(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/generation",
        json={"module": "summarize", "prompt": "Summary"},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/generation",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["prompt"] == "Summary"

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/generation/{uuid}",
        json={"module": "summarize", "prompt": "Summary 2"},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/generation",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["prompt"] == "Summary 2"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/generation/{uuid}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/generation",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_arag_workflow_postprocess(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/postprocess",
        json={"module": "external", "prompt": "External", "url": "http://example.com"},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/postprocess",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["prompt"] == "External"

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/postprocess/{uuid}",
        json={
            "module": "external",
            "prompt": "External 2",
            "url": "http://example.com",
        },
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/postprocess",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["prompt"] == "External 2"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/postprocess/{uuid}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/postprocess",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_arag_workflow_context(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context",
        json={"module": "restricted", "code": "print('hola')"},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["code"] == "print('hola')"

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context/{uuid}",
        json={"module": "restricted", "code": "print('hola2')"},
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert resp.json()[0]["code"] == "print('hola2')"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context/{uuid}",
        headers=HEADERS,
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context",
        headers=HEADERS,
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_arag_workflow_context_uuid_validation(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    context_id = "not-uuid"

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context/{context_id}",
        json={"module": "restricted", "code": "print('hola2')"},
        headers=HEADERS,
    )
    assert resp.status_code == 422

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/workflow/{WORKFLOW_ID}/context/{context_id}",
        headers=HEADERS,
    )
    assert resp.status_code == 422
