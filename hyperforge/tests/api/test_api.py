import pytest
from httpx import AsyncClient
from hyperforge.db.encryption import decrypt_data
from nucliadb_models.resource import KnowledgeBoxObj
from sqlalchemy import text
from sqlalchemy.engine.base import Connection

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["test"]),
    pytest.mark.asyncio,
]


async def test_arag_rules(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/rules",
        json={"rules": ["rule1", "rule2", {"prompt": "rule3"}]},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/rules",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert {"rules": ["rule1", "rule2", {"prompt": "rule3"}]} == resp.json()


@pytest.mark.parametrize(
    "provider,config,update_config,encrypted_fields",
    [
        (
            "nucliadb",
            {
                "url": "http://nucliadb",
                "manager": "manager1",
                "key": "ndb-key",
                "filters": ["f1", "f2"],
                "description": "desc",
                "kbid": "kbid1",
            },
            {
                "url": "http://nucliadb2",
                "manager": "manager2",
                "key": "ndb-key2",
                "filters": ["f3"],
                "description": "desc2",
                "kbid": "kbid2",
            },
            ["key"],
        ),
    ],
)
async def test_arag_driver(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    test_db: Connection,
    provider,
    config,
    update_config,
    encrypted_fields,
):
    # Create
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/drivers",
        json={
            "provider": provider,
            "name": provider,
            "config": config,
            "identifier": "identifier",
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    # Check that config is encrypted in the DB
    db_object = test_db.execute(
        text("SELECT * FROM retrieval_agents_drivers")
    ).fetchone()
    for k, v in config.items():
        if k in encrypted_fields:
            if isinstance(v, str):
                assert decrypt_data(db_object.config[k]) == v
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    assert decrypt_data(db_object.config[k][dk]) == dv
        else:
            assert db_object.config[k] == v

    # Get
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/drivers",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    driver_config = resp.json()[0]["config"]
    assert driver_config["provider"] == provider
    assert driver_config["identifier"] == "identifier"
    assert driver_config["name"] == provider
    for k, v in config.items():
        if k in encrypted_fields:
            assert (
                k not in driver_config["config"]
            )  # Encrypted fields should not be returned
        else:
            assert driver_config["config"][k] == v

    uuid = driver_config["id"]

    # Update
    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/driver/{uuid}",
        json={
            "provider": provider,
            "name": "update_name",
            "config": update_config,
            "identifier": "can't update this",
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    # Check that config is encrypted in the DB after update
    db_object = test_db.execute(
        text("SELECT * FROM retrieval_agents_drivers")
    ).fetchone()
    for k, v in update_config.items():
        if k in encrypted_fields:
            if isinstance(v, str):
                assert decrypt_data(db_object.config[k]) == v
            elif isinstance(v, dict):
                for dk, dv in v.items():
                    assert decrypt_data(db_object.config[k][dk]) == dv
        else:
            assert db_object.config[k] == v

    # Get after update
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/drivers",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    driver_config = resp.json()[0]["config"]

    assert driver_config["name"] == "update_name"
    assert driver_config["identifier"] == "identifier"
    for k, v in update_config.items():
        if k in encrypted_fields:
            assert (
                k not in driver_config["config"]
            )  # Encrypted fields should not be returned
        else:
            assert driver_config["config"][k] == v

    # Delete
    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/driver/{uuid}",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    # Get after delete
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/drivers",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200
    assert len(resp.json()) == 0


async def test_arag_driver_partial_update(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    test_db: Connection,
):
    # Create
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/drivers",
        json={
            "provider": "nucliadb",
            "name": "nucliadb",
            "config": {
                "url": "http://nucliadb",
                "manager": "manager1",
                "key": "ndb-key",
                "filters": ["f1", "f2"],
                "description": "desc",
                "kbid": "kbid1",
            },
            "identifier": "identifier",
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    uuid = body["id"]

    db_object = test_db.execute(
        text("SELECT * FROM retrieval_agents_drivers WHERE id = :id"),
        {"id": uuid},
    ).fetchone()
    encrypted_key = db_object.config["key"]
    decrypted_key = decrypt_data(encrypted_key)

    # Partial update: note that the key field is not included in the update
    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/driver/{uuid}",
        json={
            "provider": "nucliadb",
            "name": "update_name",
            "identifier": "can't update this",
            "config": {
                "url": "http://nucliadb2",
                "manager": "manager2",
                "filters": ["f3"],
                "description": "desc2",
                "kbid": "kbid2",
            },
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    # Get after update
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/drivers",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    driver_config = body[0]["config"]
    assert driver_config["config"]["url"] == "http://nucliadb2"
    assert driver_config["config"]["manager"] == "manager2"
    assert driver_config["config"]["filters"] == ["f3"]
    assert driver_config["config"]["description"] == "desc2"
    assert driver_config["config"]["kbid"] == "kbid2"

    # Check that the key field remains unchanged
    db_object = test_db.execute(
        text("SELECT * FROM retrieval_agents_drivers WHERE id = :id"),
        {"id": uuid},
    ).fetchone()
    assert decrypt_data(db_object.config["key"]) == decrypted_key


async def test_arag_preprocess(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/preprocess",
        json={
            "module": "rephrase",
            "title": "brave",
            "rules": [],
            "session_info": True,
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
        f"/api/v1/agent/{arag_kb.uuid}/preprocess",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert resp.json()[0]["title"] == "brave"
    assert resp.json()[0]["session_info"] is True

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/preprocess/{uuid}",
        json={"module": "rephrase", "title": "brave2", "rules": [], "all": False},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/preprocess",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert resp.json()[0]["title"] == "brave2"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/preprocess/{uuid}",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/preprocess",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert len(resp.json()) == 0


async def test_arag_generation(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/generation",
        json={"module": "summarize", "prompt": "Summary"},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )

    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/generation",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert resp.json()[0]["prompt"] == "Summary"

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/generation/{uuid}",
        json={"module": "summarize", "prompt": "Summary 2"},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/generation",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert resp.json()[0]["prompt"] == "Summary 2"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/generation/{uuid}",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/generation",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert len(resp.json()) == 0


async def test_arag_postprocess(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/postprocess",
        json={"module": "external", "prompt": "External", "url": "http://example.com"},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/postprocess",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert resp.json()[0]["prompt"] == "External"

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/postprocess/{uuid}",
        json={
            "module": "external",
            "prompt": "External 2",
            "url": "http://example.com",
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
        f"/api/v1/agent/{arag_kb.uuid}/postprocess",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert resp.json()[0]["prompt"] == "External 2"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/postprocess/{uuid}",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    resp = await arag_api.get(
        f"/api/v1/agent/{arag_kb.uuid}/postprocess",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    assert len(resp.json()) == 0


async def test_arag_context(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/context",
        json={"module": "restricted", "code": "print('hola')"},
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

    assert resp.json()[0]["code"] == "print('hola')"

    uuid = resp.json()[0]["id"]

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/context/{uuid}",
        json={"module": "restricted", "code": "print('hola2')"},
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

    assert resp.json()[0]["code"] == "print('hola2')"

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/context/{uuid}",
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

    assert len(resp.json()) == 0


async def test_arag_context_uuid_validation(
    arag_kb: KnowledgeBoxObj, arag_api: AsyncClient
):
    context_id = "not-uuid"

    resp = await arag_api.patch(
        f"/api/v1/agent/{arag_kb.uuid}/context/{context_id}",
        json={"module": "restricted", "code": "print('hola2')"},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 422

    resp = await arag_api.delete(
        f"/api/v1/agent/{arag_kb.uuid}/context/{context_id}",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 422


async def test_arag_context_url(arag_kb: KnowledgeBoxObj, arag_api: AsyncClient):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/context",
        json={"module": "http", "url": "http://10.0.0.1:81/get", "method": "GET"},
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )

    assert resp.status_code == 422
