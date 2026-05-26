from typing import Any, Dict, List

import pytest
from httpx import AsyncClient
from hyperforge.api.models import InteractionRequest
from hyperforge.interaction import AragAnswer
from hyperforge.server.session import SessionManager
from nucliadb_models.resource import KnowledgeBoxObj

pytestmark = [
    pytest.mark.vcr(
        ignore_localhost=True,
        ignore_hosts=[
            "generativelanguage.googleapis.com",
            "vertexaisearch.cloud.google.com",
            "test",
        ],
    ),
    pytest.mark.asyncio,
]

CONFIG = {
    "rules": {"rules": ["Be direct and clear", {"prompt": "Talk like a lawyer"}]},
    "drivers": [
        {
            "provider": "google",
            "identifier": "google-01",
            "name": "google",
            "config": {
                "vertexai": False,
                "api_key": "AIzaSyDBBq0QwyVtYauiP0D7GkqQaWm8A92kHrM",
            },
        }
    ],
    "preprocess": [{"module": "historical", "rules": [], "all": False}],
    "context": [
        {
            "module": "google",
            "title": "Google agent",
            "source": "google-01",
        },
    ],
    "generation": [
        {"module": "summarize", "title": "Summarize agent"},
    ],
    "postprocess": [
        {"module": "remi", "title": "Validation agent"},
    ],
}


async def config_arag(
    arag_api: AsyncClient, arag_kb: KnowledgeBoxObj, config: Dict[str, Any]
):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/rules",
        json=config.get("rules"),
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200

    for driver in config.get("drivers", []):
        resp = await arag_api.post(
            f"/api/v1/agent/{arag_kb.uuid}/drivers",
            json=driver,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )
        assert resp.status_code == 200

    for postprocess in config.get("postprocess", []):
        resp = await arag_api.post(
            f"/api/v1/agent/{arag_kb.uuid}/postprocess",
            json=postprocess,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )

        assert resp.status_code == 200

    for preprocess in config.get("preprocess", []):
        resp = await arag_api.post(
            f"/api/v1/agent/{arag_kb.uuid}/preprocess",
            json=preprocess,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )

        assert resp.status_code == 200

    for context in config.get("context", []):
        resp = await arag_api.post(
            f"/api/v1/agent/{arag_kb.uuid}/context",
            json=context,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )

        assert resp.status_code == 200

    for generation in config.get("generation", []):
        resp = await arag_api.post(
            f"/api/v1/agent/{arag_kb.uuid}/generation",
            json=generation,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )

        assert resp.status_code == 200


async def test_arag_interaction_e2e(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    arag_api_session: str,
    arag_server: SessionManager,
):
    await config_arag(arag_api, arag_kb, CONFIG)

    result: List[AragAnswer] = []
    async with arag_api.stream(
        "POST",
        f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}",
        json=InteractionRequest(
            question="What certifications does Nuclia have?",
        ).model_dump(),
        timeout=200,
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SESSIONMEMBER",
        },
    ) as response:
        async for json_body in response.aiter_lines():
            result.append(AragAnswer.model_validate_json(json_body))

    assert len(result) > 2
    assert result[-2].answer
    assert "27001" in result[-2].answer
