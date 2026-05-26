from collections.abc import AsyncIterator

import pytest
from httpx import AsyncClient

from nucliadb_agentic_api.tests.utils.dirty_index import mark_dirty, wait_for_sync


@pytest.fixture(scope="function")
async def knowledgebox(nucliadb_writer_manager: AsyncClient) -> AsyncIterator[str]:
    """Test knowledge box with 2 vectorsets.

    As we test against a standalone nucliadb, the knowledgebox is created
    through the /kbs endpoint, only accessible for onprem (standalone)
    deployments.

    Vectorsets and it's settings are hardcoded and tests may assume they are
    those. Be careful to change those values.

    """
    resp = await nucliadb_writer_manager.post(
        "/kbs",
        json={
            "title": "Test KB",
            "slug": "knowledgebox",
            "learning_configuration": {
                "semantic_models": [
                    "en-2024-04-24",
                    "multilingual-2024-05-06",
                ],
                "semantic_model_configs": {
                    "en-2024-04-24": {
                        "similarity": 0,  # DOT
                        "size": 768,
                        "threshold": 0.47,
                        "matryoshka_dims": [],
                    },
                    "multilingual-2024-05-06": {
                        "similarity": 0,  # DOT
                        "size": 1024,
                        "threshold": 0.4,
                        "matryoshka_dims": [],
                    },
                },
                # legacy fields
                # "semantic_model": "en-2024-04-24",
                "semantic_vector_similarity": "DOT",
            },
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    kbid = body["uuid"]

    # give room to nidx to sync the searcher and avoid shard not found errors
    await mark_dirty()
    await wait_for_sync()

    yield kbid

    resp = await nucliadb_writer_manager.delete(f"/kb/{kbid}")
    assert resp.status_code == 200


@pytest.fixture(scope="function")
async def resource(nucliadb_writer: AsyncClient, knowledgebox: str):
    kbid = knowledgebox

    resp = await nucliadb_writer.post(
        f"/kb/{kbid}/resources",
        json={
            "slug": "my-resource",
            "title": "The title",
            "summary": "The summary",
            "texts": {"text_field": {"body": "The body of the text field"}},
        },
    )
    assert resp.status_code in (200, 201)
    rid = resp.json()["uuid"]

    await mark_dirty()
    await wait_for_sync()

    yield rid
