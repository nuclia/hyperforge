from unittest.mock import patch

from fastapi import HTTPException
from httpx import AsyncClient
from nucliadb_models.search import FullResourceStrategy


async def test_search_configuration_ask(
    nucliadb_search: AsyncClient,
    nucliadb_writer_manager: AsyncClient,
    knowledgebox,
):
    kbid = knowledgebox

    resp = await nucliadb_writer_manager.post(
        f"/kb/{kbid}/search_configurations/find_config",
        json={"kind": "find", "config": {"top_k": 1, "features": ["semantic"]}},
    )
    assert resp.status_code == 201

    resp = await nucliadb_writer_manager.post(
        f"/kb/{kbid}/search_configurations/ask_config",
        json={
            "kind": "ask",
            "config": {
                "top_k": 1,
                "rag_strategies": [{"name": "full_resource", "count": 2}],
            },
        },
    )
    assert resp.status_code == 201

    async def run_ask(params):
        with patch("nuclia_arag_api.v1.ask.ask") as mock:
            mock.side_effect = HTTPException(status_code=500)
            await nucliadb_search.post(
                f"/kb/{kbid}/ask",
                json={**params, "query": "whatever"},
            )
            mock.assert_called_once()
            return mock.call_args[1]["ask_request"]

    # Default ask request (sanity check)
    request = await run_ask({})
    assert request.top_k == 20

    # Using search configuration
    request = await run_ask({"search_configuration": "ask_config"})
    assert request.top_k == 1
    assert request.rag_strategies == [
        FullResourceStrategy(
            name="full_resource",
            count=2,
            include_remaining_text_blocks=False,
            apply_to=None,
        )
    ]

    # Using search configuration and overrides
    request = await run_ask({"search_configuration": "ask_config", "top_k": 12})
    assert request.top_k == 12
    assert request.rag_strategies == [
        FullResourceStrategy(
            name="full_resource",
            count=2,
            include_remaining_text_blocks=False,
            apply_to=None,
        )
    ]

    request = await run_ask(
        {"search_configuration": "ask_config", "rag_strategies": []}
    )
    assert request.top_k == 1
    assert request.rag_strategies == []

    # Using invalid search configuration
    resp = await nucliadb_search.post(
        f"/kb/{kbid}/ask",
        json={"query": "whatever", "search_configuration": "invalid"},
    )
    assert resp.status_code == 400

    # Using find search configuration
    resp = await nucliadb_search.post(
        f"/kb/{kbid}/ask",
        json={"query": "whatever", "search_configuration": "find_config"},
    )
    assert resp.status_code == 400
