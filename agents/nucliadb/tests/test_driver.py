import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from nucliadb_models.search import AskRequest, FindRequest
from nucliadb_sdk.v2 import NucliaDBAsync

from hyperforge_nucliadb.driver import NucliaDBDriver
from hyperforge_nucliadb.sync.driver import SyncDriver


def make_driver(client, driver_type=NucliaDBDriver):
    return driver_type.model_construct(
        provider="nucliadb",
        name="test",
        config=SimpleNamespace(kbid="test-kb", filters=[]),
        driver=client,
        manager=SimpleNamespace(),
        _synonyms=None,
    )


@pytest.mark.asyncio
async def test_labels_uses_complete_labelsets_response():
    client = AsyncMock()
    client.get_labelsets.return_value = SimpleNamespace(
        labelsets={
            "type": SimpleNamespace(labels=[SimpleNamespace(title="report")]),
            "topic": SimpleNamespace(labels=[SimpleNamespace(title="science")]),
        }
    )
    driver = make_driver(client)

    assert await driver.labels() == {"type": ["report"], "topic": ["science"]}

    client.get_labelsets.assert_awaited_once_with(kbid="test-kb")
    client.get_labelset.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_type", [NucliaDBDriver, SyncDriver])
@pytest.mark.parametrize("operation", ["ask", "find", "find_raw"])
async def test_search_requests_use_agentic_client_type(driver_type, operation):
    def handle(request: httpx.Request) -> httpx.Response:
        endpoint = "ask" if operation == "ask" else "find"
        assert request.url.path.endswith(f"/kb/test-kb/{endpoint}")
        assert request.headers.get_list("x-ndb-client") == ["agentic"]
        assert request.headers["x-test-header"] == "preserved"
        assert json.loads(request.content)["query"] == "question"
        result = {"resources": {}, "total": 0}
        if operation == "ask":
            result = {
                "status": "success",
                "answer": "answer",
                "retrieval_results": result,
            }
        return httpx.Response(200, json=result)

    client = NucliaDBAsync(
        url="http://nucliadb.test/api",
        headers={"X-NDB-Client": "api", "x-test-header": "preserved"},
        _httpx_transport=httpx.MockTransport(handle),
    )
    driver = make_driver(client, driver_type)

    async with client.session:
        if operation == "ask":
            result = await driver.ask(AskRequest(query="question"))
            assert result.answer == "answer"
        elif operation == "find":
            result = await driver.find("question")
            assert result.total == 0
        else:
            result = await driver.find_raw(FindRequest(query="question"))
            assert result.total == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_header", ["X-NDB-Client", "x-ndb-client", "X-NdB-ClIeNt"]
)
async def test_ask_preserves_caller_headers_without_mutating_them(client_header):
    client = AsyncMock()
    driver = make_driver(client)
    headers = {client_header: "widget", "X-Synchronous": "true"}
    original_headers = headers.copy()
    request = AskRequest(query="question")

    await driver.ask(request, headers=headers)

    kwargs = client.ask.await_args.kwargs
    sent_headers = httpx.Headers(kwargs["headers"])
    assert sent_headers.get_list("x-ndb-client") == ["agentic"]
    assert sent_headers["X-Synchronous"] == "true"
    assert kwargs["kbid"] == "test-kb"
    assert kwargs["content"] is request
    assert headers == original_headers
