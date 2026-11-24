import json

import httpx
import pytest
from nucliadb_sdk.v2 import NucliaDBAsync

from hyperforge.memory.memory import SessionMemory
from hyperforge.models import MemoryConfig, NucliaDBMemoryConfig
from hyperforge.server.cache import NoCache


@pytest.mark.asyncio
@pytest.mark.parametrize("all_sessions", [False, True])
@pytest.mark.parametrize("internal", [False, True])
async def test_history_search_uses_agentic_client_type(
    all_sessions, internal, monkeypatch
):
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/kb/test-kb/find")
        assert request.headers.get_list("x-ndb-client") == ["agentic"]
        assert request.headers["x-test-header"] == "preserved"
        if internal:
            assert request.headers["x-nucliadb-account"] == "owning-account"
            assert request.headers["x-nucliadb-roles"] == "READER"
        else:
            assert "x-nucliadb-account" not in request.headers
        payload = json.loads(request.content)
        assert payload["query"] == "previous question"
        if all_sessions:
            assert "resource_filters" not in payload
        else:
            assert payload["resource_filters"] == ["session-1"]
        return httpx.Response(200, json={"resources": {}, "total": 0})

    clients = []

    def create_client(**kwargs):
        headers = {"X-NDB-Client": "api", "x-test-header": "preserved"}
        headers.update(kwargs.pop("headers", {}))
        client = NucliaDBAsync(
            headers=headers, _httpx_transport=httpx.MockTransport(handle), **kwargs
        )
        clients.append(client)
        return client

    monkeypatch.setattr("hyperforge.memory.memory.NucliaDBAsync", create_client)
    session = SessionMemory(
        MemoryConfig(
            nucliadb=NucliaDBMemoryConfig(
                url="http://{component}.test/api",
                kbid="test-kb",
                internal=internal,
            )
        ),
        agent_id="agent-1",
        workflow_id="default",
        cache=NoCache(),
    )
    session.init("session-1")
    session.account_id = "owning-account"

    try:
        result = await session.search_in_questions("previous question", all_sessions)
    finally:
        for client in clients:
            await client.session.aclose()

    assert result.total == 0
