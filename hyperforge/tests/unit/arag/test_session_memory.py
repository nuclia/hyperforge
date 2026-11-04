import json

import httpx
import pytest
from nucliadb_sdk.v2 import NucliaDBAsync

from hyperforge.memory.memory import SessionMemory
from hyperforge.models import MemoryConfig
from hyperforge.server.cache import NoCache


@pytest.mark.asyncio
@pytest.mark.parametrize("all_sessions", [False, True])
async def test_history_search_uses_agentic_client_type(all_sessions):
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/kb/test-kb/find")
        assert request.headers.get_list("x-ndb-client") == ["agentic"]
        assert request.headers["x-test-header"] == "preserved"
        payload = json.loads(request.content)
        assert payload["query"] == "previous question"
        if all_sessions:
            assert "resource_filters" not in payload
        else:
            assert payload["resource_filters"] == ["session-1"]
        return httpx.Response(200, json={"resources": {}, "total": 0})

    client = NucliaDBAsync(
        url="http://nucliadb.test/api",
        headers={"X-NDB-Client": "api", "x-test-header": "preserved"},
        _httpx_transport=httpx.MockTransport(handle),
    )
    session = SessionMemory(
        MemoryConfig(), agent_id="agent-1", workflow_id="default", cache=NoCache()
    )
    session.kbid = "test-kb"
    session.nucliadb_search = client
    session.init("session-1")

    async with client.session:
        result = await session.search_in_questions("previous question", all_sessions)

    assert result.total == 0
