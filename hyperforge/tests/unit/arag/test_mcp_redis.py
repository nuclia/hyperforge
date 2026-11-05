import asyncio
import json
import socket
import uuid
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from fastapi import HTTPException
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from redis.asyncio import Redis
from redis.exceptions import ConnectionError

from hyperforge.api import mcp_transport
from hyperforge.api.app import HTTPApplication
from hyperforge.api.mcp_server_pool import MCPServerPool
from hyperforge.api.mcp_sessions import MCPSessionStore
from hyperforge.api.settings import Settings
from hyperforge.api.v1 import mcp_interaction
from hyperforge.db.settings import DataManagerSettings
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
    Feedback,
    OAuthAuthenticateURL,
)
from hyperforge.workflows import WorkflowData

pytestmark = pytest.mark.asyncio
URL = "/api/v1/agent/agent/session/ephemeral/mcp"
KEY = ("account", "user", "basic", "agent", "ephemeral")
HEADERS = {
    "X-STF-ROLES": "SESSIONMEMBER",
    "X-STF-USER": "user",
    "X-STF-ACCOUNT": "account",
    "X-STF-ACCOUNT-TYPE": "basic",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-11-25",
}


@pytest.fixture
async def replicas(valkey_url):
    redis = Redis.from_url(valkey_url, decode_responses=True)
    apps = []
    headers = dict(HEADERS, **{"X-STF-USER": uuid.uuid4().hex})
    for _ in range(2):
        app = HTTPApplication(
            Settings(mcp_session_idle_ttl_seconds=60),
            DataManagerSettings(postgresql_dsn="postgresql://localhost/test"),
        )
        app.agent_manager = AsyncMock()
        app.agent_manager.workflows_list.return_value = [
            WorkflowData(id="ask", name="ask", description="Ask", parameters={})
        ]
        app.agent_manager.get_agent_config_basic.return_value = SimpleNamespace(
            instructions="test"
        )
        app.agent_manager.get_prompts.return_value = []
        app.mcp_server_pool = MCPServerPool(10, 60, 5)
        app.mcp_sessions = MCPSessionStore(redis)
        app.state.mcp_headers = headers
        apps.append(app)
    try:
        yield apps, redis
    finally:
        for app in apps:
            await app.mcp_server_pool.shutdown()
        identity = (KEY[0], headers["X-STF-USER"], *KEY[2:])
        _, prefix = apps[0].mcp_sessions.keys(identity, "")
        keys = [key async for key in redis.scan_iter(match=prefix + "*")]
        if keys:
            await redis.delete(*keys)
        await redis.aclose()


@asynccontextmanager
async def clients(apps):
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=apps[0]),
            base_url="http://test",
            headers=apps[0].state.mcp_headers,
        ) as first,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=apps[1]),
            base_url="http://test",
            headers=apps[0].state.mcp_headers,
        ) as second,
    ):
        yield first, second


async def initialize(client):
    response = await client.post(
        URL,
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-11-25",
                "capabilities": {"elicitation": {"form": {}, "url": {}}},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200, response.text
    session_id = response.headers["mcp-session-id"]
    response = await client.post(
        URL,
        headers={"Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    assert response.status_code == 202
    return session_id


async def tools(client, session_id, **headers):
    return await client.post(
        URL,
        headers={"Mcp-Session-Id": session_id, **headers},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )


@pytest.fixture
def workflow(monkeypatch):
    started, completed, cancelled = [], [], []

    async def stream(app, websocket, **kwargs):
        invocation = uuid.uuid4().hex
        started.append(invocation)
        finished = False
        try:
            yield AragAnswer(
                operation=AnswerOperation.AGENT_REQUEST,
                oauth=OAuthAuthenticateURL(oauth_url="https://example.com/authorize"),
            )
            yield AragAnswer(
                operation=AnswerOperation.AGENT_REQUEST,
                feedback=Feedback(
                    request_id=invocation,
                    question=invocation,
                    module="test",
                    agent_id="agent",
                    data=None,
                    response_schema={
                        "type": "object",
                        "properties": {
                            "choice": {"type": "string"},
                            "invocation": {"type": "string"},
                        },
                        "required": ["choice", "invocation"],
                    },
                ),
            )
            response = await websocket.queue.get()
            answer = json.loads(response.response)
            assert answer["invocation"] == invocation
            choice = answer["choice"]
            completed.append(invocation)
            finished = True
            yield AragAnswer(
                answer=json.dumps({"invocation": invocation, "choice": choice})
            )
        finally:
            if not finished:
                cancelled.append(invocation)

    monkeypatch.setattr(mcp_interaction, "stream_response", stream)
    return SimpleNamespace(started=started, completed=completed, cancelled=cancelled)


@asynccontextmanager
async def serving(apps):
    servers, sockets, urls = [], [], []
    try:
        for app in apps:
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            config = uvicorn.Config(
                app, lifespan="off", log_level="error", timeout_graceful_shutdown=1
            )
            config.load()
            server = uvicorn.Server(config)
            server.lifespan = config.lifespan_class(config)
            servers.append(server)
            await server.startup(sockets=[sock])
            urls.append(f"http://127.0.0.1:{sock.getsockname()[1]}")
        async with asyncio.timeout(5):
            while not all(server.started for server in servers):
                await asyncio.sleep(0.01)
        yield SimpleNamespace(urls=urls, headers=apps[0].state.mcp_headers)
    finally:
        for server, sock in zip(servers, sockets):
            await server.shutdown(sockets=[sock])


class BalancedTransport(httpx.AsyncBaseTransport):
    def __init__(self, urls, alternate=False):
        self.urls = urls
        self.alternate = alternate
        self.transport = httpx.AsyncHTTPTransport()
        self.requests = []

    async def handle_async_request(self, request):
        body = await request.aread()
        message = json.loads(body) if body else {}
        method = message.get("method")
        if method in {"tools/call", "tools/list"}:
            node = message["id"] % 2 if self.alternate else 1
        else:
            node = 0
        self.requests.append((node, message))
        request.url = httpx.URL(self.urls[node] + request.url.raw_path.decode())
        request.headers["host"] = request.url.netloc.decode()
        return await self.transport.handle_async_request(request)

    async def aclose(self):
        await self.transport.aclose()


@asynccontextmanager
async def mcp_client(urls, choice, entered=None, release=None, alternate=False):
    async def elicit(context, params):
        if params.mode == "url":
            if entered is not None:
                entered.set()
            if release is not None:
                await release.wait()
            return types.ElicitResult(action="accept")
        return types.ElicitResult(
            action="accept", content={"choice": choice, "invocation": params.message}
        )

    balance = BalancedTransport(urls.urls, alternate)
    async with (
        httpx.AsyncClient(transport=balance, headers=urls.headers) as http,
        streamable_http_client(urls.urls[0] + URL, http_client=http) as (
            read,
            write,
            session_id,
        ),
        ClientSession(read, write, elicitation_callback=elicit) as mcp,
    ):
        await mcp.initialize()
        yield mcp, session_id(), balance


def value(result):
    assert not result.isError, result
    return json.loads(result.content[0].text)


async def test_requests_and_delete_are_independent_across_replicas(replicas):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        a, b = await asyncio.gather(initialize(first), initialize(first))
        assert a != b
        assert (await tools(second, a)).status_code == 200
        assert (await tools(second, b)).status_code == 200
        assert (
            await second.delete(URL, headers={"Mcp-Session-Id": a})
        ).status_code == 200
        assert (await tools(first, a)).status_code == 404
        assert (await tools(first, b)).status_code == 200


@pytest.mark.parametrize(
    "header", ["X-STF-USER", "X-STF-ACCOUNT", "X-STF-ACCOUNT-TYPE"]
)
async def test_session_is_bound_to_authenticated_identity(replicas, header):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        assert (await tools(second, session_id, **{header: "other"})).status_code == 404
        assert (await tools(second, session_id)).status_code == 200


async def test_sse_and_elicitation_cross_replicas_with_real_mcp_client(
    replicas, workflow
):
    apps, redis = replicas
    identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
    _, prefix = apps[0].mcp_sessions.keys(identity, "")
    reply_pattern = prefix + "*:reply:*"
    for app in apps:
        app.mcp_settings.mcp_session_idle_ttl_seconds = 1
    entered = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()
    async with (
        serving(apps) as urls,
        mcp_client(urls, "a", entered[0], release) as first,
        mcp_client(urls, "b", entered[1], release) as second,
    ):
        assert first[1] != second[1]
        calls = [
            asyncio.create_task(client[0].call_tool("ask", {}))
            for client in (first, second)
        ]
        try:
            async with asyncio.timeout(10):
                await asyncio.gather(*(event.wait() for event in entered))
                await asyncio.sleep(1.3)
                for client in (first, second):
                    assert await redis.exists(
                        apps[0].mcp_sessions.keys(identity, client[1])[1]
                    )
                assert not workflow.completed
                assert not any(task.done() for task in calls)
                assert (
                    len([key async for key in redis.scan_iter(match=reply_pattern)])
                    == 2
                )
                release.set()
                assert [
                    value(result)["choice"] for result in await asyncio.gather(*calls)
                ] == ["a", "b"]
        finally:
            release.set()
            for task in calls:
                task.cancel()
            await asyncio.gather(*calls, return_exceptions=True)
        assert all(
            node == 0
            for client in (first, second)
            for node, message in client[2].requests
            if "result" in message
        )
        assert all(
            node == 1
            for client in (first, second)
            for node, message in client[2].requests
            if message.get("method") == "tools/call"
        )
    assert len(workflow.started) == len(workflow.completed) == 2
    assert not [key async for key in redis.scan_iter(match=reply_pattern)]


async def test_parallel_calls_share_identity_and_session_without_id_collisions(
    replicas, workflow
):
    apps, _ = replicas
    async with (
        serving(apps) as urls,
        mcp_client(urls, "same", alternate=True) as client,
    ):
        results = await asyncio.wait_for(
            asyncio.gather(*(client[0].call_tool("ask", {}) for _ in range(20))), 15
        )
        values = [value(result) for result in results]
        assert len({row["invocation"] for row in values}) == 20
        assert {row["choice"] for row in values} == {"same"}
        assert {
            node
            for node, message in client[2].requests
            if message.get("method") == "tools/call"
        } == {0, 1}
    assert len(workflow.started) == len(workflow.completed) == 20


async def test_pending_response_delivery_is_retained_and_idempotent(
    replicas,
):
    apps, redis = replicas
    identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
    store = apps[0].mcp_sessions
    params = types.InitializeRequestParams(
        protocolVersion="2025-11-25",
        capabilities=types.ClientCapabilities(),
        clientInfo=types.Implementation(name="test", version="1"),
    )
    session_id = await store.create(identity, params, 60, 2)
    token = await store.start(identity, session_id, 1, 60)
    await store.pending(identity, session_id, "wire", 60)
    reply = '{"jsonrpc":"2.0","id":"wire","result":{"action":"accept"}}'
    assert all(
        await asyncio.gather(
            *(store.reply(identity, session_id, "wire", reply, 60) for _ in range(8))
        )
    )
    with pytest.raises(HTTPException) as conflict:
        await store.reply(identity, session_id, "wire", reply + " ", 60)
    assert conflict.value.status_code == 409
    await store.finish(identity, session_id, 1, "stale", [], 60)
    update = await store.poll(identity, session_id, 1, token, ["wire"], 60)
    assert not update.cancelled
    assert update.replies == {"wire": reply}
    await store.acknowledge(identity, session_id, ["wire"], 60)
    update = await store.poll(identity, session_id, 1, token, [], 60)
    assert not update.cancelled
    assert update.replies == {}
    await store.finish(identity, session_id, 1, token, [], 60)
    assert await store.reply(identity, session_id, "wire", reply, 60) is False
    abandoned = await store.start(identity, session_id, 1, 1)
    await store.pending(identity, session_id, "expired", 1)
    await asyncio.sleep(1.1)
    assert await store.reply(identity, session_id, "expired", reply, 60) is False
    replacement = await store.start(identity, session_id, 1, 60)
    await store.cancel(identity, session_id, 1, 60)
    await store.finish(identity, session_id, 1, abandoned, [], 60)
    assert (await store.poll(identity, session_id, 1, replacement, [], 60)).cancelled
    await store.finish(identity, session_id, 1, replacement, [], 60)
    _, record = store.keys(identity, session_id)
    assert await redis.exists(record)
    assert await redis.exists(store._reply_key(identity, session_id, "wire")) == 0


async def test_session_limit_is_atomic_and_deleted_sessions_free_capacity(
    replicas,
):
    apps, _ = replicas
    identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
    store = apps[0].mcp_sessions
    params = types.InitializeRequestParams(
        protocolVersion="2025-11-25",
        capabilities=types.ClientCapabilities(),
        clientInfo=types.Implementation(name="test", version="1"),
    )
    results = await asyncio.gather(
        *(store.create(identity, params, 60, 2) for _ in range(12)),
        return_exceptions=True,
    )
    sessions = [result for result in results if isinstance(result, str)]
    assert len(set(sessions)) == 2
    assert all(
        isinstance(result, HTTPException) and result.status_code == 503
        for result in results
        if not isinstance(result, str)
    )
    assert await store.session_count(identity, 60) == 2
    await store.delete(identity, sessions[0], 60)
    assert await store.create(identity, params, 60, 2) not in sessions
    assert await store.session_count(identity, 60) == 2


async def test_session_renewal_keeps_registry_and_admission_consistent(
    replicas, monkeypatch
):
    apps, redis = replicas
    identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
    store = apps[0].mcp_sessions
    params = types.InitializeRequestParams(
        protocolVersion="2025-11-25",
        capabilities=types.ClientCapabilities(),
        clientInfo=types.Implementation(name="test", version="1"),
    )
    session_id = await store.create(identity, params, 1, 1)
    registry, record = store.keys(identity, session_id)
    renewed, release = asyncio.Event(), asyncio.Event()
    original_pipeline = redis.pipeline

    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        original_execute = pipe.execute

        async def delayed_execute(*args, **kwargs):
            renews_session = any(
                command[:2] == ("GETEX", record) for command, _ in pipe.command_stack
            )
            result = await original_execute(*args, **kwargs)
            if renews_session:
                renewed.set()
                await release.wait()
            return result

        monkeypatch.setattr(pipe, "execute", delayed_execute)
        return pipe

    monkeypatch.setattr(redis, "pipeline", pipeline)
    task = asyncio.create_task(store.get(identity, session_id, 60))
    try:
        await asyncio.wait_for(renewed.wait(), 3)
        await asyncio.sleep(1.1)
        release.set()
        assert (await asyncio.wait_for(task, 3)).params == params
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert await redis.exists(record)
    assert await store.session_count(identity, 60) == 1
    assert await redis.sismember(registry, session_id)
    with pytest.raises(HTTPException) as exhausted:
        await store.create(identity, params, 60, 1)
    assert exhausted.value.status_code == 503


async def test_short_session_renewal_does_not_shorten_registry_ttl(replicas):
    apps, redis = replicas
    identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
    store = apps[0].mcp_sessions
    params = types.InitializeRequestParams(
        protocolVersion="2025-11-25",
        capabilities=types.ClientCapabilities(),
        clientInfo=types.Implementation(name="test", version="1"),
    )
    long_session = await store.create(identity, params, 60, 2)
    short_session = await store.create(identity, params, 1, 2)
    registry, _ = store.keys(identity, long_session)
    assert (await store.get(identity, short_session, 1)).params == params
    assert await redis.ttl(registry) > 1
    await asyncio.sleep(1.1)
    assert await store.session_count(identity, 60) == 1
    assert (await store.get(identity, long_session, 60)).params == params
    with pytest.raises(HTTPException) as expired:
        await store.get(identity, short_session, 60)
    assert expired.value.status_code == 404


async def test_redis_failure_returns_503_without_local_fallback(replicas, monkeypatch):
    apps, redis = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        monkeypatch.setattr(
            redis, "execute_command", AsyncMock(side_effect=ConnectionError("offline"))
        )
        assert (await tools(second, session_id)).status_code == 503


async def test_shutdown_keeps_shared_session_available_on_other_replica(replicas):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        await apps[0].mcp_server_pool.shutdown()
        assert (await tools(second, session_id)).status_code == 200


@asynccontextmanager
async def raw_call(network, session_id):
    async def events(response):
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                yield json.loads(line[6:])

    async with (
        httpx.AsyncClient(
            headers={**network.headers, "Mcp-Session-Id": session_id}, timeout=5
        ) as http,
        http.stream(
            "POST",
            network.urls[1] + URL,
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "ask", "arguments": {}},
            },
        ) as response,
    ):
        assert response.status_code == 200
        yield http, events(response)


@pytest.mark.parametrize("operation", ["delete", "cancel"])
async def test_control_on_another_replica_cancels_only_its_call(
    replicas, workflow, operation
):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        survivor = await initialize(second)
        async with (
            serving(apps) as network,
            raw_call(network, session_id) as (http, events),
        ):
            assert (await anext(events))["method"] == "elicitation/create"
            if operation == "delete":
                response = await http.delete(network.urls[0] + URL)
            else:
                response = await http.post(
                    network.urls[0] + URL,
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": 7},
                    },
                )
            assert response.status_code in {200, 202}
            result = await asyncio.wait_for(anext(events), 3)
            assert result["error"]["message"] == "Request cancelled"
            assert (await tools(second, survivor)).status_code == 200
    assert not workflow.completed
    assert len(workflow.cancelled) == 1


@pytest.mark.parametrize("waiting_for", ["pool", "creation"])
async def test_cancellation_before_handler_acquisition_does_not_start_tool(
    replicas, workflow, monkeypatch, waiting_for
):
    apps, redis = replicas
    entered, release = asyncio.Event(), asyncio.Event()
    if waiting_for == "pool":
        target = apps[1].mcp_server_pool
        attribute = "acquire"
    else:
        target = mcp_interaction
        attribute = "_create_mcp_server"
    original = getattr(target, attribute)

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        monkeypatch.setattr(target, attribute, blocked)
        async with serving(apps) as network:

            async def call():
                async with raw_call(network, session_id) as (_, events):
                    return await anext(events)

            task = asyncio.create_task(call())
            try:
                await asyncio.wait_for(entered.wait(), 3)
                response = await first.post(
                    URL,
                    headers={"Mcp-Session-Id": session_id},
                    json={
                        "jsonrpc": "2.0",
                        "method": "notifications/cancelled",
                        "params": {"requestId": 7},
                    },
                )
                assert response.status_code == 202
                release.set()
                result = await asyncio.wait_for(task, 3)
                assert result.get("error", {}).get("message") == "Request cancelled"
                assert not workflow.started
            finally:
                release.set()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
        assert not await redis.exists(
            apps[0].mcp_sessions._call_key(identity, session_id, 7)
        )
        response = await second.post(
            URL,
            headers={"Mcp-Session-Id": session_id},
            json={"jsonrpc": "2.0", "id": 7, "method": "tools/list"},
        )
        assert response.status_code == 200


@pytest.mark.parametrize("failure", ["error", "cancel"])
async def test_interrupted_admission_releases_call_and_rejects_duplicate(
    replicas, monkeypatch, failure
):
    apps, redis = replicas
    entered, release = asyncio.Event(), asyncio.Event()
    pool = apps[1].mcp_server_pool
    original = pool.acquire

    async def blocked(*args, **kwargs):
        entered.set()
        await release.wait()
        raise HTTPException(503, "Server creation failed")

    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        monkeypatch.setattr(pool, "acquire", blocked)
        identity = (KEY[0], apps[0].state.mcp_headers["X-STF-USER"], *KEY[2:])
        call_key = apps[0].mcp_sessions._call_key(identity, session_id, 1)
        task = asyncio.create_task(tools(second, session_id))
        try:
            await asyncio.wait_for(entered.wait(), 3)
            assert await redis.exists(call_key)
            assert (await tools(second, session_id)).status_code == 409
            assert await redis.exists(call_key)
            if failure == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                release.set()
                assert (await asyncio.wait_for(task, 3)).status_code == 503
            assert not await redis.exists(call_key)
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(pool, "acquire", original)
        assert (await tools(second, session_id)).status_code == 200


async def test_coordination_failure_before_dispatch_does_not_start_tool(
    replicas, workflow, monkeypatch
):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        monkeypatch.setattr(
            apps[1].mcp_sessions,
            "poll",
            AsyncMock(side_effect=HTTPException(503, "Coordination unavailable")),
        )
        response = await second.post(
            URL,
            headers={"Mcp-Session-Id": session_id},
            json={
                "jsonrpc": "2.0",
                "id": 7,
                "method": "tools/call",
                "params": {"name": "ask", "arguments": {}},
            },
        )
        assert response.status_code == 503
        assert not workflow.started


async def test_replies_can_complete_a_tool_at_full_request_capacity(
    replicas, workflow, monkeypatch
):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        monkeypatch.setattr(mcp_transport, "MAX_ACTIVE_CALLS", 1)
        async with (
            serving(apps) as network,
            raw_call(network, session_id) as (http, events),
        ):
            url_request = await anext(events)
            assert (await tools(second, session_id)).status_code == 503
            for content in (None, {"choice": "completed"}):
                elicitation = url_request if content is None else await anext(events)
                if content is not None:
                    content["invocation"] = elicitation["params"]["message"]
                response = await http.post(
                    network.urls[1] + URL,
                    json={
                        "jsonrpc": "2.0",
                        "id": elicitation["id"],
                        "result": {
                            "action": "accept",
                            **({"content": content} if content else {}),
                        },
                    },
                )
                assert response.status_code == 202
            result = await asyncio.wait_for(anext(events), 3)
            assert "completed" in result["result"]["content"][0]["text"]
    assert len(workflow.started) == len(workflow.completed) == 1


@pytest.mark.parametrize("stalled", [False, True])
async def test_coordination_loss_cancels_running_tool_without_replay(
    replicas, workflow, monkeypatch, stalled
):
    apps, redis = replicas
    release = asyncio.Event()
    original = redis.execute_command
    original_pipeline = redis.pipeline

    async def unavailable(operation, *args, **kwargs):
        if stalled:
            await release.wait()
            return await operation(*args, **kwargs)
        raise ConnectionError("offline")

    def pipeline(*args, **kwargs):
        pipe = original_pipeline(*args, **kwargs)
        monkeypatch.setattr(pipe, "execute", partial(unavailable, pipe.execute))
        return pipe

    async with clients(apps) as (first, _):
        for app in apps:
            app.mcp_settings.mcp_session_idle_ttl_seconds = 1
        session_id = await initialize(first)
        async with (
            serving(apps) as network,
            raw_call(network, session_id) as (_, events),
        ):
            assert (await anext(events))["method"] == "elicitation/create"
            monkeypatch.setattr(
                redis, "execute_command", partial(unavailable, original)
            )
            monkeypatch.setattr(redis, "pipeline", pipeline)
            try:
                async with asyncio.timeout(6):
                    while not workflow.cancelled:
                        await asyncio.sleep(0.01)
                assert not workflow.completed
            finally:
                release.set()
                monkeypatch.setattr(redis, "execute_command", original)
                monkeypatch.setattr(redis, "pipeline", original_pipeline)
            with pytest.raises((httpx.RemoteProtocolError, StopAsyncIteration)):
                await asyncio.wait_for(anext(events), 6)
    assert len(workflow.started) == 1
