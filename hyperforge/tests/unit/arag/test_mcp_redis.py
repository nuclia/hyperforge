import asyncio
import base64
import json
import socket
from contextlib import asynccontextmanager
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ElicitResult, TextContent
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, RedisError
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import StreamingResponse

from hyperforge.api import mcp_redis
from hyperforge.api.app import HTTPApplication
from hyperforge.api.mcp_redis import RedisMCPSessionRouter
from hyperforge.api.mcp_server_pool import MCPServerPool
from hyperforge.api.settings import Settings
from hyperforge.api.v1 import mcp_interaction
from hyperforge.db.settings import DataManagerSettings
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
    "MCP-Protocol-Version": "2025-03-26",
}


@pytest.fixture
async def replicas(valkey_url):
    redis = Redis.from_url(valkey_url, decode_responses=True)
    apps = []
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
        app.mcp_session_router = RedisMCPSessionRouter(
            redis,
            partial(mcp_interaction.local_mcp_response, app),
            invalidate_sessions=app.mcp_server_pool.invalidate_sessions,
            session_ttl=60,
            max_request_bytes=1024 * 1024,
            max_response_bytes=4 * 1024 * 1024,
            lease_seconds=3,
        )
        await app.mcp_session_router.start()
        apps.append(app)
    try:
        yield apps, redis
    finally:
        for app in apps:
            await app.mcp_session_router.shutdown()
            await app.mcp_server_pool.shutdown()
        await redis.aclose()


@asynccontextmanager
async def clients(apps):
    async with (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=apps[0]),
            base_url="http://test",
            headers=HEADERS,
        ) as first,
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=apps[1]),
            base_url="http://test",
            headers=HEADERS,
        ) as second,
    ):
        yield first, second


async def initialize(client):
    response = await client.post(
        URL,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.headers["mcp-session-id"]


async def tools(client, session_id, **headers):
    return await client.post(
        URL,
        headers={"Mcp-Session-Id": session_id, **headers},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    )


async def ask(client, session_id, **headers):
    return await client.post(
        URL,
        headers={"Mcp-Session-Id": session_id, **headers},
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "ask", "arguments": {}},
        },
    )


async def wait_for_recovery(router, previous_owner):
    async with asyncio.timeout(5):
        while router.owner == previous_owner or not router._healthy:
            await asyncio.sleep(0.01)


@asynccontextmanager
async def running_request(request):
    task = asyncio.create_task(request)
    try:
        yield task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
def blocked_tool(monkeypatch):
    """Patch before initialization captures the SDK's tool callback."""
    entered, cancelled, finish = (asyncio.Event() for _ in range(3))

    async def call_tool(*args):
        entered.set()
        try:
            await finish.wait()
            return []
        finally:
            cancelled.set()

    tool = AsyncMock(side_effect=call_tool)
    monkeypatch.setattr(mcp_interaction, "call_tool", tool)
    return SimpleNamespace(
        entered=entered, cancelled=cancelled, finish=finish, mock=tool
    )


@asynccontextmanager
async def blocked_call(client, session_id, tool):
    async with running_request(ask(client, session_id)) as response:
        try:
            await asyncio.wait_for(tool.entered.wait(), 5)
            yield SimpleNamespace(
                response=response, cancelled=tool.cancelled, tool=tool.mock
            )
        finally:
            tool.finish.set()


async def relay_stream(replicas, handler, receive, send):
    apps, redis = replicas
    owner, remote = (app.mcp_session_router for app in apps)
    owner.local_handler = handler
    session_id = "streaming"
    await redis.set(owner._session_key(KEY, session_id), owner.owner, ex=60)
    request = Request(
        {
            "type": "http",
            "method": "GET",
            "path": URL,
            "headers": [(b"mcp-session-id", session_id.encode())],
        }
    )
    await remote.serve(KEY, request, b"", receive, send)


@pytest.mark.parametrize("acknowledged_at", [2.0, 4.0])
async def test_renewal_includes_round_trip_time_and_rejects_late_acknowledgements(
    monkeypatch, acknowledged_at
):
    now = 1.0
    renewals = 0
    client = AsyncMock()
    router = RedisMCPSessionRouter(
        client,
        AsyncMock(),
        invalidate_sessions=AsyncMock(),
        session_ttl=60,
        max_request_bytes=1024,
        max_response_bytes=1024,
        lease_seconds=3,
    )
    router._healthy = True
    router._lease_deadline = 3.0

    async def renew(*args):
        nonlocal now, renewals
        renewals += 1
        now = acknowledged_at
        return renewals == 1

    client.eval = renew
    monkeypatch.setattr(mcp_redis, "time", SimpleNamespace(monotonic=lambda: now))
    await asyncio.wait_for(router._heartbeat(router.owner), 3)
    if acknowledged_at < 3.0:
        assert router._lease_deadline == 4.0
        assert renewals == 2
    else:
        assert router._lease_deadline == 3.0
        assert renewals == 1


async def test_expired_local_deadline_rejects_work_before_supervisor_runs(
    replicas, monkeypatch
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    call_tool = AsyncMock(return_value=[])
    monkeypatch.setattr(mcp_interaction, "call_tool", call_tool)
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        assert await redis.exists(owner._owner_key(previous))
        owner._lease_deadline = mcp_redis.time.monotonic() - 1
        response = await ask(first, session_id)
        assert response.status_code == 503, response.text
        call_tool.assert_not_awaited()
        await wait_for_recovery(owner, previous)
        replacement = await initialize(first)
        assert (await tools(second, replacement)).status_code == 200


async def test_stalled_renewal_cancels_sdk_without_waiting_for_redis(
    replicas, monkeypatch, blocked_tool
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    eval_script = redis.eval
    renewing = asyncio.Event()

    async def stalled_renewal(script, numkeys, *args):
        if args[0] == owner._owner_key(previous) and "'EXPIRE'" in script:
            renewing.set()
            await asyncio.Event().wait()
        return await eval_script(script, numkeys, *args)

    monkeypatch.setattr(redis, "eval", stalled_renewal)
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        async with blocked_call(second, session_id, blocked_tool) as call:
            await asyncio.wait_for(renewing.wait(), 2)
            response = await asyncio.wait_for(asyncio.shield(call.response), 4)
            assert response.status_code in (404, 503), response.text
            await asyncio.wait_for(call.cancelled.wait(), 1)
            await wait_for_recovery(owner, previous)
            assert not await redis.exists(owner._owner_key(previous))
            assert not apps[0].mcp_server_pool._servers
            assert (await tools(second, session_id)).status_code == 404
            replacement = await initialize(first)
            assert (await tools(second, replacement)).status_code == 200
            call.tool.assert_awaited_once()


async def test_stalled_initial_registration_never_admits_requests(
    replicas, monkeypatch
):
    _, redis = replicas
    router = RedisMCPSessionRouter(
        redis,
        AsyncMock(),
        invalidate_sessions=AsyncMock(),
        session_ttl=60,
        max_request_bytes=1024,
        max_response_bytes=1024,
        lease_seconds=1,
    )
    set_value = redis.set

    async def delayed_acknowledgement(key, value, **kwargs):
        result = await set_value(key, value, **kwargs)
        if key == router._owner_key(router.owner):
            await asyncio.Event().wait()
        return result

    monkeypatch.setattr(redis, "set", delayed_acknowledgement)
    try:
        with pytest.raises(RedisError, match="registration"):
            await asyncio.wait_for(router.start(), 2)
        assert not router._healthy
        assert not router._workers
        assert router._supervisor is None
    finally:
        await router.shutdown()


async def test_stalled_recovery_registration_times_out_and_retries(
    replicas, monkeypatch
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    set_value = redis.set
    registering = asyncio.Event()

    async def delayed_acknowledgement(key, value, **kwargs):
        result = await set_value(key, value, **kwargs)
        if (
            key.startswith("mcp.owner.")
            and value != previous
            and not registering.is_set()
        ):
            registering.set()
            await asyncio.Event().wait()
        return result

    monkeypatch.setattr(redis, "set", delayed_acknowledgement)
    owner._workers[1].cancel()
    await asyncio.wait_for(registering.wait(), 2)
    with pytest.raises(HTTPException) as unavailable:
        owner._ensure_healthy()
    assert unavailable.value.status_code == 503
    await wait_for_recovery(owner, previous)
    assert not await redis.exists(owner._owner_key(previous))
    async with clients(apps) as (first, second):
        replacement = await initialize(first)
        assert (await tools(second, replacement)).status_code == 200


@pytest.mark.parametrize(
    ("phase", "failure"),
    [("accepted", ConnectionError), ("start", RuntimeError), ("body", ConnectionError)],
)
async def test_fatal_reply_failure_retires_owner_without_needing_an_error_frame(
    replicas, monkeypatch, phase, failure
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    append = owner._append
    failed_reply = None
    call_tool = AsyncMock(return_value=[TextContent(type="text", text="once")])

    async def failing_append(key, payload, limit):
        nonlocal failed_reply
        if key.startswith("mcp.reply."):
            message = json.loads(payload)
            if failed_reply is None and message["type"] == phase:
                failed_reply = key
            if key == failed_reply:
                raise failure("Simulated owner publication failure")
        return await append(key, payload, limit)

    monkeypatch.setattr(owner, "_append", failing_append)
    monkeypatch.setattr(mcp_interaction, "call_tool", call_tool)
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        other_owner = apps[1].mcp_session_router.owner
        other_session = await initialize(second)
        async with running_request(ask(second, session_id)) as call:
            if phase == "body":
                with pytest.raises(
                    RuntimeError, match="response already started"
                ) as stopped:
                    await asyncio.wait_for(asyncio.shield(call), 4)
                assert isinstance(stopped.value.__cause__, HTTPException)
                assert stopped.value.__cause__.status_code in (404, 503)
            else:
                response = await asyncio.wait_for(asyncio.shield(call), 4)
                assert response.status_code in (404, 503), response.text
            await wait_for_recovery(owner, previous)
            assert not await redis.exists(owner._owner_key(previous))
            assert not owner._tasks
            assert not owner._local_tasks
            assert not apps[0].mcp_server_pool._servers
            assert (await tools(first, session_id)).status_code == 404
            assert (await tools(second, session_id)).status_code == 404
            assert apps[1].mcp_session_router.owner == other_owner
            assert (await tools(first, other_session)).status_code == 200
            replacement = await initialize(first)
            assert (await tools(second, replacement)).status_code == 200
            assert call_tool.await_count == (0 if phase == "accepted" else 1)


@pytest.mark.parametrize(
    "failure", ["heartbeat", "listener", "execution", "lease", "shutdown"]
)
async def test_owner_loss_cancels_tools_and_requires_new_sessions(
    replicas, blocked_tool, failure
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        async with blocked_call(second, session_id, blocked_tool) as call:
            if failure == "shutdown":
                await owner.shutdown()
            elif failure == "lease":
                await redis.delete(owner._owner_key(previous))
            elif failure == "execution":
                next(iter(owner._tasks)).cancel()
            else:
                owner._workers[0 if failure == "heartbeat" else 1].cancel()
            response = await asyncio.wait_for(asyncio.shield(call.response), 4)
            assert response.status_code in (404, 503), response.text
            await asyncio.wait_for(call.cancelled.wait(), 2)
            if failure != "shutdown":
                await wait_for_recovery(owner, previous)
                assert (await tools(first, session_id)).status_code == 404
            else:
                assert owner._closed and not owner._healthy
                assert not owner._workers
            assert not await redis.exists(owner._owner_key(previous))
            assert not owner._tasks
            assert not owner._local_tasks
            assert not apps[0].mcp_server_pool._servers
            assert (await tools(second, session_id)).status_code == 404
            replacement = await initialize(second if failure == "shutdown" else first)
            assert (await tools(second, replacement)).status_code == 200
            call.tool.assert_awaited_once()


async def test_recovery_waits_for_sdk_invalidation_before_accepting_new_sessions(
    replicas, monkeypatch
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    invalidate = owner.invalidate_sessions
    invalidating = asyncio.Event()
    resume = asyncio.Event()

    async def paused_invalidation():
        invalidating.set()
        await resume.wait()
        await invalidate()

    monkeypatch.setattr(owner, "invalidate_sessions", paused_invalidation)
    try:
        owner._workers[1].cancel()
        await asyncio.wait_for(invalidating.wait(), 2)
        assert not owner._healthy
        assert owner.owner == previous
        async with asyncio.timeout(2):
            while await redis.exists(owner._owner_key(previous)):
                await asyncio.sleep(0.01)
        async with clients(apps) as (first, second):
            assert (await tools(first, "unavailable")).status_code == 503
            resume.set()
            await wait_for_recovery(owner, previous)
            replacement = await initialize(first)
            assert (await tools(second, replacement)).status_code == 200
    finally:
        resume.set()


async def test_stalled_revocation_cannot_delay_sdk_cancellation_or_lease_expiry(
    replicas, monkeypatch, blocked_tool
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    eval_script = redis.eval
    revoking = asyncio.Event()

    async def stalled_revocation(script, numkeys, *args):
        if args[0] == owner._owner_key(previous) and "'DEL'" in script:
            revoking.set()
            await asyncio.Event().wait()
        return await eval_script(script, numkeys, *args)

    monkeypatch.setattr(redis, "eval", stalled_revocation)
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        async with blocked_call(second, session_id, blocked_tool) as call:
            owner._workers[1].cancel()
            await asyncio.wait_for(revoking.wait(), 2)
            await asyncio.wait_for(call.cancelled.wait(), 2)
            assert not owner._healthy
            assert owner.owner == previous
            response = await asyncio.wait_for(asyncio.shield(call.response), 4)
            assert response.status_code == 404
            await wait_for_recovery(owner, previous)
            assert not await redis.exists(owner._owner_key(previous))
            replacement = await initialize(first)
            assert (await tools(second, replacement)).status_code == 200


async def test_handled_request_error_does_not_invalidate_other_sessions(replicas):
    apps, _ = replicas
    owner = apps[0].mcp_session_router
    previous = owner.owner
    handler = owner.local_handler

    async def failing_call(key, request, body):
        if json.loads(body).get("method") == "tools/call":
            raise RuntimeError("Simulated request failure")
        return await handler(key, request, body)

    owner.local_handler = failing_call
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        assert (await ask(second, session_id)).status_code == 503
        assert (await tools(second, session_id)).status_code == 200
        assert owner.owner == previous
        assert owner._healthy


async def test_requests_and_delete_can_alternate_between_replicas(replicas):
    apps, redis = replicas
    async with clients(apps) as (first, second):
        a, b = await asyncio.gather(initialize(first), initialize(second))
        assert a != b
        for client, session_id in ((second, a), (first, b), (first, a), (second, b)):
            response = await tools(client, session_id)
            assert response.status_code == 200, response.text
            assert response.json()["result"]["tools"][0]["name"] == "ask"
        response = await second.delete(URL, headers={"Mcp-Session-Id": a})
        assert response.status_code == 200, response.text
        assert (
            await first.delete(URL, headers={"Mcp-Session-Id": a})
        ).status_code == 404
        assert (await tools(first, b)).status_code == 200
        assert not await redis.exists(apps[0].mcp_session_router._session_key(KEY, a))


async def test_unknown_or_wrong_identity_cannot_use_another_session(replicas):
    apps, _ = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        response = await tools(second, session_id, **{"X-STF-USER": "someone-else"})
        assert response.status_code == 404
        response = await tools(second, "unknown")
        assert response.status_code == 404
        assert not apps[1].mcp_server_pool._servers


async def test_long_calls_use_current_credentials_without_replaying(
    replicas, monkeypatch
):
    apps, _ = replicas
    seen = []

    async def call_tool(app, server, *args):
        if not seen:
            await asyncio.sleep(7)
        seen.append(server.request_context.request.headers["authorization"])
        return []

    monkeypatch.setattr(mcp_interaction, "call_tool", call_tool)
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        for token in ("fresh-one", "fresh-two"):
            response = await ask(second, session_id, Authorization=f"Bearer {token}")
            assert response.status_code == 200, response.text
            assert response.json()["result"]["content"] == []
        assert seen == ["Bearer fresh-one", "Bearer fresh-two"]


async def test_redis_failure_returns_503_without_local_fallback(replicas, monkeypatch):
    apps, redis = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        monkeypatch.setattr(redis, "get", AsyncMock(side_effect=ConnectionError()))
        assert (await tools(second, session_id)).status_code == 503
        assert (await tools(first, session_id)).status_code == 503
        assert not apps[1].mcp_server_pool._servers


async def test_sse_and_elicitation_cross_replicas_with_real_mcp_client(
    replicas, monkeypatch
):
    apps, redis = replicas
    apps[0].mcp_session_router.max_pending_requests = 1
    elicited = []
    executions = []

    async def call_tool(app, server, *args):
        executions.append(app)
        session = server.request_context.session
        form = await session.elicit_form(
            "Enter a value",
            {"type": "object", "properties": {"value": {"type": "string"}}},
        )
        url = await session.elicit_url(
            "Authorize", "https://example.com/authorize", "oauth-request"
        )
        return [
            TextContent(
                type="text",
                text=f"{form.content['value']}:{url.action}",
            )
        ]

    async def elicit(context, params):
        elicited.append(params.mode)
        if params.mode == "url":
            return ElicitResult(action="accept")
        return ElicitResult(action="accept", content={"value": "confirmed"})

    monkeypatch.setattr(mcp_interaction, "call_tool", call_tool)
    requests = []

    async def balanced(scope, receive, send):
        # Initialize on A, then relay every subsequent request through B.
        index = int(any(name == b"mcp-session-id" for name, _ in scope["headers"]))
        requests.append((scope["method"], index))
        await apps[index](scope, receive, send)

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(balanced, lifespan="off", log_level="error"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                await asyncio.sleep(0.01)
        async with asyncio.timeout(15):
            async with httpx.AsyncClient(headers=HEADERS) as http:
                async with streamable_http_client(
                    f"http://127.0.0.1:{port}{URL}", http_client=http
                ) as (read, write, _):
                    async with ClientSession(
                        read, write, elicitation_callback=elicit
                    ) as session:
                        await session.initialize()
                        result = await session.call_tool("ask", {})
                        assert not result.isError
                        assert result.content[0].text == "confirmed:accept"
                        assert (await session.list_tools()).tools[0].name == "ask"
        assert elicited == ["form", "url"]
        assert executions == [apps[0]]
        assert ("GET", 1) in requests
        async with asyncio.timeout(5):
            while apps[0].mcp_session_router._tasks:
                await asyncio.sleep(0.05)
        assert not await redis.keys("mcp.request.*")
        assert not await redis.keys("mcp.reply.*")
        assert not await redis.keys("mcp.live.*")
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 5)
        sock.close()


async def test_response_backpressure_and_disconnect_clean_up_owner_task(replicas):
    apps, redis = replicas
    stopped = asyncio.Event()

    async def chunks():
        asyncio.current_task().add_done_callback(lambda task: stopped.set())
        while True:
            yield b"x" * 65536

    async def handler(*args):
        return StreamingResponse(chunks())

    owner = apps[0].mcp_session_router
    previous = owner.owner
    disconnect = asyncio.Event()
    sending = asyncio.Event()

    async def receive():
        await disconnect.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            sending.set()
            await asyncio.Event().wait()

    async with running_request(relay_stream(replicas, handler, receive, send)) as task:
        async with asyncio.timeout(5):
            await sending.wait()
            while True:
                keys = await redis.keys("mcp.reply.*")
                if keys and await redis.xlen(keys[0]) == 16:
                    break
                await asyncio.sleep(0.01)
        disconnect.set()
        await asyncio.wait_for(task, 5)
        await asyncio.wait_for(stopped.wait(), 5)
        async with asyncio.timeout(2):
            while owner._tasks or owner._local_tasks:
                await asyncio.sleep(0.01)
        assert owner.owner == previous
        assert owner._healthy
        assert not await redis.keys("mcp.live.*")
        assert not await redis.keys("mcp.request.*")


async def test_redis_retries_do_not_duplicate_sessions_calls_or_frames(
    replicas, monkeypatch
):
    apps, redis = replicas
    owner = apps[0].mcp_session_router
    owner.max_pending_requests = 1
    publish = owner._publish
    eval_script, set_value = redis.eval, redis.set
    call_tool = AsyncMock(return_value=[TextContent(type="text", text="once")])

    async def delayed_acceptance(request_id, message, *, sequence):
        if message["type"] == "accepted":
            await asyncio.sleep(0.1)
        await publish(request_id, message, sequence=sequence)

    async def retried_eval(script, numkeys, *args):
        result = await eval_script(script, numkeys, *args)
        if script == mcp_redis._APPEND:
            assert result
            return await eval_script(script, numkeys, *args)
        return result

    async def retried_set(name, value, **kwargs):
        result = await set_value(name, value, **kwargs)
        return await set_value(name, value, **kwargs) if kwargs.get("nx") else result

    monkeypatch.setattr(owner, "_publish", delayed_acceptance)
    monkeypatch.setattr(redis, "eval", retried_eval)
    monkeypatch.setattr(redis, "set", retried_set)
    monkeypatch.setattr(mcp_interaction, "call_tool", call_tool)
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        assert (
            sum(
                server.active_sessions
                for server in apps[0].mcp_server_pool._servers.values()
            )
            == 1
        )
        response = await ask(second, session_id)
        assert response.status_code == 200, response.text
        assert response.json()["result"]["content"][0]["text"] == "once"
        call_tool.assert_awaited_once()


async def test_expired_registry_and_oversized_headers_fail_explicitly(replicas):
    apps, redis = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        response = await tools(second, session_id, **{"X-Large": "x" * 65536})
        assert response.status_code == 431
        await redis.expire(apps[0].mcp_session_router._session_key(KEY, session_id), 1)
        await asyncio.sleep(1.1)
        assert (await tools(first, session_id)).status_code == 404
        assert (await tools(second, session_id)).status_code == 404


async def test_abandoned_request_at_capacity_does_not_stop_listener(replicas):
    apps, redis = replicas
    router = apps[0].mcp_session_router
    router.max_pending_requests = 0
    await router._append(router._inbox(router.owner), "abandoned", 1)
    async with asyncio.timeout(5):
        while await redis.xlen(router._inbox(router.owner)):
            await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert not router._workers[1].done()
    router.max_pending_requests = 100
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        assert (await tools(second, session_id)).status_code == 200


async def test_request_refreshes_registry_before_waiting_for_tool_result(
    replicas, blocked_tool
):
    apps, redis = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        registry_key = apps[0].mcp_session_router._session_key(KEY, session_id)
        await redis.expire(registry_key, 1)
        async with blocked_call(second, session_id, blocked_tool):
            assert await redis.ttl(registry_key) > 1


async def test_cancellation_reaches_owner_at_capacity(replicas, blocked_tool):
    apps, _ = replicas
    apps[0].mcp_session_router.max_pending_requests = 1
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        async with blocked_call(second, session_id, blocked_tool) as call:
            assert (await tools(second, session_id)).status_code == 503
            assert not apps[1].mcp_server_pool._servers
            response = await second.post(
                URL,
                headers={"Mcp-Session-Id": session_id},
                json={
                    "jsonrpc": "2.0",
                    "method": "notifications/cancelled",
                    "params": {"requestId": 3, "reason": "Client cancelled"},
                },
            )
            assert response.status_code == 202
            await asyncio.wait_for(call.cancelled.wait(), 5)
            response = await asyncio.wait_for(call.response, 5)
            assert response.json()["error"]["message"] == "Request cancelled"
            assert (await tools(second, session_id)).status_code == 200


async def test_expired_caller_lease_returns_503_after_acceptance(
    replicas, blocked_tool
):
    apps, redis = replicas
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        async with blocked_call(second, session_id, blocked_tool) as call:
            live_keys = await redis.keys("mcp.live.*")
            assert len(live_keys) == 1
            await redis.delete(live_keys[0])
            response = await asyncio.wait_for(asyncio.shield(call.response), 2)
            assert response.status_code == 503, response.text


async def test_expired_caller_lease_terminates_started_response(replicas):
    _, redis = replicas
    received = asyncio.Event()
    messages = []

    async def chunks():
        yield b"started"
        await asyncio.Event().wait()

    async def handler(*args):
        return StreamingResponse(chunks())

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            received.set()

    async with running_request(relay_stream(replicas, handler, receive, send)) as task:
        await asyncio.wait_for(received.wait(), 5)
        live_keys = await redis.keys("mcp.live.*")
        assert len(live_keys) == 1
        await redis.delete(live_keys[0])
        with pytest.raises(HTTPException) as exc:
            await asyncio.wait_for(asyncio.shield(task), 2)
        assert exc.value.status_code == 503
        assert [m["type"] for m in messages] == [
            "http.response.start",
            "http.response.body",
        ]
        assert messages[1]["body"] == b"started"


async def test_full_request_inbox_does_not_block_control_messages(
    replicas, monkeypatch
):
    apps, redis = replicas
    owner, remote = (app.mcp_session_router for app in apps)
    owner.max_pending_requests = remote.max_pending_requests = 1
    inbox = owner._inbox(owner.owner)
    paused = asyncio.Event()
    resume = asyncio.Event()
    xread = redis.xread

    async def paused_xread(streams, **kwargs):
        if inbox in streams:
            paused.set()
            await resume.wait()
        return await xread(streams, **kwargs)

    monkeypatch.setattr(redis, "xread", paused_xread)
    try:
        await asyncio.wait_for(paused.wait(), 5)
        assert await owner._append(inbox, "queued-request", 1)
        async with clients(apps) as (first, second):
            session_id = await initialize(first)
            assert (await tools(second, session_id)).status_code == 503
            response = await second.post(
                URL,
                headers={"Mcp-Session-Id": session_id},
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            assert response.status_code == 202, response.text
            assert await redis.xlen(inbox) == 1
    finally:
        resume.set()


async def test_control_capacity_is_bounded_and_independent_of_calls(replicas):
    apps, _ = replicas
    owner = apps[0].mcp_session_router
    owner.max_pending_requests = 1
    local_handler = owner.local_handler
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def handler(key, request, body):
        if request.method == "DELETE":
            entered.set()
            await finish.wait()
        return await local_handler(key, request, body)

    owner.local_handler = handler
    async with clients(apps) as (first, second):
        session_id = await initialize(first)
        headers = {"Mcp-Session-Id": session_id}
        delete = asyncio.create_task(second.delete(URL, headers=headers))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            assert (await second.delete(URL, headers=headers)).status_code == 503
            assert (await tools(second, session_id)).status_code == 200
        finally:
            finish.set()
            response = await asyncio.wait_for(delete, 5)
            assert response.status_code == 200, response.text


async def test_sse_pings_preserve_message_order_and_fragmentation(
    replicas, monkeypatch
):
    apps, _ = replicas
    owner = apps[0].mcp_session_router
    append = owner._append
    delayed = False

    async def delayed_append(key, payload, limit):
        nonlocal delayed
        if key.startswith("mcp.reply."):
            message = json.loads(payload)
            if (
                message["type"] == "body"
                and b"payload" in base64.b64decode(message["body"])
                and not delayed
            ):
                delayed = True
                await asyncio.sleep(0.1)
        return await append(key, payload, limit)

    monkeypatch.setattr(owner, "_append", delayed_append)
    payload = "payload" + "x" * (65536 + 1024)
    expected = ServerSentEvent(data=payload).encode()

    async def events():
        yield {"data": payload}
        await asyncio.Event().wait()

    async def handler(*args):
        return EventSourceResponse(events(), ping=0.01)

    disconnected = asyncio.Event()
    chunks = []

    async def receive():
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
            wire = b"".join(chunks)
            if expected in wire and b": ping" in wire:
                disconnected.set()

    await asyncio.wait_for(relay_stream(replicas, handler, receive, send), 5)
    wire = b"".join(chunks)
    assert delayed
    assert wire.count(expected) == 1
    assert b": ping" in wire.replace(expected, b"")
    async with asyncio.timeout(5):
        while owner._tasks or owner._local_tasks:
            await asyncio.sleep(0.01)
