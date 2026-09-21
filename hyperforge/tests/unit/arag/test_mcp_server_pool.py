import asyncio
import time
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import ValidationError
from starlette.requests import Request

from hyperforge.api.mcp_server_pool import ManagedMCPServer, MCPServerPool
from hyperforge.api.settings import Settings as ApiSettings
from hyperforge.standalone.settings import StandaloneSettings

KEY = ("account", "user", "type", "agent", "session")
OTHER_KEY = ("other-account", "user", "type", "agent", "session")


def request(*, session_id: str | None = None) -> Request:
    headers = [] if session_id is None else [(b"mcp-session-id", session_id.encode())]
    return Request({"type": "http", "method": "POST", "headers": headers})


class FakeManagedServer:
    def __init__(self):
        self.session_created = False
        self.active_requests = 0
        self.last_used = time.monotonic()
        self.task = None
        self.stopped = False

    def accept_request(self, current_request: Request) -> bool:
        if current_request.headers.get("mcp-session-id"):
            return True
        if self.session_created:
            return False
        self.session_created = True
        return True

    def is_reinitialization(self, current_request: Request, body: bytes) -> bool:
        return (
            self.session_created
            and current_request.headers.get("mcp-session-id") is None
            and b'"method":"initialize"' in body
        )

    def reserve_request(self) -> None:
        self.active_requests += 1
        self.last_used = time.monotonic()

    def release_request(self) -> None:
        self.active_requests -= 1
        self.last_used = time.monotonic()

    def close(self) -> None:
        self.stopped = True

    async def stop(self) -> None:
        self.stopped = True


async def factory(server: FakeManagedServer):
    return server


@pytest.mark.asyncio
async def test_pool_reuses_server_and_lease_releases_once():
    pool = MCPServerPool(2, 1800, 30)
    server = FakeManagedServer()

    lease = await pool.acquire(KEY, request(), b"", lambda: factory(server))
    lease.release()
    lease.release()
    reused = await pool.acquire(
        KEY, request(session_id="session-id"), b"", lambda: factory(FakeManagedServer())
    )

    assert reused.server is server
    assert server.active_requests == 1
    reused.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_rejects_concurrent_creation_for_same_key():
    pool = MCPServerPool(2, 1800, 30)
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()

    async def slow_factory():
        creation_started.set()
        await release_creation.wait()
        return FakeManagedServer()

    first = asyncio.create_task(pool.acquire(KEY, request(), b"", slow_factory))
    await creation_started.wait()

    with pytest.raises(HTTPException, match="already in progress") as error:
        await pool.acquire(KEY, request(), b"", lambda: factory(FakeManagedServer()))

    assert error.value.status_code == 409
    release_creation.set()
    lease = await first
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_creates_different_keys_concurrently():
    pool = MCPServerPool(2, 1800, 30)
    both_started = asyncio.Event()
    started = 0

    async def concurrent_factory():
        nonlocal started
        started += 1
        if started == 2:
            both_started.set()
        await both_started.wait()
        return FakeManagedServer()

    first = asyncio.create_task(pool.acquire(KEY, request(), b"", concurrent_factory))
    second = asyncio.create_task(
        pool.acquire(OTHER_KEY, request(), b"", concurrent_factory)
    )
    first_lease, second_lease = await asyncio.gather(first, second)

    first_lease.release()
    second_lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_counts_pending_creations_toward_capacity():
    pool = MCPServerPool(1, 1800, 30)
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()
    second_factory = AsyncMock()

    async def slow_factory():
        creation_started.set()
        await release_creation.wait()
        return FakeManagedServer()

    first = asyncio.create_task(pool.acquire(KEY, request(), b"", slow_factory))
    await creation_started.wait()

    with pytest.raises(HTTPException) as error:
        await pool.acquire(OTHER_KEY, request(), b"", second_factory)

    assert error.value.status_code == 503
    second_factory.assert_not_awaited()
    release_creation.set()
    lease = await first
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_preserves_recent_session_and_evicts_expired_session():
    pool = MCPServerPool(1, 60, 30)
    current = FakeManagedServer()
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(current))
    lease.release()

    with pytest.raises(HTTPException) as error:
        await pool.acquire(
            OTHER_KEY, request(), b"", lambda: factory(FakeManagedServer())
        )
    assert error.value.status_code == 503

    current.last_used = time.monotonic() - 60
    replacement = FakeManagedServer()
    replacement_lease = await pool.acquire(
        OTHER_KEY, request(), b"", lambda: factory(replacement)
    )

    assert current.stopped is True
    assert replacement_lease.server is replacement
    replacement_lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_replaces_expired_session_on_same_key():
    pool = MCPServerPool(2, 60, 30)
    expired = FakeManagedServer()
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(expired))
    lease.release()
    expired.last_used = time.monotonic() - 60
    replacement = FakeManagedServer()

    replacement_lease = await pool.acquire(
        KEY, request(), b"", lambda: factory(replacement)
    )

    assert expired.stopped is True
    assert replacement_lease.server is replacement
    replacement_lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_reaps_idle_server_without_another_acquire():
    pool = MCPServerPool(2, 0.01, 30)
    server = FakeManagedServer()
    stopped = asyncio.Event()

    async def stop():
        server.stopped = True
        stopped.set()

    server.stop = stop
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(server))
    lease.release()

    await asyncio.wait_for(stopped.wait(), timeout=1)

    assert KEY not in pool._servers
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_cleans_candidate_when_cancelled_waiting_for_lock():
    pool = MCPServerPool(1, 1800, 30)
    candidate = FakeManagedServer()
    factory_completed = asyncio.Event()
    release_factory = asyncio.Event()

    async def controlled_factory():
        await release_factory.wait()
        factory_completed.set()
        return candidate

    acquire_task = asyncio.create_task(
        pool.acquire(KEY, request(), b"", controlled_factory)
    )
    await asyncio.sleep(0)
    async with pool._lock:
        release_factory.set()
        await factory_completed.wait()
        acquire_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await acquire_task

    assert candidate.stopped is True
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_shutdown_cancels_pending_creation():
    pool = MCPServerPool(1, 1800, 30)
    creation_started = asyncio.Event()

    async def blocked_factory():
        creation_started.set()
        await asyncio.Event().wait()

    acquire_task = asyncio.create_task(
        pool.acquire(KEY, request(), b"", blocked_factory)
    )
    await creation_started.wait()
    await pool.shutdown()

    assert acquire_task.cancelled()
    with pytest.raises(HTTPException) as error:
        await pool.acquire(KEY, request(), b"", lambda: factory(FakeManagedServer()))
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_pool_remove_cancels_pending_creation():
    pool = MCPServerPool(1, 1800, 30)
    creation_started = asyncio.Event()

    async def blocked_factory():
        creation_started.set()
        await asyncio.Event().wait()

    acquire_task = asyncio.create_task(
        pool.acquire(KEY, request(), b"", blocked_factory)
    )
    await creation_started.wait()
    await pool.remove(KEY)

    assert acquire_task.cancelled()
    replacement = FakeManagedServer()
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(replacement))
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_remove_keeps_capacity_reserved_during_creation_cleanup():
    pool = MCPServerPool(1, 1800, 30)
    creation_started = asyncio.Event()
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def slow_cleanup_factory():
        creation_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            raise

    acquire_task = asyncio.create_task(
        pool.acquire(KEY, request(), b"", slow_cleanup_factory)
    )
    await creation_started.wait()
    remove_task = asyncio.create_task(pool.remove(KEY))
    await cleanup_started.wait()

    with pytest.raises(HTTPException) as error:
        await pool.acquire(
            OTHER_KEY, request(), b"", lambda: factory(FakeManagedServer())
        )

    assert error.value.status_code == 503
    release_cleanup.set()
    await remove_task
    assert acquire_task.cancelled()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_concurrent_shutdown_calls_wait_for_same_cleanup():
    pool = MCPServerPool(1, 1800, 30)
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()
    server = FakeManagedServer()

    async def stop():
        cleanup_started.set()
        await release_cleanup.wait()

    server.task = asyncio.create_task(stop())
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(server))
    lease.release()
    first_shutdown = asyncio.create_task(pool.shutdown())
    await cleanup_started.wait()
    second_shutdown = asyncio.create_task(pool.shutdown())
    await asyncio.sleep(0)

    assert not first_shutdown.done()
    assert not second_shutdown.done()
    release_cleanup.set()
    await asyncio.gather(first_shutdown, second_shutdown)


@pytest.mark.asyncio
async def test_pool_reinitializes_inactive_session():
    pool = MCPServerPool(2, 1800, 30)
    original = FakeManagedServer()
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(original))
    lease.release()
    replacement = FakeManagedServer()

    replacement_lease = await pool.acquire(
        KEY,
        request(),
        b'{"jsonrpc":"2.0","method":"initialize","id":1}',
        lambda: factory(replacement),
    )

    assert original.stopped is True
    assert replacement_lease.server is replacement
    replacement_lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_rejects_reinitialization_while_session_is_active():
    pool = MCPServerPool(2, 1800, 30)
    server = FakeManagedServer()
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(server))

    with pytest.raises(HTTPException, match="still active") as error:
        await pool.acquire(
            KEY,
            request(),
            b'{"jsonrpc":"2.0","method":"initialize","id":1}',
            lambda: factory(FakeManagedServer()),
        )

    assert error.value.status_code == 409
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_timeout_cleans_pending_creation():
    pool = MCPServerPool(1, 1800, 0.01)

    async def blocked_factory():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await pool.acquire(KEY, request(), b"", blocked_factory)

    server = FakeManagedServer()
    lease = await pool.acquire(KEY, request(), b"", lambda: factory(server))
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_managed_server_start_propagates_failure_and_cancellation():
    manager = cast(StreamableHTTPSessionManager, object())
    server = ManagedMCPServer(manager)

    async def fail_startup():
        raise RuntimeError("startup failed")

    server.run = fail_startup
    with pytest.raises(RuntimeError, match="startup failed"):
        await server.start()

    cancelled_server = ManagedMCPServer(manager)

    async def wait_forever():
        await asyncio.Event().wait()

    cancelled_server.run = wait_forever
    start_task = asyncio.create_task(cancelled_server.start())
    await asyncio.sleep(0)
    start_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await start_task

    assert cancelled_server.task is not None
    assert cancelled_server.task.done()


@pytest.mark.parametrize("settings_type", [ApiSettings, StandaloneSettings])
@pytest.mark.parametrize(
    "field",
    [
        "mcp_max_request_bytes",
        "mcp_max_response_bytes",
        "mcp_max_servers",
        "mcp_session_idle_ttl_seconds",
        "mcp_startup_timeout_seconds",
    ],
)
def test_mcp_settings_require_positive_limits(settings_type, field):
    with pytest.raises(ValidationError):
        settings_type(**{field: 0})
