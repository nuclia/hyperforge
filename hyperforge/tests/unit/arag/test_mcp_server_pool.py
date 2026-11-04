import asyncio
import time
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from hyperforge.api.app import HTTPApplication
from hyperforge.api.mcp_server_pool import ManagedMCPServer, MCPServerPool
from hyperforge.api.mcp_transport import MCPTransport
from hyperforge.api.settings import Settings as ApiSettings
from hyperforge.db.settings import DataManagerSettings
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.settings import StandaloneSettings

KEY = ("account", "user", "type", "agent", "session")
OTHER_KEY = ("other-account", "user", "type", "agent", "session")


def test_api_mcp_settings_use_the_original_configuration_before_startup():
    settings = ApiSettings(mcp_max_sessions_per_server=2)
    app = HTTPApplication(
        settings=settings,
        data_manager_settings=DataManagerSettings(
            postgresql_dsn="postgresql://localhost/test"
        ),
    )
    assert app.mcp_settings is settings
    settings.mcp_max_sessions_per_server = 3
    assert app.mcp_settings.mcp_max_sessions_per_server == 3


def test_standalone_mcp_settings_do_not_depend_on_server_startup():
    settings = StandaloneSettings(mcp_max_sessions_per_server=2)
    app = StandaloneApplication(agents_cfg={}, settings=settings)
    assert app.mcp_settings is settings
    settings.mcp_max_sessions_per_server = 3
    assert app.mcp_settings.mcp_max_sessions_per_server == 3


class FakeManagedServer:
    def __init__(self):
        self.active_sessions = 1
        self.manager = AsyncMock()
        self.manager.session_count.side_effect = lambda: self.active_sessions
        self.has_been_leased = False
        self.active_requests = 0
        self.last_used = time.monotonic()
        self.task = None
        self.stopped = False

    def reserve_request(self) -> None:
        self.has_been_leased = True
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

    lease = await pool.acquire(KEY, lambda: factory(server))
    lease.release()
    lease.release()
    reused = await pool.acquire(KEY, lambda: factory(FakeManagedServer()))

    assert reused.server is server
    assert server.active_requests == 1
    reused.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_shares_concurrent_creation_for_same_key():
    pool = MCPServerPool(2, 1800, 30)
    creation_started = asyncio.Event()
    release_creation = asyncio.Event()
    shared_server = FakeManagedServer()

    async def slow_factory():
        creation_started.set()
        await release_creation.wait()
        return shared_server

    first = asyncio.create_task(pool.acquire(KEY, slow_factory))
    await creation_started.wait()
    unused_factory = AsyncMock()
    second = asyncio.create_task(pool.acquire(KEY, unused_factory))
    await asyncio.sleep(0)
    release_creation.set()
    leases = await asyncio.gather(first, second)
    assert all(lease.server is shared_server for lease in leases)
    assert shared_server.active_requests == 2
    unused_factory.assert_not_awaited()
    for lease in leases:
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

    first = asyncio.create_task(pool.acquire(KEY, concurrent_factory))
    second = asyncio.create_task(pool.acquire(OTHER_KEY, concurrent_factory))
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

    first = asyncio.create_task(pool.acquire(KEY, slow_factory))
    await creation_started.wait()

    with pytest.raises(HTTPException) as error:
        await pool.acquire(OTHER_KEY, second_factory)

    assert error.value.status_code == 503
    second_factory.assert_not_awaited()
    release_creation.set()
    lease = await first
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_does_not_evict_a_manager_before_its_first_lease():
    pool = MCPServerPool(1, 1800, 30)
    server = FakeManagedServer()
    server.active_sessions = 0
    pool._servers[KEY] = server
    unused_factory = AsyncMock()
    with pytest.raises(HTTPException) as error:
        await pool.acquire(OTHER_KEY, unused_factory)
    assert error.value.status_code == 503
    assert pool._servers[KEY] is server
    assert server.stopped is False
    unused_factory.assert_not_awaited()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_preserves_recent_session_and_evicts_expired_session():
    pool = MCPServerPool(1, 60, 30)
    current = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(current))
    lease.release()

    with pytest.raises(HTTPException) as error:
        await pool.acquire(OTHER_KEY, lambda: factory(FakeManagedServer()))
    assert error.value.status_code == 503

    current.last_used = time.monotonic() - 60
    replacement = FakeManagedServer()
    replacement_lease = await pool.acquire(OTHER_KEY, lambda: factory(replacement))

    assert current.stopped is True
    assert replacement_lease.server is replacement
    replacement_lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_replaces_expired_session_on_same_key():
    pool = MCPServerPool(2, 60, 30)
    expired = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(expired))
    lease.release()
    expired.last_used = time.monotonic() - 60
    replacement = FakeManagedServer()

    replacement_lease = await pool.acquire(KEY, lambda: factory(replacement))

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
    lease = await pool.acquire(KEY, lambda: factory(server))
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

    acquire_task = asyncio.create_task(pool.acquire(KEY, controlled_factory))
    await asyncio.sleep(0)
    async with pool._lock:
        release_factory.set()
        await factory_completed.wait()
        acquire_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await acquire_task

    await pool.shutdown()
    assert candidate.stopped is True


@pytest.mark.asyncio
async def test_pool_shutdown_cancels_pending_creation():
    pool = MCPServerPool(1, 1800, 30)
    creation_started = asyncio.Event()

    async def blocked_factory():
        creation_started.set()
        await asyncio.Event().wait()

    acquire_task = asyncio.create_task(pool.acquire(KEY, blocked_factory))
    await creation_started.wait()
    await pool.shutdown()

    assert acquire_task.cancelled()
    with pytest.raises(HTTPException) as error:
        await pool.acquire(KEY, lambda: factory(FakeManagedServer()))
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_pool_remove_cancels_pending_creation():
    pool = MCPServerPool(1, 1800, 30)
    creation_started = asyncio.Event()

    async def blocked_factory():
        creation_started.set()
        await asyncio.Event().wait()

    acquire_task = asyncio.create_task(pool.acquire(KEY, blocked_factory))
    await creation_started.wait()
    await pool.remove(KEY)

    assert acquire_task.cancelled()
    replacement = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(replacement))
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

    acquire_task = asyncio.create_task(pool.acquire(KEY, slow_cleanup_factory))
    await creation_started.wait()
    remove_task = asyncio.create_task(pool.remove(KEY))
    await cleanup_started.wait()

    with pytest.raises(HTTPException) as error:
        await pool.acquire(OTHER_KEY, lambda: factory(FakeManagedServer()))

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
    lease = await pool.acquire(KEY, lambda: factory(server))
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
async def test_pool_reuses_manager_for_another_initialization():
    pool = MCPServerPool(2, 1800, 30)
    original = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(original))
    lease.release()
    unused_factory = AsyncMock()

    replacement_lease = await pool.acquire(
        KEY,
        unused_factory,
    )

    assert original.stopped is False
    assert replacement_lease.server is original
    unused_factory.assert_not_awaited()
    replacement_lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_accepts_initialization_while_another_request_is_active():
    pool = MCPServerPool(2, 1800, 30)
    server = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(server))

    second = await pool.acquire(
        KEY,
        lambda: factory(FakeManagedServer()),
    )
    assert second.server is server
    assert server.active_requests == 2
    second.release()
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_can_create_a_manager_for_a_shared_session():
    pool = MCPServerPool(1, 1800, 30)
    server = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(server))
    assert lease.server is server
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_cancelling_one_waiter_does_not_cancel_shared_creation():
    pool = MCPServerPool(1, 1800, 30)
    started = asyncio.Event()
    proceed = asyncio.Event()
    server = FakeManagedServer()

    async def create():
        started.set()
        await proceed.wait()
        return server

    first = asyncio.create_task(pool.acquire(KEY, create))
    await started.wait()
    second = asyncio.create_task(pool.acquire(KEY, AsyncMock()))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    proceed.set()
    lease = await second
    assert lease.server is server
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_pool_timeout_cleans_pending_creation():
    pool = MCPServerPool(1, 1800, 0.01)

    async def blocked_factory():
        await asyncio.Event().wait()

    with pytest.raises(TimeoutError):
        await pool.acquire(KEY, blocked_factory)

    server = FakeManagedServer()
    lease = await pool.acquire(KEY, lambda: factory(server))
    lease.release()
    await pool.shutdown()


@pytest.mark.asyncio
async def test_managed_server_start_propagates_failure_and_cancellation():
    manager = cast(MCPTransport, object())
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
        "mcp_max_sessions_per_server",
        "mcp_session_idle_ttl_seconds",
        "mcp_startup_timeout_seconds",
    ],
)
def test_mcp_settings_require_positive_limits(settings_type, field):
    with pytest.raises(ValidationError):
        settings_type(**{field: 0})
