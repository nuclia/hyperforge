"""Integration tests for Redis-backed A2A pending-task correlation."""

import asyncio
from uuid import uuid4

from a2a.types import a2a_pb2
from redis.asyncio import Redis

from hyperforge.a2a.task_store import (
    PendingTaskRecord,
    RedisA2ASDKTaskStore,
    RedisA2ATaskStore,
)


def _record(task_id: str = "task-1") -> PendingTaskRecord:
    return PendingTaskRecord(
        task_id=task_id,
        context_id="context-1",
        feedback_id="feedback-1",
        request_id="request-1",
    )


def _task(task_id: str, context_id: str = "context-1") -> a2a_pb2.Task:
    return a2a_pb2.Task(
        id=task_id,
        context_id=context_id,
        status=a2a_pb2.TaskStatus(state=a2a_pb2.TaskState.TASK_STATE_WORKING),
    )


async def test_redis_a2a_task_store_claims_once_and_preserves_invalid_claims(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    store = RedisA2ATaskStore(redis, f"test:a2a:task:{uuid4().hex}", 30)
    record = _record()

    try:
        await store.save_pending(record)

        assert (
            await store.claim_pending("task-1", "wrong-context", "feedback-1") is None
        )
        assert await store.get_pending("task-1") == record
        assert (
            await store.claim_pending("task-1", "context-1", "wrong-feedback") is None
        )
        assert await store.get_pending("task-1") == record

        first, second = await asyncio.gather(
            store.claim_pending("task-1", "context-1", "feedback-1"),
            store.claim_pending("task-1", "context-1", "feedback-1"),
        )

        assert [first, second].count(record) == 1
        assert await store.get_pending("task-1") is None
    finally:
        await redis.aclose()  # type: ignore[attr-defined]


async def test_redis_a2a_task_store_expires_pending_records(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    store = RedisA2ATaskStore(redis, f"test:a2a:task:{uuid4().hex}", 1)

    try:
        await store.save_pending(_record())
        assert await store.get_pending("task-1") is not None

        await asyncio.sleep(1.1)

        assert await store.get_pending("task-1") is None
    finally:
        await redis.aclose()  # type: ignore[attr-defined]


async def test_redis_a2a_task_store_tracks_execution_owner(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    store = RedisA2ATaskStore(redis, f"test:a2a:task:{uuid4().hex}", 30)

    try:
        await store.save_owner("task-1", "instance-1")
        assert await store.get_owner("task-1") == "instance-1"

        await store.remove_owner("task-1")
        assert await store.get_owner("task-1") is None
    finally:
        await redis.aclose()  # type: ignore[attr-defined]


async def test_sdk_task_store_is_shared_between_server_instances(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    prefix = f"test:a2a:sdk-task:{uuid4().hex}"

    def owner_resolver(_context):
        return "account:agent"

    first_store = RedisA2ASDKTaskStore(redis, prefix, 30, owner_resolver)
    second_store = RedisA2ASDKTaskStore(redis, prefix, 30, owner_resolver)

    try:
        await first_store.save(_task("task-1"), context=None)  # type: ignore[arg-type]

        loaded = await second_store.get("task-1", context=None)  # type: ignore[arg-type]
        assert loaded == _task("task-1")

        page = await second_store.list(
            a2a_pb2.ListTasksRequest(),
            context=None,  # type: ignore[arg-type]
        )
        assert [task.id for task in page.tasks] == ["task-1"]
        assert page.total_size == 1

        await second_store.delete("task-1", context=None)  # type: ignore[arg-type]
        assert await first_store.get("task-1", context=None) is None  # type: ignore[arg-type]
    finally:
        await redis.aclose()  # type: ignore[attr-defined]


async def test_sdk_task_store_uses_ttl_and_cluster_safe_key_tag(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    prefix = f"test:a2a:sdk-task:{uuid4().hex}"
    store = RedisA2ASDKTaskStore(redis, prefix, 1, lambda _context: "account:agent")

    try:
        await store.save(_task("task-1"), context=None)  # type: ignore[arg-type]
        assert "{account:agent}" in store._task_key("account:agent", "task-1")
        assert "{account:agent}" in store._index_key("account:agent")

        await asyncio.sleep(1.1)

        assert await store.get("task-1", context=None) is None  # type: ignore[arg-type]
        page = await store.list(
            a2a_pb2.ListTasksRequest(),
            context=None,  # type: ignore[arg-type]
        )
        assert not page.tasks
    finally:
        await redis.aclose()  # type: ignore[attr-defined]


async def test_sdk_task_store_skips_corrupt_tasks(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    prefix = f"test:a2a:sdk-task:{uuid4().hex}"
    owner = "account:agent"
    store = RedisA2ASDKTaskStore(redis, prefix, 30, lambda _context: owner)

    try:
        await store.save(_task("valid"), context=None)  # type: ignore[arg-type]
        await redis.set(store._task_key(owner, "corrupt"), "not-base64")
        await redis.zadd(store._index_key(owner), {"corrupt": 1})

        assert await store.get("corrupt", context=None) is None  # type: ignore[arg-type]
        await redis.set(store._task_key(owner, "corrupt"), "not-base64")
        await redis.zadd(store._index_key(owner), {"corrupt": 1})
        page = await store.list(
            a2a_pb2.ListTasksRequest(),
            context=None,  # type: ignore[arg-type]
        )

        assert [task.id for task in page.tasks] == ["valid"]
    finally:
        await redis.aclose()  # type: ignore[attr-defined]
