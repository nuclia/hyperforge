"""Integration tests for Redis-backed A2A pending-task correlation."""

import asyncio
from uuid import uuid4

from redis.asyncio import Redis

from hyperforge.a2a.task_store import PendingTaskRecord, RedisA2ATaskStore


def _record(task_id: str = "task-1") -> PendingTaskRecord:
    return PendingTaskRecord(
        task_id=task_id,
        context_id="context-1",
        routing={
            "account": "local",
            "agent_id": "agent-1",
            "workflow_id": "default",
            "session": "session-1",
            "headers": {"authorization": "Bearer token"},
            "arguments": {"region": "EMEA"},
        },
        feedback_id="feedback-1",
        request_id="request-1",
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
