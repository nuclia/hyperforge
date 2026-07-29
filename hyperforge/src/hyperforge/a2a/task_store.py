"""Durable correlation records for A2A tasks awaiting Hyperforge feedback."""

import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from redis.asyncio import Redis


@dataclass(frozen=True)
class PendingTaskRecord:
    """Serializable state needed to validate a later A2A feedback response."""

    task_id: str
    context_id: str
    routing: dict[str, Any]
    feedback_id: str
    request_id: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, value: str) -> "PendingTaskRecord":
        return cls(**json.loads(value))


class A2ATaskStore(Protocol):
    async def save_pending(self, record: PendingTaskRecord) -> None: ...

    async def get_pending(self, task_id: str) -> PendingTaskRecord | None: ...

    async def claim_pending(
        self, task_id: str, context_id: str, feedback_id: str
    ) -> PendingTaskRecord | None: ...

    async def remove(self, task_id: str) -> None: ...


class RedisA2ATaskStore:
    """Redis-backed pending-task correlation with an expiry matching feedback TTL."""

    def __init__(self, client: Redis, key_prefix: str, ttl_seconds: int) -> None:
        self._client = client
        self._key_prefix = key_prefix
        self._ttl_seconds = ttl_seconds

    def _key(self, task_id: str) -> str:
        return f"{self._key_prefix}:{task_id}"

    async def save_pending(self, record: PendingTaskRecord) -> None:
        await self._client.set(
            self._key(record.task_id), record.to_json(), ex=self._ttl_seconds
        )

    async def get_pending(self, task_id: str) -> PendingTaskRecord | None:
        value = await self._client.get(self._key(task_id))
        return PendingTaskRecord.from_json(value) if value else None

    async def claim_pending(
        self, task_id: str, context_id: str, feedback_id: str
    ) -> PendingTaskRecord | None:
        value = await self._client.eval(
            """
            local value = redis.call('GET', KEYS[1])
            if not value then
                return nil
            end
            local record = cjson.decode(value)
            if record.context_id ~= ARGV[1] or record.feedback_id ~= ARGV[2] then
                return nil
            end
            redis.call('DEL', KEYS[1])
            return value
            """,
            1,
            self._key(task_id),
            context_id,
            feedback_id,
        )
        return PendingTaskRecord.from_json(value) if value else None

    async def remove(self, task_id: str) -> None:
        await self._client.delete(self._key(task_id))
