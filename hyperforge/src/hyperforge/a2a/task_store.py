"""Durable Redis stores for A2A task state and feedback correlation."""

import base64
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from a2a.server.context import ServerCallContext
from a2a.server.owner_resolver import OwnerResolver, resolve_user_scope
from a2a.server.tasks import TaskStore
from a2a.types import a2a_pb2
from a2a.types.a2a_pb2 import Task
from a2a.utils.constants import DEFAULT_LIST_TASKS_PAGE_SIZE
from a2a.utils.errors import InvalidParamsError
from a2a.utils.task import decode_page_token, encode_page_token
from redis.asyncio import Redis


@dataclass(frozen=True)
class PendingTaskRecord:
    """Serializable state needed to validate a later A2A feedback response."""

    task_id: str
    context_id: str
    routing: dict[str, Any]
    feedback_id: str
    request_id: str
    owner_instance_id: str = ""

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


class RedisA2ASDKTaskStore(TaskStore):
    """TTL-backed A2A SDK task storage shared by all A2A server pods.

    Task payloads are base64 encoded because the shared broker client uses
    ``decode_responses=True``. The hash tag keeps the task and owner index in
    one Valkey Cluster slot, allowing their pipeline to remain cluster-safe.
    """

    def __init__(
        self,
        client: Redis,
        key_prefix: str,
        ttl_seconds: int,
        owner_resolver: OwnerResolver = resolve_user_scope,
    ) -> None:
        self._client = client
        self._key_prefix = key_prefix.rstrip(":")
        self._ttl_seconds = ttl_seconds
        self._owner_resolver = owner_resolver

    def _owner_tag(self, owner: str) -> str:
        return f"{{{owner}}}"

    def _task_key(self, owner: str, task_id: str) -> str:
        return f"{self._key_prefix}:{self._owner_tag(owner)}:task:{task_id}"

    def _index_key(self, owner: str) -> str:
        return f"{self._key_prefix}:{self._owner_tag(owner)}:index"

    @staticmethod
    def _score(task: Task) -> float:
        if task.HasField("status") and task.status.HasField("timestamp"):
            return task.status.timestamp.ToDatetime().timestamp()
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _serialize(task: Task) -> str:
        return base64.b64encode(task.SerializeToString()).decode("ascii")

    @staticmethod
    def _deserialize(value: str) -> Task:
        task = Task()
        task.ParseFromString(base64.b64decode(value))
        return task

    async def save(self, task: Task, context: ServerCallContext) -> None:
        owner = self._owner_resolver(context)
        task_key = self._task_key(owner, task.id)
        index_key = self._index_key(owner)
        async with self._client.pipeline(transaction=True) as pipeline:
            pipeline.set(task_key, self._serialize(task), ex=self._ttl_seconds)
            pipeline.zadd(index_key, {task.id: self._score(task)})
            pipeline.expire(index_key, self._ttl_seconds)
            await pipeline.execute()

    async def get(self, task_id: str, context: ServerCallContext) -> Task | None:
        owner = self._owner_resolver(context)
        value = await self._client.get(self._task_key(owner, task_id))
        return self._deserialize(value) if value else None

    async def list(
        self,
        params: a2a_pb2.ListTasksRequest,
        context: ServerCallContext,
    ) -> a2a_pb2.ListTasksResponse:
        owner = self._owner_resolver(context)
        task_ids = await self._client.zrevrange(self._index_key(owner), 0, -1)
        tasks: list[Task] = []
        stale_ids: list[str] = []

        for task_id in task_ids:
            value = await self._client.get(self._task_key(owner, task_id))
            if value is None:
                stale_ids.append(task_id)
                continue
            task = self._deserialize(value)
            if params.context_id and task.context_id != params.context_id:
                continue
            if params.status and task.status.state != params.status:
                continue
            if params.HasField("status_timestamp_after"):
                if not task.status.HasField("timestamp"):
                    continue
                if (
                    task.status.timestamp.ToDatetime()
                    < params.status_timestamp_after.ToDatetime()
                ):
                    continue
            tasks.append(task)

        if stale_ids:
            await self._client.zrem(self._index_key(owner), *stale_ids)

        start_index = 0
        if params.page_token:
            start_task_id = decode_page_token(params.page_token)
            for index, task in enumerate(tasks):
                if task.id == start_task_id:
                    start_index = index
                    break
            else:
                raise InvalidParamsError(f"Invalid page token: {params.page_token}")

        page_size = params.page_size or DEFAULT_LIST_TASKS_PAGE_SIZE
        page = tasks[start_index : start_index + page_size]
        next_page_token = (
            encode_page_token(tasks[start_index + page_size].id)
            if start_index + page_size < len(tasks)
            else None
        )
        return a2a_pb2.ListTasksResponse(
            tasks=page,
            total_size=len(tasks),
            page_size=page_size,
            next_page_token=next_page_token,
        )

    async def delete(self, task_id: str, context: ServerCallContext) -> None:
        owner = self._owner_resolver(context)
        async with self._client.pipeline(transaction=True) as pipeline:
            pipeline.delete(self._task_key(owner, task_id))
            pipeline.zrem(self._index_key(owner), task_id)
            await pipeline.execute()
