import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import HTTPException
from mcp.types import InitializeRequestParams, RequestId
from pydantic import BaseModel
from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

from hyperforge.api import logger

MCPServerKey = tuple[str, str, str, str, str]
MAX_ACTIVE_CALLS = 100


class MCPSession(BaseModel):
    params: InitializeRequestParams
    ready: bool = False


class MCPUpdates(BaseModel):
    cancelled: bool = False
    replies: dict[str, str] = {}


class MCPSessionStore:
    """Session metadata and TTL-backed mailboxes; executions remain local."""

    def __init__(self, client: Redis | None = None):
        self.client = client
        self._values: dict[str, tuple[float, str]] = {}
        self.active_requests: set[asyncio.Task[Any]] = set()

    @staticmethod
    def keys(key: MCPServerKey, session_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(json.dumps(key).encode()).hexdigest()
        prefix = f"mcp:{{{digest}}}"
        return prefix + ":sessions", prefix + ":" + session_id

    @asynccontextmanager
    async def _available(self, ttl: int):
        try:
            async with asyncio.timeout(min(5, ttl / 3)):
                yield
        except (RedisError, TimeoutError) as exc:
            logger.warning("MCP coordination unavailable (%s)", type(exc).__name__)
            raise HTTPException(503, "MCP coordination is unavailable") from exc

    def _prune(self) -> None:
        now = time.monotonic()
        self._values = {
            key: value for key, value in self._values.items() if value[0] > now
        }

    async def _read(self, names: list[str], ttl: int) -> list[str | None]:
        if self.client is not None:
            async with (
                self._available(ttl),
                self.client.pipeline(transaction=True) as pipe,
            ):
                for name in names:
                    pipe.getex(name, ex=ttl)
                return await pipe.execute()
        self._prune()
        result = []
        for name in names:
            value = self._values.get(name)
            result.append(value[1] if value else None)
            if value:
                self._values[name] = (time.monotonic() + ttl, value[1])
        return result

    async def _add(self, name: str, value: str, ttl: int) -> None:
        if self.client is not None:
            async with self._available(ttl):
                added = await self.client.set(name, value, ex=ttl, nx=True)
        else:
            self._prune()
            added = name not in self._values
            if added:
                self._values[name] = (time.monotonic() + ttl, value)
        if not added:
            raise HTTPException(409, "MCP request is already active")

    async def _change(
        self, name: str, change: Callable[[str | None], str | None], ttl: int
    ) -> str | None:
        if self.client is not None:
            async with (
                self._available(ttl),
                self.client.pipeline(transaction=True) as pipe,
            ):
                while True:
                    try:
                        await pipe.watch(name)
                        previous = await pipe.get(name)
                        value = change(previous)
                        if value == previous:
                            return previous
                        pipe.multi()
                        if value is None:
                            pipe.delete(name)
                        else:
                            pipe.set(name, value, keepttl=True)
                        await pipe.execute()
                        return previous
                    except WatchError:
                        continue
        self._prune()
        entry = self._values.get(name)
        previous = entry[1] if entry else None
        value = change(previous)
        if value is None:
            self._values.pop(name, None)
        elif entry:
            self._values[name] = (entry[0], value)
        return previous

    async def _delete(self, names: list[str], ttl: int) -> None:
        if self.client is not None:
            async with self._available(ttl):
                await self.client.delete(*names)
        else:
            for name in names:
                self._values.pop(name, None)

    async def session_count(self, key: MCPServerKey, ttl: int) -> int:
        registry, prefix = self.keys(key, "")
        if self.client is not None:
            async with self._available(ttl):
                ids = list(await self.client.smembers(registry))
                if not ids:
                    return 0
                expired = await self._expired(prefix, ids)
                if expired:
                    await self.client.srem(registry, *expired)
                return len(ids) - len(expired)
        self._prune()
        return sum(
            name.startswith(prefix) and len(name) == len(prefix) + 32
            for name in self._values
        )

    async def _expired(self, prefix: str, ids: list[str]) -> list[str]:
        assert self.client is not None
        async with self.client.pipeline(transaction=False) as pipe:
            for sid in ids:
                pipe.exists(prefix + sid)
            present = await pipe.execute()
        return [sid for sid, exists in zip(ids, present) if not exists]

    async def create(
        self, key: MCPServerKey, params: InitializeRequestParams, ttl: int, limit: int
    ) -> str:
        sid = uuid.uuid4().hex
        registry, name = self.keys(key, sid)
        value = MCPSession(params=params).model_dump_json(by_alias=True)
        if self.client is not None:
            async with (
                self._available(ttl),
                self.client.pipeline(transaction=True) as pipe,
            ):
                while True:
                    try:
                        await pipe.watch(registry)
                        ids = list(await pipe.smembers(registry))
                        expired = await self._expired(self.keys(key, "")[1], ids)
                        if len(ids) - len(expired) >= limit:
                            raise HTTPException(
                                503, "MCP session capacity is exhausted"
                            )
                        remaining = await pipe.ttl(registry)
                        pipe.multi()
                        if expired:
                            pipe.srem(registry, *expired)
                        pipe.set(name, value, ex=ttl)
                        pipe.sadd(registry, sid)
                        pipe.expire(registry, max(ttl, remaining))
                        await pipe.execute()
                        break
                    except WatchError:
                        continue
        else:
            if await self.session_count(key, ttl) >= limit:
                raise HTTPException(503, "MCP session capacity is exhausted")
            self._values[name] = (time.monotonic() + ttl, value)
        return sid

    async def get(self, key: MCPServerKey, sid: str, ttl: int) -> MCPSession:
        registry, name = self.keys(key, sid)
        value = (await self._read([name], ttl))[0]
        if value is None:
            raise HTTPException(404, "MCP session not found")
        if self.client is not None:
            async with self._available(ttl):
                await self.client.expire(registry, ttl, gt=True)
        return MCPSession.model_validate_json(value)

    async def ready(self, key: MCPServerKey, sid: str, ttl: int) -> None:
        session = await self.get(key, sid, ttl)
        session.ready = True
        await self._change(
            self.keys(key, sid)[1],
            lambda value: session.model_dump_json(by_alias=True) if value else value,
            ttl,
        )

    async def delete(self, key: MCPServerKey, sid: str, ttl: int) -> None:
        await self._delete([self.keys(key, sid)[1]], ttl)

    def _call_key(self, key: MCPServerKey, sid: str, request_id: RequestId) -> str:
        digest = hashlib.sha256(json.dumps(request_id).encode()).hexdigest()
        return self.keys(key, sid)[1] + ":call:" + digest

    async def start(
        self, key: MCPServerKey, sid: str, request_id: RequestId, ttl: int
    ) -> str:
        token = uuid.uuid4().hex
        await self._add(self._call_key(key, sid, request_id), token, ttl)
        return token

    async def cancel(
        self, key: MCPServerKey, sid: str, request_id: RequestId, ttl: int
    ) -> None:
        await self._change(
            self._call_key(key, sid, request_id),
            lambda value: (
                "cancelled:" + value
                if value and not value.startswith("cancelled:")
                else value
            ),
            ttl,
        )

    def _reply_key(self, key: MCPServerKey, sid: str, wire_id: str) -> str:
        return self.keys(key, sid)[1] + ":reply:" + wire_id

    async def pending(
        self, key: MCPServerKey, sid: str, wire_id: str, ttl: int
    ) -> None:
        await self._add(self._reply_key(key, sid, wire_id), "", ttl)

    async def reply(
        self, key: MCPServerKey, sid: str, wire_id: str, body: str, ttl: int
    ) -> bool:
        previous = await self._change(
            self._reply_key(key, sid, wire_id),
            lambda value: body if value == "" else value,
            ttl,
        )
        if previous not in {None, "", body}:
            raise HTTPException(409, "MCP request already has a different response")
        return previous is not None

    async def poll(
        self,
        key: MCPServerKey,
        sid: str,
        request_id: RequestId,
        token: str,
        pending: list[str],
        ttl: int,
    ) -> MCPUpdates:
        await self.get(key, sid, ttl)
        call, *replies = await self._read(
            [
                self._call_key(key, sid, request_id),
                *(self._reply_key(key, sid, wire_id) for wire_id in pending),
            ],
            ttl,
        )
        return MCPUpdates(
            cancelled=call != token or any(reply is None for reply in replies),
            replies={
                wire_id: reply for wire_id, reply in zip(pending, replies) if reply
            },
        )

    async def acknowledge(
        self, key: MCPServerKey, sid: str, ids: list[str], ttl: int
    ) -> None:
        if ids:
            await self._delete([self._reply_key(key, sid, id_) for id_ in ids], ttl)

    async def finish(
        self,
        key: MCPServerKey,
        sid: str,
        request_id: RequestId,
        token: str,
        pending: list[str],
        ttl: int,
    ) -> None:
        await self._change(
            self._call_key(key, sid, request_id),
            lambda value: None if value in {token, "cancelled:" + token} else value,
            ttl,
        )
        await self.acknowledge(key, sid, pending, ttl)
