from lru import LRU
from redis.asyncio import Redis

from hyperforge.models import HistoryQuestionAnswer, Source


class Cache:
    async def get(self, key: str) -> str | None:
        raise NotImplementedError()

    async def set(self, key: str, value: str, expire: int):
        raise NotImplementedError()

    async def get_list(self, key: str) -> list[str] | None:
        raise NotImplementedError()

    async def append(self, key: str, values: list[str], limit: int, expire: int):
        raise NotImplementedError()


class NoCache(Cache):
    async def get(self, key: str) -> str | None:
        return None

    async def get_list(self, key: str) -> list[str] | None:
        return None

    async def set(self, key: str, value: str, expire: int):
        pass

    async def append(self, key: str, values: list[str], limit: int, expire: int):
        pass


class InMemoryCache(Cache):
    _store: LRU
    _list_store: LRU

    def __init__(self, size: int = 3000):
        self._store = LRU(size)
        self._list_store = LRU(size)

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def get_list(self, key: str) -> list[str] | None:
        return self._list_store.get(key)

    async def set(self, key: str, value: str, expire: int):
        self._store[key] = value

    async def append(self, key: str, values: list[str], limit: int, expire: int):
        if key not in self._list_store:
            self._list_store[key] = []
        self._list_store[key].extend(values)
        # Enforce limit
        if len(self._list_store[key]) > limit:
            self._list_store[key] = self._list_store[key][-limit:]


class ValkeyCache(Cache):
    def __init__(self, client: Redis):
        self.client = client

    async def get(self, key: str) -> str | None:
        return await self.client.get(key)

    async def get_list(self, key: str) -> list[str] | None:
        list = await self.client.lrange(key, 0, -1)
        if len(list) == 0:
            # This can mean empty list or non-cached, be specific in return
            if await self.client.exists(key):
                return []
            else:
                return None
        else:
            return list

    async def set(self, key: str, value: str, expire: int):
        await self.client.set(key, value, ex=expire)

    async def append(self, key: str, values: list[str], limit: int, expire: int):
        async with self.client.pipeline() as multi:
            await (
                multi.rpush(key, *values)
                .ltrim(key, -limit, -1)
                .expire(key, expire)
                .execute()
            )


# The following are small wrapper classes for each cache item, to ensure they are
# always accessed in a consistent way.
#
# Key design notes:
# - Components separated by dot (.)
# - From greater to lesser specificity (so it's possible to invalidate by prefix)
# - Keys include descriptive components before IDs so it's easy to see what a random hex number means
class CachedNucliaDBSource:
    def __init__(self, cache: Cache, agent_id: str, source: str):
        self._cache = cache
        self._key = f"cache.agent.{agent_id}.source.{source}"

    async def get(self) -> Source | None:
        value = await self._cache.get(self._key)
        if value is None:
            return None
        return Source.model_validate_json(value)

    async def set(self, source: Source):
        await self._cache.set(self._key, source.model_dump_json(), expire=900)


class CachedSessionQA:
    def __init__(self, cache: Cache, agent_id: str, session: str):
        self._cache = cache
        self._key = f"cache.agent.{agent_id}.session.{session}.qa_history"

    async def get(self) -> list[HistoryQuestionAnswer] | None:
        value = await self._cache.get_list(self._key)
        if value is None:
            return None
        return [HistoryQuestionAnswer.model_validate_json(v) for v in value]

    async def append(self, qa: HistoryQuestionAnswer):
        await self.append_all([qa])

    async def append_all(self, qas: list[HistoryQuestionAnswer]):
        await self._cache.append(
            self._key, [qa.model_dump_json() for qa in qas], limit=20, expire=3600
        )
