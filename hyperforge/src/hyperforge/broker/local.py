import asyncio
from typing import AsyncIterator

from lru import LRU
from pydantic import TypeAdapter

from hyperforge.broker import AgentTimeoutError, Broker
from hyperforge.pubsub import AgentMessage, StartInteraction


class LocalBroker(Broker):
    """In-process broker implementation. No Redis required.
    API and server must share the same instance."""

    def __init__(self, keepalive_ms: int = 20000, max_streams: int = 500):
        self._keepalive_ms = keepalive_ms
        self._activation_queue: asyncio.Queue[
            tuple[StartInteraction, dict[str, str]]
        ] = asyncio.Queue()
        # topic -> (messages list, condition). Bounded by LRU to avoid unbounded growth.
        self._streams: LRU = LRU(max_streams)
        # key -> queue (single item). Deleted after receive_reply.
        self._reply_channels: dict[str, asyncio.Queue[str]] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return str(self._counter)

    def _get_or_create_stream(
        self, topic: str
    ) -> tuple[list[tuple[str, str]], asyncio.Condition]:
        if topic not in self._streams:
            self._streams[topic] = ([], asyncio.Condition())
        return self._streams[topic]

    @property
    def keepalive_seconds(self) -> float:
        return self._keepalive_ms / 1000

    async def publish_activation(
        self, msg: StartInteraction, trace: dict[str, str]
    ) -> None:
        await self._activation_queue.put((msg, trace))

    async def subscribe_activations(
        self,
    ) -> AsyncIterator[tuple[StartInteraction, dict[str, str]]]:
        while True:
            msg, trace = await self._activation_queue.get()
            yield msg, trace

    async def publish(self, topic: str, message: AgentMessage) -> None:
        messages, condition = self._get_or_create_stream(topic)
        cursor = self._next_id()
        async with condition:
            messages.append((cursor, message.model_dump_json()))
            condition.notify_all()

    async def subscribe(
        self, topic: str, from_cursor: str = "0"
    ) -> AsyncIterator[tuple[str, AgentMessage]]:
        messages, condition = self._get_or_create_stream(topic)
        adapter: TypeAdapter[AgentMessage] = TypeAdapter(AgentMessage)

        # Find the starting index from the cursor
        last_index = 0
        if from_cursor != "0":
            for i, (cursor, _) in enumerate(messages):
                if cursor == from_cursor:
                    last_index = i + 1
                    break

        while True:
            async with condition:
                if last_index < len(messages):
                    batch = messages[last_index:]
                    last_index = len(messages)
                else:
                    try:
                        await asyncio.wait_for(
                            condition.wait(),
                            timeout=self._keepalive_ms / 1000,
                        )
                    except asyncio.TimeoutError:
                        raise AgentTimeoutError(topic)
                    batch = messages[last_index:]
                    last_index = len(messages)

            for cursor, raw in batch:
                yield cursor, adapter.validate_json(raw)

    async def send_reply(self, key: str, payload: str) -> None:
        if key not in self._reply_channels:
            self._reply_channels[key] = asyncio.Queue(maxsize=1)
        await self._reply_channels[key].put(payload)

    async def receive_reply(self, key: str, timeout_ms: int) -> str | None:
        if key not in self._reply_channels:
            self._reply_channels[key] = asyncio.Queue(maxsize=1)
        try:
            return await asyncio.wait_for(
                self._reply_channels[key].get(),
                timeout=timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            return None
        finally:
            self._reply_channels.pop(key, None)

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        self._streams.clear()
        self._reply_channels.clear()
