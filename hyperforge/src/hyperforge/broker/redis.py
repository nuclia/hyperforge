import asyncio
import json
from typing import AsyncIterator, cast

import opentelemetry.propagate
from pydantic import TypeAdapter
from redis.asyncio import Redis, ResponseError

from hyperforge import logger
from hyperforge.broker import AgentTimeoutError, Broker
from hyperforge.pubsub import AgentMessage, StartInteraction
from hyperforge.redis_utils import ManualStreamKeysRedisCluster


class RedisBroker(Broker):
    def __init__(self, client: Redis, activate_subject: str, keepalive_ms: int):
        self._client = client
        self._activate_subject = activate_subject
        self._keepalive_ms = int(keepalive_ms)

    @property
    def keepalive_seconds(self) -> float:
        return self._keepalive_ms / 1000

    @classmethod
    def from_url(
        cls,
        url: str,
        activate_subject: str,
        keepalive_ms: int,
        cluster_mode: bool = False,
    ) -> "RedisBroker":
        if cluster_mode:
            client = cast(
                Redis,
                ManualStreamKeysRedisCluster.from_url(
                    url=url,
                    decode_responses=True,
                    dynamic_startup_nodes=False,
                ),
            )
        else:
            client = Redis.from_url(
                url,
                decode_responses=True,
            )
        return cls(client, activate_subject, keepalive_ms)

    async def publish_activation(
        self, msg: StartInteraction, trace: dict[str, str]
    ) -> None:
        await self._client.xadd(
            self._activate_subject,
            {"msg": msg.model_dump_json(), "trace": json.dumps(trace)},
            maxlen=100,
        )

    async def subscribe_activations(
        self,
    ) -> AsyncIterator[tuple[StartInteraction, dict[str, str]]]:
        try:
            await self._client.xgroup_create(
                name=self._activate_subject,
                groupname="arag_server",
                mkstream=True,
            )
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        while True:
            try:
                response = await self._client.xreadgroup(
                    groupname="arag_server",
                    consumername="any_server",
                    streams={self._activate_subject: ">"},
                    block=1000,
                    count=1,
                    noack=True,
                )
                if not response:
                    continue
                [_stream, messages] = response[0]
                if messages:
                    [_msgid, fields] = messages[0]
                    msg = StartInteraction.model_validate_json(fields["msg"])
                    trace = json.loads(fields.get("trace", "{}"))
                    yield msg, trace
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Activation subscription cancelled, exiting...")
                break
            except Exception:
                logger.exception("Error while subscribing to activations, retrying...")
                await asyncio.sleep(1)

    async def publish(self, topic: str, message: AgentMessage) -> None:
        async with self._client.pipeline() as pipe:
            await (
                pipe.xadd(topic, {"msg": message.model_dump_json()}, maxlen=100)
                .expire(topic, 300)
                .execute()
            )

    async def subscribe(
        self, topic: str, from_cursor: str = "0"
    ) -> AsyncIterator[tuple[str, AgentMessage]]:
        cursor = from_cursor
        retries = 0
        while True:
            try:
                response = await self._client.xread(
                    {topic: cursor},
                    block=max(50, int(self._keepalive_ms)),
                )
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info(f"Subscription to topic '{topic}' cancelled, exiting...")
                break
            except Exception:
                logger.exception(
                    f"Error while subscribing to topic '{topic}', retrying..."
                )
                retries += 1
                if retries > 5:
                    logger.error(
                        f"Too many errors while subscribing to topic '{topic}', giving up..."
                    )
                    raise AgentTimeoutError(topic)
                await asyncio.sleep(1)
                continue
            if not response:
                raise AgentTimeoutError(topic)
            [_stream, messages] = response[0]
            for msgid, fields in messages:
                cursor = msgid
                obj: AgentMessage = TypeAdapter(AgentMessage).validate_json(
                    fields["msg"]
                )
                yield cursor, obj

    async def send_reply(self, key: str, payload: str) -> None:
        trace_headers: dict[str, str] = {}
        opentelemetry.propagate.inject(trace_headers)
        await self._client.xadd(
            key, {"msg": payload, "trace": json.dumps(trace_headers)}, maxlen=100
        )

    async def receive_reply(self, key: str, timeout_ms: int) -> str | None:
        response = await self._client.xread(
            {key: "$"},
            block=timeout_ms,
            count=1,
        )
        if not response:
            return None
        return response[0][1][0][1]["msg"]

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        await self._client.aclose()  # type: ignore[attr-defined]
