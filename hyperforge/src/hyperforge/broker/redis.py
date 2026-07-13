import asyncio
import json
import time
from typing import Any, AsyncIterator, cast

import opentelemetry.propagate
from pydantic import TypeAdapter
from redis.asyncio import Redis, ResponseError
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import ConnectionError as RedisConnectionError

from hyperforge import logger
from hyperforge.broker import AgentTimeoutError, Broker
from hyperforge.pubsub import AgentMessage, StartInteraction
from hyperforge.redis_utils import ManualStreamKeysRedisCluster

_REDIS_CONNECT_TIMEOUT_SECONDS = 5
_REDIS_HEALTH_CHECK_INTERVAL_SECONDS = 30
_REDIS_RETRY_ATTEMPTS = 3
_REDIS_RETRY_BACKOFF_BASE_SECONDS = 0.1
_REDIS_RETRY_BACKOFF_CAP_SECONDS = 1.0
_REDIS_SUBSCRIBE_ACTIVATIONS_RETRY_SLEEP_SECONDS = 1
_REPLY_READ_BLOCK_MAX_MS = 2000


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
        client_kwargs = cls._client_kwargs(keepalive_ms)
        if cluster_mode:
            # redis-py cluster client does not accept retry_on_timeout.
            client_kwargs.pop("retry_on_timeout", None)
            client = cast(
                Redis,
                ManualStreamKeysRedisCluster.from_url(
                    url=url,
                    dynamic_startup_nodes=False,
                    **client_kwargs,
                ),
            )
        else:
            client = Redis.from_url(
                url,
                **client_kwargs,
            )  # type: ignore[call-overload]
        return cls(client, activate_subject, keepalive_ms)

    @staticmethod
    def _client_kwargs(keepalive_ms: int) -> dict[str, Any]:
        retry = Retry(
            backoff=ExponentialBackoff(
                base=_REDIS_RETRY_BACKOFF_BASE_SECONDS,
                cap=_REDIS_RETRY_BACKOFF_CAP_SECONDS,
            ),
            retries=_REDIS_RETRY_ATTEMPTS,
        )
        return {
            "decode_responses": True,
            "socket_timeout": keepalive_ms / 1000,
            "socket_connect_timeout": _REDIS_CONNECT_TIMEOUT_SECONDS,
            "health_check_interval": _REDIS_HEALTH_CHECK_INTERVAL_SECONDS,
            "socket_keepalive": True,
            "retry": retry,
            "retry_on_timeout": True,
        }

    async def publish_activation(
        self, msg: StartInteraction, trace: dict[str, str]
    ) -> None:
        await self._client.xadd(
            self._activate_subject,
            {"msg": msg.model_dump_json(), "trace": json.dumps(trace)},
            maxlen=100,
        )

    async def _ensure_consumer_group(self) -> None:
        """Create the stream and consumer group if they don't exist."""
        try:
            await self._client.xgroup_create(
                name=self._activate_subject,
                groupname="arag_server",
                mkstream=True,
            )
        except ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def subscribe_activations(
        self,
    ) -> AsyncIterator[tuple[StartInteraction, dict[str, str]]]:
        await self._ensure_consumer_group()

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
            except ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning("Consumer group lost, re-creating...")
                    await self._ensure_consumer_group()
                else:
                    logger.exception(
                        "Error while subscribing to activations, retrying..."
                    )
                    await asyncio.sleep(
                        _REDIS_SUBSCRIBE_ACTIVATIONS_RETRY_SLEEP_SECONDS
                    )
            except Exception:
                logger.exception("Error while subscribing to activations, retrying...")
                await asyncio.sleep(_REDIS_SUBSCRIBE_ACTIVATIONS_RETRY_SLEEP_SECONDS)

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
        deadline = time.monotonic() + (timeout_ms / 1000)
        reconnect_attempts = 0

        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                return None

            # Keep each blocking read short so transient socket closures/failovers
            # can be recovered within the same overall timeout budget.
            block_ms = min(_REPLY_READ_BLOCK_MAX_MS, max(1, int(remaining_s * 1000)))

            try:
                response = await self._client.xread(
                    {key: "$"},
                    block=block_ms,
                    count=1,
                )
            except RedisConnectionError:
                reconnect_attempts += 1
                backoff_s = min(1.0, 0.1 * reconnect_attempts)
                logger.warning(
                    "Redis connection closed while receiving reply key=%s; retrying (attempt=%d)",
                    key,
                    reconnect_attempts,
                )
                await asyncio.sleep(backoff_s)
                continue

            if not response:
                continue
            return response[0][1][0][1]["msg"]

    async def initialize(self) -> None:
        pass

    async def finalize(self) -> None:
        await self._client.aclose()  # type: ignore[attr-defined]
