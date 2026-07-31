from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from redis.asyncio import Redis

from hyperforge.broker.redis import RedisBroker


@pytest.mark.asyncio
async def test_send_reply_expires_stream():
    client = AsyncMock()
    pipeline_context = AsyncMock()
    pipeline = MagicMock()
    pipeline.execute = AsyncMock()
    pipeline_context.__aenter__.return_value = pipeline
    client.pipeline = Mock(return_value=pipeline_context)
    pipeline.xadd.return_value = pipeline
    pipeline.expire.return_value = pipeline
    broker = RedisBroker(client, "activations", keepalive_ms=20_000)

    await broker.send_reply("feedback-id", "approved")

    pipeline.xadd.assert_called_once()
    pipeline.expire.assert_called_once_with("feedback-id", 300)
    pipeline.execute.assert_awaited_once()


@pytest.mark.asyncio
async def test_receive_reply_consumes_response_published_before_waiting():
    client = AsyncMock()
    client.xread.return_value = [("feedback-id", [("1-0", {"msg": "approved"})])]
    broker = RedisBroker(client, "activations", keepalive_ms=20_000)

    result = await broker.receive_reply("feedback-id", timeout_ms=1_000)

    assert result == "approved"
    client.xread.assert_awaited_once()
    streams = client.xread.await_args.args[0]
    assert streams["feedback-id"] != "$"
    assert client.xread.await_args.kwargs["count"] == 1
    client.delete.assert_awaited_once_with("feedback-id")


@pytest.mark.asyncio
async def test_receive_reply_ignores_cleanup_failure_after_success():
    client = AsyncMock()
    client.xread.return_value = [("feedback-id", [("1-0", {"msg": "approved"})])]
    client.delete.side_effect = RuntimeError("cleanup failed")
    broker = RedisBroker(client, "activations", keepalive_ms=20_000)

    result = await broker.receive_reply("feedback-id", timeout_ms=1_000)

    assert result == "approved"


@pytest.mark.asyncio
async def test_receive_reply_cleanup_failure_does_not_mask_read_error():
    client = AsyncMock()
    client.xread.side_effect = ValueError("read failed")
    client.delete.side_effect = RuntimeError("cleanup failed")
    broker = RedisBroker(client, "activations", keepalive_ms=20_000)

    with pytest.raises(ValueError, match="read failed"):
        await broker.receive_reply("feedback-id", timeout_ms=1_000)


class CallbackBetweenReadsRedis:
    """Simulate a callback arriving after one XREAD block has expired."""

    def __init__(self) -> None:
        self.calls = 0
        self.cursors: list[str] = []
        self.deleted_keys: list[str] = []

    async def delete(self, key: str) -> None:
        self.deleted_keys.append(key)

    async def xread(
        self,
        streams: dict[str, str],
        *,
        block: int,
        count: int,
    ):
        del block, count
        self.calls += 1
        key, cursor = next(iter(streams.items()))
        self.cursors.append(cursor)

        if self.calls == 1:
            # The first blocking read expires. The OAuth callback is published
            # immediately afterwards, before receive_reply starts its next read.
            return []

        if cursor == "$":
            # Redis resolves "$" to the current stream tail, so the callback
            # published between reads is now behind the cursor and is skipped.
            return []

        return [(key, [("1-0", {"msg": "oauth-callback"})])]


@pytest.mark.asyncio
async def test_receive_reply_does_not_miss_callback_between_blocking_reads():
    client = CallbackBetweenReadsRedis()
    broker = RedisBroker(
        client=cast(Any, cast(Redis, client)),
        activate_subject="activations",
        keepalive_ms=1000,
    )

    payload = await broker.receive_reply("oauth-key", timeout_ms=20)

    assert payload == "oauth-callback"
    assert client.calls == 2
    assert client.cursors[0] == client.cursors[1]
    assert client.deleted_keys == ["oauth-key"]
