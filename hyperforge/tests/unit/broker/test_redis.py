from unittest.mock import AsyncMock

import pytest

from hyperforge.broker.redis import RedisBroker


@pytest.mark.asyncio
async def test_receive_reply_consumes_response_published_before_waiting():
    client = AsyncMock()
    client.xread.return_value = [
        ("feedback-id", [("1-0", {"msg": "approved"})])
    ]
    broker = RedisBroker(client, "activations", keepalive_ms=20_000)

    result = await broker.receive_reply("feedback-id", timeout_ms=1_000)

    assert result == "approved"
    client.xread.assert_awaited_once()
    streams = client.xread.await_args.args[0]
    assert streams == {"feedback-id": "0-0"}
    assert client.xread.await_args.kwargs["count"] == 1
    client.delete.assert_awaited_once_with("feedback-id")