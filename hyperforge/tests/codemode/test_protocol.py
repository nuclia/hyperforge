import asyncio

import pytest

from hyperforge.codemode.model import deserialize
from hyperforge.codemode.sandbox import MAX_PACKET_BYTES, SandboxReader


def test_deserialize_preserves_unknown_model_marker() -> None:
    value = {"__model__": "customer-data", "value": 1}

    assert deserialize(value) == value


@pytest.mark.asyncio
async def test_reader_rejects_oversized_packet_before_payload() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((MAX_PACKET_BYTES + 1).to_bytes(4, "little"))

    with pytest.raises(ValueError, match="maximum size"):
        await SandboxReader(reader)._read_packet()
