from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hyperforge_nucliadb.driver import NucliaDBDriver


def make_driver(client):
    return NucliaDBDriver.model_construct(
        provider="nucliadb",
        name="test",
        config=SimpleNamespace(kbid="test-kb"),
        driver=client,
        manager=SimpleNamespace(),
        _synonyms=None,
    )


@pytest.mark.asyncio
async def test_labels_uses_complete_labelsets_response():
    client = AsyncMock()
    client.get_labelsets.return_value = SimpleNamespace(
        labelsets={
            "type": SimpleNamespace(labels=[SimpleNamespace(title="report")]),
            "topic": SimpleNamespace(labels=[SimpleNamespace(title="science")]),
        }
    )
    driver = make_driver(client)

    assert await driver.labels() == {"type": ["report"], "topic": ["science"]}

    client.get_labelsets.assert_awaited_once_with(kbid="test-kb")
    client.get_labelset.assert_not_awaited()
