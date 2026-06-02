import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.minimal_fixtures import cassette_nua_key

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")
CONFIG = {
    "drivers": [],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
            {
                "prompt": "The documentation of Nuclia is hosted at https://docs.nuclia.dev"
            },
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [],
    "context": [
        {
            "module": "static",
            "title": "",
            "context": "Partner cars include brands like Peugeot, Citroen, Renault, etc.",
        }
    ],
    "postprocess": [
        {
            "module": "external",
            "method": "POST",
            "call_schema": {
                "type": "object",
                "properties": {
                    "brand": {"type": "string", "description": "Brand of car"},
                },
            },
            "headers": {"aa": "bb"},
            "url": "https://example.com/aaa",
        },
    ],
    "generation": [],
}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_external(mocker):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"oki doki"

    mock_client = AsyncMock()
    mock_client.request = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    mocker.patch(
        "hyperforge.utils.http.safe_http_client",
        return_value=mock_client,
    )

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Que modelos de coche salieron el 2020",
        config=CONFIG,
        loaded_modules=["hyperforge_external", "hyperforge_static"],
    )

    assert (
        question_memory.steps[-1].reason is not None
        and "oki doki" in question_memory.steps[-1].reason
    )
    assert len(question_memory.steps) >= 1
    assert question_memory.steps[-2].reason is not None and (
        "partner" in question_memory.steps[-2].reason
        or "Peugeot" in question_memory.steps[-2].reason
    )
    assert mock_client.request.called
