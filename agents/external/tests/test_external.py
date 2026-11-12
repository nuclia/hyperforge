import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from hyperforge.engine import main as arag_main
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import Context

from hyperforge_external.agent import ExternalCallAgent
from hyperforge_external.config import ExternalCallAgentConfig

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")

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
    request = httpx.Request("POST", "https://example.com/aaa")
    mock_response = httpx.Response(200, content=b"oki doki", request=request)

    mock_client = AsyncMock()
    mock_client.build_request.return_value = request
    mock_client.send = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    mocker.patch(
        "hyperforge_external.agent.safe_http_client",
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
    assert mock_client.send.called


@pytest.mark.asyncio
async def test_external_with_context_payload(mocker):
    config = ExternalCallAgentConfig(
        context=True,
        url="https://example.com/aaa",
    )
    external_agent = ExternalCallAgent(config)
    memory = SimpleNamespace(
        original_question="Question",
        final_answer="Answer",
        contexts=[
            Context(
                original_question_uuid=None,
                actual_question_uuid=None,
                question="Question",
                source="static",
                agent="static",
            )
        ],
        contexts_minimal=lambda: "Context",
        add_step=AsyncMock(),
    )

    mock_response = httpx.Response(
        200,
        content=b"oki doki",
        request=httpx.Request("POST", config.url),
    )
    mock_client = MagicMock()
    mock_client.build_request.side_effect = httpx.Request
    mock_client.send = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mocker.patch(
        "hyperforge_external.agent.safe_http_client",
        return_value=mock_client,
    )

    await external_agent(memory, manager=None)

    request = mock_client.send.call_args.args[0]
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content)[0]["source"] == "static"
