import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer
from hyperforge.minimal_fixtures import cassette_nua_key

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")

CONFIG = {
    "drivers": [],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
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
            "module": "http",
            "title": "HTTP Context",
            "url": "https://example.com/context",
            "method": "GET",
            "question_query_param": "q",
        }
    ],
    "generation": [
        {"module": "summarize", "title": "Summarize agent"},
    ],
    "postprocess": [],
}


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_http_static_get(mocker):
    answers = []

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"Experts from the University of Nuclia have found that waking up relaxed can be achieved by following a consistent sleep schedule, creating a calming bedtime routine, and ensuring a comfortable sleep environment. Also, if your alarm is a an BetterStack call, it will help you clear your mind"

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def callback(obj: AragAnswer):
        answers.append(obj)

    mocker.patch(
        "hyperforge.utils.http.safe_http_client",
        return_value=mock_client,
    )

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="What is the best way to wake up relaxed?",
        config=CONFIG,
        callback=callback,
        loaded_modules=["hyperforge_http", "hyperforge_summarize"],
    )
    assert "BetterStack" in question_memory.final_answer
    assert question_memory.final_answer is not None
