import json
import os
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from httpx import AsyncClient
from hyperforge.configure import GLOBAL_REGISTRY
from hyperforge.engine import engine, init
from hyperforge.interaction import AragAnswer, Feedback, OAuthAuthenticateURL
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.pubsub import UserToAgentInteraction

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")


KB_E103CAF3_F8CB_4161_A57C_AAD1192D0666 = os.environ.get(
    "KB_E103CAF3_F8CB_4161_A57C_AAD1192D0666"
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["europe-1.dp.progress.cloud"]),
    pytest.mark.asyncio,
]

CONFIG = {
    "drivers": [
        {
            "name": "nuclia-sync",
            "provider": "sync",
            "identifier": "nuclia-sync",
            "config": {
                "url": "https://europe-1.stashify.cloud/api",
                "manager": "https://europe-1.stashify.cloud/api",
                "kbid": "e103caf3-f8cb-4161-a57c-aad1192d0666",
                "key": KB_E103CAF3_F8CB_4161_A57C_AAD1192D0666,
                "filters": [],
                "description": "Nuclia Sync source for testing",
                "connection_ids": ["019cade7-c177-77c5-99c2-c8771f85cf91"],
            },
        },
    ],
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
            "module": "sync",
            "title": "",
            "sources": ["nuclia-sync"],
        },
    ],
    "generation": [
        {"module": "summarize"},
    ],
    "postprocess": [],
}

SYNC_CONFIG_ID = "019cade7-c177-77c5-99c2-c8771f85cf91"
EXTERNAL_CONNECTION_ID = "019cade7-64ee-7389-bec6-888c8ff8d604"


def _mock_sync_response(
    method: str, url: str, payload: dict[str, Any]
) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request(method, url),
    )


async def test_sync_agent(arag_api: AsyncClient):
    sync_driver_config = cast(dict[str, Any], CONFIG["drivers"][0]["config"])  # type: ignore
    sync_base_url = f"{sync_driver_config['url']}/v1/kb/{sync_driver_config['kbid']}"

    async def mock_sync_get(url: str, *args: Any, **kwargs: Any) -> httpx.Response:
        resolved_url = url if url.startswith("http") else f"{sync_base_url}{url}"
        if resolved_url.endswith(f"/sync_config/{SYNC_CONFIG_ID}"):
            return _mock_sync_response(
                "GET",
                resolved_url,
                {"external_connection": {"id": EXTERNAL_CONNECTION_ID}},
            )
        if resolved_url.endswith(f"/external_connection/{EXTERNAL_CONNECTION_ID}"):
            return _mock_sync_response(
                "GET",
                resolved_url,
                {
                    "id": EXTERNAL_CONNECTION_ID,
                    "kb_id": sync_driver_config["kbid"],
                    "created_by": "00000000-0000-0000-0000-000000000001",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "provider": "sharefile_oauth",
                },
            )
        raise AssertionError(f"Unexpected sync GET request: {resolved_url}")

    async def mock_sync_connect(*args: Any, **kwargs: Any) -> AsyncMock:
        client = AsyncMock(spec=AsyncClient)
        client.get.side_effect = mock_sync_get
        return client

    async def mock_get_oauth_url(*args: Any, **kwargs: Any) -> str:
        return "https://sharefile.example.com/oauth"

    async def mock_validate_resources(
        self: Any,
        resource_ids: list[str],
        credentials: str,
        connection_id: str,
        sync_config_id: str,
        sync_metadata_by_resource: dict[str, Any],
    ) -> list[str]:
        assert credentials == '{"credentials": "creds"}'
        assert connection_id == EXTERNAL_CONNECTION_ID
        assert sync_config_id == SYNC_CONFIG_ID
        assert sync_metadata_by_resource
        return resource_ids

    GLOBAL_REGISTRY.clear()
    with (
        patch(
            "hyperforge_nucliadb.sync.driver.sync_connect",
            new=mock_sync_connect,
        ),
        patch(
            "hyperforge_nucliadb.sync.driver.SyncDriver.get_oauth_url",
            new=mock_get_oauth_url,
        ),
        patch(
            "hyperforge_nucliadb.sync.driver.SyncDriver.validate_resources",
            new=mock_validate_resources,
        ),
    ):
        state, memory = await init(
            config=CONFIG,
            agent_id="default",
            internal_nua=False,
            external_nua_api_key=NUA_KEY,
            memory_klass=EphemeralSessionMemory,
            loaded_modules=["hyperforge_nucliadb", "hyperforge_summarize"],
        )

        answers = []
        feedbacks = []
        oauths = []
        oauth_answer: list[Any] = []

        async def callback(obj: AragAnswer):
            answers.append(obj)

        async def oauth_callback_fn(question_id: str, oauth_uuid: str) -> str | None:
            answer = oauth_answer.pop()
            return answer

        async def oauth(obj: OAuthAuthenticateURL):
            oauths.append(obj)
            if "sharefile" in obj.oauth_url:
                oauth_answer.append('{"credentials": "creds"}')

        async def feedback(obj: Feedback):
            feedbacks.append(obj)
            if obj.question == "Get credentials":
                return UserToAgentInteraction(
                    request_id=obj.request_id,
                    response=json.dumps({"existing_credentials": {}}),
                )
            if obj.question == "Send credentials":
                assert (
                    obj.credentials
                    and obj.credentials[SYNC_CONFIG_ID][EXTERNAL_CONNECTION_ID]
                    == '{"credentials": "creds"}'
                )
                return UserToAgentInteraction(
                    request_id=obj.request_id,
                    response=json.dumps(
                        {
                            "existing_credentials": {
                                SYNC_CONFIG_ID: {"credentials": "creds"}
                            }
                        }
                    ),
                )

        memory.debug = True
        question = "New employees at ADP"
        question_memory = memory.start_question(question)
        question_memory.set_callback_fn(callback)
        question_memory.set_feedback_fn(feedback)
        question_memory.set_oauth_fn(oauth)
        question_memory.set_oauth_callback_fn(oauth_callback_fn)
        try:
            await engine(
                manager=state.manager,
                agent=state.agent,
                question_memory=question_memory,
            )
        except Exception as e:
            assert b"credentials" in e.response.content  # type: ignore

    GLOBAL_REGISTRY.clear()

    assert oauths
    assert feedbacks
