import json
import os
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from httpx import AsyncClient
from hyperforge.configure import GLOBAL_REGISTRY
from hyperforge.engine import engine, init
from hyperforge.interaction import AragAnswer, Feedback, OAuthAuthenticateURL
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.pubsub import UserToAgentInteraction
from pydantic import ValidationError

from hyperforge_nucliadb.basic_ask_agent import BasicAskAgent
from hyperforge_nucliadb.sync.agent import SyncAskAgent
from hyperforge_nucliadb.sync.config import SyncAskAgentConfig
from hyperforge_nucliadb.sync.config_driver import SyncConnection
from hyperforge_nucliadb.sync.driver import SyncDriver

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.dp.progress.cloud/")


KB_E103CAF3_F8CB_4161_A57C_AAD1192D0666 = os.environ.get(
    "KB_E103CAF3_F8CB_4161_A57C_AAD1192D0666"
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True),
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


async def test_sync_connection_defaults_to_hybrid_mode():
    config = {
        "url": "https://example.com",
        "manager": "https://example.com",
        "kbid": "kb",
        "key": "key",
        "description": "test",
        "filters": [],
    }

    assert SyncConnection(**config).connection_ids == []
    with pytest.raises(ValidationError):
        SyncConnection(**config, connection_ids=[" "])


async def test_sync_driver_lazily_resolves_and_caches_connection():
    config = SyncConnection(
        url="https://example.com",
        manager="https://example.com",
        kbid="00000000-0000-0000-0000-000000000010",
        key="key",
        description="test",
        filters=[],
    )
    client = AsyncMock(spec=AsyncClient)
    client.get.side_effect = [
        _mock_sync_response(
            "GET",
            f"https://example.com/sync_config/{SYNC_CONFIG_ID}",
            {"external_connection": {"id": EXTERNAL_CONNECTION_ID}},
        ),
        _mock_sync_response(
            "GET",
            f"https://example.com/external_connection/{EXTERNAL_CONNECTION_ID}",
            {
                "id": EXTERNAL_CONNECTION_ID,
                "kb_id": "00000000-0000-0000-0000-000000000010",
                "created_by": "00000000-0000-0000-0000-000000000001",
                "created_at": "2024-01-01T00:00:00Z",
                "updated_at": "2024-01-01T00:00:00Z",
                "provider": "sharefile_oauth",
            },
        ),
    ]
    driver = SyncDriver.model_construct(
        provider="sync",
        name="sync",
        config=config,
        async_driver=client,
        information={},
        sync_configs={},
        driver=MagicMock(),
        manager=MagicMock(),
        _synonyms=None,
    )

    first = await driver.resolve_sync_config(SYNC_CONFIG_ID)
    second = await driver.resolve_sync_config(SYNC_CONFIG_ID)

    assert first is second
    assert first.provider.value == "sharefile_oauth"
    assert client.get.await_count == 2


async def test_sync_driver_does_not_expand_configured_allowlist():
    config = SyncConnection(
        url="https://example.com",
        manager="https://example.com",
        kbid="00000000-0000-0000-0000-000000000010",
        key="key",
        description="test",
        filters=[],
        connection_ids=[SYNC_CONFIG_ID],
    )
    client = AsyncMock(spec=AsyncClient)
    driver = SyncDriver.model_construct(
        provider="sync",
        name="sync",
        config=config,
        async_driver=client,
        information={},
        sync_configs={},
        driver=MagicMock(),
        manager=MagicMock(),
        _synonyms=None,
    )

    with pytest.raises(ValueError, match="not configured"):
        await driver.resolve_sync_config(EXTERNAL_CONNECTION_ID)
    client.get.assert_not_awaited()


async def test_hybrid_catalog_filter_is_preserved_after_connection_resolution():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    agent.sources = {
        "source": SimpleNamespace(
            config=SimpleNamespace(connection_ids=[]),
            sync_configs={SYNC_CONFIG_ID: [EXTERNAL_CONNECTION_ID]},
        )  # type: ignore[dict-item]
    }
    catalog_filter = MagicMock()

    assert agent.enrich_catalog_filter(catalog_filter) is catalog_filter


async def test_mixed_sources_preserve_hybrid_catalog_filter():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["hybrid", "restricted"]))
    agent.sources = {
        "hybrid": SimpleNamespace(config=SimpleNamespace(connection_ids=[])),
        "restricted": SimpleNamespace(
            config=SimpleNamespace(connection_ids=[SYNC_CONFIG_ID])
        ),
    }  # type: ignore[dict-item]
    catalog_filter = MagicMock()

    assert agent.enrich_catalog_filter(catalog_filter) is catalog_filter


async def test_search_by_title_initializes_sources_before_enriching_filter():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(config=SimpleNamespace(connection_ids=[]))
    manager = SimpleNamespace(
        drivers=SimpleNamespace(get=MagicMock(return_value=driver))
    )

    with patch.object(
        BasicAskAgent,
        "search_by_title",
        new=AsyncMock(return_value={"source": []}),
    ):
        result = await agent.search_by_title(
            memory=MagicMock(),
            manager=manager,  # type: ignore[arg-type]
            title="title",
        )

    assert result == {"source": []}
    assert agent.sources == {"source": driver}


async def test_configured_source_ignores_resources_outside_allowlist():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(
        config=SimpleNamespace(connection_ids=[SYNC_CONFIG_ID]),
        sync_configs={SYNC_CONFIG_ID: [EXTERNAL_CONNECTION_ID]},
        information={},
    )
    agent.sources = {"source": driver}  # type: ignore[assignment]
    resources = [
        SimpleNamespace(id="public-resource", origin=None),
        SimpleNamespace(
            id="other-connection-resource",
            origin=SimpleNamespace(
                source_id="sync_config_00000000-0000-0000-0000-000000000001",
                sync_metadata=SimpleNamespace(),
            ),
        ),
    ]
    ndb = SimpleNamespace(
        config=SimpleNamespace(kbid="kb"),
        driver=SimpleNamespace(get_resource_by_id=AsyncMock(side_effect=resources)),
    )
    manager = SimpleNamespace(
        drivers=SimpleNamespace(get=MagicMock(return_value=driver))
    )

    with patch("hyperforge_nucliadb.sync.agent.get_ndb_driver", return_value=ndb):
        result = await agent._post_filter_configured_resources(
            memory=MagicMock(),
            manager=manager,  # type: ignore[arg-type]
            resources={"source": [resource.id for resource in resources]},
        )

    assert result == {}


async def test_hybrid_dynamic_connection_is_authorized():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(
        config=SimpleNamespace(connection_ids=[]),
        sync_configs={SYNC_CONFIG_ID: [EXTERNAL_CONNECTION_ID]},
        information={
            EXTERNAL_CONNECTION_ID: SimpleNamespace(provider="sharefile_oauth")
        },
        validate_resources=AsyncMock(return_value=["connected-resource"]),
    )
    resource = SimpleNamespace(
        id="connected-resource",
        origin=SimpleNamespace(
            source_id=f"sync_config_{SYNC_CONFIG_ID}",
            sync_metadata=SimpleNamespace(model_dump=MagicMock(return_value={})),
        ),
    )
    ndb = SimpleNamespace(
        config=SimpleNamespace(kbid="kb"),
        driver=SimpleNamespace(get_resource_by_id=AsyncMock(return_value=resource)),
    )
    manager = SimpleNamespace(
        drivers=SimpleNamespace(get=MagicMock(return_value=driver))
    )
    memory = MagicMock()
    memory.get_session_id.return_value = "session"
    memory.send_feedback = AsyncMock(
        return_value=UserToAgentInteraction(
            request_id="session",
            response=json.dumps(
                {
                    "existing_credentials": {
                        SYNC_CONFIG_ID: {EXTERNAL_CONNECTION_ID: "credential"}
                    }
                }
            ),
        )
    )

    with patch("hyperforge_nucliadb.sync.agent.get_ndb_driver", return_value=ndb):
        result = await agent._post_filter_configured_resources(
            memory=memory,
            manager=manager,  # type: ignore[arg-type]
            resources={"source": [resource.id]},
            allowed_connection_ids={SYNC_CONFIG_ID},
        )

    assert result == {"source": {SYNC_CONFIG_ID: ["connected-resource"]}}


async def test_hybrid_dynamic_connection_ignores_unrequested_credentials():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(
        config=SimpleNamespace(connection_ids=[]),
        sync_configs={SYNC_CONFIG_ID: [EXTERNAL_CONNECTION_ID]},
        information={
            EXTERNAL_CONNECTION_ID: SimpleNamespace(provider="sharefile_oauth")
        },
        validate_resources=AsyncMock(return_value=["connected-resource"]),
    )
    resource = SimpleNamespace(
        id="connected-resource",
        origin=SimpleNamespace(
            source_id=f"sync_config_{SYNC_CONFIG_ID}",
            sync_metadata=SimpleNamespace(model_dump=MagicMock(return_value={})),
        ),
    )
    ndb = SimpleNamespace(
        config=SimpleNamespace(kbid="kb"),
        driver=SimpleNamespace(get_resource_by_id=AsyncMock(return_value=resource)),
    )
    manager = SimpleNamespace(
        drivers=SimpleNamespace(get=MagicMock(return_value=driver))
    )
    memory = MagicMock()
    memory.get_session_id.return_value = "session"
    memory.send_feedback = AsyncMock(
        return_value=UserToAgentInteraction(
            request_id="session",
            response=json.dumps(
                {
                    "existing_credentials": {
                        SYNC_CONFIG_ID: {EXTERNAL_CONNECTION_ID: "credential"},
                        "00000000-0000-0000-0000-000000000002": {
                            "unrequested-connection": "credential"
                        },
                    }
                }
            ),
        )
    )

    with patch("hyperforge_nucliadb.sync.agent.get_ndb_driver", return_value=ndb):
        result = await agent._post_filter_configured_resources(
            memory=memory,
            manager=manager,  # type: ignore[arg-type]
            resources={"source": [resource.id]},
            allowed_connection_ids={SYNC_CONFIG_ID},
        )

    assert result == {"source": {SYNC_CONFIG_ID: ["connected-resource"]}}
    driver.validate_resources.assert_awaited_once()


async def test_hybrid_resources_without_connections_skip_authorization():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(config=SimpleNamespace(connection_ids=[]))
    agent.sources = {"source": driver}  # type: ignore[assignment]
    resource = SimpleNamespace(id="public-resource", origin=None)
    ndb = SimpleNamespace(
        config=SimpleNamespace(kbid="kb"),
        driver=SimpleNamespace(get_resource_by_id=AsyncMock(return_value=resource)),
    )
    configured_filter = AsyncMock()
    agent._post_filter_configured_resources = configured_filter

    with patch("hyperforge_nucliadb.sync.agent.get_ndb_driver", return_value=ndb):
        result = await agent._post_filter_hybrid_resources(
            memory=MagicMock(),
            manager=MagicMock(),
            kb_source_id="source",
            resource_ids=["public-resource"],
        )

    assert result == {"source": {"__hybrid__": ["public-resource"]}}
    configured_filter.assert_not_awaited()


async def test_hybrid_resources_authorize_only_valid_sync_origins():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(
        config=SimpleNamespace(connection_ids=[]),
        resolve_sync_config=AsyncMock(),
    )
    agent.sources = {"source": driver}  # type: ignore[assignment]
    resources = [
        SimpleNamespace(id="public-resource", origin=None),
        SimpleNamespace(
            id="connected-resource",
            origin=SimpleNamespace(
                source_id=f"sync_config_{SYNC_CONFIG_ID}",
                sync_metadata=SimpleNamespace(),
            ),
        ),
        SimpleNamespace(
            id="missing-metadata",
            origin=SimpleNamespace(
                source_id=f"sync_config_{SYNC_CONFIG_ID}", sync_metadata=None
            ),
        ),
        SimpleNamespace(
            id="invalid-connection",
            origin=SimpleNamespace(
                source_id="sync_config_not-a-uuid",
                sync_metadata=SimpleNamespace(),
            ),
        ),
    ]
    ndb = SimpleNamespace(
        config=SimpleNamespace(kbid="kb"),
        driver=SimpleNamespace(get_resource_by_id=AsyncMock(side_effect=resources)),
    )

    async def resolve_sync_config(sync_config_id: str):
        if sync_config_id == "not-a-uuid":
            raise ValueError("invalid sync config")
        return SimpleNamespace()

    driver.resolve_sync_config.side_effect = resolve_sync_config
    agent._post_filter_configured_resources = AsyncMock(
        return_value={"source": {SYNC_CONFIG_ID: ["connected-resource"]}}
    )

    with patch("hyperforge_nucliadb.sync.agent.get_ndb_driver", return_value=ndb):
        result = await agent._post_filter_hybrid_resources(
            memory=MagicMock(),
            manager=MagicMock(),
            kb_source_id="source",
            resource_ids=[resource.id for resource in resources],
        )

    assert result == {
        "source": {
            "__hybrid__": ["public-resource", "connected-resource"],
        }
    }
    agent._post_filter_configured_resources.assert_awaited_once()


async def test_hybrid_resource_with_sync_metadata_and_invalid_source_is_excluded():
    agent = SyncAskAgent(SyncAskAgentConfig(sources=["source"]))
    driver = SimpleNamespace(
        config=SimpleNamespace(connection_ids=[]),
        resolve_sync_config=AsyncMock(),
    )
    agent.sources = {"source": driver}  # type: ignore[assignment]
    resource = SimpleNamespace(
        id="invalid-sync-resource",
        origin=SimpleNamespace(
            source_id=None,
            sync_metadata=SimpleNamespace(),
        ),
    )
    ndb = SimpleNamespace(
        config=SimpleNamespace(kbid="kb"),
        driver=SimpleNamespace(get_resource_by_id=AsyncMock(return_value=resource)),
    )

    with patch("hyperforge_nucliadb.sync.agent.get_ndb_driver", return_value=ndb):
        result = await agent._post_filter_hybrid_resources(
            memory=MagicMock(),
            manager=MagicMock(),
            kb_source_id="source",
            resource_ids=[resource.id],
        )

    assert result == {"source": {"__hybrid__": []}}
    driver.resolve_sync_config.assert_not_awaited()


def _mock_sync_response(
    method: str, url: str, payload: dict[str, Any]
) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request(method, url),
    )


async def test_sync_agent():
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
