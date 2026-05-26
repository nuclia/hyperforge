import os
from typing import Any

from httpx import AsyncClient
from hyperforge.configure import get_driver_config_instance
from hyperforge.llm import NUAConnection
from hyperforge.manager import Manager
from hyperforge.memory.memory import (
    MemoryConfig,
    SessionMemory,
)
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import Rule, Rules
from hyperforge.prompts import PromptConfig
from hyperforge.server.session import SessionManager
from hyperforge.workflows import WorkflowData
from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig, Transport
from nucliadb_models.resource import KnowledgeBoxObj

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")


KB_2603EE3A_2EE0_46BA_85A7_A1A2EC5A8FFE = os.environ.get(
    "KB_2603EE3A_2EE0_46BA_85A7_A1A2EC5A8FFE"
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

DRIVERS = [
    {
        "provider": "mcphttp",
        "identifier": "mcphttp-01",
        "name": "mcphttp",
        "config": {
            "uri": "http://localhost:3001",
            "headers": {
                "X-STF-USER": "user@example.com",
                "X-STF-ROLES": "SESSIONMEMBER",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "personal",
            },
        },
    },
]

DRIVERS_AGENT = [
    {
        "provider": "nucliadb",
        "identifier": "nucliadb-01",
        "name": "nucliadb",
        "config": {
            "description": "Products connection",
            "manager": "https://aws-us-east-2-1.rag.progress.cloud/api",
            "url": "https://aws-us-east-2-1.rag.progress.cloud/api",
            "key": KB_2603EE3A_2EE0_46BA_85A7_A1A2EC5A8FFE,
            "kbid": "2603ee3a-2ee0-46ba-85a7-a1a2ec5a8ffe",
        },
    },
]

MEMORY = {"nucliadb": {"url": "", "key": "", "kbid": ""}}

ROUTER = {
    "key": NUA_KEY,
}


RULES: Rules = Rules(
    rules=[
        Rule(prompt="Be polite"),
    ]
)

CONFIG: dict[str, Any] = {
    "prompts": [
        PromptConfig(
            name="Shooping",
            prompt="A user wants to go shopping. In order to answer check the recomendations for that product on the recomendation tool, check if the user already bought that product on their shopping history and search for the product and similar products on the search tool",  # comercial use case
            description="A prompt to help a user on a shopping platform",
            arguments=[],
        ),
    ],
    "workflows": [
        {
            "metadata": WorkflowData(
                id="recomendations",
                name="recomendations",
                description="Used to give recomendations to users",
                parameters={
                    "product": {
                        "type": "string",
                        "description": "Product to search for recomendations",
                    },
                    "original_question": {
                        "type": "string",
                        "description": "Original question from the user that might be useful to give better recomendations",
                    },
                },
                rules=Rules(rules=[]),
                required=["product"],
            ),
            "preprocess": [
                {
                    "module": "rephrase",
                    "title": "Rephrase module",
                    "rules": [
                        "Do a google search query to find recomendations of products for ths product pentioned on the question"
                    ],
                }
            ],
            "context": [
                {
                    "module": "static",
                    "title": "Google Agent",
                    "context": "google-01",
                }
            ],
            "generation": [
                {"module": "summarize"},
            ],
        },
        {
            "metadata": WorkflowData(
                id="shopping_history",
                name="shopping_history",
                description="Search user shopping history to give better answers",
                parameters={
                    "user_id": {
                        "type": "string",
                        "description": "User to search for shopping history",
                    },
                    "product": {
                        "type": "string",
                        "description": "Product to search for in shopping history",
                    },
                },
                rules=Rules(rules=[]),
                required=["user_id", "product"],
            ),
            "context": [
                {
                    "module": "static",
                    "title": "",
                    "description": "Database about history shopping",
                    "context": "sql-01",
                }
            ],
            "generation": [
                {"module": "summarize"},
            ],
        },
        {
            "metadata": WorkflowData(
                id="search_products",
                name="search_products",
                description="Search for products on the database to give better answers",
                parameters={
                    "product": {
                        "type": "string",
                        "description": "Product to search for in shopping history",
                    },
                },
                rules=Rules(rules=[]),
                required=["product"],
            ),
            "context": [
                {
                    "module": "basic_ask",
                    "title": "",
                    "description": "Database about products",
                    "sources": ["nucliadb-01"],
                }
            ],
            "generation": [
                {"module": "summarize"},
            ],
        },
    ],
}


async def _test_mcp_full_workflow(
    arag_api_http: str,
    arag_kb: KnowledgeBoxObj,
    arag_api_http_session: str,
    arag_server: SessionManager,
    disable_safe_transport,
    load_agents,
):

    http_client = AsyncClient()
    HEADERS = {
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
        "X-STF-ROLES": "SOWNER",
    }
    for driver in DRIVERS_AGENT:
        resp = await http_client.post(
            f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/drivers",
            json=driver,
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    for prompt_obj in CONFIG["prompts"]:
        resp = await http_client.post(
            f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/prompts",
            json=prompt_obj.model_dump(),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text

    DRIVERS[0]["config"]["uri"] = (  # type: ignore
        f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{arag_api_http_session}/mcp"
    )

    for workflow_obj in CONFIG["workflows"]:
        resp = await http_client.post(
            f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/workflows",
            json=workflow_obj["metadata"].model_dump(),
            headers=HEADERS,
        )
        assert resp.status_code == 200, resp.text
        resp = await http_client.get(
            f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/workflows",
            headers=HEADERS,
        )
        workflow_id = workflow_obj["metadata"].id
        for context in workflow_obj["context"]:
            resp = await http_client.post(
                f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}/context",
                json=context,
                headers=HEADERS,
            )
            assert resp.status_code == 200, resp.text
        for preprocess in workflow_obj.get("preprocess", []):
            resp = await http_client.post(
                f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}/preprocess",
                json=preprocess,
                headers=HEADERS,
            )
            assert resp.status_code == 200, resp.text
        for generation in workflow_obj["generation"]:
            resp = await http_client.post(
                f"http://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/workflow/{workflow_id}/generation",
                json=generation,
                headers=HEADERS,
            )
            assert resp.status_code == 200, resp.text

    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(driver) for driver in DRIVERS],
        nua=await NUAConnection.model_validate(ROUTER).connect(),
    )
    mcp_client = await MCPAgent.from_config(
        MCPAgentConfig(
            title="MCP NucliaDB Agent",
            transport=Transport.HTTP,
            source="mcphttp-01",
            prune_context=False,
        )
    )

    memory = SessionMemory.from_config(
        MemoryConfig.model_validate(MEMORY),
        agent_id="agent",
        workflow_id="default",
        rules=RULES,
    )
    memory.init("hola")
    question = "I want to buy a new smartphone. Can you recommend me some options? I usually buy electronics from TechStore. user_id: user1"
    question_memory = memory.start_question(question, question_id="question_id")

    await mcp_client.initialize(manager, question_memory)

    await mcp_client.get_question_context(
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
        question=question,
        flow_id="flow_id",
    )

    chunk_ids = [chunk.chunk_id for chunk in question_memory.contexts[0].chunks]
    assert any("recomendations" in cid for cid in chunk_ids), (
        f"Context from Google Recomendations workflow is not present in the question memory. chunk_ids={chunk_ids}"
    )
    assert any("shopping_history" in cid for cid in chunk_ids), (
        f"Context from SQL workflow is not present in the question memory. chunk_ids={chunk_ids}"
    )
    assert any("search_products" in cid for cid in chunk_ids), (
        f"Context from NucliaDB workflow is not present in the question memory. chunk_ids={chunk_ids}"
    )


async def test_mcp_protected_resource_metadata(
    arag_api_http: str,
):
    http_client = AsyncClient(base_url=f"http://{arag_api_http}")
    resp = await http_client.get(
        "/.well-known/oauth-protected-resource/api/v1/agent/agent-id/session/session-id/mcp"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "resource": f"https://{arag_api_http}/api/v1/agent/agent-id/session/session-id/mcp",
        "scopes_supported": ["offline_access", "openid"],
        "authorization_servers": ["https://oauth.progress.cloud"],
    }
