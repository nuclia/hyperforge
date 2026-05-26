from typing import Any

from httpx import AsyncClient
from hyperforge.configure import get_driver_config_instance
from hyperforge.llm import NUAConnection
from hyperforge.manager import Manager
from hyperforge.models import Rule, Rules
from hyperforge.memory.memory import (
    MemoryConfig,
    SessionMemory,
)
from hyperforge.prompts import PromptConfig
from hyperforge.workflows import WorkflowData
from hyperforge.server.session import SessionManager
from nucliadb_models.resource import KnowledgeBoxObj

from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig, Transport

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
        "provider": "sql",
        "identifier": "sql-01",
        "name": "sql",
        "config": {
            "name": "DB sales",
            "dsn": "postgresql://xxx:yyy@10.10.10.10/db",
            "schema": "",
        },
    },
    {
        "provider": "google",
        "identifier": "google-01",
        "name": "google",
        "config": {
            "vertexai": False,
            "api_key": "AIzaSyDBBq0QwyVtYauiP0D7GkqQaWm8A92kHrM",
        },
    },
    {
        "provider": "nucliadb",
        "identifier": "nucliadb-01",
        "name": "nucliadb",
        "config": {
            "description": "Products connection",
            "manager": "https://aws-us-east-2-1.rag.progress.cloud/api",
            "url": "https://aws-us-east-2-1.rag.progress.cloud/api",
            "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6InNhIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2F3cy11cy1lYXN0LTItMS5yYWcucHJvZ3Jlc3MuY2xvdWQvIiwiaWF0IjoxNzcwMTUwNzc0LCJzdWIiOiI4N2Q0YTQ1Ny04ZGUyLTRkMWItOWI4MC1jYWE3ZjcyNDcxOGQiLCJqdGkiOiI5MzZjYzYwMi03YzI4LTRlMzktYWUzMi05MDdjMzc4MjhhMTYiLCJleHAiOjE4MDE2ODY3NzIsImtleSI6IjQyMTJlOGVhLTc1ZjktNDI0OS04ZjlmLTAzZjI4ZWM2MWJmZCIsImtpZCI6ImRiZWVlMWRjLWYxMDktNDRjMC05Y2Q1LWI1Y2QxNDMxMDBjYSJ9.W2oSFp6CiUzdZLLqoc5SiCoXdonfz99pw9KrrWQ1zjlYfGB0iG-mTdbbG229C-bS_3oWT0Fs0ZZYpzLb5Oh4HTiUr7P0e18ckM449ILhviEQuoto14hOdwsNi30AWQMhJb6ljch-Ka8fb25u6dIL5q4D-3YY07HSX0sH7M3dqPI1tL89zYlIXHPcHBO6huvLBdSiKWnw7K1BXdyO18E15f9WXty4R3A__gCiSOJB4zY9fCyKp9QEi1-I1eebCtAGYceHbxAvTdriduoIfFgNwgYOFjHmdQXmQC6jOOMcZkenY2srXYZvs71y-VCiWMcWjMVHzu_nKXHUCKHl29cd-SLg7vxF7ZLbW_ywylA6LtjgNshafGMNlD3juuRMYqU43HoySgPcU9_suUKaAlFHcxlQTRNuWRVFOemfsF81zCPx-sO2Pzp-UQN0vnSAvBUuq_6RhQ4aUZfY4H6PN8IKGfolRDkM2K4xcSRz5wsnVURs9cgaaFEjP9BmFcL2ajYAnimoRqhGOTv3Y8AhiqI_KJYjJn2HthMgAYZJVlPtdG2VuUh6G-e266kWUSEiHW8Jx6hNNO0vdH86BjRK8uemLHCocBajXS-Qpf2MWxwVfeXLhsG8mxREOC4irBhcMfBDWJzyBb1tSawfHDThO3AcilEb8uRN1CL1RoPBqok8ogo",
            "kbid": "2603ee3a-2ee0-46ba-85a7-a1a2ec5a8ffe",
        },
    },
]

MEMORY = {"nucliadb": {"url": "", "key": "", "kbid": ""}}

ROUTER = {
    "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6Im51YSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3MzcwNjE2NjUsInN1YiI6IjAzZDQ3OTk4LWY2NzItNGE5Yi1hNTdiLWJkZTNhOWMzZWU4YyIsImp0aSI6IjQ1YTQ3NjJkLTIyMzgtNGI2OC05YTI2LWQ4N2QzNjJhZThmYiIsImV4cCI6MjUzMzcwNzY0ODAwLCJrZXkiOiI0N2JjZDU4ZC04NDExLTQ1NTgtYWIzZS0wNGMyMWI4NWI2ZWEiLCJhbGxvd19rYl9tYW5hZ2VtZW50IjpmYWxzZX0.Ljgv780vMuwviospTcRQYxrFV_H7XXR0hJeeSyFIfwVjni7hyyrxB189R5rQyLLI2n85iAdNGshvc8etDQRkXr8n8IWFsy_FOWcru-LZFZwGCpsY6hKK4TdWXR9v5sxA5xyKA7lmWw1LZ8dfNbcdx11OY15BfmGuMpiq_auIs1F90C8T8_LmXbz0SbdYzPIoEP0JFBX92jHqDoJNUTlMELUrcjupK9ao2pZahI47zQHrWjGuw2KrSjghdZgzwjC0YEa7C8quEVZ9SoLOkJvJV7XV4LrlGGcsxZzng8kLBGRBS-i8p26n5vFvMqiZKqDWpq68cVzZhAsL93wkzHVZCAHpfEsHQ4DUb-Da53xUrrnVnyl1w79iXiLYwP0wxh3b34B1b1ca3rRKuifbd1e762gf11qw6LHpJ9qKYhRv6O3KZ18_amwjLhqYna5uUfrP7f59tJZ9vzTG1oTZ5KlMBeVfu_IvhAmMbGpTygqEoxXqNrH3lWOsEPLhRVBC6D5t84xy7WLe4XsGR4xWduLWHsjxPYbmTrLMysGSqBSNGPwUi8jMTrH16-xprNJRiWVHcvgz_FGQ7sT7RucaAxhmFlZY9h3BFw7u_6awOeX4ymhH6_iDzWxBc0Fx5JsDgQm9jkhlYIHqZG36N5XfsmqfCyM12gNa37j-8MPOt7eU0XQ",
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
                    "module": "google",
                    "title": "Google Agent",
                    "source": "google-01",
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
                    "module": "sql",
                    "title": "",
                    "description": "Database about history shopping",
                    "source": "sql-01",
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


async def test_mcp_full_workflow(
    arag_api_http: str,
    arag_kb: KnowledgeBoxObj,
    arag_api_http_session: str,
    arag_server: SessionManager,
    disable_safe_transport,
    pg_shoping_example,
):
    # Create a session
    for driver in DRIVERS_AGENT:
        if driver["provider"] == "sql":
            driver["config"]["dsn"] = pg_shoping_example  # type: ignore

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
