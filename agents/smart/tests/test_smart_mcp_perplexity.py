import os

import pytest
from hyperforge.configure import get_driver_config_instance
from hyperforge.llm import NUAConnection
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory, MemoryConfig
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import Rule, Rules

from hyperforge_smart.agent import SmartAgent
from hyperforge_smart.config import SmartAgentConfig

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")


PERPLEXITY_KEY = os.environ.get("PERPLEXITY_API_KEY", "DUMMY_PERPLEXITY_KEY")

DRIVERS = [
    {
        "provider": "mcphttp",
        "identifier": "mcphttp-01",
        "name": "mcphttp",
        "config": {
            "uri": "http://localhost:3001",
            "headers": {
                "X-STF-USER": "user@example.com",
                "X-STF-ROLES": "READER",
                "X-STF-ACCOUNT": "account_id",
                "X-STF-ACCOUNT-TYPE": "personal",
            },
        },
    },
    {
        "provider": "perplexity",
        "identifier": "perplexity-01",
        "name": "perplexity",
        "config": {
            "key": PERPLEXITY_KEY,
        },
    },
]

MEMORY = {"nucliadb": {"url": "", "key": "", "kbid": ""}}

ROUTER = {
    "local_openai": "http://127.0.0.1:1234/v1",
    "key": NUA_KEY,
}

RULES: Rules = Rules(
    rules=[
        Rule(prompt="Be polite"),
    ]
)

MCP_AGENT_ID = "mcp-nucliadb-001"
PERPLEXITY_AGENT_ID = "perplexity-web-001"


@pytest.mark.skipif(
    os.environ.get("LOCAL_TESTING") is None,
    reason="Only check if LOCAL_TESTING var is enabled",
)
async def test_smart_with_mcp_and_perplexity(
    arag_api_http: str, article_dataset: str, disable_safe_transport
):
    """SmartAgent orchestrates MCPAgent (NucliaDB) + Perplexity in a multi-step flow.

    The question asks for document content AND author info, requiring:
    - MCPAgent to search and retrieve the NucliaDB document (two MCP tool steps)
    - PerplexityAgent to search the web for author information

    This exercises the new preload() integration: SmartAgent discovers MCP tools
    at runtime via preload() and routes the question accordingly.
    """
    DRIVERS[0]["config"]["uri"] = (  # type: ignore
        f"http://{arag_api_http}/api/v1/kb/{article_dataset}/mcp"
    )

    nua_driver = await NUAConnection.model_validate(ROUTER).connect()

    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(driver) for driver in DRIVERS],
        nua=nua_driver,
    )

    smart_agent = await SmartAgent.from_config(
        SmartAgentConfig.model_validate(
            {
                "id": "smart-001",
                "title": "Smart Research Agent",
                "planning_mode": "reactive",
                "executor_model": "claude-4-5-sonnet",
                "extra_prompt": (
                    "Use the MCP NucliaDB agent to search and retrieve documents from the knowledge base. "
                    "Use the Perplexity agent to search the web for information about the authors. "
                    "Always try to answer all parts of the question."
                    "Always try and find information in the  NucliaDB knowledge base using the MCP agent before searching the web with Perplexity."
                ),
                "registered_agents": [
                    {
                        "id": MCP_AGENT_ID,
                        "module": "mcp",
                        "title": "MCP NucliaDB Agent",
                        "transport": "HTTP",
                        "source": "mcphttp-01",
                        "context_validation_model": "claude-4-5-sonnet",
                        "prune_context": False,
                        "tool_choice_model": "claude-4-5-sonnet",
                    },
                    {
                        "id": PERPLEXITY_AGENT_ID,
                        "module": "perplexity",
                        "title": "Perplexity Web Search",
                        "source": "perplexity-01",
                    },
                ],
                "registered_agents_descriptions": {
                    MCP_AGENT_ID: "Searches and retrieves papers from the NucliaDB knowledge base using MCP.",
                    PERPLEXITY_AGENT_ID: "Searches the web for current information about people and topics.",
                },
            }
        )
    )

    memory = EphemeralSessionMemory.from_config(
        MemoryConfig.model_validate(MEMORY),
        agent_id="agent",
        rules=RULES,
        workflow_id="default",
    )
    memory.init("hola")

    question = "Retrieve the document about the architecture of Agents and give me info about the authors"
    question_memory = memory.start_question(question, question_id="question_id")

    contexts = await smart_agent.smart_planner(
        question=question,
        memory=question_memory,
        manager=manager,
        question_uuid="question_id",
    )

    # SmartAgent should have preloaded MCP tools and called them
    smart_steps = [step for step in question_memory.steps if step.module == "smart"]
    assert smart_steps, "SmartAgent should have recorded steps"

    all_step_values = " ".join(step.value or "" for step in question_memory.steps)
    # Verify the MCP two-step flow happened: search then retrieve
    assert "search_documents" in all_step_values, (
        "SmartAgent should have called search_documents via MCP"
    )
    assert "get_document" in all_step_values, (
        "SmartAgent should have called get_document via MCP (two-step flow)"
    )

    # Verify Perplexity was called for author info
    assert "internet_search" in all_step_values

    # Verify contexts were gathered
    assert contexts, "SmartAgent should have collected context"
    total_chunks = sum(len(context.chunks) for context in contexts)
    assert total_chunks > 0, "Should have retrieved document chunks via MCP"

    # Verify document content was retrieved
    expected_texts = [
        "architecture of Agents",
        "Agentic Context Engineering",
        "Philipp SchmidHugging Face",
    ]
    all_chunk_text = " ".join(
        chunk.text for context in contexts for chunk in context.chunks
    )
    assert any(text in all_chunk_text for text in expected_texts), (
        f"Expected document content not found in retrieved chunks. "
        f"Got: {all_chunk_text[:300]}"
    )
