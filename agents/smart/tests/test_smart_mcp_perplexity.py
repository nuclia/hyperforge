import os

import pytest
from hyperforge.configure import get_driver_config_instance
from hyperforge.llm import NUAConnection
from hyperforge.manager import Manager
from hyperforge.memory import Rule, Rules
from hyperforge.memory.memory import EphemeralSessionMemory, MemoryConfig
from nuclia import REGIONAL
from nucliadb_utils.settings import nuclia_settings

from agents.smart.src.hyperforge_smart.agent import SmartAgent
from agents.smart.src.hyperforge_smart.config import SmartAgentConfig
from nucliadb_agentic_api.src.nucliadb_agentic_api.ask.predict import (
    start_predict_engine,
    stop_predict_engine,
)
from nucliadb_agentic_api.src.nucliadb_agentic_api.ask.settings import (
    settings as nucliadb_local_settings,
)

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
            "key": "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",
        },
    },
]

MEMORY = {"nucliadb": {"url": "", "key": "", "kbid": ""}}

ROUTER = {
    "local_openai": "http://127.0.0.1:1234/v1",
    "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6Im51YSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3MzcwNjE2NjUsInN1YiI6IjAzZDQ3OTk4LWY2NzItNGE5Yi1hNTdiLWJkZTNhOWMzZWU4YyIsImp0aSI6IjQ1YTQ3NjJkLTIyMzgtNGI2OC05YTI2LWQ4N2QzNjJhZThmYiIsImV4cCI6MjUzMzcwNzY0ODAwLCJrZXkiOiI0N2JjZDU4ZC04NDExLTQ1NTgtYWIzZS0wNGMyMWI4NWI2ZWEiLCJhbGxvd19rYl9tYW5hZ2VtZW50IjpmYWxzZX0.Ljgv780vMuwviospTcRQYxrFV_H7XXR0hJeeSyFIfwVjni7hyyrxB189R5rQyLLI2n85iAdNGshvc8etDQRkXr8n8IWFsy_FOWcru-LZFZwGCpsY6hKK4TdWXR9v5sxA5xyKA7lmWw1LZ8dfNbcdx11OY15BfmGuMpiq_auIs1F90C8T8_LmXbz0SbdYzPIoEP0JFBX92jHqDoJNUTlMELUrcjupK9ao2pZahI47zQHrWjGuw2KrSjghdZgzwjC0YEa7C8quEVZ9SoLOkJvJV7XV4LrlGGcsxZzng8kLBGRBS-i8p26n5vFvMqiZKqDWpq68cVzZhAsL93wkzHVZCAHpfEsHQ4DUb-Da53xUrrnVnyl1w79iXiLYwP0wxh3b34B1b1ca3rRKuifbd1e762gf11qw6LHpJ9qKYhRv6O3KZ18_amwjLhqYna5uUfrP7f59tJZ9vzTG1oTZ5KlMBeVfu_IvhAmMbGpTygqEoxXqNrH3lWOsEPLhRVBC6D5t84xy7WLe4XsGR4xWduLWHsjxPYbmTrLMysGSqBSNGPwUi8jMTrH16-xprNJRiWVHcvgz_FGQ7sT7RucaAxhmFlZY9h3BFw7u_6awOeX4ymhH6_iDzWxBc0Fx5JsDgQm9jkhlYIHqZG36N5XfsmqfCyM12gNa37j-8MPOt7eU0XQ",
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

    if "http" in nua_driver.region:
        url = nua_driver.region.strip("/")
    else:
        url = REGIONAL.format(region=nua_driver.region).strip("/")

    nucliadb_local_settings.nucliadb_reader_address = f"http://{arag_api_http}/api"
    nucliadb_local_settings.nucliadb_search_address = f"http://{arag_api_http}/api"
    nuclia_settings.onprem = True
    nuclia_settings.nuclia_public_url = url
    nuclia_settings.nuclia_service_account = ROUTER["key"]
    nuclia_settings.nuclia_zone = nua_driver.region
    await start_predict_engine()

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

    context = await smart_agent.smart_planner(
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
    assert context, "SmartAgent should have collected context"
    total_chunks = len(context.chunks)
    assert total_chunks > 0, "Should have retrieved document chunks via MCP"

    # Verify document content was retrieved
    expected_texts = [
        "architecture of Agents",
        "Agentic Context Engineering",
        "Philipp SchmidHugging Face",
    ]
    all_chunk_text = " ".join(chunk.text for chunk in context.chunks)
    assert any(text in all_chunk_text for text in expected_texts), (
        f"Expected document content not found in retrieved chunks. "
        f"Got: {all_chunk_text[:300]}"
    )

    await stop_predict_engine()
