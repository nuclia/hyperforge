import os
from uuid import uuid4

import pytest
from hyperforge.configure import get_driver_config_instance, load_all_configurations
from hyperforge.manager import Manager
from hyperforge.memory import Context
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from nuclia.lib.nua import AsyncNuaClient

from hyperforge_perplexity_search.agent import PerplexitySearchAgent
from hyperforge_perplexity_search.config import PerplexitySearchAgentConfig

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")
PERPLEXITY_KEY = os.environ.get("PERPLEXITY_API_KEY", "DUMMY_PERPLEXITY_KEY")

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True),
    pytest.mark.asyncio,
]

DRIVERS = [
    {
        "provider": "perplexity",
        "identifier": "perplexity-01",
        "name": "perplexity",
        "config": {"key": PERPLEXITY_KEY},
    },
]


async def _run_question(
    drivers: list[dict], question: str, config_overrides: dict | None = None
) -> list[Context]:
    load_all_configurations("hyperforge_perplexity_search")
    load_all_configurations("hyperforge_perplexity")  # register the perplexity driver
    # Perplexity agent doesn't use nua — construct client directly with test values
    nua = AsyncNuaClient(region="europe-1", account="test", token=NUA_KEY)
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(d) for d in drivers],
        nua=nua,
    )
    config = PerplexitySearchAgentConfig.model_validate(
        {
            "module": "perplexity_search",
            "title": "Perplexity Search Agent",
            "source": "perplexity-01",
            **(config_overrides or {}),
        }
    )
    agent = PerplexitySearchAgent(config=config)
    await agent.inner_from_config(config)

    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    memory = session.start_question(question)
    flow_id = uuid4().hex
    await agent.get_question_context(
        memory=memory,
        manager=manager,
        question_uuid=memory.original_question_uuid,
        question=question,
        flow_id=flow_id,
    )
    return memory.get_agent_contexts(flow_id=flow_id, agent_id=agent.agent_id)


async def test_perplexity_search():
    contexts = await _run_question(
        DRIVERS,
        "What is Nuclia?",
        config_overrides={"domain": ["nuclia.com"], "max_results": 3},
    )
    assert len(contexts) > 0
    assert any(ctx.chunks for ctx in contexts)


async def test_perplexity_search_domain():
    contexts = await _run_question(
        DRIVERS,
        "What is Marklogic?",
        config_overrides={"domain": ["progress.com"], "max_results": 5},
    )
    assert len(contexts) > 0
