import os
from uuid import uuid4

import pytest
from hyperforge.configure import get_driver_config_instance, load_all_configurations
from hyperforge.manager import Manager
from hyperforge.memory import Context
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import MemoryConfig, Rules
from hyperforge_perplexity.config import PerplexityAgentConfig
from hyperforge_perplexity.perplexity import PerplexityAgent
from nuclia.lib.nua import AsyncNuaClient

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

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
    drivers: list[dict],
    question: str,
    config_overrides: dict | None = None,
) -> list[Context]:
    """Build a manager from driver configs, run the agent, return contexts."""
    load_all_configurations("hyperforge_perplexity")
    # Perplexity agent doesn't use nua — construct client directly with test values
    nua = AsyncNuaClient(region="europe-1", account="test", token=NUA_KEY)
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(d) for d in drivers],
        nua=nua,
    )
    config = PerplexityAgentConfig.model_validate(
        {
            "module": "perplexity",
            "title": "Perplexity Agent",
            "source": "perplexity-01",
            **(config_overrides or {}),
        }
    )
    agent = PerplexityAgent(config=config)
    await agent.inner_from_config(config)

    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(),
        agent_id="test",
        workflow_id="test",
        rules=Rules(),
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


async def test_perplexity():
    contexts = await _run_question(
        DRIVERS,
        "What is Nuclia?",
        config_overrides={"domain": ["nuclia.com"], "related_questions": True},
    )
    assert len(contexts) > 0
    assert any(ctx.chunks for ctx in contexts)


@pytest.mark.skipif(
    os.environ.get("LOCAL_TESTING") is None,
    reason="Only run when LOCAL_TESTING env var is set.",
)
async def test_perplexity_images():
    contexts = await _run_question(
        DRIVERS,
        "What colors are 'duppi' usually?",
        config_overrides={"images": True, "related_questions": True},
    )
    assert len(contexts) > 0
    assert any(ctx.chunks for ctx in contexts)
    assert any(ctx.images for ctx in contexts)
