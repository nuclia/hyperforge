import os
from uuid import uuid4

import pytest
from hyperforge.configure import get_driver_config_instance, load_all_configurations
from hyperforge.manager import Manager
from hyperforge.memory import Context
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from nuclia.lib.nua import AsyncNuaClient

from hyperforge_google.agent import GoogleAgent
from hyperforge_google.config import GoogleAgentConfig

NUA_KEY = os.environ.get("NUA_KEY", "DUMMY")

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True),
    pytest.mark.asyncio,
]


DRIVERS = [
    {
        "provider": "google",
        "identifier": "google-01",
        "name": "google",
        "config": {
            "vertexai": False,
            "api_key": os.environ.get("GOOGLE_API_KEY", "DUMMY_API_KEY"),
        },
    },
]


async def _run_question(
    drivers: list[dict],
    question: str,
) -> list[Context]:
    """Build a manager from driver configs, run the agent, return contexts."""
    load_all_configurations("hyperforge_google")
    # Google agent doesn't use nua — construct client directly with test values
    nua = AsyncNuaClient(region="europe-1", account="test", token=NUA_KEY)
    manager = await Manager.from_config(
        drivers=[get_driver_config_instance(d) for d in drivers],
        nua=nua,
    )
    config = GoogleAgentConfig.model_validate(
        {
            "module": "google",
            "title": "Google Agent",
            "source": "google-01",
        }
    )
    agent = GoogleAgent(config=config)
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


async def test_google():
    contexts = await _run_question(
        DRIVERS,
        "What is Nuclia?",
    )
    assert len(contexts) > 0
    all_text = " ".join(chunk.text for ctx in contexts for chunk in ctx.chunks).lower()
    summary = " ".join(ctx.summary for ctx in contexts if ctx.summary).lower()
    assert "rag" in all_text or "ai" in all_text or "rag" in summary or "ai" in summary
