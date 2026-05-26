from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from hyperforge_static.agent import StaticAgent
from hyperforge_static.config import StaticAgentConfig

pytestmark = pytest.mark.asyncio


async def test_static_agent():
    # 1. Create a mocked manager
    manager = MagicMock(spec=Manager)

    # 2. Create the configuration for StaticAgent
    config = StaticAgentConfig(
        id="test-static",
        title="Static Context",
        context="This is a test static context text.",
        structured="{'key': 'value'}",
        prune_context=False,
    )

    # 3. Create the StaticAgent instance
    agent = StaticAgent(config=config)

    # 4. Set up EphemeralSessionMemory and QuestionMemory
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    memory = session.start_question("What is the static context?")
    flow_id = uuid4().hex

    # 5. Call static_context directly
    context = await agent.static_context(
        memory=memory,
        manager=manager,
        question="What is the static context?",
    )

    # 6. Assert static_context returns Context with correct values
    assert context is not None
    assert context.agent_id == "test-static"
    assert context.title == "Static Context"
    assert len(context.chunks) == 1
    assert context.chunks[0].text == "This is a test static context text."
    assert len(context.structured) == 1
    assert context.structured[0] == "{'key': 'value'}"

    # 7. Call _get_question_context
    missing = await agent._get_question_context(
        memory=memory,
        manager=manager,
        question_uuid=memory.original_question_uuid,
        question="What is the static context?",
        flow_id=flow_id,
    )

    # 8. Assertions on the memory state and missing questions
    assert (
        missing == []
    )  # Since no validation/fallback model is used, should return empty list

    # Check that context is saved to memory under the given flow_id
    saved_contexts = memory.get_agent_contexts(flow_id=flow_id, agent_id="test-static")
    assert len(saved_contexts) == 1
    assert saved_contexts[0].chunks[0].text == "This is a test static context text."
    assert saved_contexts[0].structured[0] == "{'key': 'value'}"

    # Verify step was added to memory
    assert len(memory.steps) == 1
    step = memory.steps[0]
    assert step.module == "static"
    assert "Static context" in step.title
    assert step.value == " Static context retrieval"
