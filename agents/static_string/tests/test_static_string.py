from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from hyperforge_static_string.agent import StaticStringAgent, StaticStringAgentConfig

pytestmark = pytest.mark.asyncio


def make_config(**kwargs) -> StaticStringAgentConfig:
    """Helper to create a StaticStringAgentConfig with sensible test defaults."""
    defaults = dict(
        id="test-static-string",
        title="Static String",
        module="static_string",
        prune_context=False,  # Disable validation so no LLM calls are needed
    )
    defaults.update(kwargs)
    return StaticStringAgentConfig(**defaults)


def make_session() -> EphemeralSessionMemory:
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    return session


async def test_static_string_returns_context_string():
    """static_string() returns exactly the configured context string."""
    config = make_config(context="Hello, world!")
    agent = StaticStringAgent(config=config)
    assert agent.static_string() == "Hello, world!"


async def test_get_question_context_saves_context_to_memory():
    """_get_question_context saves a Context with the correct chunk to memory."""
    manager = MagicMock(spec=Manager)
    config = make_config(context="Some static context text.")
    agent = StaticStringAgent(config=config)

    memory = make_session().start_question("What is the answer?")
    flow_id = uuid4().hex

    missing = await agent._get_question_context(
        memory=memory,
        manager=manager,
        question_uuid=memory.original_question_uuid,
        question="What is the answer?",
        flow_id=flow_id,
    )

    # No missing context — everything was provided
    assert missing == []

    # Context was saved with the correct chunk.
    # Note: StaticStringAgent does not set agent_id on the Context object,
    # so contexts are stored under the empty-string key.
    saved = memory.get_agent_contexts(flow_id=flow_id, agent_id="")
    assert len(saved) == 1
    assert saved[0].chunks[0].chunk_id == "static_string"
    assert saved[0].chunks[0].text == "Some static context text."
    assert saved[0].chunks[0].origin_agent == config.module


async def test_get_question_context_records_step():
    """_get_question_context adds exactly one step with correct metadata."""
    manager = MagicMock(spec=Manager)
    config = make_config(context="Step test context.")
    agent = StaticStringAgent(config=config)

    memory = make_session().start_question("Any question?")
    flow_id = uuid4().hex

    await agent._get_question_context(
        memory=memory,
        manager=manager,
        question_uuid=memory.original_question_uuid,
        question="Any question?",
        flow_id=flow_id,
    )

    assert len(memory.steps) == 1
    step = memory.steps[0]
    assert step.module == "static_string"
    assert "Search results" in step.title
    assert step.value == "String done"


async def test_get_question_context_different_flow_ids():
    """Contexts are scoped to their flow_id and do not bleed across flows."""
    manager = MagicMock(spec=Manager)
    config = make_config(context="Flow-isolated context.")
    agent = StaticStringAgent(config=config)

    memory = make_session().start_question("Question?")
    flow_id_a = uuid4().hex
    flow_id_b = uuid4().hex

    await agent._get_question_context(
        memory=memory,
        manager=manager,
        question_uuid=memory.original_question_uuid,
        question="Question?",
        flow_id=flow_id_a,
    )

    saved_a = memory.get_agent_contexts(flow_id=flow_id_a, agent_id="")
    saved_b = memory.get_agent_contexts(flow_id=flow_id_b, agent_id="")
    assert len(saved_a) == 1
    assert len(saved_b) == 0
