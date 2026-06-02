from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from hyperforge.exceptions import MaxRetries
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from hyperforge_restart.agent import RestartAgent
from hyperforge_restart.config import RestartAgentConfig

pytestmark = pytest.mark.asyncio


def make_memory(restart_steps: int = 0):
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    memory = session.start_question("What is the capital of France?")
    # Inject fake restart steps into memory
    for _ in range(restart_steps):
        step = MagicMock()
        step.module = "restart"
        memory.steps.append(step)
    return memory


def make_manager(answered: bool, questions: list[str] | None = None):
    manager = MagicMock(spec=Manager)
    manager.execute_json = AsyncMock(
        return_value=(
            {
                "questions": questions or ["What language do they speak in France?"],
                "reason": "Need to confirm language spoken.",
                "answered": answered,
            },
            10,
            20,
        )
    )
    return manager


async def test_restart_agent_answered():
    """When the LLM says the question is already answered, restart should be False and no step added."""
    config = RestartAgentConfig(model="gpt-4o", retries=2)
    agent = RestartAgent(config=config)
    memory = make_memory(restart_steps=0)
    manager = make_manager(answered=True)

    await agent(memory=memory, manager=manager)

    assert memory.restart is False
    assert not any(s.module == "restart" for s in memory.steps)


async def test_restart_agent_not_answered_first_retry():
    """When not answered and retries not exhausted, a restart step should be added."""
    config = RestartAgentConfig(model="gpt-4o", retries=2)
    agent = RestartAgent(config=config)
    memory = make_memory(restart_steps=0)
    manager = make_manager(answered=False, questions=["What language do they speak?"])

    await agent(memory=memory, manager=manager)

    assert memory.restart is True
    restart_steps = [s for s in memory.steps if s.module == "restart"]
    assert len(restart_steps) == 1


async def test_restart_agent_not_answered_last_allowed_retry():
    """On the last allowed retry (retry < retries), a restart step should still be added."""
    config = RestartAgentConfig(model="gpt-4o", retries=2)
    agent = RestartAgent(config=config)
    # One restart step already in memory, retries=2 so retry=1 < 2
    memory = make_memory(restart_steps=1)
    manager = make_manager(answered=False, questions=["Another clarification?"])

    await agent(memory=memory, manager=manager)

    assert memory.restart is True
    restart_steps = [s for s in memory.steps if s.module == "restart"]
    assert len(restart_steps) == 2


async def test_restart_agent_max_retries_raises():
    """When retries are exhausted and question is not answered, MaxRetries should be raised."""
    config = RestartAgentConfig(model="gpt-4o", retries=2)
    agent = RestartAgent(config=config)
    # Two restart steps already means retry == retries
    memory = make_memory(restart_steps=2)
    manager = make_manager(answered=False)

    with pytest.raises(MaxRetries):
        await agent(memory=memory, manager=manager)


async def test_restart_agent_context_questions_updated():
    """New questions from the LLM should be added to memory context questions."""
    config = RestartAgentConfig(model="gpt-4o", retries=3)
    agent = RestartAgent(config=config)
    memory = make_memory(restart_steps=0)
    new_questions = ["What year did it become the capital?", "What is the population?"]
    manager = make_manager(answered=False, questions=new_questions)

    await agent(memory=memory, manager=manager)

    stored_questions = list(memory.context_questions.values())
    for q in new_questions:
        assert q in stored_questions
