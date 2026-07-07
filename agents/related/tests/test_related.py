from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge_related.agent import RelatedAgent
from hyperforge_related.config import RelatedAgentConfig

from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules


def make_agent(
    model: str = "chatgpt-azure-4o-mini", prompt: str | None = None
) -> RelatedAgent:
    config = RelatedAgentConfig(id="test-related", model=model, prompt=prompt)
    return RelatedAgent(config=config)


def make_memory(question: str = "What is machine learning?"):
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    return session.start_question(question)


def make_manager(related: list[str] | None = None, return_none: bool = False):
    manager = MagicMock(spec=Manager)
    questions = (
        ["How does ML work?", "What is deep learning?"] if related is None else related
    )
    payload = None if return_none else {"related": questions}
    manager.execute_json = AsyncMock(return_value=(payload, 10, 20))
    return manager


# --- Behaviour tests ---


@pytest.mark.asyncio
async def test_related_questions_added_to_future_questions():
    """Questions returned by the LLM are stored in memory.future_questions."""
    agent = make_agent()
    memory = make_memory()
    questions = ["What is supervised learning?", "What is a neural network?"]
    manager = make_manager(related=questions)

    await agent(memory=memory, manager=manager)

    stored = list(memory.future_questions.values())
    for q in questions:
        assert q in stored


@pytest.mark.asyncio
async def test_step_added_when_related_not_none():
    """A step is recorded when the LLM returns a non-None result."""
    agent = make_agent()
    memory = make_memory()
    manager = make_manager(related=["Some follow-up?"])

    await agent(memory=memory, manager=manager)

    assert len(memory.steps) == 1
    step = memory.steps[0]
    assert step.module == "related"
    assert step.title == "Related: Related questions"
    assert step.agent_path == f"/postprocess/{agent.config.id}"


@pytest.mark.asyncio
async def test_no_step_when_llm_returns_none():
    """When execute_json returns None, no step and no future questions are added."""
    agent = make_agent()
    memory = make_memory()
    manager = make_manager(return_none=True)

    await agent(memory=memory, manager=manager)

    assert len(memory.steps) == 0
    assert len(memory.future_questions) == 0


@pytest.mark.asyncio
async def test_empty_related_list_adds_step_but_no_future_questions():
    """When the LLM returns an empty list, a step is still added but future_questions stays empty."""
    agent = make_agent()
    memory = make_memory()
    manager = make_manager(related=[])

    await agent(memory=memory, manager=manager)

    assert len(memory.steps) == 1
    assert len(memory.future_questions) == 0


@pytest.mark.asyncio
async def test_execute_json_called_with_correct_model():
    """The configured model name is forwarded to manager.execute_json."""
    agent = make_agent(model="my-custom-model")
    memory = make_memory()
    manager = make_manager()

    await agent(memory=memory, manager=manager)

    call_kwargs = manager.execute_json.call_args.kwargs
    assert call_kwargs["model"] == "my-custom-model"


@pytest.mark.asyncio
async def test_custom_prompt_rendered_in_request():
    """A custom prompt from config is included in the rendered prompt sent to the LLM."""
    custom_prompt = "Focus on practical applications only."
    agent = make_agent(prompt=custom_prompt)
    memory = make_memory()
    manager = make_manager()

    await agent(memory=memory, manager=manager)

    rendered_prompt = manager.execute_json.call_args.kwargs["prompt"]
    assert custom_prompt in rendered_prompt


@pytest.mark.asyncio
async def test_question_included_in_rendered_prompt():
    """The original question is included in the prompt sent to the LLM."""
    question = "What are transformers in NLP?"
    agent = make_agent()
    memory = make_memory(question=question)
    manager = make_manager()

    await agent(memory=memory, manager=manager)

    rendered_prompt = manager.execute_json.call_args.kwargs["prompt"]
    assert question in rendered_prompt
