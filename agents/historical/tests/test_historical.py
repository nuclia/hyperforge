from unittest.mock import AsyncMock, patch

import pytest
from hyperforge.manager import Manager
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from hyperforge_historical.agent import HistoricalAgent
from hyperforge_historical.config import HistoricalAgentConfig
from nucliadb_models.search import KnowledgeboxFindResults


def make_agent(all_sessions: bool = False) -> HistoricalAgent:
    config = HistoricalAgentConfig(id="test-historical", all=all_sessions)
    return HistoricalAgent(config=config)


def make_memory(question: str = "What is the capital of France?"):
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    return session.start_question(question)


# --- Config tests ---


def test_config_module_literal():
    config = HistoricalAgentConfig(all=True)
    assert config.module == "historical"


def test_step_title():
    agent = make_agent()
    assert agent.step_title("Historical context") == "History: Historical context"


# --- Behaviour tests ---


async def test_no_history_hit_adds_step_but_no_answer():
    """When search returns no results, only a step is added, no answer."""
    agent = make_agent()
    memory = make_memory()

    # EphemeralSessionMemory already returns total=0 by default
    await agent(memory=memory, manager=None)

    assert len(memory.steps) == 1
    assert memory.steps[0].module == "historical"
    assert len(memory.answers) == 0


async def test_history_hit_adds_empty_answer_and_step():
    """When search finds results, an empty answer and a step are both recorded."""
    agent = make_agent()
    memory = make_memory()

    with patch.object(
        memory,
        "search_in_questions",
        new=AsyncMock(return_value=KnowledgeboxFindResults(total=2, resources={})),
    ):
        await agent(memory=memory, manager=None)

    assert len(memory.steps) == 1
    assert memory.steps[0].module == "historical"
    assert len(memory.answers) == 1
    # answers are stored as (text, metadata) tuples
    assert memory.answers[0][0] == ""


async def test_all_false_passed_to_search():
    """config.all=False is forwarded to search_in_questions."""
    agent = make_agent(all_sessions=False)
    memory = make_memory()

    mock_search = AsyncMock(return_value=KnowledgeboxFindResults(total=0, resources={}))
    with patch.object(memory, "search_in_questions", new=mock_search):
        await agent(memory=memory, manager=None)

    mock_search.assert_awaited_once_with(memory.original_question, False)


async def test_all_true_passed_to_search():
    """config.all=True is forwarded to search_in_questions."""
    agent = make_agent(all_sessions=True)
    memory = make_memory()

    mock_search = AsyncMock(return_value=KnowledgeboxFindResults(total=0, resources={}))
    with patch.object(memory, "search_in_questions", new=mock_search):
        await agent(memory=memory, manager=None)

    mock_search.assert_awaited_once_with(memory.original_question, True)


async def test_step_has_correct_fields():
    """Recorded step has the right module, title and agent_path."""
    agent = make_agent()
    memory = make_memory()

    await agent(memory=memory, manager=None)

    step = memory.steps[0]
    assert step.module == "historical"
    assert step.title == "History: Historical context"
    assert step.agent_path == f"/preprocess/{agent.agent_id}"


async def test_none_question_skips_search_and_adds_step():
    """When original_question is None the search is skipped but a step is still added."""
    agent = make_agent()
    memory = make_memory()
    memory.original_question = None

    mock_search = AsyncMock(return_value=KnowledgeboxFindResults(total=0, resources={}))
    with patch.object(memory, "search_in_questions", new=mock_search):
        await agent(memory=memory, manager=None)

    mock_search.assert_not_awaited()
    assert len(memory.steps) == 1
    assert len(memory.answers) == 0
