"""
Unit tests for client-managed chat history in QuestionMemory.
"""

import pytest
from nuclia.lib.nua_responses import Author

from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import HistoryQuestionAnswer, MemoryConfig, Rules
from hyperforge.server.cache import NoCache


def _make_ephemeral_session() -> EphemeralSessionMemory:
    session = EphemeralSessionMemory(
        config=MemoryConfig(),
        agent_id="agent-1",
        workflow_id="default",
        cache=NoCache(),
    )
    session.rules = Rules().rules
    session.init("session-1")
    return session


@pytest.mark.asyncio
async def test_client_history_overrides_session_history():
    """Client-provided history takes precedence over any server-side accumulated history.
    Without client history, the session's stored history is used instead."""
    session = _make_ephemeral_session()

    # Accumulate one server-side turn
    first = session.start_question("Server Q")
    first.final_answer = "Server A"
    await session.save(first)

    # Second request arrives with client-managed history (different content)
    client_history = [HistoryQuestionAnswer(question="Client Q", answer="Client A")]
    memory = session.start_question("New question", chat_history=client_history)

    context_str, count = await memory.context_history()
    messages = await memory.get_chat_history()

    # context_history: client content present, server content absent
    assert count == 1
    assert "Client Q" in context_str and "Client A" in context_str
    assert "Server Q" not in context_str and "Server A" not in context_str

    # get_chat_history: correct author ordering
    assert len(messages) == 2
    assert messages[0].author == Author.USER and messages[0].text == "Client Q"
    assert messages[1].author == Author.NUCLIA and messages[1].text == "Client A"

    # Third request with no client history falls back to server-stored history
    memory_no_history = session.start_question("Another question")

    context_str, count = await memory_no_history.context_history()

    assert count == 1
    assert "Server Q" in context_str and "Server A" in context_str


@pytest.mark.asyncio
async def test_empty_list_chat_history_overrides_server_history():
    """An explicit chat_history=[] must override server-side history (clear it),
    not fall back to it. This distinguishes 'omitted' (None) from 'intentionally empty' ([])."""
    session = _make_ephemeral_session()

    # Accumulate one server-side turn
    first = session.start_question("Server Q")
    first.final_answer = "Server A"
    await session.save(first)

    # Omitting chat_history → falls back to server-side history
    memory_omitted = session.start_question("Q")
    context_str, count = await memory_omitted.context_history()
    assert count == 1
    assert "Server Q" in context_str

    # Passing [] → overrides with no history (does not fall back to server)
    memory_empty = session.start_question("Q", chat_history=[])
    context_str, count = await memory_empty.context_history()
    messages = await memory_empty.get_chat_history()
    assert count == 0
    assert context_str == ""
    assert messages == []
