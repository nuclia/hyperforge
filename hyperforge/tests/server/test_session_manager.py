import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperforge import engine
from hyperforge.interaction import AnswerOperation
from hyperforge.pubsub import StartInteraction
from hyperforge.server.session import SessionManager


@pytest.mark.asyncio
async def test_ephemeral_session_is_not_cached():
    manager = SessionManager(
        settings=SimpleNamespace(
            answers_subject="arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.answer",
            internal_nucliadb_url="",
            internal_nua_api="",
            internal_nua=False,
            local_openai=None,
            external_nua_api_key=None,
            standalone=False,
            allow_private_network_endpoints=False,
        ),
        broker=None,  # type: ignore[arg-type]
        agent_manager=SimpleNamespace(get_agent_config=AsyncMock()),
        cache=None,  # type: ignore[arg-type]
    )
    manager.agent_manager.get_agent_config.return_value = SimpleNamespace(
        memory=SimpleNamespace(),
        rules=SimpleNamespace(rules=[]),
    )

    memory = SimpleNamespace(rules=None)
    memory.start_question = MagicMock(return_value=SimpleNamespace())
    task = MagicMock()
    message = StartInteraction(
        account="account",
        agent_id="agent",
        session="ephemeral",
        question_id="question-id",
        question="question",
    )

    with (
        patch(
            "hyperforge.server.session.get_memory",
            new_callable=AsyncMock,
            return_value=memory,
        ) as get_memory,
        patch(
            "hyperforge.server.session.get_state",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(),
        ),
        patch.object(manager, "answer", new=MagicMock(return_value=object())),
        patch("hyperforge.server.session.asyncio.create_task", return_value=task),
    ):
        await manager.activate(message)

    get_memory.assert_awaited_once()
    assert "ephemeral" not in manager.memory


@pytest.mark.asyncio
async def test_answer_closes_manager():
    manager = SessionManager(
        settings=SimpleNamespace(question_timeout_seconds=10),
        broker=SimpleNamespace(keepalive_seconds=10),
        agent_manager=None,
        cache=None,
    )
    manager.callback = AsyncMock()
    manager.send_message = AsyncMock()
    manager.process_event = MagicMock()
    state_manager = SimpleNamespace(aclose=AsyncMock())
    state = SimpleNamespace(
        manager=state_manager,
        agent=AsyncMock(),
    )
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        set_feedback_fn=MagicMock(),
        set_oauth_fn=MagicMock(),
        set_oauth_callback_fn=MagicMock(),
        session=SimpleNamespace(id="session"),
        final_answer=None,
        final_answer_citations=None,
        final_answer_urls=None,
        data_visualizations=None,
        save=AsyncMock(),
    )

    await manager.answer(
        "account", "agent", "workflow", "topic", state, question_memory
    )

    state_manager.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_answer_logs_original_exception_group():
    manager = SessionManager(
        settings=SimpleNamespace(question_timeout_seconds=10),
        broker=SimpleNamespace(keepalive_seconds=10),
        agent_manager=None,
        cache=None,
    )
    manager.callback = AsyncMock()
    manager.send_message = AsyncMock()
    original_error = ExceptionGroup("task group", [RuntimeError("provider failed")])
    state = SimpleNamespace(
        manager=SimpleNamespace(aclose=AsyncMock()),
        agent=AsyncMock(side_effect=original_error),
    )
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        set_feedback_fn=MagicMock(),
        set_oauth_fn=MagicMock(),
        set_oauth_callback_fn=MagicMock(),
        session=SimpleNamespace(id="session"),
        final_answer=None,
        final_answer_citations=None,
        final_answer_urls=None,
        data_visualizations=None,
        save=AsyncMock(),
    )

    with (
        patch("hyperforge.server.session.logger.exception") as log_exception,
        patch("hyperforge.server.session.errors.capture_exception") as capture,
    ):
        await manager.answer(
            "account", "agent", "workflow", "topic", state, question_memory
        )

    log_exception.assert_called_once_with("Answering exception")
    capture.assert_called_once_with(original_error)
    answer = manager.callback.await_args_list[-1].args[1]
    assert answer.exception.detail == "provider failed"


@pytest.mark.asyncio
async def test_answer_handles_mixed_cancellation_exception_group():
    manager = SessionManager(
        settings=SimpleNamespace(question_timeout_seconds=10),
        broker=SimpleNamespace(keepalive_seconds=10),
        agent_manager=None,
        cache=None,
    )
    manager.callback = AsyncMock()
    manager.send_message = AsyncMock()
    state = SimpleNamespace(
        manager=SimpleNamespace(aclose=AsyncMock()),
        agent=AsyncMock(
            side_effect=BaseExceptionGroup(
                "task group",
                [asyncio.CancelledError(), RuntimeError("provider failed")],
            )
        ),
    )
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        set_feedback_fn=MagicMock(),
        set_oauth_fn=MagicMock(),
        set_oauth_callback_fn=MagicMock(),
        session=SimpleNamespace(id="session"),
        final_answer=None,
        final_answer_citations=None,
        final_answer_urls=None,
        data_visualizations=None,
        save=AsyncMock(),
    )

    await manager.answer(
        "account", "agent", "workflow", "topic", state, question_memory
    )

    answer = manager.callback.await_args_list[-1].args[1]
    assert answer.operation == AnswerOperation.ERROR
    assert answer.exception.detail == "provider failed"


@pytest.mark.asyncio
async def test_answer_propagates_cancellation_only_exception_group():
    manager = SessionManager(
        settings=SimpleNamespace(question_timeout_seconds=10),
        broker=SimpleNamespace(keepalive_seconds=10),
        agent_manager=None,
        cache=None,
    )
    manager.callback = AsyncMock()
    state_manager = SimpleNamespace(aclose=AsyncMock())
    cancellation = BaseExceptionGroup("task group", [asyncio.CancelledError()])
    state = SimpleNamespace(
        manager=state_manager,
        agent=AsyncMock(side_effect=cancellation),
    )
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        set_feedback_fn=MagicMock(),
        set_oauth_fn=MagicMock(),
        set_oauth_callback_fn=MagicMock(),
        session=SimpleNamespace(id="session"),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await manager.answer(
            "account", "agent", "workflow", "topic", state, question_memory
        )

    assert raised.value is cancellation
    state_manager.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_engine_main_closes_manager(monkeypatch):
    state_manager = SimpleNamespace(aclose=AsyncMock())
    state = SimpleNamespace(manager=state_manager, agent=AsyncMock())
    question_memory = SimpleNamespace(
        set_callback_fn=MagicMock(),
        session=SimpleNamespace(user_info={}),
        headers={},
    )
    session_memory = SimpleNamespace(
        start_question=MagicMock(return_value=question_memory)
    )
    monkeypatch.setattr(
        engine,
        "init",
        AsyncMock(return_value=(state, session_memory)),
    )

    await engine.main(config={})

    state_manager.aclose.assert_awaited_once_with()
