import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from a2a.utils.errors import InvalidParamsError

from hyperforge.a2a.executor import (
    HyperforgeA2AExecutor,
    PendingA2ATask,
)
from hyperforge.a2a.settings import A2ASettings
from hyperforge.a2a.task_store import PendingTaskRecord
from hyperforge.api.v1.interaction import Shutdown, WebsocketReceiver
from hyperforge.broker import AgentTimeoutError
from hyperforge.pubsub import AgentAnswer, AgentDone


def _executor():
    broker = AsyncMock()
    task_store = AsyncMock()
    context = SimpleNamespace(
        settings=A2ASettings(a2a_task_ttl_seconds=1),
        broker=broker,
        task_store=task_store,
        agent_manager=AsyncMock(),
    )
    return HyperforgeA2AExecutor(context), broker, task_store


async def _empty_stream():
    if False:
        yield


async def test_feedback_timeout_discards_pending_task_and_waiter():
    executor, broker, task_store = _executor()
    receiver = WebsocketReceiver(websocket=None)
    executor._pending_tasks["task"] = PendingA2ATask(
        context_id="context",
        feedback_id="feedback",
        request_id="request",
        receiver=receiver,
        response_stream=_empty_stream(),
    )
    broker.receive_reply.return_value = None

    waiter = asyncio.create_task(executor._wait_for_feedback("task"))
    executor._pending_waiters["task"] = waiter
    await waiter

    assert "task" not in executor._pending_tasks
    assert "task" not in executor._pending_waiters
    assert isinstance(receiver.queue.get_nowait(), Shutdown)
    task_store.remove.assert_awaited_once_with("task")


async def test_cancel_stops_active_task_before_feedback():
    executor, _, task_store = _executor()
    receiver = WebsocketReceiver(websocket=None)
    executor._active_receivers["task"] = receiver
    context = SimpleNamespace(task_id="task", context_id="context")
    event_queue = AsyncMock()

    await executor.cancel(context, event_queue)

    assert isinstance(receiver.queue.get_nowait(), Shutdown)
    assert "task" not in executor._active_receivers
    task_store.remove.assert_awaited_once_with("task")


@pytest.mark.parametrize(
    ("response", "record"),
    [
        ("", None),
        (
            "answer",
            PendingTaskRecord(
                task_id="task",
                context_id="context",
                feedback_id="other-feedback",
                request_id="request",
            ),
        ),
    ],
)
async def test_invalid_feedback_does_not_claim_pending_task(response, record):
    executor, _broker, task_store = _executor()
    task_store.get_pending.return_value = record
    context = Mock(
        task_id="task",
        context_id="context",
        metadata={"feedback_id": "feedback"},
    )
    context.get_user_input.return_value = response

    with pytest.raises(InvalidParamsError):
        await executor._forward_feedback(context, AsyncMock())

    task_store.claim_pending.assert_not_awaited()


async def test_missing_feedback_id_does_not_fail_pending_task():
    executor, _broker, task_store = _executor()
    context = Mock(task_id="task", context_id="context", metadata={})

    with pytest.raises(InvalidParamsError):
        await executor._forward_feedback(context, AsyncMock())

    task_store.claim_pending.assert_not_awaited()


async def test_feedback_delivery_failure_terminates_pending_task():
    executor, broker, task_store = _executor()
    record = PendingTaskRecord(
        task_id="task",
        context_id="context",
        feedback_id="feedback",
        request_id="request",
        owner_instance_id="owner",
    )
    task_store.get_pending.return_value = record
    task_store.claim_pending.return_value = record
    broker.send_reply.side_effect = RuntimeError("Redis unavailable")
    context = Mock(
        task_id="task",
        context_id="context",
        metadata={"feedback_id": "feedback"},
    )
    context.get_user_input.return_value = "answer"
    updater = Mock()
    updater.start_work = AsyncMock()

    updater.failed = AsyncMock()

    await executor._forward_feedback(context, updater)

    task_store.save_pending.assert_not_awaited()
    task_store.remove.assert_awaited_once_with("task")
    updater.failed.assert_awaited_once()


async def test_relay_retries_keepalive_timeout_with_stable_cursor():
    executor, broker, _task_store = _executor()
    cursors = []
    attempts = 0

    async def subscribe(_topic, cursor):
        nonlocal attempts
        attempts += 1
        cursors.append(cursor)
        if attempts == 1:
            yield "1", AgentDone()
            raise AgentTimeoutError("relay")
        yield "2", AgentDone()

    broker.subscribe = subscribe

    messages = [message async for message in executor._subscribe_relay("relay")]

    assert len(messages) == 2
    assert cursors == ["0", "1"]


async def test_initial_stream_exhaustion_fails_and_cleans_task():
    executor, _broker, task_store = _executor()
    updater = Mock(context_id="context")
    updater.failed = AsyncMock()

    await executor._stream_until_pause(
        "task", updater, WebsocketReceiver(websocket=None), _empty_stream()
    )

    updater.failed.assert_awaited_once()
    task_store.remove.assert_awaited_once_with("task")


async def test_relay_stream_exhaustion_publishes_error_and_done():
    executor, broker, task_store = _executor()
    pending = PendingA2ATask(
        context_id="context",
        feedback_id="feedback",
        request_id="request",
        receiver=WebsocketReceiver(websocket=None),
        response_stream=_empty_stream(),
    )

    await executor._relay_until_pause("task", pending, "relay")

    published = [call.args[1] for call in broker.publish.await_args_list]
    assert isinstance(published[0], AgentAnswer)
    assert published[0].answer.exception is not None
    assert isinstance(published[1], AgentDone)
    task_store.remove.assert_awaited_once_with("task")
