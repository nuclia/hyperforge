"""A2A ``AgentExecutor`` bridging incoming A2A messages to Hyperforge."""

import asyncio
import json
from dataclasses import asdict, dataclass
from typing import Any, AsyncIterator
from uuid import uuid4

from a2a.helpers import new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import a2a_pb2

from hyperforge.a2a import logger
from hyperforge.a2a.context import A2AServerContext
from hyperforge.a2a.task_store import PendingTaskRecord
from hyperforge.api.models import InteractionRequest
from hyperforge.api.v1.interaction import Shutdown, WebsocketReceiver, stream_response
from hyperforge.db import exceptions
from hyperforge.interaction import AnswerOperation, AragAnswer
from hyperforge.pubsub import AgentAnswer, AgentDone, UserToAgentInteraction

# Metadata keys read from the incoming A2A message to route the interaction.
META_ACCOUNT = "account"
META_AGENT_ID = "agent_id"
META_WORKFLOW_ID = "workflow_id"
META_SESSION = "session"
META_HEADERS = "headers"
META_ARGUMENTS = "arguments"
META_FEEDBACK_ID = "feedback_id"
_ALLOWED_METADATA = {
    META_ACCOUNT,
    META_AGENT_ID,
    META_WORKFLOW_ID,
    META_SESSION,
    META_HEADERS,
    META_ARGUMENTS,
    META_FEEDBACK_ID,
}


@dataclass(frozen=True)
class A2ARouting:
    account: str
    agent_id: str
    workflow_id: str
    session: str
    headers: dict[str, str]
    arguments: dict[str, str]


@dataclass
class PendingA2ATask:
    context_id: str
    feedback_id: str
    request_id: str
    routing: A2ARouting
    receiver: WebsocketReceiver
    response_stream: AsyncIterator[AragAnswer]


def _owner_reply_subject(owner_instance_id: str, task_id: str) -> str:
    return f"hyperforge:a2a:feedback:{owner_instance_id}:{task_id}"


def _text_part(text: str) -> a2a_pb2.Part:
    return a2a_pb2.Part(text=text)


def _optional_string(metadata: dict[str, Any], name: str) -> str | None:
    value = metadata.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"A2A metadata '{name}' must be a non-empty string")
    return value


def _string_mapping(metadata: dict[str, Any], name: str) -> dict[str, str]:
    value = metadata.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"A2A metadata '{name}' must be an object")

    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"A2A metadata '{name}' must use non-empty string keys")
        if isinstance(item, (dict, list)) or item is None:
            raise ValueError(f"A2A metadata '{name}.{key}' must be a scalar value")
        result[key] = str(item)
    return result


def parse_routing_metadata(
    metadata: dict[str, Any], settings: Any, context_id: str
) -> A2ARouting:
    """Validate client metadata against the fixed A2A server identity."""
    unknown_keys = set(metadata).difference(_ALLOWED_METADATA)
    if unknown_keys:
        keys = ", ".join(sorted(unknown_keys))
        raise ValueError(f"Unknown A2A metadata field(s): {keys}")

    if not settings.a2a_account or not settings.a2a_agent_id:
        raise ValueError("A2A server identity is not configured")

    account = _optional_string(metadata, META_ACCOUNT)
    if account is not None and account != settings.a2a_account:
        raise ValueError("A2A metadata 'account' does not match this server")

    agent_id = _optional_string(metadata, META_AGENT_ID)
    if agent_id is not None and agent_id != settings.a2a_agent_id:
        raise ValueError("A2A metadata 'agent_id' does not match this server")

    headers = _string_mapping(metadata, META_HEADERS)
    allowed_headers = {
        header.lower() for header in settings.a2a_allowed_forwarded_headers
    }
    disallowed_headers = [
        header for header in headers if header.lower() not in allowed_headers
    ]
    if disallowed_headers:
        headers_list = ", ".join(sorted(disallowed_headers))
        raise ValueError(f"A2A metadata contains disallowed header(s): {headers_list}")

    return A2ARouting(
        account=settings.a2a_account,
        agent_id=settings.a2a_agent_id,
        workflow_id=_optional_string(metadata, META_WORKFLOW_ID) or "default",
        session=_optional_string(metadata, META_SESSION) or context_id,
        headers=headers,
        arguments=_string_mapping(metadata, META_ARGUMENTS),
    )


def arag_answer_to_parts(msg: AragAnswer) -> list[a2a_pb2.Part]:
    """Extract the human-facing text parts from an ``AragAnswer`` chunk."""
    parts: list[a2a_pb2.Part] = []
    if msg.answer:
        parts.append(_text_part(msg.answer))
    elif msg.streaming_response_chunk and msg.streaming_response_chunk.text:
        parts.append(_text_part(msg.streaming_response_chunk.text))
    elif msg.generated_text:
        parts.append(_text_part(msg.generated_text))
    return parts


class HyperforgeA2AExecutor(AgentExecutor):
    """Runs a Hyperforge interaction in response to an A2A ``message/send``."""

    def __init__(self, context: A2AServerContext):
        self.app = context
        self._pending_tasks: dict[str, PendingA2ATask] = {}
        self._pending_waiters: dict[str, asyncio.Task[None]] = {}
        self._instance_id = uuid4().hex
        self._task_store = context.task_store

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or uuid4().hex
        context_id = context.context_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)

        if context.current_task is not None:
            await self._forward_feedback(context, updater)
            return

        # A2A requires a Task to be enqueued before any status/artifact events.
        if context.current_task is None:
            await event_queue.enqueue_event(
                new_task(
                    task_id,
                    context_id,
                    a2a_pb2.TaskState.TASK_STATE_SUBMITTED,
                )
            )

        try:
            routing = parse_routing_metadata(
                dict(context.metadata or {}), self.app.settings, context_id
            )
        except ValueError as exc:
            await updater.failed(updater.new_agent_message([_text_part(str(exc))]))
            return

        question = context.get_user_input()
        if not question:
            await updater.failed(
                updater.new_agent_message(
                    [_text_part("Empty message: no text content to process.")]
                )
            )
            return

        try:
            await self.app.agent_manager.ensure_workflow_active(
                routing.account, routing.agent_id, routing.workflow_id
            )
        except exceptions.NotFoundError:
            await updater.failed(
                updater.new_agent_message(
                    [_text_part(f"Unknown workflow_id: {routing.workflow_id}")]
                )
            )
            return

        await updater.start_work()

        interaction = InteractionRequest(
            question=question,
            headers=routing.headers,
            arguments=routing.arguments,
        )
        receiver = WebsocketReceiver(websocket=None)
        response_stream = stream_response(
            self.app,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
            receiver,
            routing.account,
            routing.agent_id,
            routing.session,
            interaction,
            workflow_id=routing.workflow_id,
        )

        try:
            await self._stream_until_pause(
                task_id, updater, receiver, response_stream, routing
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("A2A interaction failed")
            self._pending_tasks.pop(task_id, None)
            await updater.failed(
                updater.new_agent_message([_text_part(f"Internal error: {exc}")])
            )

    async def _forward_feedback(
        self, context: RequestContext, updater: TaskUpdater
    ) -> None:
        task_id = context.task_id
        if task_id is None:  # pragma: no cover - RequestContext always has one here
            await updater.failed(
                updater.new_agent_message([_text_part("Missing task_id")])
            )
            return

        metadata = dict(context.metadata or {})
        feedback_id = metadata.get(META_FEEDBACK_ID)
        if not isinstance(feedback_id, str):
            await updater.failed(
                updater.new_agent_message(
                    [_text_part("Invalid or missing feedback_id")]
                )
            )
            return

        record = await self._task_store.claim_pending(
            task_id, context.context_id or "", feedback_id
        )
        if record is None:
            await updater.failed(
                updater.new_agent_message([_text_part("A2A task has expired")])
            )
            return

        response = context.get_user_input()
        if not response:
            await updater.failed(
                updater.new_agent_message([_text_part("Empty feedback response")])
            )
            return

        await updater.start_work()
        relay_topic = f"hyperforge:a2a:relay:{uuid4().hex}"
        await self.app.broker.send_reply(
            _owner_reply_subject(record.owner_instance_id, task_id),
            json.dumps(
                {
                    "request_id": record.request_id,
                    "response": response,
                    "relay_topic": relay_topic,
                }
            ),
        )

        try:
            async for _cursor, message in self.app.broker.subscribe(relay_topic):
                if isinstance(message, AgentAnswer):
                    answer = message.answer
                    if answer.operation == AnswerOperation.ERROR:
                        detail = answer.exception.detail if answer.exception else "Unknown error"
                        await updater.failed(
                            updater.new_agent_message([_text_part(detail)])
                        )
                        return
                    if answer.operation == AnswerOperation.AGENT_REQUEST and answer.feedback:
                        await updater.requires_input(
                            updater.new_agent_message(
                                [_text_part(answer.feedback.question)],
                                metadata={
                                    "feedback_id": answer.feedback.feedback_id,
                                    "response_schema": answer.feedback.response_schema,
                                    "request_id": answer.feedback.request_id,
                                },
                            )
                        )
                        return
                    parts = arag_answer_to_parts(answer)
                    if parts:
                        await updater.add_artifact(parts, name="answer")
                elif isinstance(message, AgentDone):
                    await updater.complete()
                    return
        except Exception:  # pragma: no cover - broker failure handling
            logger.exception("A2A feedback relay failed")
            await updater.failed(
                updater.new_agent_message(
                    [
                        _text_part(
                            "A2A task was interrupted by a server restart; "
                            "resend the request to continue"
                        )
                    ]
                )
            )

    async def _stream_until_pause(
        self,
        task_id: str,
        updater: TaskUpdater,
        receiver: WebsocketReceiver,
        response_stream: AsyncIterator[AragAnswer],
        routing: A2ARouting,
    ) -> None:
        async for msg in response_stream:
            if msg.operation == AnswerOperation.ERROR:
                detail = msg.exception.detail if msg.exception else "Unknown error"
                self._pending_tasks.pop(task_id, None)
                await updater.failed(updater.new_agent_message([_text_part(detail)]))
                return
            if msg.operation == AnswerOperation.AGENT_REQUEST and msg.feedback:
                self._pending_tasks[task_id] = PendingA2ATask(
                    context_id=updater.context_id,
                    feedback_id=msg.feedback.feedback_id,
                    request_id=msg.feedback.request_id,
                    routing=routing,
                    receiver=receiver,
                    response_stream=response_stream,
                )
                await self._task_store.save_pending(
                    PendingTaskRecord(
                        task_id=task_id,
                        context_id=updater.context_id,
                        routing=asdict(routing),
                        feedback_id=msg.feedback.feedback_id,
                        request_id=msg.feedback.request_id,
                        owner_instance_id=self._instance_id,
                    )
                )
                self._start_feedback_waiter(task_id)
                await asyncio.sleep(0)
                await updater.requires_input(
                    updater.new_agent_message(
                        [_text_part(msg.feedback.question)],
                        metadata={
                            "feedback_id": msg.feedback.feedback_id,
                            "response_schema": msg.feedback.response_schema,
                            "request_id": msg.feedback.request_id,
                        },
                    )
                )
                return
            if msg.operation == AnswerOperation.DONE:
                self._pending_tasks.pop(task_id, None)
                self._cancel_feedback_waiter(task_id)
                await self._task_store.remove(task_id)
                await updater.complete()
                return

            parts = arag_answer_to_parts(msg)
            if parts:
                await updater.add_artifact(parts, name="answer")

    def _start_feedback_waiter(self, task_id: str) -> None:
        self._cancel_feedback_waiter(task_id)
        self._pending_waiters[task_id] = asyncio.create_task(
            self._wait_for_feedback(task_id)
        )

    def _cancel_feedback_waiter(self, task_id: str) -> None:
        waiter = self._pending_waiters.pop(task_id, None)
        if waiter is not None and waiter is not asyncio.current_task():
            waiter.cancel()

    async def _wait_for_feedback(self, task_id: str) -> None:
        try:
            payload = await self.app.broker.receive_reply(
                _owner_reply_subject(self._instance_id, task_id),
                self.app.settings.a2a_task_ttl_seconds * 1000,
            )
            if payload is None:
                return
            pending = self._pending_tasks.get(task_id)
            if pending is None:
                return
            reply = json.loads(payload)
            await pending.receiver.queue.put(
                UserToAgentInteraction(
                    request_id=reply["request_id"], response=reply["response"]
                )
            )
            await self._relay_until_pause(task_id, pending, reply["relay_topic"])
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("A2A feedback owner failed")

    async def _relay_until_pause(
        self, task_id: str, pending: PendingA2ATask, relay_topic: str
    ) -> None:
        async for msg in pending.response_stream:
            if msg.operation == AnswerOperation.ERROR:
                self._pending_tasks.pop(task_id, None)
                await self.app.broker.publish(relay_topic, AgentAnswer(answer=msg))
                await self.app.broker.publish(relay_topic, AgentDone())
                return
            if msg.operation == AnswerOperation.AGENT_REQUEST and msg.feedback:
                self._pending_tasks[task_id] = PendingA2ATask(
                    context_id=pending.context_id,
                    feedback_id=msg.feedback.feedback_id,
                    request_id=msg.feedback.request_id,
                    routing=pending.routing,
                    receiver=pending.receiver,
                    response_stream=pending.response_stream,
                )
                await self._task_store.save_pending(
                    PendingTaskRecord(
                        task_id=task_id,
                        context_id=pending.context_id,
                        routing=asdict(pending.routing),
                        feedback_id=msg.feedback.feedback_id,
                        request_id=msg.feedback.request_id,
                        owner_instance_id=self._instance_id,
                    )
                )
                self._start_feedback_waiter(task_id)
                await asyncio.sleep(0)
                await self.app.broker.publish(relay_topic, AgentAnswer(answer=msg))
                return
            if msg.operation == AnswerOperation.DONE:
                self._pending_tasks.pop(task_id, None)
                await self._task_store.remove(task_id)
                await self.app.broker.publish(relay_topic, AgentDone())
                return
            await self.app.broker.publish(relay_topic, AgentAnswer(answer=msg))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or uuid4().hex
        context_id = context.context_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)
        pending = self._pending_tasks.pop(task_id, None)
        self._cancel_feedback_waiter(task_id)
        if pending is not None:
            await pending.receiver.queue.put(Shutdown())
        await self._task_store.remove(task_id)
        await updater.cancel()
