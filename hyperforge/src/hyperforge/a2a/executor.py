"""A2A ``AgentExecutor`` bridging incoming A2A messages to Hyperforge."""

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from a2a.helpers import new_task
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import a2a_pb2

from hyperforge.a2a import logger
from hyperforge.a2a.context import A2AServerContext
from hyperforge.api.models import InteractionRequest
from hyperforge.api.v1.interaction import WebsocketReceiver, stream_response
from hyperforge.db import exceptions
from hyperforge.interaction import AnswerOperation, AragAnswer

# Metadata keys read from the incoming A2A message to route the interaction.
META_ACCOUNT = "account"
META_AGENT_ID = "agent_id"
META_WORKFLOW_ID = "workflow_id"
META_SESSION = "session"
META_HEADERS = "headers"
META_ARGUMENTS = "arguments"
_ALLOWED_METADATA = {
    META_ACCOUNT,
    META_AGENT_ID,
    META_WORKFLOW_ID,
    META_SESSION,
    META_HEADERS,
    META_ARGUMENTS,
}


@dataclass(frozen=True)
class A2ARouting:
    account: str
    agent_id: str
    workflow_id: str
    session: str
    headers: dict[str, str]
    arguments: dict[str, str]


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

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or uuid4().hex
        context_id = context.context_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)

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

        try:
            async for msg in stream_response(
                self.app,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                WebsocketReceiver(websocket=None),
                routing.account,
                routing.agent_id,
                routing.session,
                interaction,
                workflow_id=routing.workflow_id,
            ):
                if msg.operation == AnswerOperation.ERROR:
                    detail = msg.exception.detail if msg.exception else "Unknown error"
                    await updater.failed(
                        updater.new_agent_message([_text_part(detail)])
                    )
                    return
                elif msg.operation == AnswerOperation.AGENT_REQUEST and msg.feedback:
                    # A2A input-required: surface the agent's question and yield
                    # control back to the client. Resumption is expected via a
                    # follow-up message referencing the same task.
                    await updater.requires_input(
                        updater.new_agent_message(
                            [_text_part(msg.feedback.question)],
                            metadata={
                                "response_schema": msg.feedback.response_schema,
                                "request_id": msg.feedback.request_id,
                            },
                        )
                    )
                    return
                elif msg.operation == AnswerOperation.DONE:
                    break
                else:
                    parts = arag_answer_to_parts(msg)
                    if parts:
                        await updater.add_artifact(parts, name="answer")

            await updater.complete()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("A2A interaction failed")
            await updater.failed(
                updater.new_agent_message([_text_part(f"Internal error: {exc}")])
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task_id = context.task_id or uuid4().hex
        context_id = context.context_id or uuid4().hex
        updater = TaskUpdater(event_queue, task_id, context_id)
        await updater.cancel()
