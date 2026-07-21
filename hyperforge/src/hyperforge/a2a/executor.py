"""A2A ``AgentExecutor`` bridging incoming A2A messages to Hyperforge.

The executor extracts routing information (``account``, ``agent_id``,
``workflow_id``, ``session``) from the A2A message metadata, then drives the
exact same broker-backed interaction pipeline used by the HTTP/WS API
(``stream_response``). ``AragAnswer`` chunks streamed back from the worker are
mapped onto A2A task artifacts / status updates.
"""

from typing import Any, Optional
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
from hyperforge.interaction import AnswerOperation, AragAnswer

# Metadata keys read from the incoming A2A message to route the interaction.
META_ACCOUNT = "account"
META_AGENT_ID = "agent_id"
META_WORKFLOW_ID = "workflow_id"
META_SESSION = "session"
META_HEADERS = "headers"
META_ARGUMENTS = "arguments"


def _text_part(text: str) -> a2a_pb2.Part:
    return a2a_pb2.Part(text=text)


def _headers_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    headers_raw = metadata.get(META_HEADERS) or {}
    headers: dict[str, str] = {}
    if isinstance(headers_raw, dict):
        for key, value in headers_raw.items():
            headers[str(key)] = str(value)
    return headers


def _arguments_from_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    args_raw = metadata.get(META_ARGUMENTS) or {}
    arguments: dict[str, str] = {}
    if isinstance(args_raw, dict):
        for key, value in args_raw.items():
            arguments[str(key)] = str(value)
    return arguments


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

        metadata = context.metadata or {}
        settings = self.app.settings

        account: Optional[str] = (
            metadata.get(META_ACCOUNT) or settings.a2a_default_account
        )
        agent_id: Optional[str] = metadata.get(META_AGENT_ID)
        workflow_id: str = str(metadata.get(META_WORKFLOW_ID) or "default")
        session: str = str(metadata.get(META_SESSION) or context_id)
        question = context.get_user_input()

        if not account or not agent_id:
            await updater.failed(
                updater.new_agent_message(
                    [
                        _text_part(
                            "Missing required A2A metadata: 'account' and "
                            "'agent_id' must be provided."
                        )
                    ]
                )
            )
            return

        if not question:
            await updater.failed(
                updater.new_agent_message(
                    [_text_part("Empty message: no text content to process.")]
                )
            )
            return

        await updater.start_work()

        interaction = InteractionRequest(
            question=question,
            headers=_headers_from_metadata(metadata),
            arguments=_arguments_from_metadata(metadata),
        )

        try:
            async for msg in stream_response(
                self.app,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                WebsocketReceiver(websocket=None),
                account,
                agent_id,
                session,
                interaction,
                workflow_id=workflow_id,
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
