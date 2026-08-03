"""Tests for consuming an external HTTP/JSON-RPC A2A server."""

import httpx
import hyperforge_a2a.agent as agent_module
from a2a.helpers import (
    get_message_text,
    new_task_from_user_message,
    new_text_message,
    new_text_part,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    TaskState,
)
from hyperforge_a2a.agent import A2AClientAgent
from hyperforge_a2a.client import build_a2a_client
from hyperforge_a2a.config import A2AAgentConfig
from sse_starlette.sse import AppStatus
from starlette.applications import Starlette

from hyperforge.memory.memory import NoMemorySessionMemory
from hyperforge.models import MemoryConfig
from hyperforge.server.cache import NoCache


class _JsonRpcEchoExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if context.current_task is None:
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)
        await updater.update_status(
            state=TaskState.TASK_STATE_WORKING,
            message=new_text_message("Working"),
        )
        await updater.add_artifact(
            [new_text_part(f"External answer: {get_message_text(context.message)}")]
        )
        await updater.update_status(
            state=TaskState.TASK_STATE_COMPLETED,
            message=new_text_message("Complete"),
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        await TaskUpdater(event_queue, task.id, task.context_id).cancel()


def _jsonrpc_a2a_app() -> Starlette:
    card = AgentCard(
        name="Test HTTP A2A Agent",
        description="In-process JSON-RPC A2A test server.",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=True),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        supported_interfaces=[
            AgentInterface(protocol_binding="JSONRPC", url="http://a2a.test")
        ],
        skills=[
            AgentSkill(
                id="echo",
                name="Echo",
                description="Returns the submitted question.",
                input_modes=["text/plain"],
                output_modes=["text/plain"],
            )
        ],
    )
    handler = DefaultRequestHandler(_JsonRpcEchoExecutor(), InMemoryTaskStore(), card)
    return Starlette(
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, "/"),
        ]
    )


async def test_http_agent_card_discovery_streams_into_context(monkeypatch):
    # sse-starlette caches this event process-wide, but pytest gives async tests
    # separate event loops. Do not reuse a closed loop's event.
    AppStatus.should_exit_event = None

    async def build_test_client(*_args, **_kwargs):
        http_client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_jsonrpc_a2a_app()),
            base_url="http://a2a.test",
        )
        return await build_a2a_client(
            "http://a2a.test", use_tls=False, http_client=http_client
        )

    monkeypatch.setattr(agent_module, "build_a2a_client", build_test_client)

    client_agent = await A2AClientAgent.from_config(
        A2AAgentConfig(id="http-a2a", source="http://a2a.test")
    )
    session = NoMemorySessionMemory(
        MemoryConfig(), "client-agent", "default", cache=NoCache()
    )
    session.init("http-a2a-session")
    memory = session.start_question("Is HTTP A2A supported?")

    try:
        context = await client_agent.a2a_query(
            "Is HTTP A2A supported?",
            memory,
            manager=None,  # type: ignore[arg-type]
        )
    finally:
        AppStatus.should_exit_event = None

    assert context.summary == "External answer: Is HTTP A2A supported?"
    assert [chunk.text for chunk in context.chunks] == [context.summary]
