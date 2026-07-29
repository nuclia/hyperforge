"""End-to-end test for the A2A gRPC serving interface.

Spins up a real gRPC A2A server backed by the production
``HyperforgeA2AExecutor`` (with the broker interaction pipeline mocked) and
drives it with the a2a-sdk gRPC client used by the A2A client agent.
"""

import socket
from concurrent import futures
from uuid import uuid4

import grpc
import pytest
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2_grpc
from hyperforge_a2a.agent import A2AClientAgent
from hyperforge_a2a.client import (
    build_grpc_client,
    build_send_request,
    collect_text_from_stream_response,
)
from hyperforge_a2a.config import A2AAgentConfig
from redis.asyncio import Redis

import hyperforge.a2a.executor as executor_module
import hyperforge.server.session as session_module
from hyperforge.a2a.card import build_agent_card, build_agent_skills
from hyperforge.a2a.executor import HyperforgeA2AExecutor, parse_routing_metadata
from hyperforge.a2a.settings import A2ASettings
from hyperforge.a2a.task_store import RedisA2ATaskStore
from hyperforge.broker.local import LocalBroker
from hyperforge.engine import State
from hyperforge.interaction import AnswerOperation, AragAnswer, Feedback
from hyperforge.memory.memory import NoMemorySessionMemory
from hyperforge.models import MemoryConfig
from hyperforge.pubsub import UserToAgentInteraction
from hyperforge.server.cache import NoCache
from hyperforge.server.session import SessionManager
from hyperforge.server.settings import Settings as ServerSettings
from hyperforge.standalone.agent import StaticAgentManager
from hyperforge.standalone.config import StandAloneAgentConfig, WorkflowConfig


class _FakeContext:
    def __init__(
        self, settings: A2ASettings, agent_manager=None, broker=None, task_store=None
    ):
        self.settings = settings
        self.agent_manager = agent_manager
        self.broker = broker
        self.task_store = task_store


class _FakeAgentManager:
    def __init__(self, workflows: set[str] | None = None):
        self.workflows = workflows or {"default"}

    async def ensure_workflow_active(self, account, agent_id, workflow_id):
        if workflow_id not in self.workflows:
            from hyperforge.db import exceptions

            raise exceptions.NotFoundError("Workflow not found")


class _DeterministicWorkflow:
    async def __call__(self, memory, manager):
        await memory.add_answer(
            "The deterministic workflow completed.",
            module="deterministic",
            agent_path="/generation/deterministic",
        )
        await memory.add_final_answer()


@pytest.fixture
async def a2a_task_store(valkey):
    redis = Redis(host=valkey[0], port=valkey[1], decode_responses=True)
    store = RedisA2ATaskStore(redis, f"test:a2a:grpc:{uuid4().hex}", 30)
    yield store
    await redis.aclose()  # type: ignore[attr-defined]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _serve(executor, port: int):
    settings = A2ASettings(a2a_grpc_port=port)
    handler = DefaultRequestHandler(
        executor, InMemoryTaskStore(), build_agent_card(settings)
    )
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=4))
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(GrpcHandler(handler), server)
    server.add_insecure_port(f"127.0.0.1:{port}")
    await server.start()
    return server


async def test_a2a_grpc_round_trip(monkeypatch, a2a_task_store):
    captured = {}

    async def fake_stream_response(
        app, websocket, account, agent_id, session, interaction, workflow_id="default"
    ):
        captured["account"] = account
        captured["agent_id"] = agent_id
        captured["workflow_id"] = workflow_id
        captured["session"] = session
        captured["question"] = interaction.question
        captured["headers"] = dict(interaction.headers)
        yield AragAnswer(
            operation=AnswerOperation.ANSWER,
            answer=f"Answer to: {interaction.question}",
        )
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", fake_stream_response)

    port = _free_port()
    settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="acc",
        a2a_agent_id="myagent",
        a2a_allowed_forwarded_headers=["authorization"],
    )
    executor = HyperforgeA2AExecutor(
        _FakeContext(settings, _FakeAgentManager({"wf1"}), task_store=a2a_task_store)
    )
    server = await _serve(executor, port)

    try:
        client = build_grpc_client(f"127.0.0.1:{port}", use_tls=False)
        request = build_send_request(
            "what is A2A?",
            {
                "account": "acc",
                "agent_id": "myagent",
                "workflow_id": "wf1",
                "headers": {"authorization": "Bearer token"},
            },
        )
        texts: list[str] = []
        async for response in client.send_message(request):
            texts.extend(collect_text_from_stream_response(response))
        await client.close()
    finally:
        await server.stop(grace=1)

    assert any("Answer to: what is A2A?" in t for t in texts)
    assert captured["account"] == "acc"
    assert captured["agent_id"] == "myagent"
    assert captured["workflow_id"] == "wf1"
    assert captured["question"] == "what is A2A?"
    assert captured["headers"].get("authorization") == "Bearer token"


async def test_a2a_client_agent_builds_context_from_streamed_workflow(
    monkeypatch, a2a_task_store
):
    captured = {}

    async def deterministic_workflow(
        app, websocket, account, agent_id, session, interaction, workflow_id="default"
    ):
        captured["account"] = account
        captured["agent_id"] = agent_id
        captured["workflow_id"] = workflow_id
        captured["headers"] = dict(interaction.headers)
        yield AragAnswer(operation=AnswerOperation.ANSWER, answer="deterministic ")
        yield AragAnswer(operation=AnswerOperation.ANSWER, answer="A2A response")
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", deterministic_workflow)

    port = _free_port()
    server_settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="local",
        a2a_agent_id="deterministic-agent",
        a2a_allowed_forwarded_headers=["authorization"],
    )
    server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(
                server_settings,
                _FakeAgentManager({"deterministic-workflow"}),
                task_store=a2a_task_store,
            )
        ),
        port,
    )

    try:
        client_agent = await A2AClientAgent.from_config(
            A2AAgentConfig(
                id="local-a2a-client",
                source=f"127.0.0.1:{port}",
                remote_account="local",
                remote_agent_id="deterministic-agent",
                remote_workflow_id="deterministic-workflow",
                valid_headers=["authorization"],
            )
        )
        session = NoMemorySessionMemory(
            MemoryConfig(),
            "client-agent",
            "default",
            cache=None,  # type: ignore[arg-type]
        )
        session.init("local-a2a-session")
        memory = session.start_question(
            "Run the deterministic workflow",
            headers={"authorization": "Bearer local-demo"},
        )

        context = await client_agent.a2a_query(
            "Run the deterministic workflow",
            memory,
            manager=None,  # type: ignore[arg-type]
        )
    finally:
        await server.stop(grace=1)

    assert context.summary == "deterministic \nA2A response"
    assert [chunk.text for chunk in context.chunks] == [context.summary]
    assert captured == {
        "account": "local",
        "agent_id": "deterministic-agent",
        "workflow_id": "deterministic-workflow",
        "headers": {"authorization": "Bearer local-demo"},
    }


async def test_a2a_client_server_workflow_end_to_end(monkeypatch, a2a_task_store):
    """Run client, A2A server, broker, and deterministic workflow in one process."""
    broker = LocalBroker(keepalive_ms=1_000)
    remote_agent_id = "deterministic-agent"
    workflow_id = "deterministic-workflow"
    agent_manager = StaticAgentManager(
        {
            remote_agent_id: StandAloneAgentConfig(
                workflows={workflow_id: WorkflowConfig(name="Deterministic")}
            )
        }
    )
    worker = SessionManager(
        settings=ServerSettings(health_check_enabled=False),
        broker=broker,
        agent_manager=agent_manager,
        cache=NoCache(),
    )

    async def deterministic_state(**_):
        return State(manager=None, agent=_DeterministicWorkflow())

    monkeypatch.setattr(
        session_module,
        "get_state",
        deterministic_state,
    )
    await worker.initialize(health_check=False)

    port = _free_port()
    a2a_settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="local",
        a2a_agent_id=remote_agent_id,
    )
    server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(
                a2a_settings,
                agent_manager=agent_manager,
                broker=broker,
                task_store=a2a_task_store,
            )
        ),
        port,
    )

    try:
        client_agent = await A2AClientAgent.from_config(
            A2AAgentConfig(
                id="local-a2a-client",
                source=f"127.0.0.1:{port}",
                remote_account="local",
                remote_agent_id=remote_agent_id,
                remote_workflow_id=workflow_id,
            )
        )
        session = NoMemorySessionMemory(
            MemoryConfig(),
            "client-agent",
            "default",
            cache=None,  # type: ignore[arg-type]
        )
        session.init("local-a2a-session")
        memory = session.start_question("Run the local workflow")

        context = await client_agent.a2a_query(
            "Run the local workflow",
            memory,
            manager=None,  # type: ignore[arg-type]
        )
    finally:
        await server.stop(grace=1)
        await worker.finalize()

    assert context.summary == "The deterministic workflow completed."
    assert [chunk.text for chunk in context.chunks] == [context.summary]


async def test_a2a_grpc_feedback_reply_continues_task(monkeypatch, a2a_task_store):
    captured = {}

    async def feedback_workflow(
        app, receiver, account, agent_id, session, interaction, workflow_id="default"
    ):
        feedback = Feedback(
            request_id="request-1",
            feedback_id="feedback-1",
            question="Which region should I use?",
            module="test",
            agent_id=agent_id,
            data={},
            response_schema={"type": "string"},
        )
        yield AragAnswer(operation=AnswerOperation.AGENT_REQUEST, feedback=feedback)
        reply = await receiver.receive_feedback()
        captured["request_id"] = reply.request_id
        captured["response"] = reply.response
        yield AragAnswer(
            operation=AnswerOperation.ANSWER, answer=f"Using {reply.response}"
        )
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", feedback_workflow)

    port = _free_port()
    settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="local",
        a2a_agent_id="feedback-agent",
    )
    server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(settings, _FakeAgentManager(), task_store=a2a_task_store)
        ),
        port,
    )

    try:
        client = build_grpc_client(f"127.0.0.1:{port}", use_tls=False)
        initial = build_send_request("Find sales data")
        task_id = ""
        context_id = ""
        feedback_id = ""
        states = []
        async for event in client.send_message(initial):
            which = event.WhichOneof("payload")
            if which == "status_update":
                task_id = event.status_update.task_id
                context_id = event.status_update.context_id
                states.append(event.status_update.status.state)
                if event.status_update.status.HasField("message"):
                    feedback_id = event.status_update.status.message.metadata.fields[
                        "feedback_id"
                    ].string_value
            elif which == "task":
                task_id = event.task.id
                context_id = event.task.context_id
                states.append(event.task.status.state)
                if event.task.status.HasField("message"):
                    feedback_id = event.task.status.message.metadata.fields[
                        "feedback_id"
                    ].string_value

        reply = build_send_request("EMEA", {"feedback_id": feedback_id})
        reply.message.task_id = task_id
        reply.message.context_id = context_id
        texts: list[str] = []
        async for event in client.send_message(reply):
            texts.extend(collect_text_from_stream_response(event))
            which = event.WhichOneof("payload")
            if which == "status_update":
                states.append(event.status_update.status.state)
            elif which == "task":
                states.append(event.task.status.state)
        await client.close()
    finally:
        await server.stop(grace=1)

    from a2a.types import a2a_pb2

    assert feedback_id == "feedback-1"
    assert a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED in states
    assert a2a_pb2.TaskState.TASK_STATE_COMPLETED in states
    assert captured == {"request_id": "request-1", "response": "EMEA"}
    assert any("Using EMEA" in text for text in texts)


async def test_a2a_client_agent_answers_remote_feedback(monkeypatch, a2a_task_store):
    captured = {}

    async def feedback_workflow(
        app, receiver, account, agent_id, session, interaction, workflow_id="default"
    ):
        feedback = Feedback(
            request_id="request-1",
            feedback_id="feedback-1",
            question="Which region should I use?",
            module="test",
            agent_id=agent_id,
            data={},
            response_schema={"type": "string"},
        )
        yield AragAnswer(operation=AnswerOperation.AGENT_REQUEST, feedback=feedback)
        reply = await receiver.receive_feedback()
        captured["request_id"] = reply.request_id
        captured["response"] = reply.response
        yield AragAnswer(
            operation=AnswerOperation.ANSWER, answer=f"Using {reply.response}"
        )
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", feedback_workflow)

    port = _free_port()
    settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="local",
        a2a_agent_id="feedback-agent",
    )
    server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(settings, _FakeAgentManager(), task_store=a2a_task_store)
        ),
        port,
    )

    try:
        client_agent = await A2AClientAgent.from_config(
            A2AAgentConfig(id="a2a-client", source=f"127.0.0.1:{port}")
        )
        session = NoMemorySessionMemory(
            MemoryConfig(), "client-agent", "default", cache=NoCache()
        )
        session.init("a2a-feedback-session")
        memory = session.start_question("Find sales data")
        requested_feedback = {}

        async def answer_feedback(feedback):
            requested_feedback["question"] = feedback.question
            requested_feedback["schema"] = feedback.response_schema
            requested_feedback["feedback_id"] = feedback.feedback_id
            return UserToAgentInteraction(
                request_id=feedback.request_id, response="EMEA"
            )

        memory.set_feedback_fn(answer_feedback)
        context = await client_agent.a2a_query(
            "Find sales data",
            memory,
            manager=None,  # type: ignore[arg-type]
        )
    finally:
        await server.stop(grace=1)

    assert requested_feedback == {
        "question": "Which region should I use?",
        "schema": {"type": "string"},
        "feedback_id": "feedback-1",
    }
    assert captured == {"request_id": "request-1", "response": "EMEA"}
    assert context.summary == "Using EMEA"
    assert [chunk.text for chunk in context.chunks] == ["Using EMEA"]


def test_parse_routing_metadata_defaults_and_allowed_headers():
    routing = parse_routing_metadata(
        {
            "headers": {"Authorization": "Bearer token"},
            "arguments": {"limit": 3, "include_archived": False},
        },
        A2ASettings(
            a2a_account="account",
            a2a_agent_id="research-agent",
            a2a_allowed_forwarded_headers=["authorization"],
        ),
        "a2a-context",
    )

    assert routing.account == "account"
    assert routing.agent_id == "research-agent"
    assert routing.workflow_id == "default"
    assert routing.session == "a2a-context"
    assert routing.headers == {"Authorization": "Bearer token"}
    assert routing.arguments == {"limit": "3", "include_archived": "False"}


async def test_a2a_grpc_rejects_identity_override(monkeypatch, a2a_task_store):
    async def fake_stream_response(*args, **kwargs):  # pragma: no cover - not called
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", fake_stream_response)

    port = _free_port()
    settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="account",
        a2a_agent_id="research-agent",
    )
    executor = HyperforgeA2AExecutor(
        _FakeContext(settings, _FakeAgentManager(), task_store=a2a_task_store)
    )
    server = await _serve(executor, port)

    try:
        client = build_grpc_client(f"127.0.0.1:{port}", use_tls=False)
        request = build_send_request(
            "hi", {"account": "other-account", "agent_id": "research-agent"}
        )
        texts: list[str] = []
        states = []
        async for response in client.send_message(request):
            texts.extend(collect_text_from_stream_response(response))
            which = response.WhichOneof("payload")
            if which == "status_update":
                states.append(response.status_update.status.state)
            elif which == "task":
                states.append(response.task.status.state)
        await client.close()
    finally:
        await server.stop(grace=1)

    from a2a.types import a2a_pb2

    assert a2a_pb2.TaskState.TASK_STATE_FAILED in states
    assert any("does not match this server" in text for text in texts)


async def test_a2a_grpc_rejects_unknown_workflow(monkeypatch, a2a_task_store):
    async def fake_stream_response(*args, **kwargs):  # pragma: no cover - not called
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", fake_stream_response)

    port = _free_port()
    settings = A2ASettings(
        a2a_grpc_port=port,
        a2a_account="account",
        a2a_agent_id="research-agent",
    )
    server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(
                settings,
                _FakeAgentManager({"known-workflow"}),
                task_store=a2a_task_store,
            )
        ),
        port,
    )

    try:
        client = build_grpc_client(f"127.0.0.1:{port}", use_tls=False)
        request = build_send_request("hi", {"workflow_id": "unknown-workflow"})
        texts: list[str] = []
        async for response in client.send_message(request):
            texts.extend(collect_text_from_stream_response(response))
        await client.close()
    finally:
        await server.stop(grace=1)

    assert any("Unknown workflow_id: unknown-workflow" in text for text in texts)


def test_build_agent_card_defaults():
    card = build_agent_card(A2ASettings())
    assert card.name == "Hyperforge"
    assert card.capabilities.streaming is True
    assert card.skills
    assert card.supported_interfaces[0].protocol_binding == "GRPC"


async def test_agent_card_advertises_configured_workflows():
    agent_manager = StaticAgentManager(
        {
            "research-agent": StandAloneAgentConfig(
                workflows={
                    "answer": WorkflowConfig(
                        name="Answer", description="Answer a question"
                    ),
                    "summarize": WorkflowConfig(
                        name="Summarize", description="Summarize context"
                    ),
                }
            )
        }
    )

    skills = await build_agent_skills(agent_manager, "account", "research-agent")
    card = build_agent_card(A2ASettings(), skills)

    assert [(skill.id, skill.name, skill.description) for skill in card.skills] == [
        ("research-agent:answer", "Answer", "Answer a question"),
        ("research-agent:summarize", "Summarize", "Summarize context"),
    ]
