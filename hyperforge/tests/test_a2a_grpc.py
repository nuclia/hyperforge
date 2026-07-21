"""End-to-end test for the A2A gRPC serving interface.

Spins up a real gRPC A2A server backed by the production
``HyperforgeA2AExecutor`` (with the broker interaction pipeline mocked) and
drives it with the a2a-sdk gRPC client used by the A2A client agent.
"""

from concurrent import futures

import grpc
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2_grpc
from hyperforge_a2a.client import (
    build_grpc_client,
    build_send_request,
    collect_text_from_stream_response,
)

import hyperforge.a2a.executor as executor_module
from hyperforge.a2a.card import build_agent_card
from hyperforge.a2a.executor import HyperforgeA2AExecutor
from hyperforge.a2a.settings import A2ASettings
from hyperforge.interaction import AnswerOperation, AragAnswer


class _FakeContext:
    def __init__(self, settings: A2ASettings):
        self.settings = settings
        self.agent_manager = None
        self.broker = None


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


async def test_a2a_grpc_round_trip(monkeypatch):
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

    settings = A2ASettings(a2a_grpc_port=8041)
    executor = HyperforgeA2AExecutor(_FakeContext(settings))
    server = await _serve(executor, 8041)

    try:
        client = build_grpc_client("127.0.0.1:8041", use_tls=False)
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


async def test_a2a_grpc_missing_routing_metadata(monkeypatch):
    async def fake_stream_response(*args, **kwargs):  # pragma: no cover - not called
        yield AragAnswer(operation=AnswerOperation.DONE)

    monkeypatch.setattr(executor_module, "stream_response", fake_stream_response)

    settings = A2ASettings(a2a_grpc_port=8042)
    executor = HyperforgeA2AExecutor(_FakeContext(settings))
    server = await _serve(executor, 8042)

    try:
        client = build_grpc_client("127.0.0.1:8042", use_tls=False)
        # No agent_id / account provided -> task should fail gracefully.
        request = build_send_request("hi", {"workflow_id": "default"})
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
    assert any("Missing required A2A metadata" in t for t in texts)


def test_build_agent_card_defaults():
    card = build_agent_card(A2ASettings())
    assert card.name == "Hyperforge"
    assert card.capabilities.streaming is True
    assert card.skills
    assert card.supported_interfaces[0].protocol_binding == "GRPC"
