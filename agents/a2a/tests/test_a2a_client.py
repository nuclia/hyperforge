from unittest.mock import AsyncMock, Mock, call

import pytest
from a2a.client.auth import AuthInterceptor
from a2a.client.interceptors import BeforeArgs
from a2a.types import a2a_pb2
from hyperforge.utils.http import PrivateUrlError

from hyperforge_a2a.client import (
    BearerCredentialService,
    RemoteFeedbackRequest,
    ResponseTextAccumulator,
    build_a2a_client,
    build_feedback_request,
    build_send_request,
    collect_text_from_stream_response,
    dict_to_struct,
    extract_feedback_request,
    extract_steps_from_stream_response,
    raise_for_terminal_task_error,
)
from hyperforge_a2a.config import A2AAgentConfig
from hyperforge_a2a.config_driver import A2AInnerConfig


@pytest.mark.asyncio
async def test_direct_grpc_endpoint_is_validated(monkeypatch):
    validate = AsyncMock(side_effect=PrivateUrlError("private"))
    monkeypatch.setattr("hyperforge_a2a.client.ensure_public_endpoint", validate)

    with pytest.raises(PrivateUrlError, match="private"):
        await build_a2a_client("localhost:50051", use_tls=False)

    validate.assert_awaited_once_with("localhost:50051")


@pytest.mark.asyncio
async def test_private_endpoint_can_be_allowed_by_runtime(monkeypatch):
    validate = AsyncMock()
    build_grpc_client = Mock(return_value=Mock())
    monkeypatch.setattr("hyperforge_a2a.client.ensure_public_endpoint", validate)
    monkeypatch.setattr("hyperforge_a2a.client.build_grpc_client", build_grpc_client)

    await build_a2a_client(
        "localhost:50051",
        use_tls=False,
        allow_private_network_endpoints=True,
    )

    validate.assert_not_awaited()


@pytest.mark.asyncio
async def test_rejects_private_agent_card_interface(monkeypatch):
    card = Mock(
        supported_interfaces=[
            Mock(url="http://localhost:8000", protocol_binding="JSONRPC")
        ]
    )
    validate = AsyncMock(
        side_effect=[None, PrivateUrlError("private advertised interface")]
    )
    monkeypatch.setattr("hyperforge_a2a.client.ensure_public_endpoint", validate)
    monkeypatch.setattr(
        "hyperforge_a2a.client.A2ACardResolver.get_agent_card",
        AsyncMock(return_value=card),
    )

    with pytest.raises(PrivateUrlError, match="private advertised interface"):
        await build_a2a_client(
            "https://public.example.com",
            use_tls=False,
            http_client=AsyncMock(),
        )

    validate.assert_has_awaits(
        [
            call("https://public.example.com"),
            call("http://localhost:8000"),
        ]
    )


class _Memory:
    def __init__(self, headers):
        self.headers = headers


def _make_agent(**overrides):
    from hyperforge_a2a.agent import A2AClientAgent

    config = A2AAgentConfig(source="localhost:8034", **overrides)
    a = A2AClientAgent.__new__(A2AClientAgent)
    a.config = config
    a.agent_id = "a2a-1"
    return a


def test_build_send_request_sets_text_and_metadata():
    request = build_send_request("hello?", {"account": "acc", "agent_id": "ag"})
    assert request.message.parts[0].text == "hello?"
    assert request.metadata["account"] == "acc"
    assert request.metadata["agent_id"] == "ag"


@pytest.mark.asyncio
async def test_bearer_credentials_are_added_as_transport_metadata():
    service = BearerCredentialService("Bearer secret-token")
    assert await service.get_credentials("bearer", None) == "secret-token"
    assert await service.get_credentials("other", None) is None

    card = a2a_pb2.AgentCard()
    card.security_schemes["bearer"].http_auth_security_scheme.scheme = "bearer"
    card.security_requirements.add().schemes["bearer"].SetInParent()
    args = BeforeArgs(input=None, method="SendMessage", agent_card=card)
    await AuthInterceptor(service).before(args)

    assert args.context is not None
    assert args.context.service_parameters == {"Authorization": "Bearer secret-token"}


def test_dict_to_struct_nested():
    struct = dict_to_struct({"headers": {"authorization": "Bearer x"}})
    assert struct["headers"]["authorization"] == "Bearer x"


def test_collect_text_from_artifact_update():
    from a2a.types import a2a_pb2

    response = a2a_pb2.StreamResponse(
        artifact_update=a2a_pb2.TaskArtifactUpdateEvent(
            task_id="t",
            context_id="c",
            artifact=a2a_pb2.Artifact(
                artifact_id="a",
                name="answer",
                parts=[a2a_pb2.Part(text="hello world")],
            ),
        )
    )
    assert collect_text_from_stream_response(response) == ["hello world"]


def test_extract_step_artifact_without_collecting_it_as_answer_text():
    from a2a.types import a2a_pb2

    response = a2a_pb2.StreamResponse(
        artifact_update=a2a_pb2.TaskArtifactUpdateEvent(
            task_id="t",
            context_id="c",
            artifact=a2a_pb2.Artifact(
                artifact_id="step-1",
                name="step",
                parts=[a2a_pb2.Part(text="Calling Venue")],
                metadata=dict_to_struct(
                    {
                        "module": "smart",
                        "title": "Calling Venue",
                        "reason": "Venue owns access.",
                        "value": "Check Gate C.",
                        "agent_path": "/context/coordinator",
                    }
                ),
            ),
        )
    )

    assert collect_text_from_stream_response(response) == []
    step = extract_steps_from_stream_response(response)[0]
    assert step.title == "Calling Venue"
    assert step.reason == "Venue owns access."


def test_response_text_accumulator_honors_append_and_replacement():
    from a2a.types import a2a_pb2

    accumulator = ResponseTextAccumulator()
    for text, append in (("hel", False), ("lo", True), ("replacement", False)):
        accumulator.add(
            a2a_pb2.StreamResponse(
                artifact_update=a2a_pb2.TaskArtifactUpdateEvent(
                    task_id="t",
                    context_id="c",
                    append=append,
                    artifact=a2a_pb2.Artifact(
                        artifact_id="answer-1",
                        name="answer",
                        parts=[a2a_pb2.Part(text=text)],
                    ),
                )
            )
        )

    assert accumulator.texts() == ["replacement"]


def test_response_text_accumulator_reconciles_snapshots_and_ignores_status():
    from a2a.types import a2a_pb2

    accumulator = ResponseTextAccumulator()
    artifact = a2a_pb2.Artifact(
        artifact_id="answer-1",
        name="answer",
        parts=[a2a_pb2.Part(text="answer")],
    )
    accumulator.add(
        a2a_pb2.StreamResponse(
            artifact_update=a2a_pb2.TaskArtifactUpdateEvent(artifact=artifact)
        )
    )
    accumulator.add(
        a2a_pb2.StreamResponse(
            artifact_update=a2a_pb2.TaskArtifactUpdateEvent(
                artifact=a2a_pb2.Artifact(
                    artifact_id="stale", parts=[a2a_pb2.Part(text="stale")]
                )
            )
        )
    )
    accumulator.add(
        a2a_pb2.StreamResponse(
            task=a2a_pb2.Task(
                id="t",
                context_id="c",
                artifacts=[artifact],
                status=a2a_pb2.TaskStatus(
                    state=a2a_pb2.TaskState.TASK_STATE_COMPLETED,
                    message=a2a_pb2.Message(parts=[a2a_pb2.Part(text="Complete")]),
                ),
            )
        )
    )

    assert accumulator.texts() == ["answer"]


def test_build_metadata_routes_authorization_outside_message_payload():
    agent = _make_agent(
        remote_workflow_id="wf",
        valid_headers=["authorization", "x-trace-id"],
        extra_metadata={"custom": "1"},
    )
    memory = _Memory(
        {
            "authorization": "Bearer token",
            "x-trace-id": "trace-1",
            "x-other": "nope",
        }
    )
    metadata = agent._build_metadata(memory)
    assert "account" not in metadata
    assert "agent_id" not in metadata
    assert metadata["workflow_id"] == "wf"
    assert metadata["headers"] == {"x-trace-id": "trace-1"}
    assert agent._authorization(memory) == "Bearer token"
    assert metadata["custom"] == "1"


def test_tls_client_credentials_require_tls_and_a_complete_key_pair():
    with pytest.raises(ValueError, match="configured together"):
        A2AInnerConfig(
            endpoint="a2a.example.com:443",
            use_tls=True,
            client_certificate_chain="certificate",
        )

    with pytest.raises(ValueError, match="require use_tls"):
        A2AInnerConfig(
            endpoint="a2a.example.com:443",
            ca_certificate="CA",
        )

    config = A2AInnerConfig(
        endpoint="https://a2a.example.com",
        ca_certificate="CA",
    )
    assert config.use_tls is False

    with pytest.raises(ValueError, match="requires an HTTPS"):
        A2AInnerConfig(endpoint="http://a2a.example.com", use_tls=True)


@pytest.mark.asyncio
async def test_http_client_uses_configured_tls_context(monkeypatch):
    ssl_context = Mock()
    create_default_context = Mock(return_value=ssl_context)
    monkeypatch.setattr(
        "hyperforge_a2a.client.ssl.create_default_context",
        create_default_context,
    )
    http_client = AsyncMock()
    http_client.aclose = AsyncMock()
    async_client = Mock(return_value=http_client)
    monkeypatch.setattr("hyperforge_a2a.client.httpx.AsyncClient", async_client)
    monkeypatch.setattr(
        "hyperforge_a2a.client.A2ACardResolver.get_agent_card",
        AsyncMock(side_effect=RuntimeError("stop after card lookup")),
    )

    with pytest.raises(RuntimeError, match="stop after card lookup"):
        await build_a2a_client(
            "https://a2a.example.com",
            use_tls=True,
            ca_certificate="CA",
            allow_private_network_endpoints=True,
        )

    create_default_context.assert_called_once_with(cadata="CA")
    async_client.assert_called_once_with(verify=ssl_context)
    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_https_agent_card_can_select_grpc_without_tls(monkeypatch):
    card = Mock(
        supported_interfaces=[
            Mock(protocol_binding="GRPC"),
        ]
    )
    http_client = AsyncMock()
    monkeypatch.setattr(
        "hyperforge_a2a.client.httpx.AsyncClient", Mock(return_value=http_client)
    )
    grpc_client = Mock()
    create_client = AsyncMock(return_value=grpc_client)
    secure_channel = Mock(return_value=Mock())
    insecure_channel = Mock(return_value=Mock())
    credentials = Mock()
    monkeypatch.setattr(
        "hyperforge_a2a.client.A2ACardResolver.get_agent_card",
        AsyncMock(return_value=card),
    )
    monkeypatch.setattr("hyperforge_a2a.client.create_client", create_client)
    monkeypatch.setattr(
        "hyperforge_a2a.client.grpc.ssl_channel_credentials",
        Mock(return_value=credentials),
    )
    monkeypatch.setattr("hyperforge_a2a.client.grpc.aio.secure_channel", secure_channel)
    monkeypatch.setattr(
        "hyperforge_a2a.client.grpc.aio.insecure_channel", insecure_channel
    )

    result = await build_a2a_client(
        "https://a2a.example.com",
        use_tls=False,
        allow_private_network_endpoints=True,
    )

    config = create_client.await_args.kwargs["client_config"]
    assert result is grpc_client
    assert config.supported_protocol_bindings == ["JSONRPC", "HTTP+JSON", "GRPC"]
    config.grpc_channel_factory("a2a.example.com:443")
    secure_channel.assert_not_called()
    insecure_channel.assert_called_once_with("a2a.example.com:443")
    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovered_grpc_client_closes_owned_http_client(monkeypatch):
    card = Mock(
        supported_interfaces=[
            Mock(protocol_binding="GRPC"),
        ]
    )
    http_client = AsyncMock()
    http_client.aclose = AsyncMock()
    monkeypatch.setattr(
        "hyperforge_a2a.client.httpx.AsyncClient", Mock(return_value=http_client)
    )
    monkeypatch.setattr(
        "hyperforge_a2a.client.A2ACardResolver.get_agent_card",
        AsyncMock(return_value=card),
    )
    monkeypatch.setattr(
        "hyperforge_a2a.client.create_client",
        AsyncMock(return_value=Mock()),
    )

    await build_a2a_client(
        "http://a2a.example.com",
        use_tls=False,
        allow_private_network_endpoints=True,
    )

    http_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovered_grpc_client_does_not_close_injected_http_client(monkeypatch):
    card = Mock(supported_interfaces=[Mock(protocol_binding="GRPC")])
    http_client = AsyncMock()
    http_client.aclose = AsyncMock()
    monkeypatch.setattr(
        "hyperforge_a2a.client.A2ACardResolver.get_agent_card",
        AsyncMock(return_value=card),
    )
    monkeypatch.setattr(
        "hyperforge_a2a.client.create_client",
        AsyncMock(return_value=Mock()),
    )

    await build_a2a_client(
        "http://a2a.example.com",
        use_tls=False,
        http_client=http_client,
        allow_private_network_endpoints=True,
    )

    http_client.aclose.assert_not_awaited()


def test_extract_feedback_request_and_build_continuation():
    from a2a.types import a2a_pb2

    response = a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id="task-1",
            context_id="context-1",
            status=a2a_pb2.TaskStatus(
                state=a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED,
                message=a2a_pb2.Message(
                    parts=[a2a_pb2.Part(text="Which region should I use?")],
                    metadata=dict_to_struct(
                        {
                            "feedback_id": "feedback-1",
                            "request_id": "request-1",
                            "agent_id": "venue",
                            "response_schema": {"type": "string"},
                        }
                    ),
                ),
            ),
        )
    )

    feedback = extract_feedback_request(response)

    assert feedback == RemoteFeedbackRequest(
        task_id="task-1",
        context_id="context-1",
        feedback_id="feedback-1",
        request_id="request-1",
        question="Which region should I use?",
        response_schema={"type": "string"},
        agent_id="venue",
    )
    continuation = build_feedback_request("EMEA", feedback)
    assert continuation.message.task_id == "task-1"
    assert continuation.message.context_id == "context-1"
    assert continuation.message.parts[0].text == "EMEA"
    assert continuation.metadata["feedback_id"] == "feedback-1"


def test_relayed_feedback_uses_local_request_id():
    from hyperforge_a2a.agent import build_local_feedback

    remote = RemoteFeedbackRequest(
        task_id="task-1",
        context_id="context-1",
        feedback_id="feedback-1",
        request_id="remote-request",
        question="Approve the change?",
        response_schema={"type": "string"},
    )

    feedback = build_local_feedback(
        remote,
        local_request_id="local-session",
        module="a2a",
        agent_id="remote-agent",
        timeout_ms=600_000,
    )

    assert feedback.request_id == "local-session"
    assert feedback.feedback_id == "feedback-1"
    assert feedback.timeout_ms == 600_000
    assert feedback.data == {
        "a2a_task_id": "task-1",
        "a2a_context_id": "context-1",
        "a2a_request_id": "remote-request",
    }


@pytest.mark.parametrize(
    "state",
    [
        a2a_pb2.TaskState.TASK_STATE_FAILED,
        a2a_pb2.TaskState.TASK_STATE_CANCELED,
        a2a_pb2.TaskState.TASK_STATE_REJECTED,
        a2a_pb2.TaskState.TASK_STATE_AUTH_REQUIRED,
    ],
)
def test_remote_task_error_raises_its_message(state):
    response = a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id="task-1",
            context_id="context-1",
            status=a2a_pb2.TaskStatus(
                state=state,
                message=a2a_pb2.Message(parts=[a2a_pb2.Part(text="Remote failure")]),
            ),
        )
    )

    with pytest.raises(ValueError, match="Remote failure"):
        raise_for_terminal_task_error(response)
