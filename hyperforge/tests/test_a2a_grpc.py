"""End-to-end test for the A2A gRPC serving interface.

Spins up a real gRPC A2A server backed by the production
``HyperforgeA2AExecutor`` (with the broker interaction pipeline mocked) and
drives it with the a2a-sdk gRPC client used by the A2A client agent.
"""

import socket
from concurrent import futures
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
from types import SimpleNamespace
from uuid import uuid4

import grpc
import pytest
from a2a.server.request_handlers import DefaultRequestHandler, GrpcHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import a2a_pb2_grpc
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from hyperforge_a2a.agent import A2AClientAgent
from hyperforge_a2a.client import (
    build_grpc_client,
    build_send_request,
    collect_text_from_stream_response,
    extract_steps_from_stream_response,
)
from hyperforge_a2a.config import A2AAgentConfig
from hyperforge_a2a.config_driver import A2ADriverConfig, A2AInnerConfig
from hyperforge_a2a.driver import A2ADriver
from redis.asyncio import Redis

import hyperforge.a2a.executor as executor_module
from hyperforge.a2a.card import build_agent_card, build_agent_skills
from hyperforge.a2a.executor import HyperforgeA2AExecutor, parse_routing_metadata
from hyperforge.a2a.settings import A2ASettings
from hyperforge.a2a.task_store import RedisA2ATaskStore
from hyperforge.broker.local import LocalBroker
from hyperforge.configure import load_all_configurations, scan
from hyperforge.interaction import AnswerOperation, AragAnswer, Feedback
from hyperforge.manager import Manager
from hyperforge.memory.memory import NoMemorySessionMemory
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import MemoryConfig, Step
from hyperforge.pubsub import UserToAgentInteraction
from hyperforge.server.cache import NoCache
from hyperforge.server.session import SessionManager
from hyperforge.server.settings import Settings as ServerSettings
from hyperforge.standalone.agent import StaticAgentManager
from hyperforge.standalone.config import StandAloneAgentConfig, WorkflowConfig

NUA_KEY = cassette_nua_key("https://europe-1.dp.stashify.cloud/")


async def _a2a_client(endpoint: str, **config):
    source = "remote-a2a"
    client_agent = await A2AClientAgent.from_config(
        A2AAgentConfig(source=source, **config)
    )
    driver = A2ADriver(
        name="Remote A2A",
        provider="a2a",
        config=A2AInnerConfig(endpoint=endpoint),
        allow_private_network_endpoints=True,
    )
    return client_agent, SimpleNamespace(drivers={source: driver})


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


class _InMemoryA2ATaskStore:
    def __init__(self):
        self.owners = {}
        self.pending = {}

    async def save_owner(self, task_id, owner_instance_id):
        self.owners[task_id] = owner_instance_id

    async def get_owner(self, task_id):
        return self.owners.get(task_id)

    async def remove_owner(self, task_id):
        self.owners.pop(task_id, None)

    async def save_pending(self, record):
        self.pending[record.task_id] = record

    async def get_pending(self, task_id):
        return self.pending.get(task_id)

    async def claim_pending(self, task_id, context_id, feedback_id):
        record = self.pending.get(task_id)
        if record is None:
            return None
        if record.context_id != context_id or record.feedback_id != feedback_id:
            return None
        return self.pending.pop(task_id)

    async def remove(self, task_id):
        self.pending.pop(task_id, None)


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


async def _serve(executor, port: int, credentials=None, sdk_task_store=None):
    settings = A2ASettings(a2a_grpc_host="127.0.0.1", a2a_grpc_port=port)
    handler = DefaultRequestHandler(
        executor,
        sdk_task_store or InMemoryTaskStore(),
        build_agent_card(settings),
    )
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=4))
    a2a_pb2_grpc.add_A2AServiceServicer_to_server(GrpcHandler(handler), server)
    bind_address = f"127.0.0.1:{port}"
    if credentials:
        server.add_secure_port(bind_address, credentials)
    else:
        server.add_insecure_port(bind_address)
    await server.start()
    return server


def _write_test_certificate(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc) - timedelta(minutes=1))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ip_address("127.0.0.1"))]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )
    certificate_path = tmp_path / "server-cert.pem"
    key_path = tmp_path / "server-key.pem"
    certificate_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return certificate_path, key_path


def test_tls_settings_require_key_pair_and_public_endpoint():
    with pytest.raises(ValueError, match="CERTIFICATE_CHAIN_PATH"):
        A2ASettings(a2a_tls_enabled=True)

    with pytest.raises(ValueError, match="requires A2A_TLS_ENABLED"):
        A2ASettings(a2a_tls_client_ca_path="client-ca.pem")

    with pytest.raises(ValueError, match="wildcard address"):
        build_agent_card(A2ASettings())


def test_secure_agent_card_advertises_configured_public_endpoint(tmp_path):
    certificate_path, key_path = _write_test_certificate(tmp_path)
    settings = A2ASettings(
        a2a_tls_enabled=True,
        a2a_tls_certificate_chain_path=certificate_path,
        a2a_tls_private_key_path=key_path,
        a2a_public_url="a2a.example.com:443",
    )

    card = build_agent_card(settings)

    assert card.supported_interfaces[0].url == "a2a.example.com:443"


def test_authenticated_agent_card_advertises_bearer_requirement():
    card = build_agent_card(
        A2ASettings(
            a2a_grpc_host="127.0.0.1",
            a2a_auth_enabled=True,
            a2a_authorizer_url="http://authorizer",
        )
    )

    scheme = card.security_schemes["bearer"].http_auth_security_scheme
    assert scheme.scheme == "bearer"
    assert "bearer" in card.security_requirements[0].schemes


async def test_a2a_grpc_tls_serves_agent_card(tmp_path):
    certificate_path, key_path = _write_test_certificate(tmp_path)
    credentials = grpc.ssl_server_credentials(
        [(key_path.read_bytes(), certificate_path.read_bytes())]
    )
    port = _free_port()
    settings = A2ASettings(a2a_grpc_port=port)
    server = await _serve(
        HyperforgeA2AExecutor(_FakeContext(settings, _FakeAgentManager())),
        port,
        credentials,
    )
    channel = grpc.aio.secure_channel(
        f"127.0.0.1:{port}",
        grpc.ssl_channel_credentials(root_certificates=certificate_path.read_bytes()),
    )

    try:
        stub = a2a_pb2_grpc.A2AServiceStub(channel)
        response = await stub.SendMessage(build_send_request("TLS handshake test"))
    finally:
        await channel.close()
        await server.stop(grace=1)

    assert response.WhichOneof("payload") == "task"


async def test_a2a_grpc_mtls_requires_client_certificate(tmp_path):
    server_certificate_path, server_key_path = _write_test_certificate(
        tmp_path / "server"
    )
    client_certificate_path, client_key_path = _write_test_certificate(
        tmp_path / "client"
    )
    credentials = grpc.ssl_server_credentials(
        [(server_key_path.read_bytes(), server_certificate_path.read_bytes())],
        root_certificates=client_certificate_path.read_bytes(),
        require_client_auth=True,
    )
    port = _free_port()
    settings = A2ASettings(a2a_grpc_port=port)
    server = await _serve(
        HyperforgeA2AExecutor(_FakeContext(settings, _FakeAgentManager())),
        port,
        credentials,
    )
    unauthenticated_channel = grpc.aio.secure_channel(
        f"127.0.0.1:{port}",
        grpc.ssl_channel_credentials(
            root_certificates=server_certificate_path.read_bytes()
        ),
    )
    authenticated_channel = grpc.aio.secure_channel(
        f"127.0.0.1:{port}",
        grpc.ssl_channel_credentials(
            root_certificates=server_certificate_path.read_bytes(),
            private_key=client_key_path.read_bytes(),
            certificate_chain=client_certificate_path.read_bytes(),
        ),
    )

    try:
        with pytest.raises(grpc.aio.AioRpcError):
            await a2a_pb2_grpc.A2AServiceStub(unauthenticated_channel).SendMessage(
                build_send_request("mTLS rejected client")
            )

        response = await a2a_pb2_grpc.A2AServiceStub(authenticated_channel).SendMessage(
            build_send_request("mTLS accepted client")
        )
    finally:
        await unauthenticated_channel.close()
        await authenticated_channel.close()
        await server.stop(grace=1)

    assert response.WhichOneof("payload") == "task"


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
            step=Step(
                original_question_uuid=None,
                actual_question_uuid=None,
                module="smart",
                title="Calling Logistics",
                value="Check the stage transfer schedule.",
                agent_path="/context/coordinator",
                reason="Logistics owns production timing.",
                timeit=0,
                input_nuclia_tokens=None,
                output_nuclia_tokens=None,
            )
        )
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
                "workflow_id": "wf1",
                "headers": {"authorization": "test-header-value"},
            },
        )
        texts: list[str] = []
        step_events = []
        async for response in client.send_message(request):
            step_events.extend(extract_steps_from_stream_response(response))
            texts.extend(collect_text_from_stream_response(response))
        await client.close()
    finally:
        await server.stop(grace=1)

    assert any("Answer to: what is A2A?" in t for t in texts)
    assert len(step_events) == 1
    assert step_events[0].module == "smart"
    assert step_events[0].title == "Calling Logistics"
    assert step_events[0].reason == "Logistics owns production timing."
    assert all("Calling Logistics" not in text for text in texts)
    assert captured["account"] == "acc"
    assert captured["agent_id"] == "myagent"
    assert captured["workflow_id"] == "wf1"
    assert captured["question"] == "what is A2A?"
    assert captured["headers"].get("authorization") == "test-header-value"


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
        client_agent, manager = await _a2a_client(
            f"127.0.0.1:{port}",
            id="local-a2a-client",
            remote_workflow_id="deterministic-workflow",
            valid_headers=["authorization"],
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
            headers={"authorization": "test-header-value"},
        )

        context = await client_agent.a2a_query(
            "Run the deterministic workflow",
            memory,
            manager=manager,  # type: ignore[arg-type]
        )
    finally:
        await server.stop(grace=1)

    assert context.summary == "deterministic \nA2A response"
    assert [chunk.text for chunk in context.chunks] == [context.summary]
    assert captured == {
        "account": "local",
        "agent_id": "deterministic-agent",
        "workflow_id": "deterministic-workflow",
        "headers": {},
    }


@pytest.mark.vcr(ignore_localhost=True)
@pytest.mark.asyncio
async def test_a2a_client_server_workflow_end_to_end(monkeypatch):
    """Run configured client, A2A server, broker, and NUA workflow end to end."""
    for module in ("hyperforge_static", "hyperforge_summarize"):
        scan(module)
        load_all_configurations(module)

    broker = LocalBroker(keepalive_ms=1_000)
    remote_agent_id = "remote-agent"
    workflow_id = "default"
    agent_manager = StaticAgentManager(
        {
            remote_agent_id: StandAloneAgentConfig(
                workflows={
                    workflow_id: WorkflowConfig(
                        name="A2A E2E",
                        context=[
                            {
                                "module": "static",
                                "title": "Release status",
                                "context": "The A2A release is ready for production.",
                            }
                        ],
                        generation=[{"module": "summarize"}],
                    )
                }
            )
        }
    )
    worker = SessionManager(
        settings=ServerSettings(
            health_check_enabled=False,
            external_nua_api_key=NUA_KEY,
            standalone=True,
            allow_private_network_endpoints=True,
        ),
        broker=broker,
        agent_manager=agent_manager,
        cache=NoCache(),
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
                task_store=_InMemoryA2ATaskStore(),
            )
        ),
        port,
    )

    try:
        client_agent = await A2AClientAgent.from_config(
            A2AAgentConfig(
                source="remote-a2a",
                id="local-a2a-client",
                remote_workflow_id=workflow_id,
            )
        )
        monkeypatch.setattr("hyperforge.manager.get_driver_klass", lambda _: A2ADriver)
        manager = await Manager.from_config(
            drivers=[
                A2ADriverConfig(
                    identifier="remote-a2a",
                    name="Remote A2A",
                    provider="a2a",
                    config=A2AInnerConfig(endpoint=f"127.0.0.1:{port}"),
                )
            ],
            nua=None,  # type: ignore[arg-type]
            allow_private_network_endpoints=True,
        )
        session = NoMemorySessionMemory(
            MemoryConfig(),
            "client-agent",
            "default",
            cache=None,  # type: ignore[arg-type]
        )
        session.init("local-a2a-session")
        memory = session.start_question("Is the A2A release ready?")

        context = await client_agent.a2a_query(
            "Is the A2A release ready?",
            memory,
            manager=manager,
        )
    finally:
        await server.stop(grace=1)
        await worker.finalize()

    assert "ready" in context.summary.lower()
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

    owner_port = _free_port()
    receiver_port = _free_port()
    settings = A2ASettings(
        a2a_grpc_port=owner_port,
        a2a_account="local",
        a2a_agent_id="feedback-agent",
    )
    broker = LocalBroker()
    sdk_task_store = InMemoryTaskStore()
    owner_server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(
                settings,
                _FakeAgentManager(),
                broker=broker,
                task_store=a2a_task_store,
            )
        ),
        owner_port,
        sdk_task_store=sdk_task_store,
    )
    receiver_server = await _serve(
        HyperforgeA2AExecutor(
            _FakeContext(
                settings,
                _FakeAgentManager(),
                broker=broker,
                task_store=a2a_task_store,
            )
        ),
        receiver_port,
        sdk_task_store=sdk_task_store,
    )

    try:
        owner_client = build_grpc_client(f"127.0.0.1:{owner_port}", use_tls=False)
        initial = build_send_request("Find sales data")
        task_id = ""
        context_id = ""
        feedback_id = ""
        states = []
        async for event in owner_client.send_message(initial):
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
        receiver_client = build_grpc_client(f"127.0.0.1:{receiver_port}", use_tls=False)
        async for event in receiver_client.send_message(reply):
            texts.extend(collect_text_from_stream_response(event))
            which = event.WhichOneof("payload")
            if which == "status_update":
                states.append(event.status_update.status.state)
            elif which == "task":
                states.append(event.task.status.state)
        await owner_client.close()
        await receiver_client.close()
    finally:
        await owner_server.stop(grace=1)
        await receiver_server.stop(grace=1)

    from a2a.types import a2a_pb2

    assert feedback_id == "feedback-1"
    assert a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED in states
    assert a2a_pb2.TaskState.TASK_STATE_COMPLETED in states
    assert captured == {"request_id": "request-1", "response": "EMEA"}
    assert any("Using EMEA" in text for text in texts)


async def test_a2a_client_agent_answers_nested_remote_feedback(
    monkeypatch, a2a_task_store
):
    captured = []

    async def feedback_workflow(
        app, receiver, account, agent_id, session, interaction, workflow_id="default"
    ):
        first_feedback = Feedback(
            request_id="request-1",
            feedback_id="feedback-1",
            question="Which region should I use?",
            module="test",
            agent_id=agent_id,
            data={},
            response_schema={"type": "string"},
        )
        yield AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST, feedback=first_feedback
        )
        first_reply = await receiver.receive_feedback()
        captured.append((first_reply.request_id, first_reply.response))
        second_feedback = Feedback(
            request_id="request-2",
            feedback_id="feedback-2",
            question="Which country should I use?",
            module="test",
            agent_id=agent_id,
            data={},
            response_schema={"type": "string"},
        )
        yield AragAnswer(
            operation=AnswerOperation.AGENT_REQUEST, feedback=second_feedback
        )
        second_reply = await receiver.receive_feedback()
        captured.append((second_reply.request_id, second_reply.response))
        yield AragAnswer(
            operation=AnswerOperation.ANSWER,
            answer=f"Using {second_reply.response} in {first_reply.response}",
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
            _FakeContext(
                settings,
                _FakeAgentManager(),
                broker=LocalBroker(),
                task_store=a2a_task_store,
            )
        ),
        port,
    )

    try:
        client_agent, manager = await _a2a_client(f"127.0.0.1:{port}", id="a2a-client")
        session = NoMemorySessionMemory(
            MemoryConfig(), "client-agent", "default", cache=NoCache()
        )
        session.init("a2a-feedback-session")
        memory = session.start_question("Find sales data")
        requested_feedback = []
        responses = iter(["EMEA", "Germany"])

        async def answer_feedback(feedback):
            requested_feedback.append((feedback.feedback_id, feedback.question))
            return UserToAgentInteraction(
                request_id=feedback.request_id, response=next(responses)
            )

        memory.set_feedback_fn(answer_feedback)
        context = await client_agent.a2a_query(
            "Find sales data",
            memory,
            manager=manager,  # type: ignore[arg-type]
        )
    finally:
        await server.stop(grace=1)

    assert requested_feedback == [
        ("feedback-1", "Which region should I use?"),
        ("feedback-2", "Which country should I use?"),
    ]
    assert captured == [("request-1", "EMEA"), ("request-2", "Germany")]
    assert context.summary == "Using Germany in EMEA"
    assert [chunk.text for chunk in context.chunks] == ["Using Germany in EMEA"]


def test_parse_routing_metadata_defaults_and_allowed_headers():
    routing = parse_routing_metadata(
        {
            "headers": {"Authorization": "test-header-value"},
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
    assert routing.headers == {"authorization": "test-header-value"}
    assert routing.arguments == {"limit": "3", "include_archived": "False"}


def test_parse_routing_metadata_rejects_duplicate_headers_ignoring_case():
    with pytest.raises(ValueError, match="duplicate header: authorization"):
        parse_routing_metadata(
            {
                "headers": {
                    "Authorization": "test-header-value-one",
                    "authorization": "test-header-value-two",
                }
            },
            A2ASettings(
                a2a_account="account",
                a2a_agent_id="research-agent",
                a2a_allowed_forwarded_headers=["authorization"],
            ),
            "a2a-context",
        )


async def test_a2a_grpc_rejects_identity_metadata(monkeypatch, a2a_task_store):
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
    assert any(
        "Unknown A2A metadata field(s): account, agent_id" in text for text in texts
    )


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
    card = build_agent_card(A2ASettings(a2a_grpc_host="127.0.0.1"))
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
    card = build_agent_card(A2ASettings(a2a_grpc_host="127.0.0.1"), skills)

    assert [(skill.id, skill.name, skill.description) for skill in card.skills] == [
        ("research-agent:answer", "Answer", "Answer a question"),
        ("research-agent:summarize", "Summarize", "Summarize context"),
    ]
