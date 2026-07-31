"""Thin helpers around the a2a-sdk gRPC client used by the A2A client agent."""

import ssl
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import grpc
import httpx
from a2a.client import (
    A2ACardResolver,
    Client,
    ClientConfig,
    ClientFactory,
    create_client,
    minimal_agent_card,
)
from a2a.types import a2a_pb2
from a2a.utils import TransportProtocol
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict


@dataclass(frozen=True)
class RemoteFeedbackRequest:
    """Correlation data required to resume a remote A2A task."""

    task_id: str
    context_id: str
    feedback_id: str
    request_id: str
    question: str
    response_schema: Any
    agent_id: str = ""


@dataclass(frozen=True)
class RemoteAgentStep:
    module: str
    title: str
    reason: str
    value: str
    agent_path: str


def dict_to_struct(data: dict[str, Any]) -> struct_pb2.Struct:
    struct = struct_pb2.Struct()
    struct.update(data)
    return struct


def _read_pem(path: Path | None) -> bytes | None:
    return path.read_bytes() if path else None


def build_grpc_client(
    source: str,
    use_tls: bool,
    tls_ca_certificate_path: Path | None = None,
    tls_client_certificate_chain_path: Path | None = None,
    tls_client_private_key_path: Path | None = None,
) -> Client:
    """Create an A2A gRPC client targeting ``source`` (host:port)."""

    credentials = (
        grpc.ssl_channel_credentials(
            root_certificates=_read_pem(tls_ca_certificate_path),
            private_key=_read_pem(tls_client_private_key_path),
            certificate_chain=_read_pem(tls_client_certificate_chain_path),
        )
        if use_tls
        else None
    )

    def channel_factory(url: str) -> grpc.aio.Channel:
        target = url or source
        if use_tls:
            assert credentials is not None
            return grpc.aio.secure_channel(target, credentials)
        return grpc.aio.insecure_channel(target)

    config = ClientConfig(
        streaming=True,
        grpc_channel_factory=channel_factory,
        supported_protocol_bindings=[TransportProtocol.GRPC],
        accepted_output_modes=["text/plain"],
    )
    card = minimal_agent_card(source, [TransportProtocol.GRPC])
    return ClientFactory(config).create(card)


async def build_a2a_client(
    source: str,
    use_tls: bool,
    http_client: httpx.AsyncClient | None = None,
    tls_ca_certificate_path: Path | None = None,
    tls_client_certificate_chain_path: Path | None = None,
    tls_client_private_key_path: Path | None = None,
) -> Client:
    """Create an A2A client from either a gRPC address or an HTTP Agent Card URL."""
    if not source.startswith(("http://", "https://")):
        return build_grpc_client(
            source,
            use_tls,
            tls_ca_certificate_path,
            tls_client_certificate_chain_path,
            tls_client_private_key_path,
        )

    if source.startswith("http://") and use_tls:
        raise ValueError("use_tls requires an HTTPS Agent Card URL")

    owns_http_client = http_client is None
    if http_client is None:
        if use_tls:
            ssl_context = ssl.create_default_context(
                cafile=(
                    str(tls_ca_certificate_path) if tls_ca_certificate_path else None
                )
            )
            if tls_client_certificate_chain_path and tls_client_private_key_path:
                ssl_context.load_cert_chain(
                    certfile=tls_client_certificate_chain_path,
                    keyfile=tls_client_private_key_path,
                )
            http_client = httpx.AsyncClient(verify=ssl_context)
        else:
            http_client = httpx.AsyncClient()
    try:
        card = await A2ACardResolver(
            httpx_client=http_client,
            base_url=source,
        ).get_agent_card()
        return await create_client(
            agent=card,
            client_config=ClientConfig(
                streaming=True,
                httpx_client=http_client,
                accepted_output_modes=["text/plain"],
            ),
        )
    except Exception:
        if owns_http_client:
            await http_client.aclose()
        raise


def build_message(question: str) -> a2a_pb2.Message:
    return a2a_pb2.Message(
        message_id=uuid4().hex,
        role=a2a_pb2.Role.ROLE_USER,
        parts=[a2a_pb2.Part(text=question)],
    )


def build_send_request(
    question: str, metadata: Optional[dict[str, Any]] = None
) -> a2a_pb2.SendMessageRequest:
    request = a2a_pb2.SendMessageRequest(message=build_message(question))
    if metadata:
        request.metadata.CopyFrom(dict_to_struct(metadata))
    return request


def build_feedback_request(
    response: str, feedback: RemoteFeedbackRequest
) -> a2a_pb2.SendMessageRequest:
    """Build a reply to an ``input-required`` task without changing its routing."""
    request = build_send_request(response, {"feedback_id": feedback.feedback_id})
    request.message.task_id = feedback.task_id
    request.message.context_id = feedback.context_id
    return request


def _feedback_from_status(
    task_id: str, context_id: str, status: a2a_pb2.TaskStatus
) -> RemoteFeedbackRequest | None:
    if status.state != a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED:
        return None
    if not task_id or not context_id or not status.HasField("message"):
        raise ValueError("Malformed A2A input-required event")

    message = status.message
    feedback_id = message.metadata.fields.get("feedback_id")
    request_id = message.metadata.fields.get("request_id")
    response_schema = message.metadata.fields.get("response_schema")
    agent_id = message.metadata.fields.get("agent_id")
    question = "\n".join(extract_text_from_parts(message.parts)).strip()
    if (
        feedback_id is None
        or not feedback_id.string_value
        or request_id is None
        or not request_id.string_value
        or response_schema is None
        or not question
    ):
        raise ValueError("Malformed A2A input-required feedback metadata")

    return RemoteFeedbackRequest(
        task_id=task_id,
        context_id=context_id,
        feedback_id=feedback_id.string_value,
        request_id=request_id.string_value,
        question=question,
        response_schema=MessageToDict(response_schema),
        agent_id=agent_id.string_value if agent_id is not None else "",
    )


def extract_feedback_request(
    response: a2a_pb2.StreamResponse,
) -> RemoteFeedbackRequest | None:
    """Extract an A2A ``input-required`` request from a streamed response."""
    which = response.WhichOneof("payload")
    if which == "status_update":
        update = response.status_update
        return _feedback_from_status(update.task_id, update.context_id, update.status)
    if which == "task":
        task = response.task
        return _feedback_from_status(task.id, task.context_id, task.status)
    return None


def raise_for_terminal_task_error(response: a2a_pb2.StreamResponse) -> None:
    """Raise a local error when the remote task has failed or been cancelled."""
    which = response.WhichOneof("payload")
    if which == "status_update":
        status = response.status_update.status
    elif which == "task":
        status = response.task.status
    else:
        return

    terminal_states = {
        a2a_pb2.TaskState.TASK_STATE_FAILED,
        a2a_pb2.TaskState.TASK_STATE_CANCELED,
    }
    if status.state not in terminal_states:
        return

    detail = "Remote A2A task failed"
    if status.HasField("message"):
        text = "\n".join(extract_text_from_parts(status.message.parts)).strip()
        if text:
            detail = text
    raise ValueError(detail)


def extract_text_from_parts(parts: Any) -> list[str]:
    texts: list[str] = []
    for part in parts:
        if part.text:
            texts.append(part.text)
    return texts


def collect_text_from_stream_response(response: a2a_pb2.StreamResponse) -> list[str]:
    """Pull human-facing text out of a single A2A stream response event."""
    texts: list[str] = []
    which = response.WhichOneof("payload")
    if which == "message":
        texts.extend(extract_text_from_parts(response.message.parts))
    elif which == "artifact_update":
        artifact = response.artifact_update.artifact
        if artifact.name != "step":
            texts.extend(extract_text_from_parts(artifact.parts))
    elif which == "status_update":
        status = response.status_update.status
        if status.HasField("message"):
            texts.extend(extract_text_from_parts(status.message.parts))
    elif which == "task":
        for artifact in response.task.artifacts:
            if artifact.name != "step":
                texts.extend(extract_text_from_parts(artifact.parts))
        if response.task.status.HasField("message"):
            texts.extend(extract_text_from_parts(response.task.status.message.parts))
    return texts


def extract_steps_from_stream_response(
    response: a2a_pb2.StreamResponse,
) -> list[RemoteAgentStep]:
    which = response.WhichOneof("payload")
    if which == "artifact_update":
        artifacts = iter([response.artifact_update.artifact])
    elif which == "task":
        artifacts = iter(response.task.artifacts)
    else:
        artifacts = iter([])

    steps = []
    for artifact in artifacts:
        if artifact.name != "step":
            continue
        metadata = MessageToDict(artifact.metadata)
        steps.append(
            RemoteAgentStep(
                module=str(metadata.get("module", "")),
                title=str(metadata.get("title", "")),
                reason=str(metadata.get("reason", "")),
                value=str(metadata.get("value", "")),
                agent_path=str(metadata.get("agent_path", "")),
            )
        )
    return steps
