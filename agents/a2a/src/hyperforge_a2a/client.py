"""Thin helpers around the a2a-sdk gRPC client used by the A2A client agent."""

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


def dict_to_struct(data: dict[str, Any]) -> struct_pb2.Struct:
    struct = struct_pb2.Struct()
    struct.update(data)
    return struct


def build_grpc_client(source: str, use_tls: bool) -> Client:
    """Create an A2A gRPC client targeting ``source`` (host:port)."""

    def channel_factory(url: str) -> grpc.aio.Channel:
        target = url or source
        if use_tls:
            return grpc.aio.secure_channel(target, grpc.ssl_channel_credentials())
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
) -> Client:
    """Create an A2A client from either a gRPC address or an HTTP Agent Card URL."""
    if not source.startswith(("http://", "https://")):
        return build_grpc_client(source, use_tls)

    owns_http_client = http_client is None
    if http_client is None:
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
        texts.extend(extract_text_from_parts(response.artifact_update.artifact.parts))
    elif which == "status_update":
        status = response.status_update.status
        if status.HasField("message"):
            texts.extend(extract_text_from_parts(status.message.parts))
    elif which == "task":
        for artifact in response.task.artifacts:
            texts.extend(extract_text_from_parts(artifact.parts))
        if response.task.status.HasField("message"):
            texts.extend(extract_text_from_parts(response.task.status.message.parts))
    return texts
