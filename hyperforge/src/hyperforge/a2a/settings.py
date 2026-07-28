from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class A2ASettings(BaseSettings):
    """Settings for the A2A gRPC serving interface.

    Reuses the same broker / NucliaDB / module-loading configuration as the
    rest of Hyperforge so the A2A server drives the exact same interaction
    pipeline as the HTTP/WS API.
    """

    debug: bool = False
    log_level: str = "WARNING"

    # gRPC bind address for the A2A A2AService servicer.
    a2a_grpc_host: str = "0.0.0.0"
    a2a_grpc_port: int = 8034
    a2a_grpc_max_workers: int = 10

    # Agent card metadata advertised over A2A.
    a2a_agent_name: str = "Hyperforge"
    a2a_agent_description: str = (
        "Hyperforge agentic RAG exposed over the Agent2Agent (A2A) protocol."
    )
    a2a_agent_version: str = "1.0.0"
    # Public URL clients should use to reach this gRPC endpoint (advertised in
    # the agent card interfaces). Falls back to host:port when unset.
    a2a_public_url: Optional[str] = None

    # The single Hyperforge agent represented by this A2A server. Both values
    # are required when starting the server and are advertised through its
    # workflow-derived Agent Card.
    a2a_account: Optional[str] = None
    a2a_agent_id: Optional[str] = None
    a2a_allowed_forwarded_headers: list[str] = Field(default_factory=list)

    # Broker / pubsub (must match the worker + api settings).
    valkey_url: str = "redis://arag-valkey-cluster"
    valkey_cluster_mode: bool = False
    activate_subject: str = "arag.activate"
    answers_subject: str = (
        "arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.answer"
    )
    oauth_subject: str = "arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.oauth.{oauth_uuid}"
    pubsub_keepalive_seconds: float = 20

    sentry_url: Optional[str] = None
    running_environment: str = "stage"
    zone: str = "stashify"

    load_modules: list[str] = []

    metrics_port: int = 8091
