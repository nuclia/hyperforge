from pathlib import Path
from typing import Dict, List, Literal, Optional

from hyperforge.context.config import ContextAgentConfig
from pydantic import Field, model_validator
from pydantic.config import ConfigDict


class A2AAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="A2A")
    module: Literal["a2a"] = "a2a"

    source: str = Field(
        ...,
        title="A2A server address",
        description=(
            "gRPC address (e.g. 'localhost:8034') or HTTP(S) URL that serves "
            "an A2A Agent Card (e.g. 'http://localhost:9999')."
        ),
        json_schema_extra={"show_in_node": True},
    )
    use_tls: bool = Field(
        default=False,
        title="Use TLS",
        description="Establish a secure gRPC channel to the A2A server."
        " HTTPS URLs use TLS automatically.",
    )
    tls_ca_certificate_path: Optional[Path] = Field(
        default=None,
        title="TLS CA certificate path",
        description="Optional PEM CA bundle used to verify the remote A2A server.",
    )
    tls_client_certificate_chain_path: Optional[Path] = Field(
        default=None,
        title="TLS client certificate chain path",
        description="Optional PEM client certificate chain for mTLS.",
    )
    tls_client_private_key_path: Optional[Path] = Field(
        default=None,
        title="TLS client private key path",
        description="Optional PEM client private key for mTLS.",
    )
    read_timeout_seconds: int = Field(default=120, title="Read timeout in seconds")

    # Routing metadata forwarded to the remote A2A agent (interpreted by the
    # remote server, e.g. another Hyperforge instance).
    remote_account: Optional[str] = Field(
        default=None,
        title="Remote account",
        description="'account' routing metadata sent to the remote A2A agent.",
    )
    remote_agent_id: Optional[str] = Field(
        default=None,
        title="Remote agent id",
        description="'agent_id' routing metadata sent to the remote A2A agent.",
    )
    remote_workflow_id: str = Field(
        default="default",
        title="Remote workflow id",
        description="'workflow_id' routing metadata sent to the remote A2A agent.",
    )
    extra_metadata: Dict[str, str] = Field(
        default_factory=dict,
        title="Extra metadata",
        description="Additional metadata forwarded to the remote A2A message.",
    )
    valid_headers: List[str] = Field(
        default_factory=list,
        title="Valid headers to forward to the remote A2A agent",
    )

    @model_validator(mode="after")
    def validate_tls_settings(self) -> "A2AAgentConfig":
        client_certificate_configured = (
            self.tls_client_certificate_chain_path is not None
        )
        client_key_configured = self.tls_client_private_key_path is not None
        if client_certificate_configured != client_key_configured:
            raise ValueError(
                "TLS client certificate chain and private key must be configured together"
            )
        if (
            self.tls_ca_certificate_path
            or client_certificate_configured
            or client_key_configured
        ) and not self.use_tls:
            raise ValueError("TLS client credentials require use_tls to be true")
        if self.source.startswith("http://") and self.use_tls:
            raise ValueError("use_tls requires an HTTPS Agent Card URL")
        return self
