from typing import Dict, List, Literal, Optional

from hyperforge.context.config import ContextAgentConfig
from pydantic import Field
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
