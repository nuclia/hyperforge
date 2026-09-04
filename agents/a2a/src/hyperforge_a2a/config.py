from typing import Dict, List, Literal

from hyperforge.context.config import ContextAgentConfig
from pydantic import Field
from pydantic.config import ConfigDict


class A2AAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="A2A")
    module: Literal["a2a"] = "a2a"

    source: str = Field(
        ...,
        title="A2A driver",
        description="Identifier of the configured A2A driver.",
        json_schema_extra={"show_in_node": True},
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
        description=(
            "Request headers allowed for delegation. 'authorization' is sent as "
            "A2A transport authentication and is never included in message metadata."
        ),
    )
