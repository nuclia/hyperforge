from typing import Literal

from hyperforge.agent import AgentConfig
from pydantic import Field
from pydantic.config import ConfigDict


class PassthroughAgentConfig(AgentConfig):
    """Configuration for the passthrough generation agent."""

    model_config = ConfigDict(title="Passthrough")
    module: Literal["passthrough"] = "passthrough"
    rich_context: bool = Field(
        default=False,
        title="Rich context output",
        description=(
            "When enabled, context results are emitted as structured MCP content blocks "
            "(chunks, images, structured data) via their existing callback messages instead "
            "of being concatenated into a plain-text answer. "
            "Use this when the output is complex."
        ),
    )
