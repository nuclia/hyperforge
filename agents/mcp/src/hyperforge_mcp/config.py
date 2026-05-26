from enum import Enum
from typing import Dict, List, Literal

from hyperforge.context.config import ContextAgentConfig
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class Transport(str, Enum):
    STDIO = "STDIO"
    HTTP = "HTTP"


class MCPAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="MCP")
    module: Literal["mcp"] = "mcp"
    transport: Transport = Field(Transport.HTTP, title="Proper transport mechanism")
    source: str = Field(
        ...,
        json_schema_extra={
            "show_in_node": True,
        },
    )
    tool_choice_model: str = Field(
        default="chatgpt-4.1",
        title="Tool choice model",
        description="Model used to choose the tool to use",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    valid_headers: List[str] = Field(
        default_factory=list, title="Valid headers to forward to the agent"
    )
    read_timeout_seconds: int = Field(default=400, title="Read timeout in seconds")
    roots: Dict[str, str] = Field(
        default_factory=dict, title="Available roots format name: url}"
    )
    interaction: bool = Field(default=False, title="Enable interaction with the user")
    feedback_timeout: int = Field(
        default=10000, title="Feedback timeout in milliseconds"
    )
    progress_feedback: bool = Field(default=True, title="Enable progress feedback")
    work_chain: bool = Field(default=True, title="Enable loop on tool")
    max_turns: int = Field(
        default=5, title="Maximum number of tool calls before stopping"
    )
    sampling_model: str = Field(
        default="gemini-2.5-flash",
        title="Sampling model",
        description="Model used for sampling",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    include_mcp_prompts: bool = Field(
        default=False,
        title="If a prompt was selected, include it in the context to generate a partial answer",
    )
    expose_prompts_as_tools: bool = Field(
        default=True,
        title="Expose MCP prompts as callable tools for SmartAgent",
        description=(
            "When registered inside a SmartAgent, also expose each MCP prompt "
            "as a published function so the planner can fetch and inject prompt "
            "content as context."
        ),
    )


class MultiMCPAgentConfig(ContextAgentConfig):
    module: Literal["multi_mcp"] = "multi_mcp"
    configs: List[MCPAgentConfig] = Field(
        default_factory=list, title="List of MCP agent configurations"
    )
    summarize_model: str = "gemini-2.5-flash"
    feedback_timeout: int = Field(
        default=10000, title="Feedback timeout in milliseconds"
    )
    interaction: bool = Field(default=True, title="Enable interaction with the user")
    tool_choice_model: str = Field(
        default="chatgpt-4.1",
        title="Tool choice model",
        description="Model used to choose the tool to use",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    work_chain: bool = Field(default=True, title="Enable loop on tool")
    max_turns: int = Field(
        default=5, title="Maximum number of tool calls before stopping"
    )
