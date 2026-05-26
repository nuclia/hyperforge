from typing import Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class SummarizeAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Summarize")
    module: Literal["summarize"] = "summarize"
    system_prompt: Optional[str] = Field(
        default=None,
        title="System prompt",
        description="System prompt to guide the model's behavior and response style",
        json_schema_extra={
            "show_in_node": True,
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    prompt: Optional[str] = Field(
        default=None,
        json_schema_extra={
            "show_in_node": True,
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    model: str = Field(
        default="chatgpt-azure-4o-mini",
        title="Generative model",
        description="Model used to generate the response",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    images: bool = False
    conversational: bool = False
    include_mcp_prompts: bool = Field(
        default=False,
        title="If MCP prompts were used during the context steps, include them in the prompt to generate the final answer",
    )
    citations: bool = Field(
        default=False,
        title="Whether to include markdown citations in the generated answer.",
    )
    force_chunk_level_citations: bool = Field(
        default=False,
        title="Whether to always use chunk-level citations instead of context-level citations.",
    )
    history: bool = Field(
        default=False,
        title="Session history",
        description="Include previous Q&A history from the current session in the context provided to the summarize agent",
    )
