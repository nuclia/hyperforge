from typing import Literal, Optional, Tuple

from hyperforge.context.config import ContextAgentConfig
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class StaticAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Static data")
    module: Literal["static"] = "static"
    published_functions: Optional[Tuple[str, ...]] = Field(
        default=("static_context",),
        title="Published functions",
        description="List of functions published by this agent to be used by other agents in the chain",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    context: Optional[str] = Field(
        None,
        description="Static context to be used by the agent",
        json_schema_extra={
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    structured: Optional[str] = Field(
        None,
        description="Structured data in JSON format to be used by the agent",
        json_schema_extra={
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
