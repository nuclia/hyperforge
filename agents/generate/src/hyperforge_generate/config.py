from typing import Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class GenerateAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Generate")
    module: Literal["generate"] = "generate"
    prompt: Optional[str] = Field(
        None,
        json_schema_extra={
            "show_in_node": True,
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
        description="Model used to generate the response",
    )
    images: bool = False
    generate_image: bool = False
