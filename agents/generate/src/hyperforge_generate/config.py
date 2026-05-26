from typing import Literal, Optional

from hyperforge.agent import AgentConfig
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
    model: str = Field(
        default="chatgpt-azure-4o-mini",
        title="Generative model",
        description="Model used to generate the response",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    images: bool = False
    generate_image: bool = False
