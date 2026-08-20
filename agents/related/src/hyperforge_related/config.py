from typing import Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from pydantic import Field
from pydantic.config import ConfigDict


class RelatedAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Related")
    module: Literal["related"] = "related"
    prompt: Optional[str] = None
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
        description="Model used to generate related questions",
    )
    images: bool = False
