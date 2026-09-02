from typing import Literal

from hyperforge.agent import AgentConfig
from hyperforge.llm_config import LLMField
from pydantic import Field
from pydantic.config import ConfigDict


class RestartAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Restart")
    module: Literal["restart"] = "restart"
    model: LLMField = Field(
        title="Generative model",
        description="Model used by the restart agent",
    )
    retries: int = 2
