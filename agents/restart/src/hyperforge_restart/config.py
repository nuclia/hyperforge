from typing import Literal

from hyperforge.agent import AgentConfig
from pydantic.config import ConfigDict


class RestartAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Restart")
    module: Literal["restart"] = "restart"
    model: str
    retries: int = 2
