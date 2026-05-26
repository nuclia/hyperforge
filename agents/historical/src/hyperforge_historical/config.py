from typing import Literal

from hyperforge.agent import AgentConfig
from pydantic.config import ConfigDict


class HistoricalAgentConfig(AgentConfig):
    model_config = ConfigDict(title="History")
    all: bool
    module: Literal["historical"] = "historical"
