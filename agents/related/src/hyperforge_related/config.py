from typing import Literal, Optional

from hyperforge.agent import AgentConfig
from pydantic.config import ConfigDict


class RelatedAgentConfig(AgentConfig):
    model_config = ConfigDict(title="Related")
    module: Literal["related"] = "related"
    prompt: Optional[str] = None
    model: str = "chatgpt-azure-4o-mini"
    images: bool = False
