from enum import Enum
from typing import Literal

from hyperforge.agent import AgentConfig
from pydantic import Field
from pydantic.config import ConfigDict


class ContextGranularity(str, Enum):
    FULL = "full"
    PARTIAL_ANSWERS = "partial_answers"


class RemiAgentConfig(AgentConfig):
    model_config = ConfigDict(title="REMi evaluation")
    module: Literal["remi"] = "remi"
    context_granularity: ContextGranularity = Field(
        default=ContextGranularity.FULL,
        title="Granularity of the contexts pieces",
        description="Granularity of the context pieces sent to REMi for groundedness evaluation. "
        "If 'partial_answers', the evaluation will use agent-level answer attempts to the question when available for a speedier analysis. "
        "If 'full', the evaluation will use all individual text chunks that each agent generated for a more detailed analysis.",
    )
