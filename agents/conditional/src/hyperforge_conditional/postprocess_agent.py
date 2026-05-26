from typing import List, Literal, Optional

from hyperforge.agent import Agent, AgentConfig
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from pydantic import Field
from pydantic.config import ConfigDict

from hyperforge_conditional.conditional import (
    Conditional,
    ConditionalAgentConfig,
)


class PostprocessConditionalAgentConfig(ConditionalAgentConfig):
    model_config = ConfigDict(title="Condition")
    module: Literal["post_conditional"] = "post_conditional"
    then: List["AgentConfig"] = Field(
        default_factory=list,
        title="Then",
        description="List of agents to run in case the condition is  valid",
    )
    else_: List["AgentConfig"] = Field(
        default_factory=list,
        title="Else",
        description="List of agents to run in case the condition is not valid",
    )


@agent(
    id="postprocess_conditional",
    agent_type="postprocess",
    title="Postprocess Conditional",
    description="Agent that performs conditional postprocessing.",
    config_schema=PostprocessConditionalAgentConfig,
)
class PostprocessConditional(Agent[PostprocessConditionalAgentConfig], Conditional):
    arag_step: str = "postprocess"

    async def inner_from_config(
        self, config: PostprocessConditionalAgentConfig, agent_id: Optional[str] = None
    ):
        await self.conditional_from_config(config)

    async def __call__(self, memory: QuestionMemory, manager: Manager):
        await Conditional.__call__(self, memory, manager)
