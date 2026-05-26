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


class PreprocessConditionalAgentConfig(ConditionalAgentConfig):
    model_config = ConfigDict(title="Condition")
    module: Literal["pre_conditional"] = "pre_conditional"
    then: List[AgentConfig] = Field(
        default_factory=list,
        title="Then",
        description="List of agents to run in case the condition is  valid",
    )
    else_: List[AgentConfig] = Field(
        default_factory=list,
        title="Else",
        description="List of agents to run in case the condition is not valid",
    )


@agent(
    id="preprocess_conditional",
    agent_type="preprocess",
    title="Preprocess Conditional",
    description="Conditional preprocessing agent that decides which preprocessing strategy to use based on the context.",
    config_schema=PreprocessConditionalAgentConfig,
)
class PreprocessConditional(Agent[PreprocessConditionalAgentConfig], Conditional):
    arag_step: str = "preprocess"

    async def inner_from_config(
        self, config: PreprocessConditionalAgentConfig, agent_id: Optional[str] = None
    ):
        await self.conditional_from_config(config)

    async def __call__(self, memory: QuestionMemory, manager: Manager):
        await Conditional.__call__(self, memory, manager)
