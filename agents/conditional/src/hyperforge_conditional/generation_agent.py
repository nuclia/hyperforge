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


class GenerationConditionalAgentConfig(ConditionalAgentConfig):
    model_config = ConfigDict(title="Condition")
    module: Literal["generation_conditional"] = "generation_conditional"
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
    id="generation_conditional",
    agent_type="generation",
    title="Generation Conditional",
    description="Conditional generation agent that decides which generation strategy to use based on the context.",
    config_schema=GenerationConditionalAgentConfig,
)
class GenerationConditional(Agent[GenerationConditionalAgentConfig], Conditional):
    arag_step: str = "generation"

    async def inner_from_config(
        self, config: GenerationConditionalAgentConfig, agent_id: Optional[str] = None
    ):
        # Build then and else branches
        await self.conditional_from_config(config)

    async def __call__(self, memory: QuestionMemory, manager: Manager):
        await Conditional.__call__(self, memory, manager)
