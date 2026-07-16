import asyncio
from time import time
from typing import Any, Dict, List, Literal, Optional, cast

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.agent import Agent, AgentConfig
from hyperforge.configure import get_agent_config_klass, get_agent_klass
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.trace import trace_agent
from hyperforge.utils import WidgetType
from pydantic import BaseModel, Field, field_serializer, field_validator

CONDITIONAL_AGENT = """
Given a prompt that indicates a condition , assess whether a given text fulfills it or not.
# PROMPT:
{{prompt}}


# TEXT:
{{text}}

{% if similarity %}
# SIMILAR QUERIES:
Similar queries that also fulfill that condition
{% for similar in similarity_examples %}
- {{similar}}
{% endfor -%}
{% endif %}

# Important rules to follow

{% for rule in rules %}
{{rule}}
{% endfor -%}

# Output definition:

Your output should always follow the following schema in a JSON format (Not markdon, please, start with b...).

{{json_schema}}
"""

CONDITIONAL_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(CONDITIONAL_AGENT)


CONDITIONAL_SCHEMA = {
    "title": "yes_no",
    "description": "Choose yes or no, depending on whether the condition is fulfilled",
    "parameters": {
        "type": "object",
        "properties": {
            "yes": {"type": "boolean"},
            "reason": {"type": "string", "description": "reasoning behind the answer"},
        },
    },
}


class ConditionalAgentConfig(AgentConfig):
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
    prompt: Optional[str] = Field(
        default=None,
        title="Condition Prompt",
        description="Prompt to evaluate the condition",
        json_schema_extra={
            "show_in_node": True,
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    has_keywords: Optional[List[str]] = Field(
        default=None,
        title="Keywords",
        description="List of keywords to evaluate the condition",
    )
    similarity: Optional[List[str]] = Field(
        default=None,
        title="Similar Queries",
        description="List of similar queries to evaluate the condition",
    )
    on: Literal["QUESTION", "ANSWER", "CONTEXT"] = Field(
        default="QUESTION",
        title="Evaluate condition on",
        description="Source to evaluate the condition on. ",
    )
    model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.reasoning),
        title="Generative model",
        description="Model used to assess the condition",
    )

    @field_serializer("then", "else_")
    def serialize_conditional_agent(
        self, field: list[BaseModel]
    ) -> Optional[List[Dict[str, Any]]]:
        if field is None:
            return field
        return [agent.model_dump() for agent in field]

    @field_validator("then", "else_", mode="before")
    @classmethod
    def is_conditional_agent(cls, value: list[Dict[str, Any]]) -> list[BaseModel]:
        if value is None:
            return value
        result: list[BaseModel] = []
        for agent_cfg in value:
            module = agent_cfg.get("module")
            if module is None:
                raise ValueError("Invalid agent config: missing 'module' field")

            agent_config_klass = get_agent_config_klass(module)
            agent_config_instance = agent_config_klass.model_validate(agent_cfg)
            result.append(cast(BaseModel, agent_config_instance))
        return result


class Conditional:
    then: Optional[list[Agent]] = None
    else_: Optional[list[Agent]] = None
    arag_step: str
    config: Any
    agent_id: Any

    async def conditional_from_config(self, config: ConditionalAgentConfig):
        agent_module = None
        then_agents_obj_list = []
        for then_agent in config.then:
            if then_agent:
                agent_module = then_agent.module
            if agent_module is None:
                raise Exception("No agent found")

            agent_klass = get_agent_klass(agent_module)
            # We dump and load to create the agent via from_config
            then_agents_obj_list.append((agent_klass, then_agent))

        agent_id = None
        else_agents_obj_list = []
        for else_agent in config.else_:
            if else_agent is not None:
                agent_id = else_agent.module
            if agent_id is None:
                raise Exception("No agent found")

            agent_class = get_agent_klass(agent_id)
            # We dump and load to create the agent via from_config
            else_agents_obj_list.append((agent_class, else_agent))

        self.then = await asyncio.gather(
            *[klass.from_config(cfg) for klass, cfg in then_agents_obj_list]
        )
        self.else_ = (
            await asyncio.gather(
                *[klass.from_config(cfg) for klass, cfg in else_agents_obj_list]
            )
            if else_agents_obj_list
            else None
        )

    async def make_decision(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        title: Optional[str] = None,
    ) -> bool:
        t0 = time()
        config: ConditionalAgentConfig = cast(ConditionalAgentConfig, self.config)

        prompt = CONDITIONAL_AGENT_TEMPLATE.render(
            prompt=config.prompt,
            rules=memory.get_rules(),
            text=question,
            similarity_examples=config.similarity,
        )

        sources, input, output = await manager.execute_json(
            prompt=prompt,
            schema=CONDITIONAL_SCHEMA,
            user_id="conditional",
            model=config.model,
            tracking=memory.get_tracking_info(),
        )
        condition = sources.get("yes", False)
        reason = sources.get("reason")

        await memory.add_step(
            step_module=config.module,
            step_title=f"{title}: Condition check",
            step_value=str(condition),
            step_reason=reason,
            timeit=time() - t0,
            input_nuclia_tokens=input,
            output_nuclia_tokens=output,
            step_agent_path=f"/{self.arag_step}/{self.agent_id}",
        )
        return condition

    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        question = ""
        if self.config.on == "QUESTION" and memory.actual_question:
            question = memory.actual_question
        elif self.config.on == "ANSWER" and memory.final_answer:
            question = memory.final_answer
        elif self.config.on == "CONTEXT":
            question = "\n".join([x.summary for x in memory.contexts])

        condition = await self.make_decision(
            memory=memory,
            manager=manager,
            question=question,
        )

        if condition and self.then is not None:
            for then_agent in self.then:
                await then_agent(memory, manager)
        elif condition is False and self.else_ is not None:
            for else_agent in self.else_:
                await else_agent(memory, manager)
