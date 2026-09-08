from typing import Any, Dict, List, Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.configure import get_agent_config_klass
from hyperforge.context.config import ContextAgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType
from pydantic import BaseModel, Field, field_serializer, field_validator
from pydantic.config import ConfigDict


class PythonAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Python task")
    module: Literal["restricted"] = "restricted"

    agents: Optional[List["ContextAgentConfig"]] = Field(
        None,
        title="Available agents",
        description="Agent to run after executing the code",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )

    code: str = Field(
        ...,
        title="Python code",
        description="The Python code to execute",
        json_schema_extra={
            "show_in_node": True,
        },
    )

    parameters: Dict[str, str] = Field(
        default_factory=dict,
        title="Parameters",
        description="Parameters for the Python code",
        json_schema_extra={
            "show_in_node": True,
        },
    )

    decision_model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.reasoning),
        title="Generative model",
        description="Model used to assess the condition",
    )

    needs_rephrase: bool = Field(
        default=False,
        title="Needs rephrase",
        description="Indicates if the first question needs to be rephrased based on context",
    )

    @field_validator("agents", mode="before")
    @classmethod
    def is_restricted_agent(cls, value: list[Dict[str, Any]]) -> list[AgentConfig]:
        if value is None:
            return value
        result = []
        for agent_cfg in value:
            module = agent_cfg.get("module")
            if module is None:
                raise ValueError("Invalid agent config: missing 'module' field")

            agent_config_klass = get_agent_config_klass(module)
            agent_config_instance = agent_config_klass.model_validate(agent_cfg)
            result.append(agent_config_instance)
        return result

    @field_serializer("agents")
    def serialize_restricted_agent(
        self, field: list[BaseModel]
    ) -> Optional[List[Dict[str, Any]]]:
        if field is None:
            return field
        return [agent.model_dump() for agent in field]
