from typing import Any, Dict, List, Literal, Optional, Tuple

from hyperforge.configure import get_agent_config_klass
from hyperforge.context.config import ContextAgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType
from pydantic import BaseModel, Field, field_serializer, field_validator
from pydantic.config import ConfigDict

PlanningMode = Literal["reactive", "plan_execute"]


class SmartAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Smart agent")
    module: Literal["smart"] = "smart"
    planning_mode: PlanningMode = Field(
        default="reactive",
        title="Planning mode",
        description=(
            "How the smart agent reasons about which tools to call. "
            "'reactive' (default): iterative LLM tool selection — the LLM picks tools each turn, "
            "executes them, and loops until task_complete. "
            "'plan_execute': a planner LLM first drafts a structured step-by-step plan, then an "
            "executor follows the plan and calls tools; the planner is re-invoked to assess progress "
            "and decide whether more retrieval is needed."
        ),
        json_schema_extra={"widget": WidgetType.ENUM_SELECT},
    )
    enable_user_feedback: bool = Field(
        default=False,
        title="Enable user feedback tool",
        description="Allow the LLM to call a user_feedback tool to ask the user clarifying questions during execution.",
    )
    feedback_timeout: int = Field(
        default=10_000,
        title="Feedback timeout (ms)",
        description="How long to wait for a user feedback response before giving up, in milliseconds.",
    )
    published_functions: Optional[Tuple[str, ...]] = Field(
        default=("smart_planner",),
        title="Published functions",
        description="List of functions published by this agent to be used by other agents in the chain",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    # Registered agents, list of Context agents, their descriptions and schemas to be used by the planner
    registered_agents: List[ContextAgentConfig] = Field(
        default_factory=list,
        title="Registered agents",
        description="List of context agents available for the smart agent to use",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    registered_agents_descriptions: Optional[Dict[str, str]] = Field(
        default=None,
        title="Registered agents descriptions",
        description="Descriptions of the registered agents for the planner",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    registered_agents_exposed_functions: Optional[Dict[str, List[str]]] = Field(
        default=None,
        title="Registered agents exposed functions",
        description="Exposed functions of the registered agents for the planner",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    planner_model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.smart),
        title="Planner model",
        description="Model used to plan the actions to take",
    )
    executor_model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.smart),
        title="Executor model",
        description=("Model used to select and execute the tools."),
    )
    max_iterations: int = Field(
        default=5,
        title="Max iterations",
        description="Maximum number of planning and execution iterations before stopping",
    )
    extra_prompt: Optional[str] = Field(
        None,
        title="Extra prompt",
        description="Extra prompt to provide to the planner",
        json_schema_extra={"widget": WidgetType.EXPANDABLE_TEXTAREA},
    )

    history: bool = Field(
        default=False,
        title="Session history",
        description="Include previous Q&A history from the current session in the context provided to the planner",
    )
    # Commented out for now
    # planner_reasoning: bool = Field(
    #     default=False,
    #     title="Planner reasoning",
    #     description=(
    #         "Enable extended reasoning for the planner LLM. "
    #         "When enabled, the planner uses medium reasoning effort. "
    #         "Only effective if the chosen planner model supports reasoning."
    #     ),
    # )

    @field_serializer("registered_agents")
    def serialize_smart_agent(
        self, field: list[BaseModel]
    ) -> Optional[List[Dict[str, Any]]]:
        if field is None:
            return field
        return [agent.model_dump() for agent in field]

    @field_validator("registered_agents", mode="before")
    @classmethod
    def is_smart_agent(cls, value: list[Dict[str, Any]]) -> list[BaseModel]:
        if value is None:
            return value
        result: list[BaseModel] = []
        for agent_cfg in value:
            module = agent_cfg.get("module")
            if module is None:
                raise ValueError("Invalid agent config: missing 'module' field")

            agent_config_klass = get_agent_config_klass(module)
            agent_config_instance = agent_config_klass.model_validate(agent_cfg)
            result.append(agent_config_instance)
        return result
