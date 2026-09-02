from typing import Any

from pydantic import BaseModel, Field, field_serializer, field_validator

from hyperforge.agent import AgentConfig
from hyperforge.configure import get_agent_config_klass, get_driver_config_klass
from hyperforge.driver import DriverConfig
from hyperforge.harness_sdk import SYSTEM_PROMPT, ReasoningEffort, UsageLimits
from hyperforge.models import MemoryConfig, Rules


class HarnessAgentConfig(BaseModel):
    model: str
    agents: list[AgentConfig] = Field(default_factory=list)
    drivers: list[DriverConfig] = Field(default_factory=list)
    system_prompt: str = SYSTEM_PROMPT
    reasoning_effort: ReasoningEffort | None = None
    disabled_core_tools: list[str] = Field(default_factory=list)
    feedback_enabled: bool = False
    usage_limits: UsageLimits = Field(default_factory=UsageLimits)
    rules: Rules = Field(default_factory=Rules)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @field_validator("agents", mode="before")
    @classmethod
    def validate_agents(cls, value: list[Any]) -> list[AgentConfig]:
        return [
            item
            if isinstance(item, AgentConfig)
            else get_agent_config_klass(item["module"]).model_validate(item)
            for item in value
        ]

    @field_validator("drivers", mode="before")
    @classmethod
    def validate_drivers(cls, value: list[Any]) -> list[DriverConfig]:
        return [
            item
            if isinstance(item, DriverConfig)
            else get_driver_config_klass(item["provider"]).model_validate(item)
            for item in value
        ]

    @field_serializer("agents", "drivers")
    def serialize_config_items(self, items: list[BaseModel]) -> list[dict[str, Any]]:
        return [item.model_dump() for item in items]
