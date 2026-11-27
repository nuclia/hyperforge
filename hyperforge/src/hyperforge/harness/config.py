import datetime
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    field_serializer,
    field_validator,
    model_validator,
)

from hyperforge.agent import AgentConfig
from hyperforge.configure import (
    get_agent_config_klass,
    get_driver_config_klass,
    get_toolkit_config_klass,
)
from hyperforge.driver import DriverConfig
from hyperforge.harness_sdk import (
    SYSTEM_PROMPT,
    CodeModeLimits,
    HarnessSkill,
    ReasoningEffort,
    ToolkitConfig,
    UsageLimits,
)
from hyperforge.models import MemoryConfig, Rules
from hyperforge.procedural import ProceduralGuidanceConfig
from hyperforge.prompts import PromptConfig


class HarnessAgentConfig(BaseModel):
    execution_mode: Literal["loop", "code"] = "loop"
    model: str = ""
    workflow_code: str | None = None
    agents: list[AgentConfig] = Field(default_factory=list)
    toolkits: list[ToolkitConfig] = Field(default_factory=list)
    drivers: list[DriverConfig] = Field(default_factory=list)
    skills: list[HarnessSkill] = Field(default_factory=list)
    prompts: list[PromptConfig] = Field(default_factory=list)
    prompt_id: str | None = None
    expose_prompts_as_skills: bool = False
    codemode_enabled: bool = False
    codemode_limits: CodeModeLimits | None = None
    system_prompt: str = SYSTEM_PROMPT
    reasoning_effort: ReasoningEffort | None = None
    disabled_core_tools: list[str] = Field(default_factory=list)
    feedback_enabled: bool = False
    usage_limits: UsageLimits = Field(default_factory=UsageLimits)
    procedural_guidance: ProceduralGuidanceConfig | None = None
    rules: Rules = Field(default_factory=Rules)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @model_validator(mode="after")
    def validate_execution_mode(self) -> "HarnessAgentConfig":
        if self.execution_mode == "loop" and not self.model.strip():
            raise ValueError("model is required when execution_mode is 'loop'")
        if self.execution_mode == "code" and (
            self.workflow_code is None or not self.workflow_code.strip()
        ):
            raise ValueError("workflow_code is required when execution_mode is 'code'")
        return self

    @field_validator("agents", mode="before")
    @classmethod
    def validate_agents(cls, value: list[Any]) -> list[AgentConfig]:
        return [
            item
            if isinstance(item, AgentConfig)
            else get_agent_config_klass(item["module"]).model_validate(item)
            for item in value
        ]

    @field_validator("toolkits", mode="before")
    @classmethod
    def validate_toolkits(cls, value: list[Any]) -> list[ToolkitConfig]:
        return [
            item
            if isinstance(item, ToolkitConfig)
            else get_toolkit_config_klass(item["toolkit"]).model_validate(item)
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

    @field_serializer("agents", "toolkits", "drivers")
    def serialize_config_items(self, items: list[BaseModel]) -> list[dict[str, Any]]:
        return [item.model_dump() for item in items]

    def is_empty(self) -> bool:
        return (
            len(self.drivers) == 0
            and len(self.toolkits) == 0
            and len(self.agents) == 0
            and len(self.skills) == 0
            and not (self.workflow_code and self.workflow_code.strip())
        )


class HarnessAgentExportV2(BaseModel):
    version: Literal["2"] = Field(default="2")
    agent_config: HarnessAgentConfig | None = None
    agent_config_workflows: dict[str, HarnessAgentConfig] = Field(default_factory=dict)
    skills: list[HarnessSkill] = Field(default_factory=list)
    prompts: list[PromptConfig] = Field(default_factory=list)
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        description="Timestamp of when the export was created in UTC",
    )


def _export_union_adapter() -> TypeAdapter[Any]:
    from hyperforge.retrieval.config import RetrievalAgentExportV1

    return TypeAdapter(
        Annotated[
            Union[RetrievalAgentExportV1, HarnessAgentExportV2],
            Field(discriminator="version"),
        ]
    )
