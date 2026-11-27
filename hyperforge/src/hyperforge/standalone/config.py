"""
Standalone agent configuration.

Supports native Harness workflows (`execution_mode`, `model`, `toolkits`, `skills`,
`workflow_code`, `procedural_guidance`) while transparently accepting legacy
4-stage (`preprocess`, `context`, `generation`, `postprocess`) configurations.
"""

from typing import Any, Dict, Literal, Optional

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
from hyperforge.harness_sdk import HarnessSkill, ToolkitConfig, UsageLimits
from hyperforge.models import Rules
from hyperforge.procedural import ProceduralGuidanceConfig
from hyperforge.prompts import PromptConfig
from hyperforge.workflows import WorkflowData


class StandaloneMCPAuthConfig(BaseModel):
    """Optional OAuth/JWT protection for the standalone MCP endpoint."""

    enabled: bool = False
    authorization_server: Optional[str] = None
    protected_resource_metadata_url: Optional[str] = None
    protected_resource: Optional[str] = None
    scopes_supported: list[str] = Field(default_factory=list)
    required_scopes: list[str] = Field(default_factory=list)
    jwks_url: Optional[str] = None
    issuer: Optional[str] = None
    audience: Optional[str] = None

    @model_validator(mode="after")
    def validate_enabled_config(self):
        if not self.enabled:
            return self

        missing = []
        if self.authorization_server is None:
            missing.append("authorization_server")
        if self.jwks_url is None:
            missing.append("jwks_url")
        if missing:
            raise ValueError(
                f"Missing required MCP auth setting(s) when enabled: {', '.join(missing)}"
            )
        return self


class WorkflowConfig(BaseModel):
    """Configuration for a single named workflow in standalone mode."""

    name: str = "default"
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    required: list[str] = Field(default_factory=list)
    rules: Rules = Field(default_factory=Rules)

    # Native Harness fields
    execution_mode: Optional[Literal["loop", "code"]] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    toolkits: list[ToolkitConfig] = Field(default_factory=list)
    skills: list[HarnessSkill] = Field(default_factory=list)
    workflow_code: Optional[str] = None
    feedback_enabled: bool = False
    codemode_enabled: bool = False
    usage_limits: UsageLimits = Field(default_factory=UsageLimits)
    procedural_guidance: Optional[ProceduralGuidanceConfig] = None

    # Legacy 4-stage compatibility fields (auto-transpiled to Harness at load time)
    preprocess: list = Field(default_factory=list)
    context: list = Field(default_factory=list)
    generation: list = Field(default_factory=list)
    postprocess: list = Field(default_factory=list)

    def as_workflow_data(self, workflow_id: str) -> WorkflowData:
        return WorkflowData(
            id=workflow_id,
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            rules=self.rules,
            required=self.required,
            execution_mode=self.execution_mode or "loop",
            workflow_code=self.workflow_code,
        )

    @field_validator("toolkits", mode="before")
    @classmethod
    def validate_toolkits(cls, value: list[Any]) -> list[ToolkitConfig]:
        if not value:
            return []
        return [
            item
            if isinstance(item, ToolkitConfig)
            else get_toolkit_config_klass(item["toolkit"]).model_validate(item)
            for item in value
        ]

    @field_validator(
        "preprocess", "context", "generation", "postprocess", mode="before"
    )
    def validate_agents(cls, value: list[Dict[str, Any]], field):
        if len(value) == 0:
            return []
        if all([isinstance(agent, AgentConfig) for agent in value]):
            return value
        result = []
        for agent_config in value:
            module = agent_config["module"]
            agent_klass = get_agent_config_klass(module)
            result.append(agent_klass.model_validate(agent_config))

        return result

    @field_serializer("preprocess", "context", "generation", "postprocess", "toolkits")
    def serialize_agents(self, agents: list[BaseModel]) -> list[Dict[str, Any]]:
        return [agent.model_dump() for agent in agents]


class StandAloneAgentConfig(BaseModel):
    """Configuration for a single agent instance."""

    title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None

    drivers: list[DriverConfig[Any]] = Field(default_factory=list)
    rules: Rules = Field(default_factory=Rules)
    workflows: dict[str, WorkflowConfig] = Field(default_factory=dict)
    prompts: list[PromptConfig] = Field(default_factory=list)
    skills: list[HarnessSkill] = Field(default_factory=list)
    mcp_auth: Optional[StandaloneMCPAuthConfig] = None

    @field_validator("drivers", mode="before")
    def validate_drivers(cls, value: list[Dict[str, Any]], field):
        if len(value) == 0:
            return []
        if all([isinstance(agent, DriverConfig) for agent in value]):
            return value
        result = []
        for agent_config in value:
            module = agent_config["provider"]
            agent_klass = get_driver_config_klass(module)
            result.append(agent_klass.model_validate(agent_config))

        return result

    @field_serializer("drivers")
    def serialize_drivers(self, agents: list[DriverConfig]) -> list[Dict[str, Any]]:
        return [agent.model_dump() for agent in agents]


StandaloneConfig = TypeAdapter(dict[str, StandAloneAgentConfig])
