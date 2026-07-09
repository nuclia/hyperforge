"""
Standalone agent configuration.

A single JSON file describing one or more agents.  Runtime settings (NUA key,
HTTP host/port, log level, …) live in StandaloneSettings and are loaded from
environment variables or the command line — not from this file.

The top-level keys become the agent_id values used in API paths
(``/api/v1/agent/{agent_id}/...``).  The account is always ``"local"`` in
standalone mode.

Minimal valid example
---------------------
{
    "my-agent": {
        "workflows": {
            "default": {
                "name": "Default",
                "context": [
                    {"module": "google", "title": "Google Search", "source": "google-01"}
                ],
                "generation": [
                    {"module": "summarize"}
                ]
            }
        },
        "drivers": [
            {
                "identifier": "google-01",
                "name": "google",
                "provider": "google",
                "config": {"vertexai": false, "api_key": "..."}
            }
        ]
    }
}
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, TypeAdapter, field_serializer, field_validator

from hyperforge.agent import AgentConfig
from hyperforge.configure import get_agent_config_klass, get_driver_config_klass
from hyperforge.driver import DriverConfig
from hyperforge.models import Rules
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
    forward_authorization_header: bool = True


class WorkflowConfig(BaseModel):
    """Pipeline steps for a single named workflow."""

    name: str = "default"
    description: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None
    required: list[str] = Field(default_factory=list)
    rules: Rules = Field(default_factory=Rules)

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
        )

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

    @field_serializer("preprocess", "context", "generation", "postprocess")
    def serialize_agents(self, agents: list[AgentConfig]) -> list[Dict[str, Any]]:
        return [agent.model_dump() for agent in agents]


class StandAloneAgentConfig(BaseModel):
    """Configuration for a single agent instance."""

    title: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None

    # Drivers (LLM providers, search connectors, …) shared across all workflows.
    # Use DriverConfig[Any] so the generic `config` field preserves arbitrary dicts.
    drivers: list[DriverConfig[Any]] = Field(default_factory=list)

    # Top-level rules applied to all workflows.
    rules: Rules = Field(default_factory=Rules)

    # Named workflows. A "default" entry is required for the default workflow.
    workflows: dict[str, WorkflowConfig] = Field(default_factory=dict)

    # Prompts exposed via the MCP server.
    prompts: list[PromptConfig] = Field(default_factory=list)

    # Optional OAuth/JWT protection for this agent's standalone MCP endpoint.
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


# The config file is a plain JSON object whose keys are agent IDs.
# Use StandaloneConfig.validate_json(path.read_text()) to load it.
StandaloneConfig = TypeAdapter(dict[str, StandAloneAgentConfig])
