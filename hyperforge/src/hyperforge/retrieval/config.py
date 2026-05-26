import datetime
from typing import Annotated, Any, Dict, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, field_serializer, field_validator

from hyperforge.agent import AgentConfig
from hyperforge.configure import get_agent_config_klass, get_driver_config_klass
from hyperforge.driver import DriverConfig
from hyperforge.models import MemoryConfig, Rules
from hyperforge.prompts import PromptConfig
from hyperforge.workflows import WorkflowData


class RetrievalAgentConfig(BaseModel):
    drivers: list[DriverConfig]
    rules: Rules
    memory: MemoryConfig
    workflow: WorkflowData

    preprocess: list[AgentConfig]
    context: list[AgentConfig]
    generation: list[AgentConfig]
    postprocess: list[AgentConfig]

    @field_serializer("preprocess", "context", "generation", "postprocess", "drivers")
    def serialize_agents(self, agents: list[AgentConfig]) -> list[Dict[str, Any]]:
        return [agent.model_dump() for agent in agents]

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

    def is_empty(self) -> bool:
        return (
            len(self.drivers) == 0
            and len(self.preprocess) == 0
            and len(self.context) == 0
            and len(self.generation) == 0
            and len(self.postprocess) == 0
        )


class RetrievalAgentExportV1(BaseModel):
    version: Literal["1"] = Field(default="1")
    agent_config: RetrievalAgentConfig | None
    agent_config_workflows: dict[str, RetrievalAgentConfig] = Field(
        default_factory=dict
    )
    prompts: list[PromptConfig]
    timestamp: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat(
            timespec="seconds"
        ),
        description="Timestamp of when the export was created in UTC",
    )


RetrievalAgentExport = Annotated[
    Union[RetrievalAgentExportV1], Field(discriminator="version")
]
retrievalAgentAdapter = TypeAdapter(RetrievalAgentExport)


class RetrievalAgentExportRequest(BaseModel):
    passphrase: str = Field(
        description="Passphrase to encrypt the exported configuration. Will be required for import.",
        min_length=16,
    )
