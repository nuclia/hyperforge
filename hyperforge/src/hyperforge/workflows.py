from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hyperforge.models import Rules


class WorkflowData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    name: str
    description: str | None
    parameters: dict[str, Any] | None
    rules: Rules | None = Field(default_factory=Rules)
    required: list[str] = Field(default_factory=list)


class WorkflowInput(BaseModel):
    id: str
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    rules: Rules = Field(default_factory=Rules)
    required: list[str] = Field(default_factory=list)


class WorkflowUpdate(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str] = Field(default_factory=list)
    rules: Rules | None = Field(default_factory=Rules)


class RetrievalAgent(BaseModel):
    account: str
    agent_id: str
    memory: dict[str, Any] | None = None
    description: str | None = None
    title: str | None = None
    instructions: str | None = None
    created: datetime
    modified: datetime | None = None
