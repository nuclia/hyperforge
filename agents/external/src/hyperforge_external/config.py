from enum import Enum
from typing import Any, Dict, Literal, Optional

from hyperforge.agent import AgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType, sync_dns_validation
from pydantic import Field, field_validator
from pydantic.config import ConfigDict


class Method(str, Enum):
    POST = "POST"
    GET = "GET"
    PATCH = "PATCH"


class ExternalCallAgentConfig(AgentConfig):
    model_config = ConfigDict(title="External call")
    module: Literal["external"] = "external"
    prompt: Optional[str] = Field(
        None,
        title="Extra Prompt",
        description="Extra prompt to provide more clues to extract parameters",
        json_schema_extra={
            "show_in_node": True,
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    method: Method = Field(
        Method.POST,
        title="Request method",
        description="POST, GET and PATCH are supported",
    )
    description: Optional[str] = Field(
        None,
        title="Description of the operation ",
        description="Description to help the LLM to extract the parameters",
    )
    call_schema: Optional[Dict[str, Any]] = Field(
        None,
        title="JSON Schema to compute parameters ",
        description="Valid JSON Schema to define the parameters to call the URL Its incompatible with call_obj and context",
    )
    call_obj: Optional[Dict[str, Any]] = Field(
        None,
        title="Object to call the endpoint ",
        description="Object that will be used to call the endpoint adding the answer and the question to it. Its incompatible with call_schema and context",
    )
    headers: Dict[str, str] = Field(
        {},
        title="Headers to use on the API call ",
    )
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.reasoning),
        title="Generative model",
        description="Model used to extract the parameters to call the URL",
    )
    context: bool = Field(
        False,
        title="Use context as payload",
        description="Use the context as payload. Its incompatible with call_schema and call_obj",
    )
    url: str = Field(
        ...,
        title="URL to call",
        description="",
    )

    @field_validator("url")
    def validate_url(cls, v):
        return sync_dns_validation(v)
