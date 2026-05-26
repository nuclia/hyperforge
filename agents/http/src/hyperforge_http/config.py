from typing import Literal, Optional

from hyperforge.context.agent import ContextAgentConfig
from hyperforge.utils import WidgetType, sync_dns_validation
from pydantic import Field, field_validator
from pydantic.config import ConfigDict


class HTTPStaticAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="HTTP call")
    module: Literal["http"] = "http"
    method: Literal["GET", "POST"] = Field(
        default="GET", description="HTTP method to use for the request"
    )
    url: str = Field(..., description="The URL to fetch context from")

    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Optional headers to include in the HTTP request",
        json_schema_extra={"widget": WidgetType.KEY_VALUE_FIELD},
    )
    question_query_param: Optional[str] = Field(
        None,
        description="Optional query parameter to include the question in the request",
    )
    question_post_field: Optional[str] = Field(
        None,
        description="Optional POST field id to include the question in the request body",
    )
    timeout: int = Field(
        default=10, description="Timeout for the HTTP request in seconds"
    )

    @field_validator("url")
    def validate_url(cls, v):
        return sync_dns_validation(v)
