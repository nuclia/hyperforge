from typing import ClassVar, Literal, Optional, Tuple

from hyperforge.context.config import ContextAgentConfig
from hyperforge.driver import DriverConfig, EncryptedPayload
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class GoogleInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = ["api_key", "credentials"]

    api_key: Optional[str] = None
    credentials: Optional[str] = None
    vertexai: bool = False
    project: Optional[str] = None
    location: Optional[str] = None


class GoogleDriverConfig(DriverConfig[GoogleInnerConfig]):
    model_config = ConfigDict(title="Google Gemini")
    provider: Literal["google"] = "google"
    config: GoogleInnerConfig


class GoogleAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Google Gemini")
    module: Literal["google"] = "google"
    gen_model_id: str = "gemini-2.5-flash"
    source: str = "google"
    published_functions: Optional[Tuple[str, ...]] = Field(
        default=("internet_search",),
        title="Published functions",
        description="List of functions published by this agent to be used by other agents in the chain",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
