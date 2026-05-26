from enum import Enum
from typing import ClassVar, Dict, Literal, Optional

from hyperforge.driver import DriverConfig, EncryptedPayload
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class MCPHTTPInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = []

    uri: str
    timeout: float = 60 * 5
    headers: Dict[str, str] = Field(default_factory=dict)
    ca_certificate: Optional[str] = Field(
        default=None,
        title="CA certificate for HTTPS",
        json_schema_extra={"widget": WidgetType.EXPANDABLE_TEXTAREA},
    )
    crt_certificate: Optional[str] = Field(
        default=None,
        title="CRT certificate for HTTPS",
        json_schema_extra={"widget": WidgetType.EXPANDABLE_TEXTAREA},
    )

    server_url: Optional[str] = Field(
        default=None, title="OAuth Authorization Server URL"
    )
    redirect_uris: list[str] = Field(default_factory=list, title="OAuth Redirect URIs")
    auth_server_url: Optional[str] = Field(
        default=None, title="OAuth Authorization Server URL"
    )
    grant_types: list[str] = Field(
        default_factory=list,
        title="OAuth Grant Types",
        description="Default: ['authorization_code', 'refresh_token']",
    )
    response_types: list[str] = Field(
        default_factory=list,
        title="OAuth Response Types",
        description="Default: ['code']",
    )
    scope: str = Field(
        default="user", title="OAuth Scopes", description="Default: 'user'"
    )


class MCPHTTPDriverConfig(DriverConfig[MCPHTTPInnerConfig]):
    model_config = ConfigDict(title="MCP HTTP")
    provider: Literal["mcphttp"]
    config: MCPHTTPInnerConfig


class MCPStdioServer(str, Enum):
    GITHUB = "github"


class MCPStdioInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = []

    server: MCPStdioServer
    env: dict[str, str] | None = None
    """
    The environment to use when spawning the process.

    If not specified, the result of get_default_environment() will be used.
    """


class MCPStdioDriverConfig(DriverConfig[MCPStdioInnerConfig]):
    id: Optional[str] = None
    provider: Literal["mcpstdio"]
    identifier: str = "mcpstdio"
    config: MCPStdioInnerConfig
