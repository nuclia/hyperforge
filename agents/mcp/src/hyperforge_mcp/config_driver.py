from enum import Enum
from typing import Any, ClassVar, Dict, Literal, Optional

from hyperforge.driver import DriverConfig, EncryptedPayload
from hyperforge.settings import OAuthSettings
from hyperforge.utils import WidgetType
from pydantic import Field, model_validator
from pydantic.config import ConfigDict


def _redirect_uris_schema_default(schema: Dict[str, Any]) -> None:
    try:
        callback_url = OAuthSettings().mcp_callback_url
        if callback_url:
            schema["default"] = [callback_url]
    except Exception:
        pass


class MCPHTTPInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = [
        "client_secret",
        "headers",
        "ca_certificate",
        "crt_certificate",
    ]

    uri: str
    timeout: float = 60 * 5
    headers: Dict[str, str] = Field(default_factory=dict)
    sse_read_timeout: float = Field(default=300, title="SSE read timeout in seconds")
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
    redirect_uris: list[str] = Field(
        default_factory=list,
        title="OAuth Redirect URI",
        description="The callback URL registered in your OAuth Connected App. Auto-filled from the server configuration - do not change.",
        json_schema_extra=_redirect_uris_schema_default,
    )
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
    client_id: Optional[str] = Field(
        default=None,
        title="OAuth Client ID",
        description="Pre-registered client ID. If set, skips Dynamic Client Registration.",
    )
    client_secret: Optional[str] = Field(
        default=None,
        title="OAuth Client Secret",
        description="Pre-registered client secret. Required when the AS is not a public client.",
    )
    authorization_endpoint: Optional[str] = Field(
        default=None,
        title="OAuth Authorization Endpoint Override",
        description=(
            "Override the authorization endpoint discovered via RFC 8414 metadata. "
            "Use when the AS advertises a non-functional /authorize path."
        ),
    )
    token_endpoint: Optional[str] = Field(
        default=None,
        title="OAuth Token Endpoint Override",
        description=(
            "Override the token endpoint discovered via RFC 8414 metadata. "
            "Use when the AS uses a non-standard token path (e.g. uses "
            "/services/oauth2/token instead of /token)."
        ),
    )
    pkce: bool = Field(
        default=True,
        title="Enable PKCE",
        description=(
            "Whether to use PKCE (Proof Key for Code Exchange) in the OAuth 2.0 flow. "
            "Set to false for Authorization Servers that do not support PKCE "
            "(e.g. Connected Apps without PKCE enabled)."
        ),
    )

    @model_validator(mode="after")
    def _force_redirect_uris(self) -> "MCPHTTPInnerConfig":
        """Always override redirect_uris with the zone callback URL."""
        try:
            callback_url = OAuthSettings().mcp_callback_url
            if callback_url:
                self.redirect_uris = [callback_url]
        except Exception:
            pass
        return self


class MCPHTTPDriverConfig(DriverConfig[MCPHTTPInnerConfig]):
    model_config = ConfigDict(title="MCP HTTP")
    provider: Literal["mcphttp"]
    config: MCPHTTPInnerConfig


class MCPStdioServer(str, Enum):
    GITHUB = "github"


class MCPStdioInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = ["env"]

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
