import ssl
import tempfile
from functools import partial
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from httpx import Auth, Timeout
from hyperforge.configure import driver
from hyperforge.driver import Driver
from hyperforge.interaction import Feedback
from hyperforge.memory import QuestionMemory

# from mcp.shared.auth import OAuthClientMetadata
from hyperforge.utils.http import SafeTransport

# from httpx import BasicAuth, DigestAuth
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

from hyperforge import logger
from hyperforge_mcp.config_driver import MCPHTTPDriverConfig, MCPHTTPInnerConfig


def create_mcp_http_client(
    ca_cert: str | None = None,
    certificate: str | None = None,
    headers: dict[str, str] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    """Create a standardized httpx AsyncClient with MCP defaults.

    This function provides common defaults used throughout the MCP codebase:
    - follow_redirects=True (always enabled)
    - Default timeout of 30 seconds if not specified

    Args:
        headers: Optional headers to include with all requests.
        timeout: Request timeout as httpx.Timeout object.
            Defaults to 30 seconds if not specified.
        auth: Optional authentication handler.

    Returns:
        Configured httpx.AsyncClient instance with MCP defaults.

    Note:
        The returned AsyncClient must be used as a context manager to ensure
        proper cleanup of connections.

    Examples:
        # Basic usage with MCP defaults
        async with create_mcp_http_client() as client:
            response = await client.get("https://api.example.com")

        # With custom headers
        headers = {"Authorization": "Bearer token"}
        async with create_mcp_http_client(headers) as client:
            response = await client.get("/endpoint")

        # With both custom headers and timeout
        timeout = httpx.Timeout(60.0, read=300.0)
        async with create_mcp_http_client(headers, timeout) as client:
            response = await client.get("/long-request")

        # With authentication
        from httpx import BasicAuth
        auth = BasicAuth(username="user", password="pass")
        async with create_mcp_http_client(headers, timeout, auth) as client:
            response = await client.get("/protected-endpoint")
    """
    # Set MCP defaults
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
    }

    # Handle timeout
    if timeout is None:
        kwargs["timeout"] = httpx.Timeout(200.0)
    else:
        kwargs["timeout"] = timeout

    # Handle headers
    if headers is not None:
        kwargs["headers"] = headers

    # Handle authentication
    if auth is not None:
        kwargs["auth"] = auth
    safe_transport: SafeTransport
    if ca_cert is not None or certificate is not None:
        ssl_context = ssl.create_default_context()
        if ca_cert is not None:
            ssl_context.load_verify_locations(cadata=ca_cert)

        if certificate is not None:
            if len(certificate) > 32_000:
                raise ValueError("Certificate is too large")
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as cert_file:
                cert_file.write(certificate)

            ssl_context.load_cert_chain(certfile=cert_file.name)

        safe_transport = SafeTransport(verify=ssl_context)
    else:
        safe_transport = SafeTransport()

    return httpx.AsyncClient(transport=safe_transport, **kwargs)


class InMemoryTokenStorage(TokenStorage):
    """Demo In-memory token storage implementation."""

    def __init__(self):
        self.tokens: OAuthToken | None = None
        self.client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        """Get stored tokens."""
        return self.tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Store tokens."""
        self.tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Get stored client information."""
        return self.client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Store client information."""
        self.client_info = client_info


async def handle_redirect(
    memory: QuestionMemory, module: str, agent_id: str, request_id: str, auth_url: str
) -> None:
    resp = await memory.send_feedback(
        Feedback(
            request_id=request_id,
            question=f"Visit: {auth_url}",
            module=module,
            agent_id=agent_id,
            data={"auth_url": auth_url},
            response_schema=None,
        )
    )
    logger.info("Redirect feedback sent:", resp)


async def handle_callback(
    memory: QuestionMemory, module: str, agent_id: str, request_id: str
) -> tuple[str, str | None]:
    callback_url = await memory.send_feedback(
        Feedback(
            request_id=request_id,
            question="Callback",
            module=module,
            agent_id=agent_id,
            data=None,
            response_schema=None,
        )
    )
    if callback_url:
        params = parse_qs(urlparse(callback_url.response).query)
        return params["code"][0], params.get("state", [None])[0]
    else:
        raise ValueError("No callback URL received")


@driver(
    id="mcphttp",
    title="MCP HTTP Driver",
    description="Driver for interacting with the MCP HTTP API.",
    config_schema=MCPHTTPDriverConfig,
)
class MCPHTTPDriver(Driver):
    config: MCPHTTPInnerConfig
    auth: Optional[Auth] = None

    @classmethod
    async def init(cls, driver: MCPHTTPDriverConfig) -> "MCPHTTPDriver":
        obj = cls(
            config=driver.config,
            name=driver.name,
            provider=driver.provider,
        )

        # TODO: Implement basic auth ?

        return obj

    def client(
        self,
        memory: QuestionMemory,
        module: str,
        agent_id: str,
        request_id,
        headers: Optional[dict[str, str]] = None,
    ):
        # Returns the HTTP client context manager
        new_headers = {}
        new_headers.update(self.config.headers)
        if headers is not None:
            new_headers.update(headers)

        if self.config.auth_server_url is not None:
            auth: Auth | None = OAuthClientProvider(
                server_url=self.config.auth_server_url,
                client_metadata=OAuthClientMetadata(
                    client_name="ARAG MCP Client",
                    redirect_uris=[AnyUrl(x) for x in self.config.redirect_uris],
                    grant_types=self.config.grant_types,
                    response_types=self.config.response_types,
                    scope=self.config.scope,
                ),
                storage=InMemoryTokenStorage(),
                redirect_handler=partial(
                    handle_redirect, memory, module, agent_id, request_id
                ),
                callback_handler=partial(
                    handle_callback, memory, module, agent_id, request_id
                ),
            )
        else:
            auth = self.auth

        client = create_mcp_http_client(
            self.config.ca_certificate,
            self.config.crt_certificate,
            new_headers,
            Timeout(self.config.timeout),
            auth,
        )
        return streamable_http_client(self.config.uri, http_client=client)

    async def sse_reader(self):
        pass

    async def post_writer(self):
        pass

    async def initialize(self):
        pass

    async def finalize(self):
        pass
