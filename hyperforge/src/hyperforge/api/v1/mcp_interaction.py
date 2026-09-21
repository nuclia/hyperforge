import asyncio
import json
import uuid
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from fastapi import Header, HTTPException
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.lowlevel.server import lifespan as default_lifespan
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import (
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)
from nucliadb_sdk import NucliaDBAsync
from pydantic import AnyUrl
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import Response

from hyperforge.api.authentication import requires_one
from hyperforge.api.mcp_server_pool import (
    ManagedMCPServer,
    MCPRequestLease,
    MCPServerKey,
)
from hyperforge.api.models import InteractionRequest
from hyperforge.api.settings import Settings as ApiSettings
from hyperforge.api.v1.interaction import WebsocketReceiver, stream_response
from hyperforge.db.agents import AgentManager
from hyperforge.interaction import AnswerOperation, Provider
from hyperforge.prompts import PromptConfig
from hyperforge.pubsub import UserToAgentInteraction
from hyperforge.standalone.oauth import force_https_metadata, get_enabled_mcp_auth
from hyperforge.workflows import WorkflowData

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication
from hyperforge.api import logger
from hyperforge.api.models import (
    AgentRole,
)
from hyperforge.api.v1.mcp_content import convert_arag_answer_to_content
from hyperforge.api.v1.router import router

SUPPORTED_OAUTH_CREDENTIAL_PROVIDERS = frozenset({Provider.SHAREFILE_OAUTH})


async def list_tools(workflows: list[WorkflowData]) -> list[Tool]:
    return [
        Tool(
            name=workflow.name,
            description=workflow.description,
            inputSchema={
                "type": "object",
                "required": workflow.required,
                "properties": workflow.parameters,
            },
        )
        for workflow in workflows
    ]


async def list_prompts(prompts: list[PromptConfig]) -> list[Prompt]:
    """List all available prompts."""
    return [Prompt(**prompt.model_dump()) for prompt in prompts]


async def call_tool(
    app: "HTTPApplication",
    mcp_server: MCPServer,
    x_stf_account: str,
    agent_id: str,
    session: str,
    workflows: list[WorkflowData],
    headers: Headers,
    name: str,
    arguments: dict[str, Any],
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Call a tool by name with arguments."""
    workflow = next((w for w in workflows if w.name == name), None)
    if workflow is None:
        raise ResourceError(f"Unknown tool: {name}")

    for parameter in workflow.required:
        if parameter not in arguments:
            raise ResourceError(f"Missing required parameter: {parameter}")

    question = f"Calling tool: {workflow.description or workflow.name} with arguments: {arguments}"
    request_headers = headers
    if isinstance(app.settings, ApiSettings):
        current_request = mcp_server.request_context.request
        if not isinstance(current_request, Request):
            raise ResourceError("Current MCP HTTP request is unavailable")
        request_headers = current_request.headers
    interaction_headers = _prepare_interaction_headers(app, agent_id, request_headers)
    user_id = interaction_headers.get("x-stf-user")
    if not user_id:
        raise ResourceError("Authenticated user identity is required")

    interaction = InteractionRequest(
        question=question, headers=interaction_headers, arguments=arguments
    )
    mcp_session = mcp_server.request_context.session
    websocket = WebsocketReceiver(websocket=None)

    messages = []
    requested_credentials: dict[str, Provider] = {}
    async for msg in stream_response(
        app,
        websocket,
        account=x_stf_account,
        agent_id=agent_id,
        session=session,
        interaction=interaction,
        workflow_id=workflow.id,
    ):
        if msg.operation == AnswerOperation.AGENT_REQUEST and msg.oauth:
            result = await mcp_session.elicit_url(
                message="Authenticate with ShareFile to access the requested content.",
                url=msg.oauth.oauth_url,
                elicitation_id=uuid.uuid4().hex,
            )
            if result.action != "accept":
                raise ResourceError(
                    f"ShareFile authentication was {result.action} by the user"
                )
        elif msg.operation == AnswerOperation.AGENT_REQUEST and msg.feedback:
            feedback = msg.feedback
            if feedback.get_credentials is not None:
                requested_credentials = feedback.get_credentials
                unsupported = {
                    provider
                    for provider in requested_credentials.values()
                    if provider not in SUPPORTED_OAUTH_CREDENTIAL_PROVIDERS
                }
                if unsupported:
                    raise ResourceError(
                        "MCP credential storage does not support the requested provider"
                    )

                existing_credentials: dict[str, dict[str, str]] = {}
                for sync_config_id, provider in requested_credentials.items():
                    credentials = await app.agent_manager.get_sync_oauth_credentials(
                        account=x_stf_account,
                        user_id=user_id,
                        agent_id=agent_id,
                        provider=provider.value,
                        sync_config_id=sync_config_id,
                    )
                    if credentials is not None:
                        existing_credentials[sync_config_id] = credentials

                if len(existing_credentials) != len(requested_credentials):
                    existing_credentials.clear()

                websocket.queue.put_nowait(
                    UserToAgentInteraction(
                        request_id=feedback.request_id,
                        response=json.dumps(
                            {"existing_credentials": existing_credentials}
                        ),
                    )
                )
            elif feedback.credentials is not None:
                if set(feedback.credentials) != set(requested_credentials):
                    raise ResourceError(
                        "Received credentials that do not match the requested Sync configurations"
                    )

                for sync_config_id, credentials in feedback.credentials.items():
                    if not isinstance(credentials, dict) or not all(
                        isinstance(key, str) and isinstance(value, str)
                        for key, value in credentials.items()
                    ):
                        raise ResourceError("Received invalid Sync OAuth credentials")
                    provider = requested_credentials[sync_config_id]
                    await app.agent_manager.upsert_sync_oauth_credentials(
                        account=x_stf_account,
                        user_id=user_id,
                        agent_id=agent_id,
                        provider=provider.value,
                        sync_config_id=sync_config_id,
                        credentials=credentials,
                    )

                websocket.queue.put_nowait(
                    UserToAgentInteraction(
                        request_id=feedback.request_id,
                        response=json.dumps(
                            {"existing_credentials": feedback.credentials}
                        ),
                    )
                )
            else:
                result = await mcp_session.elicit_form(
                    message=feedback.question,
                    requestedSchema=feedback.response_schema,
                    related_request_id=feedback.request_id,
                )
                websocket.queue.put_nowait(
                    UserToAgentInteraction(
                        request_id=feedback.request_id,
                        response=result.content,  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
                    )
                )
        elif msg.operation == AnswerOperation.ANSWER:
            result_contents = convert_arag_answer_to_content(msg)
            for content in result_contents:
                if isinstance(content, TextContent):
                    logger.debug(f"Tool output text: {content.text}")
                messages.append(content)
        elif msg.operation == AnswerOperation.ERROR:
            detail = msg.exception.detail if msg.exception else "Agent execution failed"
            raise ResourceError(detail)

    return messages


async def list_resources(agent_id: str) -> list[Resource]:
    # TODO : Resource 1 : The list of tools ??
    return []


async def list_resource_templates() -> list[ResourceTemplate]:
    return []


async def get_prompt(
    prompts_list: list[PromptConfig], name: str, arguments: dict[str, Any] | None = None
) -> GetPromptResult:
    """Get a prompt by name with arguments."""
    prompt = next((p for p in prompts_list if p.name == name), None)
    if prompt is None:
        raise ResourceError(f"Unknown prompt: {name}")
    message = prompt.prompt.format(**(arguments or {}))
    return GetPromptResult(
        description=prompt.description,
        messages=[
            PromptMessage(role="user", content=TextContent(type="text", text=message))
        ],
    )


async def read_resource(
    ndb: NucliaDBAsync, kbid: str, uri: AnyUrl | str
) -> Iterable[ReadResourceContents]:
    """Read a resource by URI."""

    raise ResourceError(f"Unknown uri: {uri}")


def _prepare_interaction_headers(
    app: "HTTPApplication", agent_id: str, headers: Headers
) -> dict[str, str]:
    interaction_headers = dict(headers.items())
    authorization = headers.get("authorization")
    if authorization is not None:
        interaction_headers["authorization"] = authorization

    return interaction_headers


def _default_oauth_metadata(app: "HTTPApplication") -> tuple[list[str], list[str]]:
    if isinstance(app.settings, ApiSettings):
        return [app.settings.hydra_public_url], app.settings.hydra_scopes_supported
    return [], []


def _get_mcp_auth_config(app: "HTTPApplication", agent_id: str):
    return get_enabled_mcp_auth(app._agents_cfg, agent_id)


def _get_first_enabled_mcp_auth_config(app: "HTTPApplication"):
    agents_cfg = getattr(app, "_agents_cfg", None)
    if isinstance(agents_cfg, dict):
        for agent_id in agents_cfg:
            auth_config = get_enabled_mcp_auth(agents_cfg, agent_id)
            if auth_config is not None:
                return auth_config
    return None


async def _create_mcp_server(
    app: "HTTPApplication",
    request: Request,
    agent_id: str,
    session: str,
    account: str,
    max_request_bytes: int,
) -> ManagedMCPServer:
    agent_manager: AgentManager = request.app.agent_manager
    workflows, agent_config, prompts = await asyncio.gather(
        agent_manager.workflows_list(account=account, agent_id=agent_id),
        agent_manager.get_agent_config_basic(account=account, agent_id=agent_id),
        agent_manager.get_prompts(account=account, agent_id=agent_id),
    )
    mcp_server = MCPServer(
        name=agent_id,
        version="1.0.0",
        instructions=agent_config.instructions,
        lifespan=default_lifespan,
    )
    mcp_server.list_tools()(partial(list_tools, workflows))
    mcp_server.call_tool()(
        partial(
            call_tool,
            app,
            mcp_server,
            account,
            agent_id,
            session,
            workflows,
            request.headers,
        )
    )
    mcp_server.list_prompts()(partial(list_prompts, prompts=prompts))
    mcp_server.get_prompt()(partial(get_prompt, prompts))
    manager = StreamableHTTPSessionManager(
        app=mcp_server,
        json_response=True,
        stateless=False,
        security_settings=None,
        max_request_body_size=max_request_bytes,
    )
    managed_server = ManagedMCPServer(manager)
    await managed_server.start()
    return managed_server


class _MCPTransportResponse(Response):
    def __init__(
        self,
        request: Request,
        lease: MCPRequestLease,
        body: bytes,
        max_response_bytes: int,
    ):
        super().__init__()
        self.request = request
        self.lease = lease
        self.managed_server = lease.server
        self.body = body
        self.max_response_bytes = max_response_bytes

    async def __call__(self, scope, receive, send) -> None:
        try:
            await self._handle_request(scope, receive, send)
        finally:
            self.lease.release()

    async def _handle_request(self, scope, receive, send) -> None:
        body_sent = False

        async def patched_receive():
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {
                    "type": "http.request",
                    "body": self.body,
                    "more_body": False,
                }
            return await receive()

        if self.request.method == "GET":
            await self.managed_server.manager.handle_request(
                scope, patched_receive, send
            )
            return

        response_status = 200
        response_headers: dict[str, str] = {}
        body_chunks: list[bytes] = []
        response_bytes = 0
        response_too_large = False

        async def intercepting_send(message) -> None:
            nonlocal response_status, response_bytes, response_too_large
            if message["type"] == "http.response.start":
                response_status = message["status"]
                response_headers.update(
                    {
                        key.decode(): value.decode()
                        for key, value in message.get("headers", [])
                    }
                )
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                response_bytes += len(chunk)
                if response_bytes > self.max_response_bytes:
                    response_too_large = True
                    body_chunks.clear()
                elif not response_too_large:
                    body_chunks.append(chunk)

        await self.managed_server.manager.handle_request(
            scope, patched_receive, intercepting_send
        )
        if response_too_large:
            response = Response(content="MCP response is too large", status_code=502)
        else:
            response = Response(
                content=b"".join(body_chunks),
                status_code=response_status,
                headers=response_headers,
            )
        await response(scope, receive, send)


@router.get("/.well-known/oauth-protected-resource")
async def mcp_interaction_protected_resource_metadata_root(
    request: Request,
):
    """
    Root protected resource metadata endpoint for MCP client compatibility.
    """
    app: "HTTPApplication" = request.app
    auth_config = _get_first_enabled_mcp_auth_config(app)
    default_authorization_server, default_scopes_supported = _default_oauth_metadata(
        app
    )
    authorization_servers = (
        [auth_config.authorization_server]
        if auth_config is not None and auth_config.authorization_server is not None
        else default_authorization_server
    )
    scopes_supported = (
        auth_config.scopes_supported
        if auth_config is not None
        else default_scopes_supported
    )
    resource = (
        auth_config.protected_resource
        if auth_config is not None and auth_config.protected_resource is not None
        else str(request.base_url).rstrip("/")
    )
    return {
        "resource": resource,
        "scopes_supported": scopes_supported,
        "authorization_servers": authorization_servers,
    }


@router.get(
    "/.well-known/oauth-protected-resource/api/v1/agent/{agent_id}/session/{session}/mcp"
)
async def mcp_interaction_protected_resource_metadata(
    request: Request,
    agent_id: str,
    session: str,
):
    """
    Protected resource metadata discovery endpoint for MCP server authorization.
    See https://datatracker.ietf.org/doc/html/rfc9728 for details on the OAuth-protected resource metadata format and discovery process.
    """
    app: "HTTPApplication" = request.app
    mcp_url = request.url_for(
        "interaction_mcp_handler", agent_id=agent_id, session=session
    )
    force_https = force_https_metadata(app)
    auth_config = _get_mcp_auth_config(app, agent_id)
    resource = (
        auth_config.protected_resource
        if auth_config is not None and auth_config.protected_resource is not None
        else str(mcp_url.replace(scheme="https"))
        if force_https
        else str(mcp_url)
    )
    default_authorization_server, default_scopes_supported = _default_oauth_metadata(
        app
    )
    authorization_servers = (
        [auth_config.authorization_server]
        if auth_config is not None and auth_config.authorization_server is not None
        else default_authorization_server
    )
    scopes_supported = (
        auth_config.scopes_supported
        if auth_config is not None
        else default_scopes_supported
    )
    return {
        "resource": resource,
        "scopes_supported": scopes_supported,
        "authorization_servers": authorization_servers,
    }


@router.delete("/api/v1/agent/{agent_id}/session/{session}/mcp", tags=["MCP"])
@requires_one([AgentRole.MEMBER])
async def mcp_handler_delete(
    request: Request,
    agent_id: str,
    session: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    key = (x_stf_account, x_stf_user, x_stf_account_type, agent_id, session)
    await app.mcp_server_pool.remove(key)


@router.get("/api/v1/agent/{agent_id}/session/{session}/mcp", tags=["MCP"])
@router.post("/api/v1/agent/{agent_id}/session/{session}/mcp", tags=["MCP"])
@requires_one([AgentRole.MEMBER])
async def interaction_mcp_handler(
    request: Request,
    agent_id: str,
    session: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app
    runtime_settings = getattr(app, "_standalone_settings", app.settings)
    max_request_bytes = runtime_settings.mcp_max_request_bytes
    max_response_bytes = runtime_settings.mcp_max_response_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_request_bytes:
                raise HTTPException(status_code=413, detail="MCP request is too large")
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid Content-Length"
            ) from None

    body_bytes = bytearray()
    async for chunk in request.stream():
        if len(body_bytes) + len(chunk) > max_request_bytes:
            raise HTTPException(status_code=413, detail="MCP request is too large")
        body_bytes.extend(chunk)

    key: MCPServerKey = (
        x_stf_account,
        x_stf_user,
        x_stf_account_type,
        agent_id,
        session,
    )

    async def create_server() -> ManagedMCPServer:
        return await _create_mcp_server(
            app,
            request,
            agent_id,
            session,
            x_stf_account,
            max_request_bytes,
        )

    lease = await app.mcp_server_pool.acquire(
        key, request, bytes(body_bytes), create_server
    )
    return _MCPTransportResponse(request, lease, bytes(body_bytes), max_response_bytes)
