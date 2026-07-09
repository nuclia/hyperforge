import asyncio
from collections.abc import MutableMapping
from functools import partial
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import anyio
from fastapi import Header
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.lowlevel.server import lifespan as default_lifespan
from mcp.server.streamable_http import (
    MCP_SESSION_ID_HEADER,
    StreamableHTTPServerTransport,
)
from mcp.server.transport_security import TransportSecuritySettings
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
from hyperforge.api.models import InteractionRequest
from hyperforge.api.v1.interaction import WebsocketReceiver, stream_response
from hyperforge.db.agents import AgentManager
from hyperforge.interaction import AnswerOperation
from hyperforge.prompts import PromptConfig
from hyperforge.pubsub import UserToAgentInteraction
from hyperforge.standalone.oauth import get_enabled_mcp_auth
from hyperforge.workflows import WorkflowData

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication
from anyio.abc import TaskStatus

from hyperforge.api import logger
from hyperforge.api.models import (
    AgentRole,
)
from hyperforge.api.v1.mcp_content import convert_arag_answer_to_content
from hyperforge.api.v1.router import router


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
    interaction_headers = _prepare_interaction_headers(app, agent_id, headers)

    interaction = InteractionRequest(
        question=question, headers=interaction_headers, arguments=arguments
    )
    mcp_session = mcp_server.request_context.session
    websocket = WebsocketReceiver(websocket=None)

    messages = []
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
            pass
        elif msg.operation == AnswerOperation.AGENT_REQUEST and msg.feedback:
            # DO NOTHING FOR NOW
            result = await mcp_session.elicit_form(
                message=msg.feedback.question,
                requestedSchema=msg.feedback.response_schema,
                related_request_id=msg.feedback.request_id,
            )
            websocket.queue.put_nowait(
                UserToAgentInteraction(
                    request_id=msg.feedback.request_id, response=result.content
                )
            )
        elif msg.operation == AnswerOperation.ANSWER:
            result_contents = convert_arag_answer_to_content(msg)
            for content in result_contents:
                if isinstance(content, TextContent):
                    logger.debug(f"Tool output text: {content.text}")
                messages.append(content)

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
    auth_config = get_enabled_mcp_auth(getattr(app, "_agents_cfg", {}), agent_id)

    authorization = headers.get("authorization")
    if auth_config is not None and not auth_config.forward_authorization_header:
        interaction_headers.pop("authorization", None)
        interaction_headers.pop("Authorization", None)
    elif authorization is not None:
        interaction_headers["authorization"] = authorization

    return interaction_headers


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
    auth_config = get_enabled_mcp_auth(getattr(app, "_agents_cfg", {}), agent_id)
    resource = (
        auth_config.protected_resource
        if auth_config is not None and auth_config.protected_resource is not None
        else str(mcp_url.replace(scheme="https"))
    )
    authorization_servers = (
        [auth_config.authorization_server]
        if auth_config is not None and auth_config.authorization_server is not None
        else [app.settings.hydra_public_url]
    )
    scopes_supported = (
        auth_config.scopes_supported
        if auth_config is not None
        else app.settings.hydra_scopes_supported
    )
    return {
        "resource": resource,
        "scopes_supported": scopes_supported,
        "authorization_servers": authorization_servers,
    }


@router.delete("/api/v1/agent/{agent_id}/session/{session}/mcp")
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
    if (agent_id, session) in app.sses:
        await app.sses[(agent_id, session)].terminate()
    if (agent_id, session) in app.sses:
        del app.sses[(agent_id, session)]


@router.get("/api/v1/agent/{agent_id}/session/{session}/mcp")
@router.post("/api/v1/agent/{agent_id}/session/{session}/mcp")
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
    agent_manager: AgentManager = request.app.agent_manager
    request._headers._list.append((MCP_SESSION_ID_HEADER.encode(), session.encode()))

    # No session ID needed in stateless mode
    security_settings: TransportSecuritySettings | None = None
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=session,  # No session tracking in stateless mode
        is_json_response_enabled=True,
        event_store=None,  # No event store in stateless mode
        security_settings=security_settings,
    )

    workflows, agent_config, prompts = await asyncio.gather(
        agent_manager.workflows_list(account=x_stf_account, agent_id=agent_id),
        agent_manager.get_agent_config_basic(account=x_stf_account, agent_id=agent_id),
        agent_manager.get_prompts(account=x_stf_account, agent_id=agent_id),
    )

    mcp_server = MCPServer(
        name=agent_id,
        version="1.0.0",
        instructions=agent_config.instructions,
        lifespan=default_lifespan,
    )

    list_tools_partial = partial(list_tools, workflows)
    mcp_server.list_tools()(list_tools_partial)

    call_tool_partial = partial(
        call_tool,
        app,
        mcp_server,
        x_stf_account,
        agent_id,
        session,
        workflows,
        request.headers,
    )
    mcp_server.call_tool()(call_tool_partial)

    list_prompts_partial = partial(list_prompts, prompts=prompts)
    mcp_server.list_prompts()(list_prompts_partial)

    get_prompt_partial = partial(get_prompt, prompts)
    mcp_server.get_prompt()(get_prompt_partial)

    # Start server in a new task
    async def run_stateless_server(
        *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED
    ):
        async with http_transport.connect() as streams:
            read_stream, write_stream = streams
            task_status.started()
            try:
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                    stateless=True,
                )
            except Exception:
                logger.exception("Stateless session crashed")

    # Intercept ASGI send messages so FastAPI doesn't attempt to send a
    # second response after the transport has already sent one (which would
    # cause: RuntimeError: Unexpected ASGI message 'http.response.start'
    # sent, after response already completed).
    response_status = 200
    response_headers: dict[str, str] = {}
    body_chunks: list[bytes] = []

    async def intercepting_send(message: MutableMapping[str, Any]) -> None:
        nonlocal response_status
        if message["type"] == "http.response.start":
            response_status = message["status"]
            response_headers.update(
                {k.decode(): v.decode() for k, v in message.get("headers", [])}
            )
        elif message["type"] == "http.response.body":
            body_chunks.append(message.get("body", b""))

    async with anyio.create_task_group() as tg:
        # Start the server task
        await tg.start(run_stateless_server)

        # Handle the HTTP request via the intercepting send
        await http_transport.handle_request(
            request.scope, request._receive, intercepting_send
        )

        # Terminate the transport after the request is handled
        await http_transport.terminate()

    return Response(
        content=b"".join(body_chunks),
        status_code=response_status,
        headers=response_headers,
    )
