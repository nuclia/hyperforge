import json
from asyncio import gather
from collections.abc import MutableMapping
from dataclasses import dataclass
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import anyio
import pydantic_core
from fastapi import Header
from hyperforge.api import logger
from hyperforge.api.authentication import requires_one
from mcp.server.fastmcp.exceptions import ResourceError
from mcp.server.fastmcp.utilities.types import Image
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.lowlevel.server import Server as MCPServer
from mcp.server.lowlevel.server import lifespan as default_lifespan
from mcp.types import (
    Annotations,
    EmbeddedResource,
    GetPromptResult,
    ImageContent,
    Prompt,
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)
from nucliadb_sdk import NucliaDBAsync
from nucliadb_sdk.v2.exceptions import AccountLimitError, NotFoundError, RateLimitError
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, ValidationError
from starlette.requests import Request
from starlette.responses import Response

if TYPE_CHECKING:
    from hyperforge.api.app import HTTPApplication
from anyio.abc import TaskStatus
from hyperforge.api.models import (
    NucliaDBRoles,
)
from hyperforge.api.v1.router import router
from mcp.server.streamable_http import (
    StreamableHTTPServerTransport,
)
from mcp.server.transport_security import TransportSecuritySettings
from nucliadb_models.resource import (
    ConversationFieldData,
    FileFieldData,
    GenericFieldData,
    LinkFieldData,
    TextFieldData,
)
from nucliadb_models.search import (
    AskRequest,
    CitationsType,
    FieldExtensionStrategy,
    Filter,
    MetadataExtensionStrategy,
    MetadataExtensionType,
    NeighbouringParagraphsStrategy,
    NucliaDBClientType,
)

from nucliadb_agentic_api.ask.exceptions import (
    InvalidQueryError,
    KnowledgeBoxNotFound,
    NoRetrievalResultsError,
)
from nucliadb_agentic_api.ask.search.ask import AskResult, ask

BATCH_GET_DOCUMENTS_MAX = 20


@dataclass
class MCPContext:
    ndb_reader: NucliaDBAsync
    ndb_search: NucliaDBAsync
    kbid: str
    x_nucliadb_user: str
    x_ndb_client: NucliaDBClientType
    x_forwarded_for: str


class SearchDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        description="Required. The raw query string provided by the user.",
    )
    search_configuration: str | None = Field(
        None,
        description="Optional. The name of the Search Configuration to use. If not provided, default settings are used.",
    )
    filters: list[Filter] | None = Field(
        None,
        description="Optional. A list of filter objects to narrow results. Each filter can include 'all' (AND), 'any' (OR), and/or 'not_all' (NOT AND) label paths like '/l/labelset/label'.",
    )


class GetDocumentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        min_length=1,
        description="Required. The resource ID of the document to retrieve, as returned in the 'parent' field by the 'search_documents' tool.",
    )


class BatchGetDocumentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    names: list[str] = Field(
        ...,
        min_length=1,
        max_length=BATCH_GET_DOCUMENTS_MAX,
        description=f"Required. Resource IDs of up to {BATCH_GET_DOCUMENTS_MAX} documents to retrieve, as returned in the 'parent' field by the 'search_documents' tool.",
    )


@dataclass
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Any


async def get_resource(
    ndb_search: NucliaDBAsync, kbid: str, parent: str
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Get a resource by rid."""
    converted_result: list[TextContent | ImageContent | EmbeddedResource] = []
    try:
        resp = await ndb_search.get_resource_by_id(
            rid=parent, kbid=kbid, query_params={"show": ["basic", "extracted"]}
        )
    except NotFoundError as e:
        raise ResourceError(f"Document not found: '{parent}'") from e
    except Exception as e:
        logger.exception(f"Error fetching document '{parent}'")
        raise ResourceError(
            f"Failed to fetch document '{parent}'" + (f": {e}" if str(e) else "")
        ) from e
    if resp.data is not None:
        field_collections: list[
            dict[str, GenericFieldData]
            | dict[str, TextFieldData]
            | dict[str, FileFieldData]
            | dict[str, LinkFieldData]
            | dict[str, ConversationFieldData]
            | None,
        ] = [
            resp.data.generics,
            resp.data.texts,
            resp.data.files,
            resp.data.links,
            resp.data.conversations,
        ]

        for fields in field_collections:
            if fields is not None:
                for key, field in fields.items():
                    converted_result.append(
                        TextContent(
                            type="text",
                            text=field.extracted.text.text
                            if field.extracted
                            and field.extracted.text
                            and field.extracted.text.text
                            else "",
                            annotations=Annotations(audience=["user", "assistant"]),
                            _meta={"field_id": key},
                        )
                    )
    return converted_result


def _extract_resource_id(citation: str) -> str:
    resource_id = citation.split("/", 2)[0]
    return resource_id


def _get_document_ids(
    citation_footnote_to_context: dict[str, str] | None,
) -> dict[str, str]:
    if not citation_footnote_to_context:
        return {}

    return {
        ident: _extract_resource_id(citation)
        for ident, citation in citation_footnote_to_context.items()
    }


def _append_document_ids_to_answer(
    answer: str | None, document_ids: dict[str, str]
) -> str:
    answer_text = answer or ""
    if not document_ids:
        return answer_text

    document_ids_text = "\n".join(
        f"{ident}: {resource_id}" for ident, resource_id in document_ids.items()
    )
    return f"{answer_text}\n\nDocument IDs:\n{document_ids_text}"


async def _tool_search_documents(
    context: MCPContext,
    args: SearchDocumentsArgs,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    ask_request = AskRequest(query=args.query)
    ask_request.audit_metadata = {"source": "mcp_tool"}

    ask_request.rag_strategies = [
        FieldExtensionStrategy(fields=["a/title", "a/summary"]),
        NeighbouringParagraphsStrategy(before=5, after=5),
        MetadataExtensionStrategy(
            types=[
                MetadataExtensionType.CLASSIFICATION_LABELS,
                MetadataExtensionType.ORIGIN,
            ]
        ),
    ]

    if args.filters is not None:
        ask_request.filters = args.filters

    if args.search_configuration is not None:
        ask_request.search_configuration = args.search_configuration

    # citations does not work with rag strategies yet

    ask_request.generate_answer = True
    ask_request.citations = CitationsType.LLM_FOOTNOTES
    ask_request.debug = True
    try:
        ask_result: AskResult = await ask(
            search_sdk=context.ndb_search,
            reader_sdk=context.ndb_reader,
            kbid=context.kbid,
            ask_request=ask_request,
            user_id=context.x_nucliadb_user,
            client_type=context.x_ndb_client,
            origin=context.x_forwarded_for,
            extra_predict_headers={
                "X-Show-Consumption": "True",
            },
        )
    except KnowledgeBoxNotFound as e:
        raise ResourceError(f"Knowledge base '{context.kbid}' was not found") from e
    except InvalidQueryError as e:
        raise ResourceError(f"Invalid search query — {e}") from e
    except NoRetrievalResultsError as e:
        raise ResourceError("No documents matched the search query") from e
    except (RateLimitError, AccountLimitError) as e:
        raise ResourceError(
            "Service limit reached" + (f": {e}" if str(e) else "")
        ) from e
    except NotFoundError as e:
        raise ResourceError(
            "Resource not found during search" + (f": {e}" if str(e) else "")
        ) from e
    except Exception as e:
        logger.exception("Unexpected error in search_documents")
        raise ResourceError("Search failed" + (f": {e}" if str(e) else "")) from e

    resp = await ask_result.model_dump()
    document_ids = _get_document_ids(resp.citation_footnote_to_context)

    return _build_citation_blocks(resp) + [
        TextContent(
            type="text",
            text=_append_document_ids_to_answer(resp.answer, document_ids),
            annotations=Annotations(audience=["user", "assistant"]),
            _meta={
                "type": "answer",
                "document_ids": document_ids,
            },
        )
    ]


def _build_citation_blocks(
    resp: Any,
) -> list[TextContent | ImageContent | EmbeddedResource]:
    """Convert citation footnotes from an ask response into TextContent blocks."""
    result: list[TextContent | ImageContent | EmbeddedResource] = []
    if not resp.citation_footnote_to_context:
        return result

    for ident, citation in resp.citation_footnote_to_context.items():
        is_field: bool | None = None
        if citation.count("/") == 4:
            resource_id, field_type, field, split, positions = citation.split("/")
        elif citation.count("/") == 3:
            resource_id, field_type, field, positions = citation.split("/")
        elif citation.count("/") == 2:
            resource_id, field_type, field = citation.split("/")
            is_field = True
        else:
            raise ResourceError(
                f"Unexpected citation format for '{ident}'"
                f" (got {citation.count('/')} slashes): {citation!r}"
            )
        try:
            resource = resp.retrieval_results.resources[resource_id]
        except KeyError:
            logger.warning(
                f"Citation '{ident}' references unknown resource '{resource_id}', skipping"
            )
            continue

        meta = {
            "type": "citations",
            "paragraph_id": ident,
            "resource_labels": resource.computedmetadata.field_classifications
            if resource.computedmetadata
            else [],
            "resource_title": resource.title,
            "parent": resource_id,
            "field_type": field_type,
            "field": field,
        }

        if (
            is_field is True
            and resp.augmented_context is not None
            and citation in resp.augmented_context.fields
        ):
            block = resp.augmented_context.fields[citation]
            result.append(
                TextContent(
                    type="text",
                    text=block.text,
                    annotations=Annotations(audience=["user", "assistant"]),
                    _meta=meta,
                )
            )
        elif (
            resp.augmented_context is not None
            and citation in resp.augmented_context.paragraphs
        ):
            block = resp.augmented_context.paragraphs[citation]
            result.append(
                TextContent(
                    type="text",
                    text=block.text,
                    annotations=Annotations(audience=["user", "assistant"]),
                    _meta=meta,
                )
            )

        # if paragraph.reference:
        #     data = await ndb_search.get_image_data(paragraph.reference)
        #     result.append(
        #         ImageContent(
        #             type="image",
        #             mimeType="image/png",
        #             data=data,
        #             annotations=Annotations(audience=["user"]),
        #             _meta={
        #                 "type": "citations_image",
        #                 "paragraph_id": ident,
        #                 "resource_labels": resource.computedmetadata.field_classifications,
        #                 "resource_title": resource.title,
        #             },
        #         )
        #     )

    return result


async def _tool_get_document(
    context: MCPContext,
    args: GetDocumentArgs,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    return await get_resource(context.ndb_reader, context.kbid, args.name)


async def _tool_batch_get_documents(
    context: MCPContext,
    args: BatchGetDocumentsArgs,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    tasks = [
        get_resource(context.ndb_reader, context.kbid, name) for name in args.names
    ]
    results = await gather(*tasks, return_exceptions=True)
    converted_result: list[TextContent | ImageContent | EmbeddedResource] = []
    for doc_name, result in zip(args.names, results):
        if isinstance(result, BaseException) and not isinstance(result, Exception):
            raise result
        elif isinstance(result, Exception):
            logger.warning(f"Failed to fetch document '{doc_name}': {result}")
            converted_result.append(
                TextContent(
                    type="text",
                    text=f"[Error fetching '{doc_name}'"
                    + (f": {result}" if str(result) else "")
                    + "]",
                    annotations=Annotations(audience=["user", "assistant"]),
                )
            )
        else:
            converted_result.extend(result)
    return converted_result


_TOOL_DEFINITIONS: list[ToolDefinition] = [
    ToolDefinition(
        name="search_documents",
        description=(
            "Use this tool to find documents about {kb_description}. "
            "This tool returns chunks of text, names and URLs for matching documents. "
            "If the returned chunks are not detailed enough to answer the user's question "
            "use 'get_document' or 'batch_get_documents' with the 'parent' from this tool's "
            "output to retrieve the full document content. You can narrow results using "
            "optional 'filters' to match on classification labels or metadata values."
        ),
        args_model=SearchDocumentsArgs,
        handler=_tool_search_documents,
    ),
    ToolDefinition(
        name="get_document",
        description=(
            "Use this tool to get the full content of a single document. "
            "The document ID should be obtained from the 'parent' field "
            "of results from a call to the 'search_documents' tool. "
            "If you need to retrieve multiple documents use 'batch_get_documents' instead."
        ),
        args_model=GetDocumentArgs,
        handler=_tool_get_document,
    ),
    ToolDefinition(
        name="batch_get_documents",
        description=(
            f"Use this tool to retrieve the full content of up to {BATCH_GET_DOCUMENTS_MAX} "
            "documents in a single call. The document IDs should be obtained from the 'parent' "
            "field of results from a call to the 'search_documents' tool. "
            "If you only need to retrieve a single document use 'get_document' instead."
        ),
        args_model=BatchGetDocumentsArgs,
        handler=_tool_batch_get_documents,
    ),
]

_TOOL_REGISTRY: dict[str, ToolDefinition] = {td.name: td for td in _TOOL_DEFINITIONS}


async def list_tools(description: str) -> list[Tool]:
    return [
        Tool(
            name=td.name,
            description=td.description.format(kb_description=description),
            inputSchema=td.args_model.model_json_schema(),
        )
        for td in _TOOL_DEFINITIONS
    ]


async def call_tool(
    x_stf_account: str,
    x_nucliadb_user: str,
    x_ndb_client: NucliaDBClientType,
    x_forwarded_for: str,
    ndb_reader: NucliaDBAsync,
    ndb_search: NucliaDBAsync,
    kbid: str,
    name: str,
    arguments: dict[str, Any],
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Dispatch an MCP tool call to its handler."""
    if name not in _TOOL_REGISTRY:
        raise ResourceError(f"Unknown tool: {name!r}")
    tool_def = _TOOL_REGISTRY[name]
    try:
        args = tool_def.args_model.model_validate(arguments)
    except ValidationError as e:
        error_details = "; ".join(
            f"'{'.'.join(str(p) for p in err['loc'])}': {err['msg']}"
            for err in e.errors()
        )
        raise ResourceError(f"Invalid arguments for '{name}': {error_details}") from e
    context = MCPContext(
        ndb_reader=ndb_reader,
        ndb_search=ndb_search,
        kbid=kbid,
        x_nucliadb_user=x_nucliadb_user,
        x_ndb_client=x_ndb_client,
        x_forwarded_for=x_forwarded_for,
    )
    return await tool_def.handler(context, args)


async def list_resources(ndb: NucliaDBAsync, kbid: str) -> list[Resource]:
    resources = await ndb.list_resources(kbid=kbid)

    # TODO : Resource 1 : The list of tools ??
    return [
        Resource(
            uri=AnyUrl(f"nucliadb://{kbid}/{resource.id}"),
            name=resource.title if resource.title is not None else "",
            description=resource.summary,
            mimeType=resource.icon,
            size=0,
        )
        for resource in resources.resources
    ]


async def list_resource_templates() -> list[ResourceTemplate]:
    return []


async def list_prompts() -> list[Prompt]:
    """List all available prompts."""
    return []


async def get_prompt(
    name: str, arguments: dict[str, Any] | None = None
) -> GetPromptResult:
    """Get a prompt by name with arguments."""
    raise ResourceError(f"Unknown prompt: {name!r}")


async def read_resource(
    ndb: NucliaDBAsync, kbid: str, uri: AnyUrl | str
) -> Iterable[ReadResourceContents]:
    """Read a resource by URI."""
    str_uri = str(uri)
    if str_uri.startswith("nucliadb://"):
        str_uri = str_uri.replace("nucliadb://", "")
        parts = str_uri.split("/")
        if len(parts) != 2:
            raise ResourceError(
                f"Malformed resource URI, expected 'nucliadb://{{kbid}}/{{rid}}': {uri!r}"
            )
        kbid_uri, rid = parts
        try:
            resource = await ndb.get_resource_by_id(
                kbid=kbid,
                rid=rid,
                query_params={"show": ["basic", "extracted"], "extracted": ["text"]},
            )
        except NotFoundError as e:
            raise ResourceError(f"Resource not found: {uri!r}") from e
        except Exception as e:
            logger.exception(f"Error fetching resource {uri!r}")
            raise ResourceError(
                f"Failed to fetch resource {uri!r}" + (f": {e}" if str(e) else "")
            ) from e

        if not resource:
            raise ResourceError(f"Resource not found: {uri!r}")

        try:
            content = ""
            if resource.data:
                if resource.data.files:
                    for file_field in resource.data.files.values():
                        if (
                            file_field.extracted
                            and file_field.extracted.text
                            and file_field.extracted.text.text
                        ):
                            content += file_field.extracted.text.text
                            content += " \n\n "
                if resource.data.links:
                    for link_field in resource.data.links.values():
                        if (
                            link_field.extracted
                            and link_field.extracted.text
                            and link_field.extracted.text.text
                        ):
                            content += link_field.extracted.text.text
                            content += " \n\n "
                if resource.data.texts:
                    for text_field in resource.data.texts.values():
                        if (
                            text_field.extracted
                            and text_field.extracted.text
                            and text_field.extracted.text.text
                        ):
                            content += text_field.extracted.text.text
                            content += " \n\n "
                if resource.data.conversations:
                    for conversation_field in resource.data.conversations.values():
                        if (
                            conversation_field.extracted
                            and conversation_field.extracted.text
                        ):
                            if conversation_field.extracted.text.split_text:
                                for split_text in conversation_field.extracted.text.split_text.values():
                                    content += split_text
                                    content += " \n\n "
                            if conversation_field.extracted.text.text:
                                content += conversation_field.extracted.text.text
                                content += " \n\n "
            return [ReadResourceContents(content=content, mime_type=resource.icon)]
        except ResourceError:
            raise
        except Exception as e:
            logger.exception(f"Error reading resource {uri!r}")
            raise ResourceError(
                f"Failed to read resource {uri!r}" + (f": {e}" if str(e) else "")
            ) from e
    else:
        raise ResourceError(
            f"Unknown resource URI scheme (expected 'nucliadb://'): {uri!r}"
        )


@router.get("/.well-known/oauth-protected-resource/api/v1/kb/{kbid}/mcp")
async def mcp_protected_resource_metadata(
    request: Request,
    kbid: str,
):
    """
    Protected resource metadata discovery endpoint for MCP server authorization.
    See https://datatracker.ietf.org/doc/html/rfc9728 for details on the OAuth-protected resource metadata format and discovery process.
    """
    app: HTTPApplication = request.app
    mcp_url = request.url_for("mcp_handler", kbid=kbid)
    mcp_url_https = str(mcp_url).replace(
        "http://", "https://"
    )  # Ensure the URL uses https
    return {
        "resource": mcp_url_https,
        "scopes_supported": app.settings.hydra_scopes_supported,
        "authorization_servers": [app.settings.hydra_public_url],
    }


@router.get("/api/v1/kb/{kbid}/mcp")
@router.post("/api/v1/kb/{kbid}/mcp")
@requires_one([NucliaDBRoles.READER])
async def mcp_handler(
    request: Request,
    kbid: str,
    x_stf_user: str = Header(..., include_in_schema=False),
    x_ndb_client: NucliaDBClientType = Header(NucliaDBClientType.API),
    x_nucliadb_user: str = Header(""),
    x_forwarded_for: str = Header(""),
    x_stf_account: str = Header(..., include_in_schema=False),
    x_stf_account_type: str = Header(..., include_in_schema=False),
):
    app: HTTPApplication = request.app

    logger.debug("Stateless mode: Creating new transport for this request")
    # No session ID needed in stateless mode
    security_settings: TransportSecuritySettings | None = None
    http_transport = StreamableHTTPServerTransport(
        mcp_session_id=None,  # No session tracking in stateless mode
        is_json_response_enabled=True,
        event_store=None,  # No event store in stateless mode
        security_settings=security_settings,
    )

    if kbid not in app.mcp_servers:
        mcp_server = MCPServer(
            name=kbid,
            instructions="Instructions for the MCP server",
            lifespan=default_lifespan,
        )
        ndb_reader: NucliaDBAsync = app.arag_reader
        ndb_search: NucliaDBAsync = app.arag_search
        list_resources_partial = partial(list_resources, ndb_reader, kbid)
        call_tool_partial = partial(
            call_tool,
            x_stf_account,
            x_nucliadb_user,
            x_ndb_client,
            x_forwarded_for,
            ndb_reader,
            ndb_search,
            kbid,
        )
        # Ensure MCP framework calls call_tool_partial with name and arguments as positional arguments
        # If not, use: partial(call_tool, x_stf_account, ndb_reader, ndb_search, kbid, name=..., arguments=...)
        read_resource_partial = partial(read_resource, ndb_reader, kbid)
        list_tools_partial = partial(
            list_tools, description=f"the knowledge in the {kbid} knowledge base"
        )

        mcp_server.list_tools()(list_tools_partial)
        mcp_server.call_tool()(call_tool_partial)
        mcp_server.list_resources()(list_resources_partial)
        mcp_server.read_resource()(read_resource_partial)
        mcp_server.list_prompts()(list_prompts)
        mcp_server.get_prompt()(get_prompt)
        mcp_server.list_resource_templates()(list_resource_templates)

        app.mcp_servers[kbid] = mcp_server

    mcp_server = app.mcp_servers[kbid]

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

    # Assert task group is not None for type checking
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


def _convert_to_content(
    result: Any,
) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
    """Convert a result to a sequence of content objects."""
    if result is None:
        return []

    if isinstance(result, (TextContent, ImageContent, EmbeddedResource)):
        return [result]

    if isinstance(result, Image):
        return [result.to_image_content()]

    if isinstance(result, (list, tuple)):
        return list(chain.from_iterable(_convert_to_content(item) for item in result))

    if not isinstance(result, str):
        try:
            result = json.dumps(pydantic_core.to_jsonable_python(result))
        except Exception:
            result = str(result)

    return [TextContent(type="text", text=result)]
