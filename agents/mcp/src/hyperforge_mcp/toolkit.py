from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from contextlib import asynccontextmanager
from datetime import timedelta
from functools import partial
from math import ceil
from types import GenericAlias
from typing import Any, Literal, cast
from uuid import uuid4

import mcp.shared.exceptions as mcp_exceptions
import mcp.types as types
from hyperforge.configure import toolkit
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessContextType,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    ToolkitRuntime,
    tool,
)
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.manager import Manager, build_reasoning
from hyperforge.memory.memory import BaseSessionMemory
from hyperforge.models import Chunk, Context, MemoryConfig, Rules
from hyperforge.result_payload import (
    BYTES_PER_KB,
    ResultPayloadSettings,
    budget_from_config,
)
from hyperforge.toolkits.common import (
    RetrievalOutput,
    resolve_manager,
    retrieval_context_factory,
)
from mcp import ClientSession, CreateMessageResult, ErrorData
from mcp.shared.context import RequestContext
from nuclia.lib.nua_responses import Author, ChatModel, Image, Message
from pydantic import BaseModel, Field, FileUrl, create_model

from .config import Transport
from .http import MCPHTTPDriver
from .schema import IncompatibleToolSchema, normalize_tool_schema
from .stdio import MCPStdioDriver

logger = logging.getLogger(__name__)
_DEFAULT_BUDGET = ResultPayloadSettings()


def _safe_identifier(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw).strip("_")
    if not cleaned:
        return "item"
    if cleaned[0].isdigit():
        return f"n_{cleaned}"
    return cleaned


def _json_schema_python_type(schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    if kind == "string":
        return str
    if kind == "integer":
        return int
    if kind == "number":
        return float
    if kind == "boolean":
        return bool
    if kind == "array":
        items = schema.get("items")
        inner = _json_schema_python_type(items) if isinstance(items, dict) else Any
        return GenericAlias(list, inner)
    if kind == "object":
        return dict[str, Any]
    return Any


def _build_mcp_input_model(
    tool_name: str, input_schema: dict[str, Any]
) -> type[BaseModel]:
    properties = input_schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = set(input_schema.get("required") or [])
    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        spec = prop_schema if isinstance(prop_schema, dict) else {}
        py_type = _json_schema_python_type(spec)
        if prop_name in required:
            default: Any = ...
        else:
            default = spec.get("default", None)
            py_type = py_type | None
        fields[prop_name] = (
            py_type,
            Field(default=default, description=spec.get("description")),
        )
    model_name = f"MCP{_safe_identifier(tool_name).title().replace('_', '')}Input"
    return create_model(model_name, **fields)


class ReadResourceInput(BaseModel):
    uri: str = Field(description="Exact resource URI advertised by the MCP server.")


class GetMCPPromptInput(BaseModel):
    name: str = Field(description="Exact MCP prompt name.")
    arguments: dict[str, str] = Field(
        default_factory=dict,
        description="String arguments passed to the MCP prompt.",
    )


class MCPToolkitConfig(ToolkitConfig):
    """Configuration for `MCPToolkit` (pure MCP tool/prompt/resource provider)."""

    toolkit: Literal["mcp"] = "mcp"
    source: str = Field(
        ...,
        title="MCP driver identifier",
        description="Identifier of the configured MCP HTTP or STDIO driver.",
    )
    transport: Transport = Field(default=Transport.HTTP)
    valid_headers: list[str] = Field(default_factory=list)
    read_timeout_seconds: int = Field(default=400, ge=1)
    roots: dict[str, str] = Field(default_factory=dict)
    feedback_timeout_ms: int = Field(default=60_000, ge=100)
    sampling_model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.fast),
        title="MCP sampling model",
    )
    expose_prompts_as_skills: bool = True
    expose_prompt_tool: bool = True
    max_tool_result_kb: int = Field(
        default=ceil(_DEFAULT_BUDGET.max_bytes / BYTES_PER_KB),
        ge=1,
    )
    max_tool_result_item_kb: int = Field(
        default=ceil(_DEFAULT_BUDGET.max_item_bytes / BYTES_PER_KB),
        ge=1,
    )


@toolkit(
    id="mcp",
    title="Model Context Protocol (MCP) Toolkit",
    description=(
        "Expose remote or local MCP server tools, resources, and prompts directly "
        "to AgentHarness and Code Mode without an internal agent loop."
    ),
    config_schema=MCPToolkitConfig,
)
class MCPToolkit(BaseToolkit[MCPToolkitConfig]):
    """Pure MCP connector toolkit.

    Separates MCP protocol transport, tool execution, schema normalization,
    elicitation/sampling callbacks, and dynamic skill ingestion from any
    agentic loop so every MCP tool can be called directly in both `AgentHarness`
    and Code Mode.
    """

    toolkit_id = "mcp"

    def __init__(
        self, config: MCPToolkitConfig, *, namespace: str | None = None
    ) -> None:
        super().__init__(config, namespace=namespace)
        self.tools: list[types.Tool] = []
        self.prompts: list[types.Prompt] = []
        self.resources: list[types.Resource] = []
        self.server_instructions: str = ""
        self._preloaded = False

    def _forwarded_headers(self, execution_context: Any) -> dict[str, str]:
        raw_headers: dict[str, str] = {}
        if isinstance(execution_context, dict):
            mem = execution_context.get("memory")
            if mem is not None and hasattr(mem, "headers"):
                raw_headers.update(mem.headers)
            ctx_headers = execution_context.get("headers")
            if isinstance(ctx_headers, dict):
                raw_headers.update(ctx_headers)
        return {
            h: raw_headers[h] for h in self.config.valid_headers if h in raw_headers
        }

    def _compat_question_memory(self, execution_context: Any) -> Any:
        if isinstance(execution_context, dict):
            mem = execution_context.get("memory")
            if mem is not None:
                return mem
        session = BaseSessionMemory.from_config(
            MemoryConfig(),
            agent_id=self.config.id,
            workflow_id="default",
            rules=Rules(),
        )
        session.init("mcp-toolkit")
        return session.start_question("mcp", headers={})

    async def _sampling_callback(
        self,
        manager: Manager,
        context: RequestContext[ClientSession, Any, Any],
        params: types.CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        del context
        messages: list[Message] = []
        images: dict[str, Image] = {}
        for msg in params.messages:
            if isinstance(msg.content, types.ImageContent):
                images[uuid4().hex] = Image(
                    b64encoded=msg.content.data,
                    content_type=msg.content.mimeType,
                )
            elif isinstance(msg.content, types.TextContent):
                messages.append(
                    Message(
                        author=(
                            Author.NUCLIA if msg.role == "assistant" else Author.USER
                        ),
                        text=msg.content.text,
                    )
                )
        item = ChatModel(
            question="",
            user_id=self.config.id,
            query_context_images=images,
            generative_model=self.config.sampling_model.model_id,
            reasoning=build_reasoning(self.config.sampling_model),
            format_prompt=False,
            system=params.systemPrompt,
            context=messages,
            max_tokens=params.maxTokens,
        )
        resp, _, _ = await manager.execute_raw(item)
        return CreateMessageResult(
            role="user",
            content=types.TextContent(type="text", text=resp.answer or ""),
            model=self.config.sampling_model.model_id,
        )

    async def _elicitation_callback(
        self,
        call_context: ToolCallContext | None,
        context: RequestContext[ClientSession, Any, Any],
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | ErrorData:
        del context
        if call_context is None:
            return ErrorData(
                code=400, message="Interactive elicitation unavailable during preload"
            )
        if isinstance(params, types.ElicitRequestFormParams):
            response = await call_context.harness.request_feedback(
                question=params.message,
                response_schema=params.requestedSchema,
                timeout_ms=self.config.feedback_timeout_ms,
            )
            if response is not None:
                parsed = (
                    json.loads(response)
                    if response.startswith("{")
                    else {"value": response}
                )
                return types.ElicitResult(action="accept", content=parsed)
            return ErrorData(code=408, message="No feedback received")
        return ErrorData(code=400, message="URL mode elicitation is not supported")

    async def _list_roots_callback(
        self, context: RequestContext[ClientSession, Any, Any]
    ) -> types.ListRootsResult | ErrorData:
        del context
        return types.ListRootsResult(
            roots=[
                types.Root(name=key, uri=FileUrl(url=value))
                for key, value in self.config.roots.items()
            ]
        )

    @asynccontextmanager
    async def _open_session(
        self,
        drivers: Any,
        manager: Manager,
        execution_context: Any,
        call_context: ToolCallContext | None = None,
    ):
        driver = drivers.get(self.config.source) if isinstance(drivers, dict) else None
        if driver is None:
            raise ValueError(f"MCP driver {self.config.source!r} is not configured")

        timeout = timedelta(seconds=self.config.read_timeout_seconds)
        client_info = types.Implementation(
            name="Hyperforge Harness",
            title="Hyperforge MCP Toolkit",
            version="1.0.0",
        )
        if self.config.transport == Transport.HTTP:
            http_driver = cast(MCPHTTPDriver, driver)
            compat_mem = self._compat_question_memory(execution_context)
            headers = self._forwarded_headers(execution_context)
            async with http_driver.client(
                headers=headers,
                memory=compat_mem,
                module="mcp",
                agent_id=self.config.id,
                request_id=compat_mem.get_session_id(),
            ) as (read_stream, write_stream, _):
                if read_stream is None or write_stream is None:
                    raise RuntimeError("MCP HTTP streams unavailable")
                session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timeout,
                    sampling_callback=partial(self._sampling_callback, manager),
                    elicitation_callback=partial(
                        self._elicitation_callback, call_context
                    ),
                    list_roots_callback=self._list_roots_callback,
                    client_info=client_info,
                )
                async with session:
                    init_res = await session.initialize()
                    yield session, init_res
        else:
            stdio_driver = cast(MCPStdioDriver, driver)
            async with stdio_driver.client() as (read_stream, write_stream):
                if read_stream is None or write_stream is None:
                    raise RuntimeError("MCP STDIO streams unavailable")
                session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timeout,
                    sampling_callback=partial(self._sampling_callback, manager),
                    elicitation_callback=partial(
                        self._elicitation_callback, call_context
                    ),
                    list_roots_callback=self._list_roots_callback,
                    client_info=client_info,
                )
                async with session:
                    init_res = await session.initialize()
                    yield session, init_res

    def _filter_compatible_tools(
        self, raw_tools: Sequence[types.Tool]
    ) -> list[types.Tool]:
        compatible: list[types.Tool] = []
        for item in raw_tools:
            try:
                schema = normalize_tool_schema(item.inputSchema)
            except IncompatibleToolSchema as exc:
                logger.warning(
                    "Ignoring incompatible MCP tool: server=%r tool=%r path=%r reason=%r",
                    self.config.source,
                    item.name,
                    exc.json_path,
                    exc.reason,
                )
                continue
            compatible.append(item.model_copy(update={"inputSchema": schema}))
        return compatible

    async def preload(self, runtime: ToolkitRuntime) -> None:
        if self._preloaded:
            return
        manager = Manager()
        if runtime.nua_client is not None:
            manager.nua = runtime.nua_client
        manager.drivers.update(runtime.drivers)

        try:
            async with self._open_session(
                runtime.drivers, manager, dict(runtime.execution_context)
            ) as (session, init_res):
                self.server_instructions = str(
                    getattr(init_res, "instructions", None) or ""
                )
                try:
                    tools_resp = await session.list_tools()
                    self.tools = self._filter_compatible_tools(tools_resp.tools)
                    while tools_resp.nextCursor:
                        tools_resp = await session.list_tools(
                            params=types.PaginatedRequestParams(
                                cursor=tools_resp.nextCursor
                            )
                        )
                        self.tools.extend(
                            self._filter_compatible_tools(tools_resp.tools)
                        )
                except mcp_exceptions.McpError:
                    self.tools = []

                try:
                    prompts_resp = await session.list_prompts()
                    self.prompts = list(prompts_resp.prompts)
                    while prompts_resp.nextCursor:
                        prompts_resp = await session.list_prompts(
                            params=types.PaginatedRequestParams(
                                cursor=prompts_resp.nextCursor
                            )
                        )
                        self.prompts.extend(prompts_resp.prompts)
                except mcp_exceptions.McpError:
                    self.prompts = []

                try:
                    res_resp = await session.list_resources()
                    self.resources = list(res_resp.resources)
                    while res_resp.nextCursor:
                        res_resp = await session.list_resources(
                            params=types.PaginatedRequestParams(
                                cursor=res_resp.nextCursor
                            )
                        )
                        self.resources.extend(res_resp.resources)
                except mcp_exceptions.McpError:
                    self.resources = []
        except Exception as exc:
            logger.warning(
                "MCPToolkit %r could not preload from %r: %s",
                self.config.id,
                self.config.source,
                exc,
            )
        self._preloaded = True

    def _build_remote_tool(
        self, mcp_tool: types.Tool
    ) -> HarnessTool[Any, RetrievalOutput]:
        input_model = _build_mcp_input_model(mcp_tool.name, mcp_tool.inputSchema)
        budget = budget_from_config(self.config)
        remote_name = mcp_tool.name

        async def execute(
            call_context: ToolCallContext, input_value: BaseModel
        ) -> RetrievalOutput:
            manager = resolve_manager(call_context)
            arguments = input_value.model_dump(exclude_unset=True)
            ctx = Context(
                agent_id=self.config.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=remote_name,
                source=self.config.source,
                agent="mcp",
                title=f"MCP: {remote_name}",
            )
            async with self._open_session(
                manager.drivers,
                manager,
                call_context.harness.execution_context,
                call_context=call_context,
            ) as (session, _):
                result = await session.call_tool(
                    name=remote_name,
                    arguments=arguments,
                    read_timeout_seconds=timedelta(
                        seconds=self.config.read_timeout_seconds
                    ),
                )
                if result.isError:
                    err_texts = [
                        b.text
                        for b in result.content
                        if isinstance(b, types.TextContent)
                    ]
                    raise RuntimeError(
                        "; ".join(err_texts)
                        if err_texts
                        else f"MCP tool {remote_name} failed"
                    )
                if result.structuredContent is not None:
                    ctx.structured.append(
                        json.dumps(result.structuredContent, default=str)
                    )
                for idx, block in enumerate(result.content):
                    if isinstance(block, types.TextContent):
                        ctx.chunks.append(
                            Chunk(
                                chunk_id=f"mcp_{remote_name}_{idx}_{uuid4().hex[:6]}",
                                title=remote_name,
                                text=block.text,
                                metadata=block.meta,
                                origin_agent="mcp",
                            )
                        )
                    elif isinstance(block, types.ImageContent):
                        ctx.images[f"mcp_{remote_name}_{idx}"] = Image(
                            content_type=block.mimeType,
                            b64encoded=block.data,
                        )
                    elif isinstance(block, types.ResourceLink):
                        linked = await session.read_resource(block.uri)
                        for r_idx, content in enumerate(linked.contents):
                            if isinstance(content, types.TextResourceContents):
                                ctx.chunks.append(
                                    Chunk(
                                        chunk_id=f"mcp_{remote_name}_res_{r_idx}",
                                        title=remote_name,
                                        text=content.text,
                                        url=[str(content.uri)] if content.uri else [],
                                        origin_agent="mcp",
                                    )
                                )
                    elif isinstance(block, types.EmbeddedResource) and isinstance(
                        block.resource, types.TextResourceContents
                    ):
                        ctx.chunks.append(
                            Chunk(
                                chunk_id=f"mcp_{remote_name}_emb_{idx}",
                                title=remote_name,
                                text=block.resource.text,
                                url=(
                                    [str(block.resource.uri)]
                                    if block.resource.uri
                                    else []
                                ),
                                origin_agent="mcp",
                            )
                        )
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        execute.__name__ = _safe_identifier(remote_name)
        execute.__annotations__["input_value"] = input_model
        return HarnessTool(
            name=_safe_identifier(remote_name),
            description=mcp_tool.description or remote_name,
            handler=execute,
            context_type=HarnessContextType.RETRIEVAL,
            parameters_schema=mcp_tool.inputSchema,
            context_factory=retrieval_context_factory,
            payload_budget=budget,
        )

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        built: list[HarnessTool[Any, Any]] = [
            self._build_remote_tool(item) for item in self.tools
        ]
        budget = budget_from_config(self.config)

        if self.resources:

            @tool(
                name="read_mcp_resource",
                description="Read a resource exposed by the MCP server by URI.",
                context_type=HarnessContextType.RETRIEVAL,
                context_factory=retrieval_context_factory,
                payload_budget=budget,
            )
            async def read_mcp_resource(
                call_context: ToolCallContext, value: ReadResourceInput
            ) -> RetrievalOutput:
                manager = resolve_manager(call_context)
                ctx = Context(
                    agent_id=self.config.id,
                    original_question_uuid=None,
                    actual_question_uuid=None,
                    question=value.uri,
                    source=self.config.source,
                    agent="mcp",
                    title=f"MCP Resource: {value.uri}",
                )
                async with self._open_session(
                    manager.drivers,
                    manager,
                    call_context.harness.execution_context,
                    call_context=call_context,
                ) as (session, _):
                    res = await session.read_resource(
                        FileUrl(url=value.uri)
                        if value.uri.startswith("file://")
                        else value.uri
                    )  # type: ignore[arg-type]
                    for idx, content in enumerate(res.contents):
                        if isinstance(content, types.TextResourceContents):
                            ctx.chunks.append(
                                Chunk(
                                    chunk_id=f"mcp_res_{idx}",
                                    title=value.uri,
                                    text=content.text,
                                    url=[str(content.uri)] if content.uri else [],
                                    origin_agent="mcp",
                                )
                            )
                return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

            built.append(read_mcp_resource)

        if self.prompts and self.config.expose_prompt_tool:

            @tool(
                name="get_mcp_prompt",
                description="Fetch and render a named prompt template from the MCP server.",
                context_type=HarnessContextType.RETRIEVAL,
                context_factory=retrieval_context_factory,
                payload_budget=budget,
            )
            async def get_mcp_prompt(
                call_context: ToolCallContext, value: GetMCPPromptInput
            ) -> RetrievalOutput:
                manager = resolve_manager(call_context)
                ctx = Context(
                    agent_id=self.config.id,
                    original_question_uuid=None,
                    actual_question_uuid=None,
                    question=value.name,
                    source=self.config.source,
                    agent="mcp",
                    title=f"MCP Prompt: {value.name}",
                )
                async with self._open_session(
                    manager.drivers,
                    manager,
                    call_context.harness.execution_context,
                    call_context=call_context,
                ) as (session, _):
                    prompt_res = await session.get_prompt(
                        name=value.name,
                        arguments=value.arguments or None,
                    )
                    for idx, msg in enumerate(prompt_res.messages):
                        if isinstance(msg.content, types.TextContent):
                            ctx.chunks.append(
                                Chunk(
                                    chunk_id=f"mcp_prompt_{value.name}_{idx}",
                                    title=value.name,
                                    text=msg.content.text,
                                    origin_agent="mcp",
                                )
                            )
                return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

            built.append(get_mcp_prompt)

        return built

    def build_skills(self) -> Sequence[HarnessSkill]:
        skills: list[HarnessSkill] = []
        tool_names = tuple(_safe_identifier(t.name) for t in self.tools)
        summary_lines = [
            f"Connected MCP server `{self.config.source}` provides external tools."
        ]
        if self.server_instructions.strip():
            summary_lines.append(
                f"Server instructions:\n{self.server_instructions.strip()}"
            )
        if self.resources:
            uris = ", ".join(str(r.uri) for r in self.resources[:10])
            summary_lines.append(
                f"Available resources (read via `{{namespace_prefix}}read_mcp_resource`): {uris}"
            )
        if self.tools or self.server_instructions or self.resources:
            skills.append(
                HarnessSkill(
                    id="mcp_connector_guide",
                    name=f"MCP Connector ({self.config.source})",
                    description=(
                        f"Guidance and server instructions for the {self.config.source} MCP connector."
                    ),
                    instructions="\n\n".join(summary_lines),
                    recommended_tools=tool_names[:10],
                )
            )

        if self.config.expose_prompts_as_skills:
            for prompt in self.prompts:
                arg_specs = ", ".join(
                    f"`{a.name}` ({'required' if a.required else 'optional'}: {a.description or ''})"
                    for a in (prompt.arguments or [])
                )
                body = (
                    f"MCP Server Prompt `{prompt.name}`: "
                    f"{prompt.description or 'Server-provided workflow prompt.'}"
                )
                if arg_specs:
                    body += f"\nArguments: {arg_specs}"
                if self.config.expose_prompt_tool:
                    body += (
                        f"\nFetch the rendered prompt messages via "
                        f"`{{namespace_prefix}}get_mcp_prompt` with `name={prompt.name!r}`."
                    )
                skills.append(
                    HarnessSkill(
                        id=f"prompt_{_safe_identifier(prompt.name)}",
                        name=f"MCP Prompt: {prompt.name}",
                        description=prompt.description or f"MCP prompt {prompt.name}",
                        instructions=body,
                        recommended_tools=(
                            ("get_mcp_prompt",)
                            if self.config.expose_prompt_tool
                            else ()
                        ),
                    )
                )
        return skills


__all__ = [
    "GetMCPPromptInput",
    "MCPToolkit",
    "MCPToolkitConfig",
    "ReadResourceInput",
]
