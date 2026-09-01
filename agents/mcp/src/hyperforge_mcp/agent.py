import json
import logging
from contextlib import _AsyncGeneratorContextManager, asynccontextmanager
from datetime import timedelta
from functools import partial
from time import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, assert_never, cast
from uuid import uuid4

import mcp.shared.exceptions as exceptions
import mcp.types as types
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from hyperforge import logger
from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.interaction import (
    Feedback,
    PromptFeedbackSchema,
    ValidationFeedbackSchema,
)
from hyperforge.manager import Manager, build_reasoning
from hyperforge.memory import QuestionMemory
from hyperforge.models import Chunk, Context, Prompt, TrackingInfo
from hyperforge.result_payload import budget_from_config, inspect_text_blocks
from hyperforge.utils import iterate_tools_resp
from mcp import ClientSession, CreateMessageResult, ErrorData
from mcp.client.streamable_http import GetSessionIdCallback
from mcp.shared.context import RequestContext
from mcp.shared.message import SessionMessage
from mcp.shared.session import RequestResponder
from nuclia.lib.nua_responses import (
    Author,
    ChatModel,
    Image,
    Message,
    Tool,
    ToolChoiceAuto,
    UserPrompt,
)
from nuclia_models.predict.generative_responses import GenerativeFullResponse
from pydantic import FileUrl

from hyperforge_mcp.config import MCPAgentConfig, Transport
from hyperforge_mcp.http import MCPHTTPDriver
from hyperforge_mcp.stdio import MCPStdioDriver
from hyperforge_mcp.tools import (
    MCP_ROUTER_PROMPT_TEMPLATE,
    PROMPT_CHOOSE_TEMPLATE,
    SIMPLE_TOOL_CHOICE_PROMPT,
)


def _tool_parameters(input_schema: Any) -> Dict[str, Any]:
    properties = (
        input_schema.get("properties", {}) if isinstance(input_schema, dict) else {}
    )
    if not isinstance(properties, dict):
        return {}
    return {
        name: dict(schema) if isinstance(schema, dict) else {"type": "string"}
        for name, schema in properties.items()
    }


MAX_NUM_TURNS = 10


EXIT_LOOP_TOOLS = [
    Tool(
        name="task_complete",
        description="Call this tool when the task given by the user is complete",
        parameters={
            "type": "object",
            "properties": {},
        },
    ),
]


def _short_uid() -> str:
    """Return a short unique suffix for chunk IDs."""
    return uuid4().hex[:4]


def _content_payload(content: Any) -> str | None:
    if isinstance(content, (types.TextContent, types.TextResourceContents)):
        payload = content.text
        if content.meta is not None:
            payload += "\nMetadata: " + json.dumps(content.meta)
        return payload
    if isinstance(content, (types.ImageContent, types.AudioContent)):
        return content.data
    if isinstance(content, types.BlobResourceContents):
        return content.blob
    if isinstance(content, types.EmbeddedResource):
        return _content_payload(content.resource)
    return None


@agent(
    id="mcp",
    agent_type="context",
    title="MCP Agent",
    description="Use MCP to analyze data and answer questions.",
    config_schema=MCPAgentConfig,
)
class MCPAgent(ContextAgent, Agent[MCPAgentConfig]):
    __published_functions__: Dict[str, FunctionDefinition]  # type: ignore[misc] # ty: ignore[invalid-attribute-override]  # instance-level, not ClassVar
    driver: Union[MCPStdioDriver, MCPHTTPDriver, None]
    resources: List[types.Resource]
    prompts: List[types.Prompt]
    tools: List[types.Tool]
    session_id: Any
    session: Optional[ClientSession]
    _mcp_preloaded: bool

    headers: Dict[str, str]
    driver_context_manager: Optional[
        _AsyncGeneratorContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
            ]
            | None,
        ]
    ]

    def __init__(self, config: MCPAgentConfig, agent_id: Optional[str] = None):
        super().__init__(config, agent_id)
        self.driver = None
        self.resources = []
        self.prompts = []
        self.tools = []
        self.session_id = None
        self.session = None
        self._mcp_preloaded = False
        self.headers = {}
        self.driver_context_manager = None
        self.__published_functions__: Dict[str, FunctionDefinition] = {}

    async def get_tool_selection_prompt(
        self, manager: Manager, question: str, memory: QuestionMemory, context: Context
    ) -> Tuple[List[Message], List[Image]]:
        """Get the prompt to select the tool to use for the question."""
        if self.session is None:
            raise Exception("MCP session not initialized")
        t0 = time()
        description = "Choose a tool for the task"
        messages = []
        images: List[Any] = []
        audios = []
        final_prompt = None

        prompt = MCP_ROUTER_PROMPT_TEMPLATE.render(
            user=memory.user_info(),
            context=memory.contexts_markdown(),
            question=question,
        )

        messages.append(Message(author=Author.USER, text=prompt))

        if self.config.interaction and len(self.prompts) > 0:
            """ Interaction to choose the prompt
            """
            feedback = Feedback(
                request_id=memory.get_session_id(),
                question="Choose proper use case",
                module=self.config.module,
                agent_id=self.config.id or "default",
                data=self.prompts,
                timeout_ms=self.config.feedback_timeout,
                response_schema=PromptFeedbackSchema.model_json_schema(),
            )
            answer = await memory.send_feedback(feedback)
            if answer is not None and answer.request_id == memory.get_session_id():
                prompt_feedback = PromptFeedbackSchema.model_validate_json(
                    answer.response
                )  # Validate JSON
                final_prompt = await self.session.get_prompt(
                    name=prompt_feedback.prompt_id, arguments=prompt_feedback.data
                )

        if final_prompt is None:
            # We will use LLM to choose the prompt
            prompt_feedback_str = PROMPT_CHOOSE_TEMPLATE.render(
                prompts=self.prompts, task_description=question
            )
            (
                resp,
                input_tokens,
                output_tokens,
                reasoning,
            ) = await manager.execute_json_reasoning(
                model=self.config.tool_choice_model,
                prompt=prompt_feedback_str,
                user_id="mcp_no_feedback",
                schema={
                    "type": "object",
                    "title": "PromptSelection",
                    "description": "Select the most appropriate prompt for the task",
                    "properties": {
                        "prompt_id": {
                            "type": "string",
                            "description": "id of the prompt to use",
                        },
                    },
                },
                tracking=memory.get_tracking_info(),
            )
            prompt_id: str = resp["prompt_id"]
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Prompt selection"),
                step_reason="",
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                step_value="mcp prompt selected: " + prompt_id
                if prompt_id
                else "no prompt selected",
                timeit=time() - t0,
                input_nuclia_tokens=input_tokens,
                output_nuclia_tokens=output_tokens,
            )

            try:
                prompt_obj = next((p for p in self.prompts if p.name == prompt_id))
            except StopIteration:
                return messages, images

            if prompt_obj.arguments is not None and len(prompt_obj.arguments) > 0:
                arguments_schema = {
                    prompt_argument.name: {
                        "type": "string",
                        "description": prompt_argument.description or "",
                    }
                    for prompt_argument in prompt_obj.arguments
                }
                resp, input_tokens, output_tokens = await manager.execute_json(
                    model=self.config.tool_choice_model,
                    prompt=prompt_feedback_str,
                    user_id="mcp_no_feedback_arguments",
                    schema={
                        "name": "choose_arguments",
                        "title": "choose proper arguments",
                        "description": "Choose the most appropriate arguments for the selected prompt",
                        "parameters": {
                            "type": "object",
                            "properties": arguments_schema,
                        },
                    },
                    tracking=memory.get_tracking_info(),
                )

                await memory.add_step(
                    step_module=self.config.module,
                    step_title=self.step_title("Prompt arguments"),
                    step_reason="",
                    step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                    step_value="mcp_no_feedback_arguments",
                    timeit=time() - t0,
                    input_nuclia_tokens=input_tokens,
                    output_nuclia_tokens=output_tokens,
                )
            else:
                resp = {}
            try:
                final_prompt = await self.session.get_prompt(
                    name=prompt_id, arguments=resp
                )
            except Exception as e:
                logger.error(f"Error getting prompt: {e}")
                final_prompt = None

            logger.debug("No user feedback received, continuing without it")

        description = (
            final_prompt.description
            if final_prompt and final_prompt.description
            else "No description available"
        )
        if final_prompt:
            description += " Any partial answer or summary should  use the information in the prompt."
            selected_prompt = Prompt(
                prompt="",
                description=final_prompt.description,
            )
            # At the moment we will only support text in the stored prompts
            for message in final_prompt.messages:
                if isinstance(message.content, types.TextContent):
                    messages.append(
                        Message(author=Author.NUCLIA, text=message.content.text)
                    )
                    selected_prompt.prompt += message.content.text + "\n"
                if isinstance(message.content, types.ImageContent):
                    images.append(
                        Image(
                            content_type=message.content.mimeType,
                            b64encoded=message.content.data,
                        )
                    )
                if isinstance(message.content, types.AudioContent):
                    audios.append(
                        Image(
                            content_type=message.content.mimeType,
                            b64encoded=message.content.data,
                        )
                    )
                if isinstance(message.content, types.ResourceLink):
                    resource = await self.session.read_resource(message.content.uri)
                    for content in resource.contents:
                        if isinstance(content, types.TextResourceContents):
                            messages.append(
                                Message(author=Author.NUCLIA, text=content.text)
                            )
                            selected_prompt.links.append(content.text)
                        elif isinstance(content, types.BlobResourceContents):
                            images.append(
                                Image(
                                    content_type=content.mimeType
                                    if content.mimeType is not None
                                    else "application/octet-stream",
                                    b64encoded=content.blob,
                                )
                            )

                if isinstance(message.content, types.EmbeddedResource):
                    if isinstance(message.content.resource, types.TextResourceContents):
                        messages.append(
                            Message(
                                author=Author.NUCLIA,
                                text=message.content.resource.text,
                            )
                        )
                        selected_prompt.resources.append(message.content.resource.text)
                    elif isinstance(
                        message.content.resource, types.BlobResourceContents
                    ):
                        images.append(
                            Image(
                                content_type=message.content.resource.mimeType
                                if message.content.resource.mimeType is not None
                                else "application/octet-stream",
                                b64encoded=message.content.resource.blob,
                            )
                        )
            context.prompts.append(selected_prompt)

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Prompt completion"),
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value=description,
            timeit=time() - t0,
        )
        return messages, images

    async def choose_tool(
        self,
        manager: Manager,
        images: List[Image],
        messages: List[Message],
        extra_tools: List[Tool] = [],
        tracking: TrackingInfo | None = None,
    ) -> Tuple[GenerativeFullResponse, float, float]:
        # Convert MCP tools into a format the LLM can understand and use
        tools = [
            Tool(
                name=mtl.name,
                description=mtl.description or "",
                parameters=mtl.inputSchema,
            )
            for mtl in self.tools
        ]
        tools.extend(extra_tools)
        logger.debug(f"Available tools: {tools}")

        # TODO: Add model back to the agent config and validation
        item = ChatModel(
            question="",
            user_id="mcp_agent",
            query_context_images=images,
            generative_model=self.config.tool_choice_model.model_id,
            reasoning=build_reasoning(self.config.tool_choice_model),
            tools=tools,
            tool_choice=ToolChoiceAuto(),
            user_prompt=UserPrompt(
                prompt="Choose the best tool or tools for the task, select task_complete if no more tools are needed according to the user request and previous interactions"
            ),
            format_prompt=False,
            system=SIMPLE_TOOL_CHOICE_PROMPT,
            chat_history=messages,
        )
        resp, input_tokens, output_tokens = await manager.execute_raw(
            item, tracking=tracking
        )

        logger.debug(f"Tool and parameters to use: {resp}")
        return resp, input_tokens, output_tokens

    async def progress_callback(
        self,
        memory: QuestionMemory,
        progress: float | None,
        total: float | None,
        message: str | None,
    ) -> None:
        if self.config.progress_feedback:
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Progress"),
                step_reason=message,
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                step_value=f"{progress}/{total}",
                timeit=0,
            )

    async def process_tool(
        self,
        memory: QuestionMemory,
        tool_name: str,
        tool_arguments: Dict[str, Any],
        context: Context,
        messages: List[Message],
        images: List[Image],
        session: Optional[ClientSession] = None,
    ) -> None:
        active_session = session or self.session
        if active_session is None:
            raise Exception("MCP session not initialized")

        t0 = time()

        async def reject_overflow(overflow) -> None:
            error_text = overflow.render()
            context.chunks.append(
                Chunk(
                    chunk_id=f"mcp_{self.config.id}_{tool_name}_oversized_{_short_uid()}",
                    title=f"MCP tool result too large: {tool_name}",
                    text=error_text,
                    origin_agent=self.config.module,
                )
            )
            messages.append(Message(author=Author.NUCLIA, text=error_text))
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Tool result rejected"),
                step_reason=(
                    f"Tool {tool_name} exceeded the configured LLM context safety budget."
                ),
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                step_value=overflow.trace_value(),
                timeit=time() - t0,
            )

        if self.config.interaction:
            feedback = Feedback(
                request_id=memory.get_session_id(),
                question="Execute tool",
                module=self.config.module,
                agent_id=self.config.id or "default",
                data=self.prompts,
                timeout_ms=self.config.feedback_timeout,
                response_schema=ValidationFeedbackSchema.model_json_schema(),
            )
            answer = await memory.send_feedback(feedback)
            if answer is None:
                logger.debug("No user feedback received, continuing without it")
            else:
                if answer.request_id == memory.get_session_id():
                    validation = ValidationFeedbackSchema.model_validate_json(
                        answer.response
                    )  # Validate JSON
                    if validation.call_tool is False:
                        logger.debug("User cancelled tool execution")
                        return

        progress_callback_memory = partial(self.progress_callback, memory)
        tool_result = await active_session.call_tool(
            name=tool_name,
            arguments=tool_arguments,
            read_timeout_seconds=timedelta(seconds=self.config.read_timeout_seconds),
            progress_callback=progress_callback_memory,
        )
        logger.debug(f"Tool {tool_name} results: {tool_result}")

        if tool_result.isError:
            # Extract error text from content blocks; fall back to meta if none present
            error_texts = [
                block.text
                for block in tool_result.content
                if isinstance(block, types.TextContent)
            ]
            error_message = (
                "; ".join(error_texts) if error_texts else str(tool_result.meta)
            )
            error_overflow = inspect_text_blocks(
                [error_message], budget_from_config(self.config)
            )
            if error_overflow is not None:
                error_message = error_overflow.render()
                await memory.add_step(
                    step_module=self.config.module,
                    step_title=self.step_title("Tool result rejected"),
                    step_reason=(
                        f"Tool {tool_name} exceeded the configured LLM context safety budget."
                    ),
                    step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                    step_value=error_overflow.trace_value(),
                    timeit=time() - t0,
                )
            logger.error(f"Tool {tool_name} encountered an error: {error_message}")
            context.chunks.append(
                Chunk(
                    chunk_id=f"mcp_{self.config.id}_{tool_name}_error_{_short_uid()}",
                    title=f"MCP tool error: {tool_name}",
                    text=(f"Tool: {tool_name}\nError: {error_message}"),
                    origin_agent=self.config.module,
                )
            )
            messages.append(
                Message(
                    author=Author.NUCLIA,
                    text=f"Tool {tool_name} failed: {error_message}",
                )
            )
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Tool error"),
                error=f"Tool {tool_name} encountered an error: {error_message}",
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                timeit=time() - t0,
            )
            return
        text_blocks = sum(
            1 for block in tool_result.content if isinstance(block, types.TextContent)
        )
        image_blocks = sum(
            1 for block in tool_result.content if isinstance(block, types.ImageContent)
        )
        resource_blocks = sum(
            1 for block in tool_result.content if isinstance(block, types.ResourceLink)
        )

        trace_lines = [
            f"Tool: {tool_name}",
            f"is_error: {tool_result.isError}",
            f"text_blocks: {text_blocks}",
            f"image_blocks: {image_blocks}",
            f"resource_links: {resource_blocks}",
        ]
        structured = (
            json.dumps(tool_result.structuredContent, indent=2, default=str)
            if tool_result.structuredContent is not None
            else None
        )
        direct_texts = [
            payload
            for block in tool_result.content
            if not isinstance(block, types.ResourceLink)
            if (payload := _content_payload(block)) is not None
        ]
        if structured is not None:
            direct_texts.append(structured)
        direct_overflow = inspect_text_blocks(
            direct_texts, budget_from_config(self.config)
        )
        if direct_overflow is not None:
            await reject_overflow(direct_overflow)
            return
        linked_resources = []
        for block in tool_result.content:
            if isinstance(block, types.ResourceLink):
                resource = await active_session.read_resource(block.uri)
                linked_resources.append(resource)
                direct_texts.extend(
                    payload
                    for content in resource.contents
                    if (payload := _content_payload(content)) is not None
                )
        resource_overflow = inspect_text_blocks(
            direct_texts, budget_from_config(self.config)
        )
        if resource_overflow is not None:
            await reject_overflow(resource_overflow)
            return
        if structured is not None:
            context.structured.append(structured)
            trace_lines.append(f"structured_bytes: {len(structured.encode('utf-8'))}")
        messages.append(
            Message(author=Author.NUCLIA, text=f"Tool {tool_name} executed")
        )
        resource_results = iter(linked_resources)
        for block in tool_result.content:
            if isinstance(block, types.TextContent):
                block_text = block.text
                if block.meta is not None:
                    block_text += "\nMetadata: " + json.dumps(block.meta)
                context.chunks.append(
                    Chunk(
                        chunk_id=f"mcp_{self.config.id}_{tool_name}_{_short_uid()}",
                        title=f"Calling {tool_name}",
                        text=block.text,
                        metadata=block.meta,
                        origin_agent=self.config.module,
                    )
                )
                messages.append(Message(author=Author.NUCLIA, text=block_text))
            elif isinstance(block, types.ImageContent):
                context.images[f"mcp_{self.config.id}_{tool_name}"] = Image(
                    content_type=block.mimeType, b64encoded=block.data
                )
                images.append(Image(content_type=block.mimeType, b64encoded=block.data))
            elif isinstance(block, types.AudioContent):
                pass
            elif isinstance(block, types.ResourceLink):
                resource = next(resource_results)
                for index, content in enumerate(resource.contents):
                    if isinstance(content, types.TextResourceContents):
                        # Use the resource URI as the chunk URL source
                        resource_urls = [str(content.uri)] if content.uri else []
                        block_text = content.text
                        if content.meta is not None:
                            block_text += "\nMetadata: " + json.dumps(content.meta)
                        context.chunks.append(
                            Chunk(
                                chunk_id=f"mcp_{self.config.id}_{tool_name}_{index}",
                                title=f"Calling {tool_name}",
                                text=content.text,
                                metadata=content.meta,
                                origin_agent=self.config.module,
                                url=resource_urls,
                            )
                        )
                        messages.append(Message(author=Author.NUCLIA, text=block_text))
                    elif isinstance(content, types.BlobResourceContents):
                        context.images[f"mcp_{self.config.id}_{tool_name}_{index}"] = (
                            Image(
                                content_type=content.mimeType
                                if content.mimeType is not None
                                else "application/octet-stream",
                                b64encoded=content.blob,
                            )
                        )
                        images.append(
                            Image(
                                content_type=content.mimeType
                                if content.mimeType is not None
                                else "application/octet-stream",
                                b64encoded=content.blob,
                            )
                        )

            elif isinstance(block, types.EmbeddedResource):
                if isinstance(block.resource, types.TextResourceContents):
                    # Use the embedded resource URI as the chunk URL source
                    embedded_urls = (
                        [str(block.resource.uri)] if block.resource.uri else []
                    )
                    block_text = block.resource.text
                    if block.resource.meta is not None:
                        block_text += "\nMetadata: " + json.dumps(block.resource.meta)
                    context.chunks.append(
                        Chunk(
                            chunk_id=f"mcp_{self.config.id}_{tool_name}_{_short_uid()}",
                            title=f"Calling {tool_name}",
                            text=block.resource.text,
                            metadata=block.resource.meta,
                            origin_agent=self.config.module,
                            url=embedded_urls,
                        )
                    )
                    messages.append(
                        Message(
                            author=Author.NUCLIA,
                            text=block_text,
                        )
                    )
                elif isinstance(block.resource, types.BlobResourceContents):
                    context.images[f"mcp_{self.config.id}_{tool_name}"] = Image(
                        content_type=block.resource.mimeType
                        if block.resource.mimeType is not None
                        else "application/octet-stream",
                        b64encoded=block.resource.blob,
                    )
                    images.append(
                        Image(
                            content_type=block.resource.mimeType
                            if block.resource.mimeType is not None
                            else "application/octet-stream",
                            b64encoded=block.resource.blob,
                        )
                    )

        messages.append(Message(author=Author.NUCLIA, text="\n".join(trace_lines)))

        step_value = (
            f"Used tool: {tool_name} with arguments: {tool_arguments}"
            if tool_arguments
            else "No tool used"
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Tool result"),
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value=step_value,
            timeit=time() - t0,
        )

    @asynccontextmanager
    async def _open_session(self, manager: Manager, memory: QuestionMemory):
        """Open a fresh MCP session for a single operation (reconnect-per-call model).

        Yields an initialised ``ClientSession`` regardless of transport type.
        The session is closed automatically when the context manager exits.
        """
        if self.config.transport == Transport.HTTP:
            # Ensure headers are initialized for HTTP transport
            if not self.headers:
                for valid_headers in self.config.valid_headers:
                    if valid_headers in memory.headers:
                        self.headers[valid_headers] = memory.headers[valid_headers]
            async with self.http_streaming_session_ctx(
                manager=manager, memory=memory
            ) as session:
                yield session
        else:
            if self.config.transport == Transport.STDIO:
                driver = cast(MCPStdioDriver, manager.drivers[self.config.source])
            async with driver.client() as (read_stream, write_stream):  # type: ignore[union-attr]
                if read_stream is None or write_stream is None:
                    raise Exception("No read or write stream available")
                client_session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=self.config.read_timeout_seconds
                    ),
                    sampling_callback=partial(self.sampling_callback, manager, memory),
                    elicitation_callback=partial(self.elicitation_callback, memory),
                    list_roots_callback=self.list_roots_callback,
                    logging_callback=self.logging,
                    message_handler=partial(self.message_handler, memory),
                    client_info=types.Implementation(
                        name="Nuclia ARAG", title="Nuclia ARAG client", version="1.0.0"
                    ),
                )
                async with client_session as session:
                    await session.initialize()
                    yield session

    def _make_tool_caller(self, tool_name: str) -> Callable:
        """Return an async callable that opens a fresh session and executes *tool_name*.

        The signature matches what SmartAgent expects:
        ``async (memory, manager, **kwargs) -> Context``.
        Each call opens its own MCP session so concurrent SmartAgent tool
        calls are fully isolated.
        """

        async def call_mcp_tool(
            memory: QuestionMemory, manager: Manager, **kwargs: Any
        ) -> Context:
            context = Context(
                agent_id=self.agent_id,
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=memory.original_question_uuid,
                question=tool_name,
                source=self.config.source,
                agent="mcp",
            )
            try:
                async with self._open_session(manager, memory) as session:
                    await self.process_tool(
                        memory=memory,
                        tool_name=tool_name,
                        tool_arguments=kwargs,
                        context=context,
                        messages=[],
                        images=[],
                        session=session,
                    )
            except Exception as e:
                logger.error(
                    f"MCPAgent {self.agent_id!r}: error calling tool {tool_name!r}: {e}"
                )
            return context

        return call_mcp_tool

    def _make_prompt_caller(self, prompt_name: str) -> Callable:
        """Return an async callable that fetches an MCP prompt and injects its text as context.

        The signature matches what SmartAgent expects:
        ``async (memory, manager, **kwargs) -> Context``.
        ``kwargs`` are forwarded as prompt arguments to the MCP server.
        Each call opens its own MCP session.
        """

        async def call_mcp_prompt(
            memory: QuestionMemory, manager: Manager, **kwargs: Any
        ) -> Context:
            context = Context(
                agent_id=self.agent_id,
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=memory.original_question_uuid,
                question=prompt_name,
                source=self.config.source,
                agent="mcp",
            )
            t0 = time()
            try:
                async with self._open_session(manager, memory) as session:
                    prompt_result = await session.get_prompt(
                        name=prompt_name,
                        arguments={k: str(v) for k, v in kwargs.items()} or None,
                    )
                    for message in prompt_result.messages:
                        if isinstance(message.content, types.TextContent):
                            context.chunks.append(
                                Chunk(
                                    chunk_id=f"mcp_prompt_{self.config.id}_{prompt_name}_{_short_uid()}",
                                    title=f"MCP Prompt: {prompt_name}",
                                    text=message.content.text,
                                    origin_agent=self.config.module,
                                )
                            )
                        elif isinstance(message.content, types.ImageContent):
                            context.images[
                                f"mcp_prompt_{self.config.id}_{prompt_name}"
                            ] = Image(
                                content_type=message.content.mimeType,
                                b64encoded=message.content.data,
                            )
                        elif isinstance(
                            message.content, types.EmbeddedResource
                        ) and isinstance(
                            message.content.resource, types.TextResourceContents
                        ):
                            context.chunks.append(
                                Chunk(
                                    chunk_id=f"mcp_prompt_{self.config.id}_{prompt_name}_{_short_uid()}",
                                    title=f"MCP Prompt: {prompt_name}",
                                    text=message.content.resource.text,
                                    origin_agent=self.config.module,
                                )
                            )
            except Exception as e:
                logger.error(
                    f"MCPAgent {self.agent_id!r}: error fetching prompt {prompt_name!r}: {e}"
                )
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title(f"Prompt: {prompt_name}"),
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                step_value=f"Fetched MCP prompt {prompt_name!r} with arguments {kwargs}",
                timeit=time() - t0,
            )
            return context

        return call_mcp_prompt

    async def preload(self, manager: Manager, memory: QuestionMemory) -> None:
        """Discover MCP tools and prompts and expose them as ``__published_functions__``.

        Called by SmartAgent before tool selection so that this agent's MCP
        tools (and optionally prompts) appear alongside the published functions
        of all other registered agents.

        - Runs once per instance; subsequent calls are no-ops.
        - On any connection failure: logs a warning and registers zero tools so
          the SmartAgent can continue with the remaining agents.
        """
        if self._mcp_preloaded:
            return

        try:
            async with self._open_session(manager, memory) as session:
                self.session = session
                try:
                    await self.preload_tools()
                except exceptions.McpError as e:
                    logger.warning(
                        f"MCPAgent {self.agent_id!r}: failed to list tools: {e}"
                    )
                    self.tools = []

                if self.config.expose_prompts_as_tools:
                    try:
                        await self.preload_prompts()
                    except exceptions.McpError as e:
                        logger.warning(
                            f"MCPAgent {self.agent_id!r}: failed to list prompts: {e}"
                        )
                        self.prompts = []
                self.session = None
        except Exception as e:
            logger.warning(
                f"MCPAgent {self.agent_id!r}: could not connect to MCP server for preload "
                f"({self.config.source}): {e}. Registering with zero published functions."
            )
            self.__published_functions__ = {}
            self._mcp_preloaded = True
            return

        published: Dict[str, FunctionDefinition] = {}

        for tool in self.tools:
            published[tool.name] = FunctionDefinition(
                name=tool.name,
                description=tool.description or tool.name,
                parameters=_tool_parameters(tool.inputSchema),
            )
            setattr(self, tool.name, self._make_tool_caller(tool.name))

        if self.config.expose_prompts_as_tools:
            for prompt in self.prompts:
                params = {
                    arg.name: {
                        "type": "string",
                        "description": arg.description or "",
                    }
                    for arg in (prompt.arguments or [])
                }
                published[prompt.name] = FunctionDefinition(
                    name=prompt.name,
                    description=f"[MCP Prompt] {prompt.description or prompt.name}",
                    parameters=params,
                )
                setattr(self, prompt.name, self._make_prompt_caller(prompt.name))

        self.__published_functions__ = published
        self._mcp_preloaded = True
        logger.info(
            f"MCPAgent {self.agent_id!r}: preloaded {len(self.tools)} tool(s) and "
            f"{len(self.prompts) if self.config.expose_prompts_as_tools else 0} prompt(s) "
            f"as published functions."
        )

    async def mcp_interaction(
        self, memory: QuestionMemory, manager: Manager, question: str, context: Context
    ) -> Tuple[float, float]:
        """
        Interact with the MCP server to get the context for the question.
        This method will use the MCP server to call tools and get structured data.
        """
        total_input_tokens = 0.0
        total_output_tokens = 0.0

        messages, images = await self.get_tool_selection_prompt(
            manager, question, memory, context
        )
        resp, input_tokens, output_tokens = await self.choose_tool(
            manager,
            images,
            messages,
            tracking=memory.get_tracking_info(),
        )

        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        tool_calls = list(iterate_tools_resp(resp))
        for tool_name, tool_arguments in tool_calls:
            await self.process_tool(
                memory, tool_name, tool_arguments, context, messages, images
            )

        if self.config.work_chain is False or not tool_calls:
            logger.debug("Exiting loop on tool")
            return total_input_tokens, total_output_tokens

        count = 0
        finished = False
        while count < self.config.max_turns and finished is False:
            count += 1
            resp, input_tokens, output_tokens = await self.choose_tool(
                manager,
                images,
                messages,
                EXIT_LOOP_TOOLS,
                tracking=memory.get_tracking_info(),
            )
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            tool_calls = list(iterate_tools_resp(resp))
            if not tool_calls:
                logger.debug("Exiting loop because no tool was selected")
                break
            for tool_name, tool_arguments in tool_calls:
                if tool_name == "task_complete":
                    logger.debug("Exiting loop on task_complete tool")
                    finished = True
                    break
                await self.process_tool(
                    memory, tool_name, tool_arguments, context, messages, images
                )

        return total_input_tokens, total_output_tokens

    async def sampling_callback(
        self,
        manager: Manager,
        memory: QuestionMemory,
        context: RequestContext[ClientSession, Any, Any],
        params: types.CreateMessageRequestParams,
    ) -> CreateMessageResult | ErrorData:
        t0 = time()
        new_messages = []
        images = {}
        for message in params.messages:
            if isinstance(message.content, types.ImageContent):
                images[uuid4().hex] = Image(
                    b64encoded=message.content.data,
                    content_type=message.content.mimeType,
                )
            elif isinstance(message.content, types.TextContent):
                new_messages.append(
                    Message(
                        author=Author.NUCLIA
                        if message.role == "assistant"
                        else Author.USER,
                        text=message.content.text,
                    )
                )
            elif isinstance(message.content, types.AudioContent):
                pass

        item = ChatModel(
            question="",
            user_id=self.config.id or "mcp_agent",
            query_context_images=images,
            generative_model=self.config.sampling_model.model_id,
            reasoning=build_reasoning(self.config.sampling_model),
            format_prompt=False,
            system=params.systemPrompt,
            context=new_messages,
            max_tokens=params.maxTokens,
        )
        resp, input_tokens, output_tokens = await manager.execute_raw(
            item,
            tracking=memory.get_tracking_info(),
        )

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Sampling done"),
            step_reason="",
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value="mcp_sampling_done",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )

        return CreateMessageResult(
            role="user",
            content=types.TextContent(type="text", text=resp.answer),
            model=self.config.sampling_model.model_id,
        )

    async def elicitation_callback(
        self,
        memory: QuestionMemory,
        context: RequestContext[ClientSession, Any, Any],
        params: types.ElicitRequestParams,
    ) -> types.ElicitResult | ErrorData:
        # Handle form mode elicitation (with requestedSchema)
        if isinstance(params, types.ElicitRequestFormParams):
            feedback = Feedback(
                request_id=memory.get_session_id(),
                question=params.message,
                module=self.config.module,
                agent_id=self.config.id or "default",
                data=self.prompts,
                timeout_ms=self.config.feedback_timeout,
                response_schema=params.requestedSchema,
            )
            answer = await memory.send_feedback(feedback)
            if answer is not None and answer.request_id == memory.get_session_id():
                return types.ElicitResult(
                    action="accept",
                    content=json.loads(answer.response),
                )

            return ErrorData(code=666, message="No feedback received")

        # Handle URL mode elicitation
        elif isinstance(params, types.ElicitRequestURLParams):
            # URL mode elicitation is not supported in this implementation
            return ErrorData(code=400, message="URL mode elicitation is not supported")

        else:
            assert_never()

    async def list_roots_callback(
        self, context: RequestContext[ClientSession, Any, Any]
    ) -> types.ListRootsResult | ErrorData:
        return types.ListRootsResult(
            roots=[
                types.Root(name=key, uri=FileUrl(url=value))
                for key, value in self.config.roots.items()
            ]
        )

    async def logging(self, params: types.LoggingMessageNotificationParams):
        logger = logging.getLogger(params.logger or "mcp")
        match params.level:
            case "critical":
                logger.critical(params.data)
            case "error":
                logger.error(params.data)
            case "warning":
                logger.warning(params.data)
            case "info":
                logger.info(params.data)
            case "debug":
                logger.debug(params.data)
            case _:
                logger.info(params.data)

    async def message_handler(
        self,
        memory: QuestionMemory,
        message: RequestResponder[types.ServerRequest, types.ClientResult]
        | types.ServerNotification
        | Exception,
    ):
        if isinstance(message, Exception):
            logger.error(f"MCP client error: {message}")
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Client error"),
                error=str(message),
                step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
                timeit=0,
            )
        elif isinstance(message, types.ServerNotification):
            logger.debug(f"MCP notification: {message}")
        elif isinstance(message, types.ServerRequest):
            logger.debug(f"MCP request: {message}")
            # Handle specific requests if needed
        # TODO
        pass

    async def initialize(self, manager: Manager, memory: QuestionMemory) -> None:
        if self.session is not None:
            return

        if self.config.transport == Transport.STDIO:
            self.driver = cast(MCPStdioDriver, manager.drivers[self.config.source])
            self.driver_context_manager = self.driver.client()

        elif self.config.transport == Transport.HTTP:
            self.headers: Dict[str, str] = {}
            for valid_headers in self.config.valid_headers:
                if valid_headers in memory.headers:
                    self.headers[valid_headers] = memory.headers[valid_headers]

        else:
            raise Exception(f"Unknown transport {self.config.transport}")

    async def preload_tools(self):
        if self.session is None:
            raise Exception("MCP session not initialized")
        self.tools = []
        tools = await self.session.list_tools()
        # Make sure we store a summary of the tools available
        # Context: There are many tools available for a user. However, the number of tools can be large, and it is not always practical to present all of them at once. We need to create a summary of them that accurately reflects the capabilities they provide.
        # The user presents you with the tools available to them, and you must create a summary of the tools that is accurate and comprehensive. The summary should include the capabilities of the tools and when they should be used.

        self.tools.extend(tools.tools)
        while tools.nextCursor:
            params: types.PaginatedRequestParams = types.PaginatedRequestParams(
                cursor=tools.nextCursor
            )
            tools = await self.session.list_tools(params=params)
            self.tools.extend(tools.tools)

    async def preload_prompts(self):
        if self.session is None:
            raise Exception("MCP session not initialized")

        self.prompts = []
        prompts = await self.session.list_prompts()

        self.prompts.extend(prompts.prompts)
        while prompts.nextCursor:
            params: types.PaginatedRequestParams = types.PaginatedRequestParams(
                cursor=prompts.nextCursor
            )
            prompts = await self.session.list_prompts(params=params)
            self.prompts.extend(prompts.prompts)

    async def preload_resources(self):
        if self.session is None:
            raise Exception("MCP session not initialized")

        self.resources = []
        resources = await self.session.list_resources()
        self.resources.extend(resources.resources)
        while resources.nextCursor:
            params: types.PaginatedRequestParams = types.PaginatedRequestParams(
                cursor=resources.nextCursor
            )
            resources = await self.session.list_resources(params=params)
            self.resources.extend(resources.resources)

    @asynccontextmanager
    async def http_streaming_session_ctx(
        self, manager: Manager, memory: QuestionMemory
    ):
        self.driver = cast(MCPHTTPDriver, manager.drivers[self.config.source])
        http_context_manager: _AsyncGeneratorContextManager[
            tuple[
                MemoryObjectReceiveStream[SessionMessage | Exception],
                MemoryObjectSendStream[SessionMessage],
                GetSessionIdCallback,
            ],
        ]

        http_context_manager = self.driver.client(
            headers=self.headers,
            memory=memory,
            module=self.config.module,
            agent_id=self.config.id or "default",
            request_id=memory.get_session_id(),
        )
        async with http_context_manager as (read_stream, write_stream, session_id_fn):
            if read_stream is None or write_stream is None:
                raise Exception("No read or write stream available")
            self.session_id = session_id_fn()
            client_session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                read_timeout_seconds=timedelta(
                    seconds=self.config.read_timeout_seconds
                ),
                sampling_callback=partial(self.sampling_callback, manager, memory),
                elicitation_callback=partial(self.elicitation_callback, memory),
                list_roots_callback=self.list_roots_callback,
                logging_callback=self.logging,
                message_handler=partial(self.message_handler, memory),
                client_info=types.Implementation(
                    name="Nuclia ARAG", title="Nuclia ARAG client", version="1.0.0"
                ),
            )
            async with client_session as session:
                await session.initialize()
                yield session

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        try:
            await self.initialize(manager, memory)
        except KeyError:
            raise Exception("No MCP source found")

        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            structured=[],
            source=self.config.source,
            agent="mcp",
        )

        t0 = time()
        input_tokens = 0.0
        output_tokens = 0.0

        if self.config.transport == Transport.HTTP:
            loaded_tools = False
            max_retries = 2
            for attempt in range(max_retries):
                interaction_completed = False
                try:
                    async with self.http_streaming_session_ctx(
                        manager=manager, memory=memory
                    ) as session:
                        self.session = session

                        try:
                            if loaded_tools is False:
                                await self.preload_tools()
                                loaded_tools = True
                        except exceptions.McpError as e:
                            logger.warning(
                                f"Failed to preload tools from MCP server: {e}"
                            )
                            self.tools = []
                        if self.tools and loaded_tools and attempt == 0:
                            tools_text = "\n".join(
                                f"- {t.name}: {t.description or '(no description)'}"
                                for t in self.tools
                            )
                            context.chunks.append(
                                Chunk(
                                    chunk_id=f"mcp_{self.config.id}_tools_list",
                                    title="Available MCP tools",
                                    text=f"The following tools are available:\n{tools_text}",
                                    origin_agent=self.config.module,
                                )
                            )
                        if attempt == 0:
                            # Only preload prompts and resources on the first attempt
                            try:
                                await self.preload_prompts()
                            except exceptions.McpError as e:
                                logger.warning(
                                    f"Failed to preload prompts from MCP server: {e}"
                                )
                                self.prompts = []

                            try:
                                await self.preload_resources()
                            except exceptions.McpError as e:
                                logger.warning(
                                    f"Failed to preload resources from MCP server: {e}"
                                )
                                self.resources = []

                        (
                            input_tokens,
                            output_tokens,
                        ) = await self.mcp_interaction(
                            memory, manager, question, context
                        )
                        interaction_completed = True
                    break  # Success, exit retry loop
                except Exception as e:
                    if interaction_completed:
                        logger.warning(
                            "Ignoring MCP HTTP teardown error after successful interaction: %s",
                            repr(e),
                        )
                        break

                    logger.exception(
                        f"Error during MCP HTTP interaction (attempt {attempt + 1}/{max_retries})"
                    )

                    if attempt + 1 == max_retries:
                        raise e
                finally:
                    self.session = None
        elif self.driver_context_manager is not None:
            async with self.driver_context_manager as (read_stream, write_stream):  # type: ignore
                if read_stream is None or write_stream is None:
                    raise Exception("No read or write stream available")
                self.session_id = None
                client_session = ClientSession(
                    read_stream=read_stream,
                    write_stream=write_stream,
                    read_timeout_seconds=timedelta(
                        seconds=self.config.read_timeout_seconds
                    ),
                    sampling_callback=partial(self.sampling_callback, manager, memory),
                    elicitation_callback=partial(self.elicitation_callback, memory),
                    list_roots_callback=self.list_roots_callback,
                    logging_callback=self.logging,
                    message_handler=partial(self.message_handler, memory),
                    client_info=types.Implementation(
                        name="Nuclia ARAG", title="Nuclia ARAG client", version="1.0.0"
                    ),
                )
                async with client_session as self.session:
                    try:
                        await self.session.initialize()

                        try:
                            await self.preload_tools()
                        except exceptions.McpError as e:
                            logger.warning(
                                f"Failed to preload tools from MCP server: {e}"
                            )
                            self.tools = []
                        try:
                            await self.preload_prompts()
                        except exceptions.McpError as e:
                            logger.warning(
                                f"Failed to preload prompts from MCP server: {e}"
                            )
                            self.prompts = []

                        try:
                            await self.preload_resources()
                        except exceptions.McpError as e:
                            logger.warning(
                                f"Failed to preload resources from MCP server: {e}"
                            )
                            self.resources = []

                    except Exception as e:
                        logger.exception(
                            "MCP session initialization cancelled: No MCP Server found"
                        )
                        raise e

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Interaction finished"),
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value="mcp_interaction_finished",
            step_reason="",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )

        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            use_stored_context_prompts=self.config.include_mcp_prompts,
            flow_id=flow_id,
        )
        return [missing] if missing is not None else []
