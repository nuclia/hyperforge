import json
import string
import uuid
from copy import deepcopy
from enum import Enum
from functools import lru_cache
from typing import Annotated, Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import httpx
from hyperforge.json_schema import convert_json_schema
import jsonschema
from nuclia.lib.nua_responses import ChatModel, CitationsType, Tool
from pydantic import BaseModel, Field

from hyperforge import logger

NormalizedReasoningEffort = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]

openai_chat_system_prompt = "You are a helpful assistant."


openai_chat_prompt = """Answer the following question using **only** the provided context, along with any previous messages and images from our conversation. Refrain from incorporating any outside knowledge. If the context is insufficient to answer the question comprehensively, respond with: "Not enough data to answer this."

[START OF CONTEXT]
{context}
[END OF CONTEXT]

Question: {question}

# Notes
- Use the context provided without being overly selective.
- Please try to answer if possible, even if it requires to make a bit of a deduction.
- If they are images attached, look through them carefully.
- For questions related to charts or images, pay extra attention to extracting details and interpreting data presented visually. Think about it carefully before giving inaccurate interpretations
"""


def format_prompt(
    prompt: str, format_prompt: bool, *, question: str, context: str
) -> str:
    if not format_prompt:
        return prompt
    return prompt.format(question=question, context=context)


MARKDOWN_CITATIONS_PROMPT_ADJUSTMENT = """
You are given source blocks with IDs like: block-AB, block-BA, block-CD, etc.
When producing an answer, cite these sources precisely using markdown footnotes.

CITATION RULES

1. In the main body, cite sources only with bracketed Arabic numerals: [1], [2], [3], etc.
  - Never put a block ID directly in brackets (e.g. NO: [AB], [block-AB], [BA]).
  - Never mix styles (no superscripts, no inline block names, no [Ref 1], etc.).
2. Numbering is assigned in order of FIRST USE of a unique block ID.
  - The first time you need info from a block, assign it [1].
  - The first time you need info from a never-before-used block, assign it the next unused number (e.g. [2]).
  - If you later cite the SAME block again, REUSE its existing number (do NOT create a new one).
  - This guarantees there are no duplicate footnote definitions and no gaps.
3. If facts in a sentence come from different blocks, you may concatenate citations WITH spaces: like [1] [3]. Do NOT merge them (no ranges like [1-3]) or output them without spaces (no [1][3]).
4. At the end, output section consisting ONLY of the unique citation mappings, one per line, in ascending numeric order
  - Don't title this section or add any extra text, just the mappings.
  - No duplicates.
  - No skipped numbers.
  - ONLY include blocks actually cited in the body.
5. Do NOT hallucinate block IDs. Only use those provided in the context.

FORMATTING CONTRACT

* Body: free text with numeric citations as specified.
* A blank line.
* References section (if any) exactly as described, no heading.


Example format:

---
"The OP-1 has a built-in tape feature with 6 minutes of recording time [1] [2]. You can record to any of the 4 individual tracks [1]."

[1]: block-AB
[2]: block-FZ
---

"""


type ReasoningEffort = Literal["minimal", "low", "medium", "high"]
type ToolChoice = Literal["auto", "none", "required"] | dict[str, Any]
type ResponseInputParam = list[dict[str, Any]]
type ToolParam = dict[str, Any]


class NormalizedTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class NormalizedSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class NormalizedReasoning(BaseModel):
    enabled: bool = False
    display: bool = True
    effort: Optional[NormalizedReasoningEffort] = None
    budget_tokens: Optional[int] = None


class ChatCompletionResponseFormat(BaseModel):
    type: str
    json_schema: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    """Provider-neutral request for an OpenAI-compatible chat endpoint."""

    messages: list[dict[str, Any]] = Field(min_length=1)
    model: str | None = None
    stream: bool = True
    temperature: float | None = None
    max_tokens: int = 50_000
    top_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    stop: str | list[str] | None = None
    response_format: ChatCompletionResponseFormat | None = None
    json_schema: dict[str, Any] | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: ToolChoice | None = None
    reasoning_effort: ReasoningEffort | None = None
    user: str | None = None
    stream_options: dict[str, bool] = Field(
        default_factory=lambda: {"include_usage": True}
    )


class ChatCompletionToolCallFunctionDelta(BaseModel):
    name: str | None = None
    arguments: str | None = None


class Author(str, Enum):
    NUCLIA = "NUCLIA"
    USER = "USER"


class MessageToolFunction(BaseModel):
    name: str
    arguments: Any = None


class MessageToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: MessageToolFunction


class AssistantMessage(BaseModel):
    type: Literal["assistant"] = "assistant"
    author: Author = Author.NUCLIA
    text: str = ""
    content: Any = None
    tool_calls: List[MessageToolCall] = Field(default_factory=list)


class ChatCompletionToolCallDelta(BaseModel):
    index: int
    id: str | None = None
    type: Literal["function"] | None = None
    function: ChatCompletionToolCallFunctionDelta | None = None


class ChatCompletionDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning_content: str | None = None
    refusal: str | None = None
    tool_calls: list[ChatCompletionToolCallDelta] = Field(default_factory=list)


class ChatCompletionChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta = Field(default_factory=ChatCompletionDelta)
    finish_reason: str | None = None


class ChatCompletionUsage(BaseModel):
    prompt_tokens: float = 0
    completion_tokens: float = 0
    total_tokens: float = 0
    nuclia_input_tokens: float | None = None
    nuclia_output_tokens: float | None = None
    model_input_tokens: float | None = None
    model_output_tokens: float | None = None
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None

    @property
    def input_tokens(self) -> float:
        return (
            self.model_input_tokens
            if self.model_input_tokens is not None
            else self.prompt_tokens
        )

    @property
    def output_tokens(self) -> float:
        return (
            self.model_output_tokens
            if self.model_output_tokens is not None
            else self.completion_tokens
        )


class ToolMessage(BaseModel):
    type: Literal["tool"] = "tool"
    author: Author = Author.USER
    text: str = ""
    tool_call_id: str
    name: Optional[str] = None
    content: Any = None


class ChatCompletionChunk(BaseModel):
    id: str | None = None
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    created: int | None = None
    model: str | None = None
    object: str = "chat.completion.chunk"
    usage: ChatCompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class Message(BaseModel):
    type: Literal["message"] = "message"
    author: Author
    text: str


RichMessage = Annotated[
    Union[AssistantMessage, ToolMessage, Message], Field(discriminator="type")
]


class Image(BaseModel):
    content_type: str
    b64encoded: str


class ToolChoiceAuto(BaseModel):
    type: Literal["auto"] = "auto"


class ToolChoiceNone(BaseModel):
    type: Literal["none"] = "none"


class ToolChoiceRequired(BaseModel):
    type: Literal["required"] = "required"


class ToolChoiceForced(BaseModel):
    type: Literal["forced"] = "forced"
    name: str


class NucliaChatCompletionsError(RuntimeError):
    def __init__(
        self, message: str, *, provider_data: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.provider_data = provider_data or {}


@lru_cache(maxsize=64)
def generate_ctx_block_id(n: int) -> str:
    """Generates block ids in the form block-AA, block-AB, ..., block-AZ, block-BA, ..., block-ZZ"""
    # XXX: We use letters to identify block since uuids confuse the LLMs and with numbers it confuses the block numbers with footnote numbers
    if n < 0:
        raise ValueError("Number must be non-negative")
    if n >= 26 * 26:
        logger.warning(
            "Block ID exceeds maximum limit for citations,", extra={"block_id": n}
        )
        n = n % (26 * 26)
    letters = string.ascii_uppercase
    first = letters[(n // 26)]
    second = letters[n % 26]
    return "block-" + first + second


def get_ordered_context(
    context: Dict[str, str],
    context_order: Dict[str, int],
) -> Tuple[List[str], List[str]]:
    """returns context sorted by provided order"""
    if not context_order:
        return list(context.values()), list(context.keys())
    context_ordered: List[Tuple[str, int]] = sorted(
        context_order.items(), key=lambda x: x[1]
    )
    ordered_ids: List[str] = [context_id for context_id, order in context_ordered]
    ordered_context: List[str] = [context[context_id] for context_id in ordered_ids]
    return ordered_context, ordered_ids


def join_and_preprocess_contexts(item: ChatModel) -> str:
    if isinstance(item.query_context, dict):
        if item.query_context != {}:
            contexts, _ = get_ordered_context(
                context=item.query_context, context_order=item.query_context_order
            )
        else:
            contexts = list(item.query_context.values())

        # Citation context adjustment
        if item.citations == CitationsType.LLM_FOOTNOTES:
            # XXX: Hey developer! This has to match the example in the prompt in prompts.py
            contexts = [
                f"\n**{generate_ctx_block_id(i)}**\n\n{v}\n\n---"
                for i, v in enumerate(contexts)
            ]
        context = "\n".join(contexts)
    else:
        contexts = item.query_context

        # Citation context adjustment
        if item.citations == CitationsType.LLM_FOOTNOTES:
            contexts = [
                f"\n**{generate_ctx_block_id(i)}**\n\n{v}\n\n---"
                for i, v in enumerate(item.query_context)
            ]
        context = "\n".join(contexts)
    return context


def openai_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.rstrip()
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        return "\n".join(text_parts).rstrip()
    return str(content).rstrip()


def openai_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def rich_messages_to_openai_responses(
    messages: Sequence[Any],
) -> ResponseInputParam:
    converted: ResponseInputParam = []
    for message in messages:
        if message.type == "assistant":
            text = openai_content_to_text(
                message.content if message.content is not None else message.text
            )
            if text:
                converted.append({"content": text, "role": "assistant"})
            for tool_call in message.tool_calls:
                converted.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": json.dumps(
                            openai_tool_arguments(tool_call.function.arguments)
                        ),
                    }
                )
        elif message.type == "tool":
            converted.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": openai_content_to_text(
                        message.content if message.content is not None else message.text
                    ),
                }
            )
        else:
            converted.append(
                {
                    "content": openai_content_to_text(message.text),
                    "role": "user" if message.author == Author.USER else "assistant",
                }
            )
    return converted


def request_error_detail(
    exc: httpx.RequestError | httpx.HTTPStatusError,
) -> tuple[str, dict[str, Any]]:
    response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
    status = response.status_code if response is not None else None
    response_body = response.text.strip()[:2000] if response is not None else ""
    request = exc.request
    url = str(request.url) if request is not None else "unknown URL"
    detail = str(exc).strip() or repr(exc)
    parts = [f"{type(exc).__name__} for {url}"]
    if status is not None:
        parts.append(f"status={status}")
    if response_body:
        parts.append(f"response={response_body}")
    parts.append(f"error={detail}")
    return "; ".join(parts), {
        "http_status": status,
        "url": url,
        "response_body": response_body or None,
        "error_type": type(exc).__name__,
        "error": detail,
    }


def prioritize_and_adapt_prompts(
    default_user_prompt: str,
    default_system_prompt: str | None,
    item: ChatModel,
) -> Tuple[str, str | None]:
    """
    Setup user prompt and system prompt.
    Priorities are 1. item (ChatModel), 2. user_prompts (learning_config), 3. default

    If necessary, adapts the prompts to include citations in markdown format and also the context blocks
    """
    # Default
    user = default_user_prompt
    system = default_system_prompt

    # Item
    if item.user_prompt is not None and item.user_prompt.prompt != "":
        user = item.user_prompt.prompt
    if item.system is not None and item.system != "":
        system = item.system

    if item.citations == CitationsType.LLM_FOOTNOTES:
        user += MARKDOWN_CITATIONS_PROMPT_ADJUSTMENT

    return user, system


def normalize_tool_name(name: str) -> str:
    return name.replace(" ", "_")


def add_additional_properties_and_required(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Add additionalProperties: False and required fields to the parameters schema if not present.
    This is needed for OpenAI function calling to work properly.
    """
    parameters.setdefault("properties", {})
    if "type" not in parameters or parameters["type"] != "object":
        return parameters

    for kind in ["properties", "$defs"]:
        if kind not in parameters or not isinstance(parameters[kind], dict):
            continue
        for prop_name, prop_schema in parameters[kind].items():
            if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                parameters[kind][prop_name] = add_additional_properties_and_required(
                    prop_schema
                )
            elif (
                isinstance(prop_schema, dict)
                and prop_schema.get("type") == "array"
                and isinstance(prop_schema.get("items"), dict)
                and prop_schema["items"].get("type") == "object"
            ):
                prop_schema["items"] = add_additional_properties_and_required(
                    prop_schema["items"]
                )
            elif isinstance(prop_schema, dict):
                for combinator in ("anyOf", "oneOf", "allOf"):
                    if combinator in prop_schema and isinstance(
                        prop_schema[combinator], list
                    ):
                        for i, variant in enumerate(prop_schema[combinator]):
                            if (
                                isinstance(variant, dict)
                                and variant.get("type") == "object"
                            ):
                                prop_schema[combinator][i] = (
                                    add_additional_properties_and_required(variant)
                                )
                            elif (
                                isinstance(variant, dict)
                                and variant.get("type") == "array"
                                and isinstance(variant.get("items"), dict)
                                and variant["items"].get("type") == "object"
                            ):
                                variant["items"] = (
                                    add_additional_properties_and_required(
                                        variant["items"]
                                    )
                                )

    parameters["additionalProperties"] = False
    if "properties" in parameters:
        parameters["required"] = list(parameters["properties"].keys())

    return parameters


def normalize_tools(
    tools: list[Tool], *, ensure_openai_strict_schema: bool = True
) -> list[NormalizedTool]:
    return [
        NormalizedTool(
            name=normalize_tool_name(tool.name),
            description=tool.description,
            parameters=(
                add_additional_properties_and_required(deepcopy(tool.parameters))
                if ensure_openai_strict_schema
                else deepcopy(tool.parameters)
            ),
        )
        for tool in tools
    ]


def normalize_openai_compatible_tool_choice(
    tool_choice: ToolChoiceAuto
    | ToolChoiceNone
    | ToolChoiceRequired
    | ToolChoiceForced,
    tools: list[NormalizedTool],
) -> Optional[ToolChoice]:
    if not tools:
        return None
    if isinstance(tool_choice, ToolChoiceRequired):
        return "required"
    if isinstance(tool_choice, ToolChoiceForced):
        return {
            "name": normalize_tool_name(tool_choice.name),
            "type": "function",
        }
    return "auto"


def should_append_query_message(
    item: ChatModel,
    context: str,
    images: Sequence[Image] | None = None,
) -> bool:
    has_user_prompt = bool(item.user_prompt and item.user_prompt.prompt.strip())
    has_query_payload = bool(item.question or context or images or has_user_prompt)
    return has_query_payload


def normalize_json_schema(
    schema: dict[str, Any],
    *,
    additional_properties: Literal[
        "default_to_false", "unsupported", "noop"
    ] = "default_to_false",
    required: Literal["set_if_no_default", "force", "noop"] = "set_if_no_default",
) -> NormalizedSchema:
    jsonschema.Draft202012Validator.check_schema(schema)
    converted_schema = convert_json_schema(
        schema,
        additional_properties=additional_properties,
        required=required,
    )
    return NormalizedSchema(
        name=converted_schema["name"],
        description=converted_schema["description"],
        parameters=converted_schema["parameters"],
    )


def transform_chat_to_openai_messages(
    item: ChatModel,
) -> Tuple[
    ResponseInputParam,
    List[ToolParam],
    Optional[ToolChoice],
    Optional[dict[str, Any]],
]:

    normalized_tools = normalize_tools(item.tools)
    openai_tools: list[ToolParam] = [
        {
            "name": tool.name,
            "parameters": tool.parameters,
            "type": "function",
            "description": tool.description,
            "strict": True,
        }
        for tool in normalized_tools
    ]

    tool_choice: Optional[ToolChoice] = normalize_openai_compatible_tool_choice(
        item.tool_choice, normalized_tools
    )

    user, system = prioritize_and_adapt_prompts(
        openai_chat_prompt,
        openai_chat_system_prompt,
        item,
    )
    context = join_and_preprocess_contexts(item)
    # We don't raise errors for system prompt support because system prompt can be defined KB-wise and that would block the user from using the model

    schema = None
    if item.json_schema:
        schema = normalize_json_schema(item.json_schema, required="force").model_dump()
        if schema is not None:
            openai_tools.append(
                {
                    "name": schema["name"],
                    "parameters": schema["parameters"],
                    "description": schema["description"],
                    "type": "function",
                    "strict": True,
                }
            )
            tool_choice = {
                "name": schema.get("name", uuid.uuid4().hex),
                "type": "function",
            }

    messages: ResponseInputParam = []
    messages.append(
        {
            "content": system,
            "role": "system",
        }
    )

    messages.extend(rich_messages_to_openai_responses(item.chat_history))

    image_list: List[Image] = []
    if isinstance(item.query_context_images, dict):
        image_list = list(item.query_context_images.values())
    else:
        image_list = item.query_context_images

    for image in image_list:
        messages.append(
            {
                "content": [
                    {
                        "image_url": f"data:{image.content_type};base64,{image.b64encoded}",
                        "detail": "auto",
                        "type": "input_image",
                    }
                ],
                "role": "user",
            }
        )

    if should_append_query_message(item, context, image_list):
        query = format_prompt(
            user, item.format_prompt, question=item.question, context=context
        )
        messages.append(
            {
                "content": query,
                "role": "user",
            }
        )

    if item.image_generation is True:
        openai_tools.append(
            {
                "type": "image_generation",
                "background": "auto",
                "size": "auto",
                "output_format": "png",
            }
        )

    return messages, openai_tools, tool_choice, schema
