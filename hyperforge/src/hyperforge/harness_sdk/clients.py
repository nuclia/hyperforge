from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel, Field

from hyperforge.llm import AsyncNuaClient, NUAConnection

from .models import HarnessMessage, HarnessToolCall
from .tools import HarnessTool

logger = logging.getLogger(__name__)

PUBLIC_CHAT_COMPLETIONS_PATH = "/api/v1/predict/compat/chat/completions"
INTERNAL_CHAT_COMPLETIONS_PATH = "/api/internal/predict/compat/chat/completions"

type ReasoningEffort = Literal["minimal", "low", "medium", "high"]
type ToolChoice = Literal["auto", "none", "required"] | dict[str, Any]


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
    prompt_tokens_details: dict[str, Any] | None = None
    completion_tokens_details: dict[str, Any] | None = None


class ChatCompletionChunk(BaseModel):
    id: str | None = None
    choices: list[ChatCompletionChoice] = Field(default_factory=list)
    created: int | None = None
    model: str | None = None
    object: str = "chat.completion.chunk"
    usage: ChatCompletionUsage | None = None
    system_fingerprint: str | None = None
    service_tier: str | None = None


class NucliaChatCompletionsError(RuntimeError):
    def __init__(
        self, message: str, *, provider_data: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.provider_data = provider_data or {}


def _request_error_detail(
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


class NucliaChatCompletionsClient:
    """Chat-completions transport backed by Hyperforge's shared NUA client."""

    def __init__(
        self,
        nua: AsyncNuaClient,
        *,
        timeout: float = 5 * 60,
        owns_client: bool = False,
    ) -> None:
        self.nua = nua
        self.timeout = timeout
        self._owns_client = owns_client

    @classmethod
    async def from_api_key(
        cls,
        api_key: str,
        *,
        base_url: str | None = None,
        timeout: float = 5 * 60,
    ) -> NucliaChatCompletionsClient:
        nua = await NUAConnection(key=api_key).connect(base_url=base_url)
        return cls(nua, timeout=timeout, owns_client=True)

    @classmethod
    def in_cluster(
        cls,
        *,
        url: str = "http://predict.learning.svc.cluster.local:8080",
        account: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 5 * 60,
    ) -> NucliaChatCompletionsClient:
        return cls(
            AsyncNuaClient.internal(
                url=url, account=account, headers=dict(headers or {})
            ),
            timeout=timeout,
            owns_client=True,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self.nua.aclose()

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = request.model_dump(exclude_none=True)
        for attempt in range(1, 4):
            yielded_chunk = False
            try:
                async for data in self.nua.chat_completions_stream(
                    payload, timeout=self.timeout
                ):
                    if data.get("type") == "status" and data.get("code") == "ERROR":
                        raise NucliaChatCompletionsError(
                            data.get("details") or "Generation failed",
                            provider_data=data,
                        )
                    chunk = ChatCompletionChunk.model_validate(data)
                    yielded_chunk = True
                    yield chunk
                return
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                detail, provider_data = _request_error_detail(exc)
                error = NucliaChatCompletionsError(
                    f"Nuclia chat completions request failed: {detail}",
                    provider_data=provider_data,
                )
            except NucliaChatCompletionsError as exc:
                error = exc

            status = error.provider_data.get("http_status")
            retryable = status is None or status in {408, 429} or 500 <= status < 600
            if yielded_chunk or not retryable or attempt == 3:
                raise error
            delay = 0.5 * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            logger.warning(
                "Retrying Nuclia generation after pre-output failure: attempt=%d/%d model=%s delay=%.2fs error=%s",
                attempt,
                3,
                request.model,
                delay,
                error,
                extra={
                    "attempt": attempt,
                    "max_attempts": 3,
                    "model": request.model,
                    "message_count": len(request.messages),
                    "tool_count": len(request.tools),
                    "provider_data": error.provider_data,
                    "error": str(error),
                },
            )
            await asyncio.sleep(delay)


@dataclass
class ModelDelta:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[HarnessToolCall] = field(default_factory=list)
    input_tokens: float = 0
    output_tokens: float = 0
    trace_id: str | None = None
    model: str | None = None


class ModelClient(Protocol):
    def stream(
        self,
        *,
        model: str,
        reasoning_effort: ReasoningEffort | None,
        messages: list[HarnessMessage],
        tools: list[HarnessTool],
        execution_context: dict[str, object],
    ) -> AsyncIterator[ModelDelta]: ...


class NucliaModelClient:
    """Harness adapter for Nuclia's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        client: NucliaChatCompletionsClient,
        *,
        reasoning_effort: ReasoningEffort = "medium",
        max_tokens: int = 50_000,
        temperature: float | None = None,
        top_p: float | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: str | list[str] | None = None,
        response_format: ChatCompletionResponseFormat | None = None,
        json_schema: dict[str, Any] | None = None,
        tool_choice: ToolChoice = "auto",
    ) -> None:
        self.client = client
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.stop = stop
        self.response_format = response_format
        self.json_schema = json_schema
        self.tool_choice = tool_choice

    @classmethod
    async def from_api_key(
        cls,
        api_key: str,
        *,
        base_url: str | None = None,
        **options: Any,
    ) -> NucliaModelClient:
        nua = await NUAConnection(key=api_key).connect(base_url=base_url)
        return cls(NucliaChatCompletionsClient(nua, owns_client=True), **options)

    @classmethod
    async def in_cluster(
        cls,
        *,
        url: str = "http://predict.learning.svc.cluster.local:8080",
        account: str | None = None,
        headers: Mapping[str, str] | None = None,
        **options: Any,
    ) -> NucliaModelClient:
        nua = AsyncNuaClient.internal(
            url=url, account=account, headers=dict(headers or {})
        )
        return cls(NucliaChatCompletionsClient(nua, owns_client=True), **options)

    @property
    def nua(self) -> AsyncNuaClient:
        return self.client.nua

    async def aclose(self) -> None:
        await self.client.aclose()

    async def stream(
        self,
        *,
        model: str,
        reasoning_effort: ReasoningEffort | None,
        messages: list[HarnessMessage],
        tools: list[HarnessTool],
        execution_context: dict[str, object],
    ) -> AsyncIterator[ModelDelta]:
        converted_messages = self._messages(messages)
        self._validate_tool_history(converted_messages)
        request = ChatCompletionRequest(
            messages=converted_messages,
            user=str(execution_context.get("user_id", "system")),
            model=model,
            reasoning_effort=reasoning_effort or self.reasoning_effort,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            stop=self.stop,
            response_format=self.response_format,
            json_schema=self.json_schema,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            tool_choice=self.tool_choice,
        )
        pending_calls: dict[int, dict[str, str]] = {}
        emitted_calls: set[int] = set()

        def completed_calls() -> list[HarnessToolCall]:
            completed: list[HarnessToolCall] = []
            for index, pending in sorted(pending_calls.items()):
                if index in emitted_calls or not pending["name"]:
                    continue
                try:
                    arguments = json.loads(pending["arguments"] or "{}")
                except json.JSONDecodeError as exc:
                    arguments = {"_tool_error": f"Malformed tool arguments: {exc}"}
                if not isinstance(arguments, dict):
                    arguments = {
                        "_tool_error": "Tool arguments must decode to a JSON object"
                    }
                completed.append(
                    HarnessToolCall(
                        id=pending["id"] or None,
                        name=pending["name"],
                        arguments=arguments,
                    )
                )
                emitted_calls.add(index)
            return completed

        async for chunk in self.client.stream(request):
            if not chunk.choices:
                yield ModelDelta(
                    input_tokens=chunk.usage.prompt_tokens if chunk.usage else 0,
                    output_tokens=chunk.usage.completion_tokens if chunk.usage else 0,
                    trace_id=chunk.id,
                    model=chunk.model,
                )
                continue
            for choice in chunk.choices:
                for call in choice.delta.tool_calls:
                    pending = pending_calls.setdefault(
                        call.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if call.id:
                        pending["id"] = call.id
                    if call.function is not None:
                        if call.function.name:
                            pending["name"] += call.function.name
                        if call.function.arguments:
                            pending["arguments"] += call.function.arguments
                yield ModelDelta(
                    text=choice.delta.content or choice.delta.refusal or "",
                    reasoning=choice.delta.reasoning_content or "",
                    tool_calls=completed_calls()
                    if choice.finish_reason == "tool_calls"
                    else [],
                    input_tokens=chunk.usage.prompt_tokens if chunk.usage else 0,
                    output_tokens=chunk.usage.completion_tokens if chunk.usage else 0,
                    trace_id=chunk.id,
                    model=chunk.model,
                )
        remaining = completed_calls()
        if remaining:
            yield ModelDelta(tool_calls=remaining)

    @staticmethod
    def _messages(messages: Sequence[HarnessMessage]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "assistant":
                value: dict[str, Any] = {
                    "role": "assistant",
                    "content": message.content or None,
                }
                if message.tool_calls:
                    value["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in message.tool_calls
                    ]
                converted.append(value)
            elif message.role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }
                )
            else:
                converted.append({"role": message.role, "content": message.content})
        return converted

    @staticmethod
    def _validate_tool_history(messages: Sequence[dict[str, Any]]) -> None:
        pending: set[str] = set()
        for index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                if pending:
                    raise ValueError(
                        f"Incomplete tool history before message {index}; missing outputs for {sorted(pending)}"
                    )
                pending = {
                    str(call["id"])
                    for call in message.get("tool_calls", [])
                    if isinstance(call, dict) and call.get("id")
                }
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if tool_call_id not in pending:
                    raise ValueError(
                        f"Unexpected tool output at message {index}: {tool_call_id}"
                    )
                pending.remove(str(tool_call_id))
            elif pending:
                raise ValueError(
                    f"Incomplete tool history before message {index}; missing outputs for {sorted(pending)}"
                )
        if pending:
            raise ValueError(
                f"Incomplete tool history at end of messages; missing outputs for {sorted(pending)}"
            )
