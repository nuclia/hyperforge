from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx

from hyperforge.completions import ChatCompletionChoice as ChatCompletionChoice
from hyperforge.completions import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponseFormat,
    NucliaChatCompletionsError,
    ReasoningEffort,
    ToolChoice,
    request_error_detail,
)
from hyperforge.completions import (
    ChatCompletionDelta as ChatCompletionDelta,
)
from hyperforge.completions import (
    ChatCompletionToolCallDelta as ChatCompletionToolCallDelta,
)
from hyperforge.completions import (
    ChatCompletionToolCallFunctionDelta as ChatCompletionToolCallFunctionDelta,
)
from hyperforge.completions import (
    ChatCompletionUsage as ChatCompletionUsage,
)
from hyperforge.exceptions import OpenAIChatCompletionsError
from hyperforge.llm import AsyncNuaClient, NUAConnection

from .models import HarnessMessage, HarnessToolCall
from .tools import HarnessTool

logger = logging.getLogger(__name__)

PUBLIC_CHAT_COMPLETIONS_PATH = "/api/v1/predict/compat/chat/completions"
INTERNAL_CHAT_COMPLETIONS_PATH = "/api/internal/predict/compat/chat/completions"


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
                detail, provider_data = request_error_detail(exc)
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


class OpenAIChatCompletionsClient:
    """Streaming transport for a standard OpenAI chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 5 * 60,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        default_headers = {"accept": "text/event-stream"}
        if api_key:
            default_headers["authorization"] = f"Bearer {api_key}"
        default_headers.update(headers or {})
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = default_headers
        self.timeout = timeout
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http_client.aclose()

    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = request.model_dump(exclude_none=True)
        try:
            async with self.http_client.stream(
                "POST",
                self.url,
                json=payload,
                headers=self.headers,
                timeout=self.timeout,
            ) as response:
                if response.is_error:
                    await response.aread()
                response.raise_for_status()
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":") or line.startswith("event:"):
                        continue
                    if line == "data: [DONE]":
                        return
                    if line.startswith("data:"):
                        line = line[5:].lstrip()
                    yield ChatCompletionChunk.model_validate_json(line)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            detail, _ = request_error_detail(exc)
            raise OpenAIChatCompletionsError(
                f"OpenAI chat completions request failed: {detail}"
            ) from exc


class ChatCompletionsClient(Protocol):
    async def aclose(self) -> None: ...

    def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]: ...


@dataclass
class ModelDelta:
    text: str = ""
    reasoning: str = ""
    tool_calls: list[HarnessToolCall] = field(default_factory=list)
    input_tokens: float = 0
    output_tokens: float = 0
    nuclia_input_tokens: float = 0
    nuclia_output_tokens: float = 0
    model_input_tokens: float = 0
    model_output_tokens: float = 0
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
        client: ChatCompletionsClient,
        *,
        reasoning_effort: ReasoningEffort | None = "medium",
        max_tokens: int | None = 50_000,
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
        return cast(NucliaChatCompletionsClient, self.client).nua

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
            tool_choice=self.tool_choice if tools else None,
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
                    input_tokens=chunk.usage.input_tokens if chunk.usage else 0,
                    output_tokens=chunk.usage.output_tokens if chunk.usage else 0,
                    nuclia_input_tokens=(chunk.usage.nuclia_input_tokens or 0)
                    if chunk.usage
                    else 0,
                    nuclia_output_tokens=(chunk.usage.nuclia_output_tokens or 0)
                    if chunk.usage
                    else 0,
                    model_input_tokens=(chunk.usage.model_input_tokens or 0)
                    if chunk.usage
                    else 0,
                    model_output_tokens=(chunk.usage.model_output_tokens or 0)
                    if chunk.usage
                    else 0,
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
                    input_tokens=chunk.usage.input_tokens if chunk.usage else 0,
                    output_tokens=chunk.usage.output_tokens if chunk.usage else 0,
                    nuclia_input_tokens=(chunk.usage.nuclia_input_tokens or 0)
                    if chunk.usage
                    else 0,
                    nuclia_output_tokens=(chunk.usage.nuclia_output_tokens or 0)
                    if chunk.usage
                    else 0,
                    model_input_tokens=(chunk.usage.model_input_tokens or 0)
                    if chunk.usage
                    else 0,
                    model_output_tokens=(chunk.usage.model_output_tokens or 0)
                    if chunk.usage
                    else 0,
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


class OpenAIModelClient(NucliaModelClient):
    """Harness adapter for any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 5 * 60,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        **options: Any,
    ) -> None:
        options.setdefault("reasoning_effort", None)
        options.setdefault("max_tokens", None)
        super().__init__(
            OpenAIChatCompletionsClient(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
                headers=headers,
                http_client=http_client,
            ),
            **options,
        )
