import json
from asyncio import Timeout
from collections.abc import AsyncIterator
from typing import Any, Mapping, Optional, Self

import httpx
from nuclia.lib.nua import (
    AsyncGenerateStream,
    ContextItem,
    GenerateStreamResponse,
    NuaEndpoint,
    QueryRequest,
    RephraseRequest,
)
from nuclia.lib.nua import (
    AsyncNuaClient as NucliaAsyncNuaClient,
)
from nuclia.lib.nua_responses import (
    ChatModel,
    ChatResponse,
    ProcessRequestStatus,
    ProcessRequestStatusResults,
    PushResponseV2,
    QueryInfo,
    RephraseModel,
    RerankModel,
    RerankResponse,
    Sentence,
    SummarizedModel,
    Tokens,
)
from nuclia.sdk import AsyncNucliaAuth
from nuclia_models.common.consumption import Consumption
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    ConsumptionGenerative,
    FootnoteCitationsGenerativeResponse,
    GenerativeChunk,
    GenerativeFullResponse,
    JSONGenerativeResponse,
    MetaGenerativeResponse,
    ReasoningGenerativeResponse,
    StatusGenerativeResponse,
    TextGenerativeResponse,
    ToolsGenerativeResponse,
)
from nuclia_models.predict.remi import RemiRequest, RemiResponse
from pydantic import BaseModel

from hyperforge.completions import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    request_error_detail,
    transform_chat_to_openai_messages,
)
from hyperforge.exceptions import OpenAIChatCompletionsError


class NuaBaseModel(BaseModel):
    async def connect(self):
        raise NotImplementedError("Must implement connect method in subclass")

    @classmethod
    async def connect_internal(cls, kbid: str | None, account: str | None, url: str):
        raise NotImplementedError("Must implement connect_internal method in subclass")


class AsyncNuaClient(NucliaAsyncNuaClient):
    """Hyperforge NUA client, including the chat-completions compatibility API."""

    @classmethod
    def internal(
        cls,
        url: str,
        kbid: str | None = None,
        account: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> Self:
        return cls(
            region=url,
            account=account or "",
            headers=headers,
            endpoint=NuaEndpoint.INTERNAL,
            kbid=kbid,
        )

    async def chat_completions_stream(
        self, payload: dict[str, Any], *, timeout: float = 5 * 60
    ) -> AsyncIterator[dict[str, Any]]:
        path = (
            "/api/internal/predict/compat/chat/completions"
            if self.endpoint == NuaEndpoint.INTERNAL
            else "/api/v1/predict/compat/chat/completions"
        )
        request = {**payload, "stream": True}
        async with self.stream_client.stream(
            "POST",
            f"{self.url}{path}",
            json=request,
            headers={"accept": "text/event-stream"},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                detail = (await response.aread()).decode(errors="replace")
                error = httpx.HTTPStatusError(
                    f"Nuclia chat completions API error: {response.status_code} - {detail}",
                    request=response.request,
                    response=response,
                )
                raise error
            async for line in response.aiter_lines():
                line = line.strip()
                if not line or line.startswith(":") or line.startswith("event:"):
                    continue
                if line == "data: [DONE]":
                    return
                if line.startswith("data:"):
                    line = line[5:].lstrip()
                yield json.loads(line)


class NoopNuaClient(AsyncNuaClient):  # pragma: no cover
    """
    A no-op NUA client used when no LLM backend is configured.

    Agents that don't need LLM calls (e.g. the ``static`` context agent) work
    fine with this client.  Any method that actually tries to call the NUA API
    will raise a ``RuntimeError`` with a clear message so users know they need
    to configure an LLM backend.
    """

    def __init__(self) -> None:  # type: ignore[override]
        # Intentionally do NOT call super().__init__() — we have no real
        # token/account/region and don't want side-effects from the parent.
        pass

    def _not_configured(self, method: str) -> None:
        raise RuntimeError(
            f"NoopNuaClient: '{method}' was called but no LLM backend is "
            "configured.  Set EXTERNAL_NUA_API_KEY, INTERNAL_NUA=true, "
            "or LOCAL_OPENAI to enable LLM-based agents."
        )

    async def aclose(self) -> None:
        pass

    async def generate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("generate")

    async def chat(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("chat")

    async def summarize(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("summarize")

    async def rephrase(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("rephrase")

    async def sentence(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("sentence")

    async def tokens(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("tokens")

    async def query(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self._not_configured("query")

    async def chat_completions_stream(
        self, payload: dict[str, Any], *, timeout: float = 5 * 60
    ) -> AsyncIterator[dict[str, Any]]:
        self._not_configured("chat_completions_stream")
        if False:
            yield {}


class NUAConnection(NuaBaseModel):
    key: str

    async def connect(self, *, base_url: str | None = None):
        na = AsyncNucliaAuth()
        client_id, account_type, account, base_region = await na.validate_nua(self.key)
        if account is None or base_region is None:
            raise Exception("Could not connect to NUA")
        return AsyncNuaClient(
            token=self.key, account=account, region=base_url or base_region
        )

    @classmethod
    async def connect_internal(cls, kbid: str | None, account: str | None, url: str):
        return AsyncNuaClient.internal(url=url, kbid=kbid, account=account)


class AsyncLocalOpenAIClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 5 * 60,
        headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
        model: str | None = None,
    ):
        default_headers = {"accept": "text/event-stream"}
        if api_key:
            default_headers["authorization"] = f"Bearer {api_key}"
        default_headers.update(headers or {})
        self.base_url = base_url
        self.headers = default_headers
        self.timeout = timeout
        self.model = model
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.AsyncClient()

    @classmethod
    def internal(
        cls,
        url: str,
        kbid: str | None = None,
        account: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> "AsyncNuaClient":
        """Create a client for the hosted internal Predict API."""
        raise NotImplementedError("Not supported")

    @classmethod
    def onprem(
        cls,
        public_url: str,
        service_account: str | None = None,
        zone: str | None = None,
        kbid: str | None = None,
        local_predict: bool = False,
        local_predict_headers: dict[str, str] | None = None,
    ) -> "AsyncNuaClient":
        """Create a client for the public, KB-scoped on-prem Predict API."""
        raise NotImplementedError("Not supported")

    async def aclose(self) -> None:
        if self._owns_client:
            await self.http_client.aclose()

    async def __aenter__(self) -> "AsyncLocalOpenAIClient":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:
        await self.aclose()

    async def add_config_predict(self, kbid: str, config: Any):
        raise NotImplementedError("Not available")

    async def del_config_predict(self, kbid: str):
        raise NotImplementedError("Not available")

    async def update_config_predict(self, kbid: str, config: Any):
        raise NotImplementedError("Not available")

    async def schema_predict(self, kbid: Optional[str] = None) -> Any:
        raise NotImplementedError("Not available")

    async def config_predict(self, kbid: Optional[str] = None) -> Any:
        raise NotImplementedError("Not available")

    async def sentence_predict(
        self,
        text: str,
        model: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> Sentence:
        raise NotImplementedError("Not implemented")

    async def query_predict(
        self,
        request: str | QueryRequest,
        semantic_model: str | None = None,
        token_model: str | None = None,
        generative_model: str | None = None,
        extra_headers: Optional[dict[str, str]] = None,
        *,
        kbid: str | None = None,
        timeout: int = 60,
    ) -> QueryInfo:
        """Call the Predict query endpoint."""
        raise NotImplementedError("Not implemented")

    async def tokens_predict(
        self,
        text: str,
        model: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        *,
        kbid: str | None = None,
        timeout: int = 60,
    ) -> Tokens:
        """Call Predict's token endpoint for a hosted or on-prem KB."""
        raise NotImplementedError("Not implemented")

    async def _chat_completions_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        payload = request.model_dump(exclude_none=True)
        if not payload["tools"]:
            payload.pop("tools")
        try:
            async with self.http_client.stream(
                "POST",
                f"{self.base_url.rstrip('/')}/chat/completions",
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

    def _request(self, body: ChatModel, model: str | None) -> ChatCompletionRequest:
        messages, tools, tool_choice, _ = transform_chat_to_openai_messages(body)
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            message_data = dict(message)
            message_type = message_data.get("type")
            if message_type == "function_call":
                tool_call = {
                    "id": message_data["call_id"],
                    "type": "function",
                    "function": {
                        "name": message_data["name"],
                        "arguments": message_data["arguments"],
                    },
                }
                if chat_messages and chat_messages[-1].get("role") == "assistant":
                    chat_messages[-1].setdefault("tool_calls", []).append(tool_call)
                else:
                    chat_messages.append(
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [tool_call],
                        }
                    )
                continue
            if message_type == "function_call_output":
                chat_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message_data["call_id"],
                        "content": message_data["output"],
                    }
                )
                continue
            content = message_data.get("content")
            if isinstance(content, list):
                content = [
                    (
                        {
                            "type": "image_url",
                            "image_url": {"url": part["image_url"]},
                        }
                        if part.get("type") == "input_image"
                        else part
                    )
                    for part in content
                ]
            chat_messages.append({"role": message_data["role"], "content": content})
        chat_tools = []
        for tool in tools:
            tool_data = dict(tool)
            if tool_data.get("type") == "function":
                chat_tools.append(
                    {
                        "type": "function",
                        "function": {
                            key: value
                            for key, value in tool_data.items()
                            if key != "type"
                        },
                    }
                )
        chat_tool_choice: Any = tool_choice
        if isinstance(tool_choice, dict):
            chat_tool_choice = "required"
        # if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        #     chat_tool_choice = {
        #         "type": "function",
        #         "function": {"name": tool_choice["name"]},
        #     }
        response_format = None
        if body.json_schema is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": body.json_schema,
                    "strict": True,
                },
            }
        reasoning_effort = None
        if body.reasoning and not isinstance(body.reasoning, bool):
            reasoning_effort = body.reasoning.effort
        return ChatCompletionRequest(
            messages=chat_messages,
            model=model or body.generative_model or self.model,
            stream=True,
            response_format=response_format,
            tools=chat_tools,
            tool_choice=chat_tool_choice,
            max_tokens=body.max_tokens or 50_000,
            reasoning_effort=reasoning_effort,
        )

    async def _generative_stream(
        self, body: ChatModel, model: str | None
    ) -> AsyncIterator[GenerativeChunk]:
        content = ""
        tool_calls: dict[int, dict[str, Any]] = {}
        usage = None
        async for chunk in self._chat_completions_stream(self._request(body, model)):
            usage = chunk.usage or usage
            for choice in chunk.choices:
                delta = choice.delta
                if delta.reasoning_content:
                    yield GenerativeChunk(
                        chunk=ReasoningGenerativeResponse(text=delta.reasoning_content)
                    )
                if delta.content:
                    content += delta.content
                    if body.json_schema is None:
                        yield GenerativeChunk(
                            chunk=TextGenerativeResponse(text=delta.content)
                        )
                for call in delta.tool_calls:
                    current = tool_calls.setdefault(
                        call.index, {"id": call.id, "name": "", "arguments": ""}
                    )
                    if call.id:
                        current["id"] = call.id
                    if call.function:
                        current["name"] += call.function.name or ""
                        current["arguments"] += call.function.arguments or ""

        if body.json_schema is not None and content:
            yield GenerativeChunk(
                chunk=JSONGenerativeResponse(object=json.loads(content))
            )
        if tool_calls:
            tools: dict[str, list[dict[str, Any]]] = {}
            for call in tool_calls.values():
                name = call["name"]
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                tools.setdefault(name, []).append(
                    {
                        "id": call["id"],
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            yield GenerativeChunk(chunk=ToolsGenerativeResponse(tools=tools))
        if usage is not None:
            token_detail = {
                "input": usage.prompt_tokens,
                "output": usage.completion_tokens,
                "image": 0,
            }
            yield GenerativeChunk(
                chunk=ConsumptionGenerative(
                    normalized_tokens=token_detail,
                    customer_key_tokens=token_detail,
                )
            )

    async def generate(
        self,
        body: ChatModel,
        model: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        timeout: int = 300,
    ) -> GenerativeFullResponse:

        result = GenerativeFullResponse(answer="")
        async for chunk in self._generative_stream(body, model):
            if isinstance(chunk.chunk, TextGenerativeResponse):
                result.answer += chunk.chunk.text
            elif isinstance(chunk.chunk, ReasoningGenerativeResponse):
                result.reasoning = (result.reasoning or "") + chunk.chunk.text
            elif isinstance(chunk.chunk, JSONGenerativeResponse):
                result.object = chunk.chunk.object
            elif isinstance(chunk.chunk, MetaGenerativeResponse):
                result.input_tokens = chunk.chunk.input_tokens
                result.output_tokens = chunk.chunk.output_tokens
                result.input_nuclia_tokens = chunk.chunk.input_nuclia_tokens
                result.output_nuclia_tokens = chunk.chunk.output_nuclia_tokens
                result.timings = chunk.chunk.timings
            elif isinstance(chunk.chunk, CitationsGenerativeResponse):
                result.citations = chunk.chunk.citations
            elif isinstance(chunk.chunk, FootnoteCitationsGenerativeResponse):
                result.citation_footnote_to_context = chunk.chunk.footnote_to_context
            elif isinstance(chunk.chunk, StatusGenerativeResponse):
                result.code = chunk.chunk.code
            elif isinstance(chunk.chunk, ToolsGenerativeResponse):
                result.tools = chunk.chunk.tools
            elif isinstance(chunk.chunk, ConsumptionGenerative):
                result.consumption = Consumption(
                    normalized_tokens=chunk.chunk.normalized_tokens,
                    customer_key_tokens=chunk.chunk.customer_key_tokens,
                )

        return result

    async def _open_generate_stream(
        self,
        body: ChatModel,
        model: str | None,
        extra_headers: dict[str, str] | None,
        timeout: Timeout | float | None,
        kbid: str | None,
    ) -> GenerateStreamResponse[AsyncIterator[GenerativeChunk]]:
        return GenerateStreamResponse(
            "unknown",
            model or body.generative_model or self.model or "unknown",
            self._generative_stream(body, model),
        )

    def generate_stream(
        self,
        body: ChatModel,
        model: str | None = None,
        extra_headers: Optional[dict[str, str]] = None,
        timeout: Timeout | float | None = 300,
        *,
        kbid: str | None = None,
        return_metadata: bool = False,
    ) -> AsyncGenerateStream:
        if return_metadata and timeout == 300:
            timeout = Timeout(30.0)
        return AsyncGenerateStream(
            lambda: self._open_generate_stream(
                body, model, extra_headers, timeout, kbid
            )
        )

    async def summarize(
        self,
        documents: dict[str, str],
        model: Optional[str] = None,
        extra_headers: Optional[dict[str, str]] = None,
        timeout: int = 300,
    ) -> SummarizedModel:
        raise NotImplementedError("Not available")

    async def rephrase(
        self,
        question: str | RephraseRequest,
        user_context: list[str] | None = None,
        context: list[dict[Any, Any] | ContextItem] | None = None,
        model: str | None = None,
        prompt: str | None = None,
        *,
        kbid: str | None = None,
        extra_headers: Optional[dict[str, str]] = None,
        timeout: int = 120,
    ) -> RephraseModel:
        """Call Predict's rephrase endpoint."""
        raise NotImplementedError("Process link not available")

    async def remi(
        self,
        request: RemiRequest,
        extra_headers: Optional[dict[str, str]] = None,
        timeout: int = 120,
    ) -> RemiResponse:
        raise NotImplementedError("Remi not available in OpenAI")

    async def generate_retrieval(
        self,
        question: str,
        context: list[str],
        model: Optional[str] = None,
    ) -> ChatResponse:
        raise NotImplementedError("Not available")

    async def process_link(
        self,
        url: str,
        kbid: Optional[str] = None,
        headers: dict[str, str] = {},
        cookies: dict[str, str] = {},
        localstorage: dict[str, str] = {},
    ) -> PushResponseV2:
        raise NotImplementedError("Process link not available")

    async def process_file(
        self, path: str, kbid: Optional[str] = None
    ) -> PushResponseV2:
        raise NotImplementedError("Process link not available")

    async def wait_for_processing(
        self, response: PushResponseV2, timeout: int = 30
    ) -> Optional[Any]:
        raise NotImplementedError("Processing not available")

    async def processing_status(self) -> ProcessRequestStatusResults:
        raise NotImplementedError("Processing not available")

    async def processing_id_status(self, process_id: str) -> ProcessRequestStatus:
        raise NotImplementedError("Processing not available")

    async def rerank(
        self,
        model: RerankModel,
        extra_headers: Optional[dict[str, str]] = None,
        *,
        kbid: str | None = None,
        timeout: int = 60,
    ) -> RerankResponse:
        """Call Predict's rerank endpoint."""
        raise NotImplementedError("Rerank not available")
