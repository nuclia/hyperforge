from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Optional,
    Type,
    TypeVar,
    Union,
)

import backoff
from deprecated import deprecated
from httpx import AsyncClient
from nuclia.exceptions import NuaAPIException
from nuclia.lib.nua_responses import (
    ChatModel,
    ChatResponse,
    QueryInfo,
    RephraseModel,
    RerankModel,
    RerankResponse,
    Sentence,
    SummarizedModel,
    SummarizeModel,
    SummarizeResource,
    Tokens,
)
from nuclia_models.common.consumption import Consumption, ConsumptionGenerative
from nuclia_models.predict.generative_responses import (
    CitationsGenerativeResponse,
    GenerativeChunk,
    GenerativeFullResponse,
    JSONGenerativeResponse,
    MetaGenerativeResponse,
    StatusGenerativeResponse,
    TextGenerativeResponse,
    ToolsGenerativeResponse,
)
from nuclia_models.predict.remi import RemiRequest, RemiResponse
from pydantic import BaseModel

MB = 1024 * 1024
CHUNK_SIZE = 10 * MB
SENTENCE_PREDICT = "/api/v1/internal/predict/sentence"
CHAT_PREDICT = "/api/v1/internal/predict/chat"
SUMMARIZE_PREDICT = "/api/v1/internal/predict/summarize"
REPHRASE_PREDICT = "/api/v1/internal/predict/rephrase"
TOKENS_PREDICT = "/api/v1/internal/predict/tokens"
QUERY_PREDICT = "/api/v1/internal/predict/query"
REMI_PREDICT = "/api/v1/internal/predict/remi"
AGENTS_PREDICT = "/api/v1/internal/predict/run-agents"
RERANK = "/api/v1/internal/predict/rerank"

ConvertType = TypeVar("ConvertType", bound=BaseModel)


class Author(str, Enum):
    NUCLIA = "NUCLIA"
    USER = "USER"


class ContextItem(BaseModel):
    author: Author
    text: str


class RetriableRequestException(NuaAPIException):
    pass


class AsyncInternalNuaClient:
    def __init__(
        self,
        kbid: str | None,
        account: str | None,
        url: str,
    ):
        self.headers = {"X-STF-KBID": kbid} if kbid else {}
        if account:
            self.headers["X-STF-ACCOUNT"] = account

        self.stream_headers = self.headers.copy()
        self.stream_headers["Accept"] = "application/x-ndjson"

        self.url = url
        self.client = AsyncClient(headers=self.headers, base_url=url)
        self.stream_client = AsyncClient(headers=self.stream_headers, base_url=url)

    @backoff.on_exception(
        backoff.expo,
        (RetriableRequestException,),
        max_time=60,
        jitter=backoff.full_jitter,
    )
    async def _request(
        self,
        method: str,
        url: str,
        output: Type[ConvertType],
        payload: Optional[dict[Any, Any]] = None,
        timeout: int = 60,
    ) -> ConvertType:
        resp = await self.client.request(method, url, json=payload, timeout=timeout)
        if resp.status_code in (429, 512):
            raise RetriableRequestException(
                code=resp.status_code, detail=resp.content.decode()
            )
        if resp.status_code > 299:
            raise NuaAPIException(code=resp.status_code, detail=resp.content.decode())
        try:
            data = output.model_validate(resp.json())
        except Exception:
            data = output.model_validate(resp.content)
        return data

    @backoff.on_exception(
        backoff.expo,
        (RetriableRequestException,),
        max_time=60,
        jitter=backoff.full_jitter,
    )
    async def _stream(
        self,
        method: str,
        url: str,
        payload: Optional[dict[Any, Any]] = None,
        timeout: int = 60,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> AsyncIterator[GenerativeChunk]:
        async with self.stream_client.stream(
            method,
            url,
            json=payload,
            timeout=timeout,
            headers=extra_headers,
        ) as response:
            if response.status_code in (429, 512):
                raise RetriableRequestException(
                    code=response.status_code,
                    detail=(await response.aread()).decode(errors="ignore"),
                )
            elif response.status_code > 299:
                raise NuaAPIException(
                    code=response.status_code,
                    detail=(await response.aread()).decode(errors="ignore"),
                )
            async for json_body in response.aiter_lines():
                yield GenerativeChunk.model_validate_json(json_body)

    async def sentence_predict(
        self, text: str, model: Optional[str] = None
    ) -> Sentence:
        endpoint = f"{self.url}{SENTENCE_PREDICT}?text={text}"
        if model:
            endpoint += f"&model={model}"
        return await self._request("GET", endpoint, output=Sentence)

    async def tokens_predict(self, text: str, model: Optional[str] = None) -> Tokens:
        endpoint = f"{self.url}{TOKENS_PREDICT}?text={text}"
        if model:
            endpoint += f"&model={model}"
        return await self._request("GET", endpoint, output=Tokens)

    async def query_predict(
        self,
        text: str,
        semantic_model: Optional[str] = None,
        token_model: Optional[str] = None,
        generative_model: Optional[str] = None,
    ) -> QueryInfo:
        endpoint = f"{self.url}{QUERY_PREDICT}?text={text}"
        if semantic_model:
            endpoint += f"&semantic_model={semantic_model}"
        if token_model:
            endpoint += f"&token_model={token_model}"
        if generative_model:
            endpoint += f"&generative_model={generative_model}"
        return await self._request("GET", endpoint, output=QueryInfo)

    @deprecated(version="2.1.0", reason="You should use generate function")
    async def generate_predict(
        self, body: ChatModel, model: Optional[str] = None, timeout: int = 300
    ) -> ChatResponse:
        endpoint = f"{self.url}{CHAT_PREDICT}"
        if model:
            endpoint += f"?model={model}"

        return await self._request(
            "POST",
            endpoint,
            payload=body.model_dump(),
            output=ChatResponse,
            timeout=timeout,
        )

    async def generate(
        self,
        body: ChatModel,
        model: Optional[str] = None,
        timeout: int = 300,
        extra_headers: Optional[dict[str, str]] = None,
    ) -> GenerativeFullResponse:
        endpoint = f"{self.url}{CHAT_PREDICT}"
        if model:
            endpoint += f"?model={model}"
        result = GenerativeFullResponse(answer="")
        async for chunk in self._stream(
            "POST",
            endpoint,
            payload=body.model_dump(),
            timeout=timeout,
            extra_headers=extra_headers,
        ):
            if isinstance(chunk.chunk, TextGenerativeResponse):
                result.answer += chunk.chunk.text
            elif isinstance(chunk.chunk, JSONGenerativeResponse):
                result.object = chunk.chunk.object
            elif isinstance(chunk.chunk, MetaGenerativeResponse):
                result.timings = chunk.chunk.timings
            elif isinstance(chunk.chunk, CitationsGenerativeResponse):
                result.citations = chunk.chunk.citations
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

    async def generate_stream(
        self, body: ChatModel, model: Optional[str] = None, timeout: int = 300
    ) -> AsyncIterator[GenerativeChunk]:
        endpoint = f"{self.url}{CHAT_PREDICT}"
        if model:
            endpoint += f"?model={model}"

        async for gr in self._stream(
            "POST",
            endpoint,
            payload=body.model_dump(),
            timeout=timeout,
        ):
            yield gr

    async def summarize(
        self, documents: dict[str, str], model: Optional[str] = None, timeout: int = 300
    ) -> SummarizedModel:
        endpoint = f"{self.url}{SUMMARIZE_PREDICT}"
        if model:
            endpoint += f"?model={model}"

        body = SummarizeModel(
            resources={
                key: SummarizeResource(fields={"field": document})
                for key, document in documents.items()
            }
        )
        return await self._request(
            "POST",
            endpoint,
            payload=body.model_dump(),
            output=SummarizedModel,
            timeout=timeout,
        )

    async def rephrase(
        self,
        question: str,
        user_context: Optional[list[str]] = None,
        context: Optional[list[Union[dict, ContextItem]]] = None,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> RephraseModel:
        endpoint = f"{self.url}{REPHRASE_PREDICT}"
        if model:
            endpoint += f"?model={model}"

        body: dict[str, Any] = {
            "question": question,
            "user_context": user_context,
            "user_id": "USER",
        }
        if prompt:
            body["prompt"] = prompt
        if context:
            body["context"] = [
                c.model_dump(mode="json") if isinstance(c, BaseModel) else c
                for c in context
            ]
        return await self._request(
            "POST",
            endpoint,
            payload=body,
            output=RephraseModel,
        )

    async def remi(self, request: RemiRequest) -> RemiResponse:
        endpoint = f"{self.url}{REMI_PREDICT}"
        return await self._request(
            "POST",
            endpoint,
            payload=request.model_dump(),
            output=RemiResponse,
        )

    async def generate_retrieval(
        self,
        question: str,
        context: list[str],
        model: Optional[str] = None,
    ) -> ChatResponse:
        endpoint = f"{self.url}{CHAT_PREDICT}"
        if model:
            endpoint += f"?model={model}"
        body = ChatModel(
            question=question,
            retrieval=True,
            user_id="Nuclia PY CLI",
            query_context=context,
        )
        return await self._request(
            "POST", endpoint, payload=body.model_dump(), output=ChatResponse
        )

    async def rerank(self, model: RerankModel) -> RerankResponse:
        endpoint = f"{self.url}{RERANK}"
        return await self._request(
            "POST", endpoint, payload=model.model_dump(), output=RerankResponse
        )
