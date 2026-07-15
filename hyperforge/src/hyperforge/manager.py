import json
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, Tuple, Union

from nuclia.lib.nua import AsyncNuaClient
from nuclia.lib.nua_responses import (
    ChatModel,
    Image,
    Message,
    Reasoning,
    RerankModel,
    RerankResponse,
    Tokens,
    UserPrompt,
)
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
from pydantic_core import ErrorDetails, ValidationError

from hyperforge.configure import get_driver_klass
from hyperforge.driver import Driver, DriverConfig
from hyperforge.interaction import StreamingChunk
from hyperforge.llm_config import LLMConfig
from hyperforge.models import TrackingInfo

# Type alias for parameters that accept either a plain model ID string
# or a structured LLMConfig object. This keeps backwards compatibility
# with agents that still pass raw strings.
ModelParam = Union[str, LLMConfig]


def _resolve_model_id(model: ModelParam) -> str:
    """Extract the model identifier string from a ModelParam.

    If `model` is already a string, return it as-is.
    If it's an LLMConfig, return its `model_id` field.
    """
    if isinstance(model, str):
        return model
    return model.model_id


def build_reasoning(model: ModelParam) -> Union[Reasoning, bool]:
    """Build a NUA Reasoning object from a ModelParam.

    If `model` is a plain string or has no reasoning configured, returns False
    (reasoning disabled). Otherwise, builds a `Reasoning` object from the
    LLMConfig's effective reasoning settings. NUA handles any necessary
    effort/budget_tokens normalization server-side.
    """
    if isinstance(model, str):
        return False
    effective = model.get_effective_reasoning()
    if effective is None:
        return False
    kwargs: dict[str, Any] = {}
    if effective.effort is not None:
        kwargs["effort"] = effective.effort.value
    if effective.budget_tokens is not None:
        kwargs["budget_tokens"] = effective.budget_tokens
    if not kwargs:
        return False
    return Reasoning(**kwargs)


class StreamCallback(Protocol):
    """Protocol for objects that can receive streaming chunks (e.g. memory)."""

    async def emit_streaming_chunk(
        self,
        chunk: StreamingChunk | None = None,
        *,
        reasoning: bool = False,
        agent_request: str | None = None,
    ) -> None: ...

    def get_streaming(self) -> bool: ...


def convert_errors(e: ValidationError):
    new_errors: list[ErrorDetails] = e.errors()
    for error in new_errors:
        print(json.dumps(error, indent=1))


class Manager:
    drivers: Dict[str, Driver]
    nua: AsyncNuaClient

    def __init__(self) -> None:
        self.drivers: Dict[str, Driver] = {}

    @classmethod
    async def from_config(
        cls,
        drivers: List[DriverConfig],
        nua: AsyncNuaClient,
    ):
        manager = cls()

        manager.nua = nua
        for driver in drivers:
            driver_class = get_driver_klass(
                driver.provider
            )  # Check if driver provider is valid
            manager.drivers[driver.identifier] = await driver_class.init(driver)

        return manager

    def nucliadbs(self) -> list[str]:
        result = []
        for key, value in self.drivers.items():
            if value.provider == "nucliadb":
                result.append(key)
        return result

    async def rerank(self, rerank: RerankModel) -> RerankResponse:
        return await self.nua.rerank(rerank)

    async def tokens_predict(self, text: str, model: str) -> Tokens:
        return await self.nua.tokens_predict(text=text, model=model)

    async def remi(
        self, question: str | None, answer: str | None, contexts: List[str] | None
    ) -> RemiResponse:
        # Evaluate if we answered
        remi_response = await self.nua.remi(
            RemiRequest(
                user_id="arag_evaluate",
                question=question,
                answer=answer,
                contexts=contexts,
            )
        )
        return remi_response

    def _build_extra_headers(
        self,
        tracking: TrackingInfo | None = None,
    ) -> dict[str, str]:
        headers: dict[str, str] = {"x-show-consumption": "true", "x-origin": "RAO"}
        if tracking is not None:
            headers["x-client-ident"] = tracking.rao_id
            headers["x-session"] = tracking.session
            headers["x-message"] = tracking.message
        return headers

    async def execute_raw(
        self,
        item: ChatModel,
        tracking: TrackingInfo | None = None,
        memory: Optional[StreamCallback] = None,
        module: str = "",
        agent_path: str = "",
        disable_streaming: bool = False,
    ) -> Tuple[GenerativeFullResponse, float, float]:
        """Execute a chat model request.

        When *memory* is provided, ``memory.get_streaming()`` returns ``True``,
        and *disable_streaming* is ``False``, the response is streamed to the
        caller via :meth:`execute_raw_streaming` — no per-agent conditional is
        needed.  All existing callers that omit *memory* are unaffected.
        """
        if memory is not None and not disable_streaming and memory.get_streaming():
            return await self.execute_raw_streaming(
                item, memory, module=module, agent_path=agent_path
            )

        try:
            resp = await self.nua.generate(
                body=item,
                extra_headers=self._build_extra_headers(tracking),
            )
        except ValidationError as e:
            convert_errors(e)
            raise

        if resp.answer is None:
            raise Exception("No object")

        if resp.consumption is None or resp.consumption.normalized_tokens is None:
            input_tokens = 0.0
            output_tokens = 0.0
        else:
            input_tokens = resp.consumption.normalized_tokens.input
            output_tokens = resp.consumption.normalized_tokens.output
        return (
            resp,
            input_tokens,
            output_tokens,
        )

    async def execute_raw_streaming(
        self,
        item: ChatModel,
        callback: StreamCallback,
        module: str = "",
        agent_path: str = "",
    ) -> Tuple[GenerativeFullResponse, float, float]:
        """Like execute_raw but streams chunks to the callback as they arrive,
        while accumulating the full GenerativeFullResponse.

        Calls callback.emit_streaming_chunk with source metadata before streaming,
        and callback.emit_streaming_chunk(last=True) when done."""
        result = GenerativeFullResponse(answer="")
        input_tokens = 0.0
        output_tokens = 0.0

        if module or agent_path:
            await callback.emit_streaming_chunk(agent_request=f"{module}@{agent_path}")

        async for chunk in self.nua.generate_stream(
            body=item, extra_headers={"x-show-consumption": "true", "x-origin": "RAO"}
        ):
            c = chunk.chunk
            if isinstance(c, TextGenerativeResponse):
                result.answer = (result.answer or "") + c.text
                await callback.emit_streaming_chunk(StreamingChunk(text=c.text))
            elif isinstance(c, ReasoningGenerativeResponse):
                result.reasoning = (result.reasoning or "") + c.text
                await callback.emit_streaming_chunk(
                    StreamingChunk(text=c.text), reasoning=True
                )
            elif isinstance(c, JSONGenerativeResponse):
                result.object = c.object
            elif isinstance(c, MetaGenerativeResponse):
                result.timings = c.timings
                result.learning_id = c.learning_id
                result.model_name = c.model_name
                result.trace_id = c.trace_id
            elif isinstance(c, CitationsGenerativeResponse):
                result.citations = c.citations
            elif isinstance(c, FootnoteCitationsGenerativeResponse):
                result.citation_footnote_to_context = c.footnote_to_context
            elif isinstance(c, StatusGenerativeResponse):
                result.code = c.code
            elif isinstance(c, ToolsGenerativeResponse):
                result.tools = c.tools
            elif isinstance(c, ConsumptionGenerative):
                result.consumption = Consumption(
                    normalized_tokens=c.normalized_tokens,
                    customer_key_tokens=c.customer_key_tokens,
                )
                if c.normalized_tokens is not None:
                    input_tokens = c.normalized_tokens.input
                    output_tokens = c.normalized_tokens.output

        await callback.emit_streaming_chunk(StreamingChunk(text="", last=True))
        return result, input_tokens, output_tokens

    async def execute_stream(
        self,
        prompt: str,
        user_id: str,
        model: ModelParam,
        query_context_images: Dict[str, Image] = {},
        system: Optional[str] = None,
        max_tokens: int = 2000,
        chat_history: List[Message] = [],
    ) -> AsyncIterator[GenerativeChunk]:
        """Streaming variant of execute.

        Yields GenerativeChunk objects as they arrive from the LLM. Callers
        are responsible for accumulating text/reasoning tokens and reading
        consumption metadata from the final chunks.
        """
        async for chunk in self.nua.generate_stream(
            body=ChatModel(
                system=system,
                user_id=user_id,
                question="",
                user_prompt=UserPrompt(prompt=prompt),
                format_prompt=False,
                generative_model=_resolve_model_id(model),
                reasoning=build_reasoning(model),
                query_context_images=query_context_images,
                max_tokens=max_tokens,
                chat_history=chat_history,
            ),
            extra_headers={"x-show-consumption": "true", "x-origin": "RAO"},
        ):
            yield chunk

    async def execute(
        self,
        prompt: str,
        user_id: str,
        model: ModelParam,
        query_context_images: Dict[str, Image] = {},
        system: Optional[str] = None,
        max_tokens: int = 2000,
        chat_history: List[Message] = [],
        tracking: TrackingInfo | None = None,
    ) -> Tuple[str, float, float, str | None]:
        try:
            resp = await self.nua.generate(
                body=ChatModel(
                    system=system,
                    user_id=user_id,
                    question="",
                    user_prompt=UserPrompt(prompt=prompt),
                    format_prompt=False,
                    generative_model=_resolve_model_id(model),
                    reasoning=build_reasoning(model),
                    query_context_images=query_context_images,
                    max_tokens=max_tokens,
                    chat_history=chat_history,
                ),
                extra_headers=self._build_extra_headers(tracking),
            )
        except ValidationError as e:
            convert_errors(e)
            raise

        if resp is None or resp.answer is None:
            raise Exception("No object")
        if resp.consumption is None or resp.consumption.normalized_tokens is None:
            input_tokens = 0.0
            output_tokens = 0.0
        else:
            input_tokens = resp.consumption.normalized_tokens.input
            output_tokens = resp.consumption.normalized_tokens.output
        return (
            resp.answer,
            input_tokens,
            output_tokens,
            resp.code,
        )

    async def execute_from_context(
        self,
        prompt: str,
        user_id: str,
        model: ModelParam,
        images: Dict[str, Image],
        system: Optional[str] = None,
        tracking: TrackingInfo | None = None,
    ) -> Tuple[str, float, float]:
        try:
            resp = await self.nua.generate(
                body=ChatModel(
                    question="",
                    user_id=user_id,
                    user_prompt=UserPrompt(prompt=prompt),
                    format_prompt=False,
                    generative_model=_resolve_model_id(model),
                    reasoning=build_reasoning(model),
                    query_context_images=images,
                    system=system,
                ),
                extra_headers=self._build_extra_headers(tracking),
            )
        except ValidationError as e:
            convert_errors(e)
            raise

        if resp.answer is None:
            raise Exception("No object")
        if resp.consumption is None or resp.consumption.normalized_tokens is None:
            input_tokens = 0.0
            output_tokens = 0.0
        else:
            input_tokens = resp.consumption.normalized_tokens.input
            output_tokens = resp.consumption.normalized_tokens.output

        return (
            resp.answer,
            input_tokens,
            output_tokens,
        )

    async def execute_json(
        self,
        prompt: str,
        user_id: str,
        schema: Dict[str, Any],
        model: ModelParam,
        images: Dict[str, Image] = {},
        system: Optional[str] = None,
        max_tokens: int = 8192,
        tracking: TrackingInfo | None = None,
    ) -> Tuple[Dict[str, Any], float, float]:
        try:
            resp = await self.nua.generate(
                body=ChatModel(
                    user_id=user_id,
                    question="",
                    user_prompt=UserPrompt(prompt=prompt),
                    generative_model=_resolve_model_id(model),
                    reasoning=build_reasoning(model),
                    format_prompt=False,
                    query_context_images=images,
                    json_schema=schema,
                    system=system,
                    max_tokens=max_tokens,
                    citations=False,
                ),
                extra_headers=self._build_extra_headers(tracking),
            )

        except ValidationError as e:
            convert_errors(e)
            raise

        if resp.object is None:
            raise Exception("No object")

        if resp.consumption is None or resp.consumption.normalized_tokens is None:
            input_tokens = 0.0
            output_tokens = 0.0
        else:
            input_tokens = resp.consumption.normalized_tokens.input
            output_tokens = resp.consumption.normalized_tokens.output

        return (
            resp.object,
            input_tokens,
            output_tokens,
        )

    async def execute_json_citation(
        self,
        question: str,
        user_id: str,
        schema: Dict[str, Any],
        model: ModelParam,
        contexts: List[str] = [],
        images: Dict[str, Image] = {},
        system: Optional[str] = None,
        tracking: TrackingInfo | None = None,
    ) -> Tuple[Dict[str, Any], float, float, Dict[str, Any] | None]:
        try:
            resp = await self.nua.generate(
                body=ChatModel(
                    user_id=user_id,
                    question=question,
                    query_context=contexts,
                    generative_model=_resolve_model_id(model),
                    reasoning=build_reasoning(model),
                    format_prompt=True,
                    query_context_images=images,
                    json_schema=schema,
                    system=system,
                    max_tokens=2000,
                    citations=True,
                ),
                extra_headers=self._build_extra_headers(tracking),
            )

        except ValidationError as e:
            convert_errors(e)
            raise

        if resp.object is None:
            raise Exception("No object")

        if resp.consumption is None or resp.consumption.normalized_tokens is None:
            input_tokens = 0.0
            output_tokens = 0.0
        else:
            input_tokens = resp.consumption.normalized_tokens.input
            output_tokens = resp.consumption.normalized_tokens.output

        return (
            resp.object,
            input_tokens,
            output_tokens,
            resp.citations,
        )
