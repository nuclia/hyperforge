import json
import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum

import backoff
import httpx
from nuclia_models.predict.generative_responses import GenerativeChunk
from nucliadb_models.internal.predict import (
    QueryInfo,
    RerankModel,
    RerankResponse,
)
from nucliadb_models.search import ChatModel, RephraseModel
from nucliadb_protos.utils_pb2 import RelationNode
from nucliadb_telemetry import errors, metrics
from nucliadb_utils.exceptions import LimitsExceededError
from nucliadb_utils.settings import nuclia_settings
from nucliadb_utils.utilities import Utility, clean_utility, get_utility, set_utility
from pydantic import ValidationError

from nucliadb_agentic_api.ask import logger
from nucliadb_agentic_api.ask.predict_models import QueryModel


class SendToPredictError(Exception):
    pass


class ProxiedPredictAPIError(Exception):
    def __init__(self, status: int, detail: str = ""):
        self.status = status
        self.detail = detail


class NUAKeyMissingError(Exception):
    pass


class RephraseError(Exception):
    pass


class RephraseMissingContextError(Exception):
    pass


PUBLIC_PREDICT = "/api/v1/predict"
PRIVATE_PREDICT = "/api/internal/predict"
SENTENCE = "/sentence"
TOKENS = "/tokens"
QUERY = "/query"
SUMMARIZE = "/summarize"
CHAT = "/chat"
REPHRASE = "/rephrase"
FEEDBACK = "/feedback"
RERANK = "/rerank"

NUCLIA_LEARNING_ID_HEADER = "NUCLIA-LEARNING-ID"
NUCLIA_LEARNING_MODEL_HEADER = "NUCLIA-LEARNING-MODEL"
NUCLIA_LEARNING_CHAT_HISTORY_HEADER = "NUCLIA-LEARNING-CHAT-HISTORY"

predict_observer = metrics.Observer(
    "predict_engine",
    labels={"type": ""},
    error_mappings={
        "over_limits": LimitsExceededError,
        "predict_api_error": SendToPredictError,
    },
)


RETRIABLE_EXCEPTIONS = (httpx.ConnectError, httpx.ConnectTimeout)
MAX_TRIES = 2


class AnswerStatusCode(str, Enum):
    SUCCESS = "0"
    ERROR = "-1"
    NO_CONTEXT = "-2"
    NO_RETRIEVAL_DATA = "-3"

    def prettify(self) -> str:
        return {
            AnswerStatusCode.SUCCESS: "success",
            AnswerStatusCode.ERROR: "error",
            AnswerStatusCode.NO_CONTEXT: "no_context",
            AnswerStatusCode.NO_RETRIEVAL_DATA: "no_retrieval_data",
        }[self]


@dataclass
class RephraseResponse:
    rephrased_query: str
    use_chat_history: bool | None


def get_predict() -> "PredictEngine":
    return get_utility(Utility.PREDICT)  # type: ignore


async def start_predict_engine():
    predict_util = PredictEngine(
        nuclia_settings.nuclia_inner_predict_url,
        nuclia_settings.nuclia_public_url,
        nuclia_settings.nuclia_service_account,
        nuclia_settings.nuclia_zone,
        nuclia_settings.onprem,
        nuclia_settings.local_predict,
        nuclia_settings.local_predict_headers,
    )
    await predict_util.initialize()
    set_utility(Utility.PREDICT, predict_util)


async def stop_predict_engine():
    predict_util = get_utility(Utility.PREDICT)
    if predict_util is None:
        return

    clean_utility(Utility.PREDICT)
    await predict_util.finalize()


def convert_relations(data: dict[str, list[dict[str, str]]]) -> list[RelationNode]:
    result = []
    for token in data["tokens"]:
        text = token["text"]
        klass = token["ner"]
        result.append(
            RelationNode(value=text, ntype=RelationNode.NodeType.ENTITY, subtype=klass)
        )
    return result


class PredictEngine:
    def __init__(
        self,
        cluster_url: str | None = None,
        public_url: str | None = None,
        nuclia_service_account: str | None = None,
        zone: str | None = None,
        onprem: bool = False,
        local_predict: bool = False,
        local_predict_headers: dict[str, str] | None = None,
    ):
        self.nuclia_service_account = nuclia_service_account
        self.cluster_url = cluster_url
        if public_url is not None:
            self.public_url: str | None = public_url.format(zone=zone)
        else:
            self.public_url = None
        self.zone = zone
        self.onprem = onprem
        self.local_predict = local_predict
        self.local_predict_headers = local_predict_headers
        self.session: httpx.AsyncClient | None = None

    async def initialize(self):
        limits = httpx.Limits(max_connections=150, max_keepalive_connections=50)
        timeout = httpx.Timeout(30.0)
        self.session = httpx.AsyncClient(
            limits=limits,
            timeout=timeout,
        )

    async def finalize(self):
        if self.session:
            await self.session.aclose()

    def check_nua_key_is_configured_for_onprem(self):
        if self.onprem and (
            self.nuclia_service_account is None and self.local_predict is False
        ):
            raise NUAKeyMissingError()

    def get_predict_url(self, endpoint: str, kbid: str) -> str:
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        if self.onprem:
            # On-prem NucliaDB uses the public URL for the predict API. Examples:
            # /api/v1/predict/chat/{kbid}
            # /api/v1/predict/rephrase/{kbid}
            return f"{self.public_url}{PUBLIC_PREDICT}{endpoint}/{kbid}"
        else:
            return f"{self.cluster_url}{PRIVATE_PREDICT}{endpoint}"

    def get_predict_headers(self, kbid: str) -> dict[str, str]:
        if self.onprem:
            headers = {"X-STF-NUAKEY": f"Bearer {self.nuclia_service_account}"}
            if self.local_predict_headers is not None:
                headers.update(self.local_predict_headers)
            return headers
        else:
            return {"X-STF-KBID": kbid}

    async def check_response(
        self, kbid: str, resp: httpx.Response, expected_status: int = 200
    ) -> None:
        if resp.status_code == expected_status:
            return

        # Ensure the body is loaded before reading it (needed for streaming responses)
        if not resp.is_stream_consumed:
            await resp.aread()

        if resp.status_code == 402:
            data = resp.json()
            raise LimitsExceededError(402, data["detail"])

        try:
            data = resp.json()
            try:
                detail = data["detail"]
            except (KeyError, TypeError):
                detail = data
        except (json.decoder.JSONDecodeError, ValueError):
            detail = resp.text

        is_5xx_error = resp.status_code > 499
        # NOTE: 512 is a special status code sent by learning predict api indicating that the error
        # is related to an external generative model, so we don't want to log it as an error
        is_external_generative_error = resp.status_code == 512
        log_level = (
            logging.ERROR
            if is_5xx_error and not is_external_generative_error
            else logging.INFO
        )
        logger.log(
            log_level,
            "Predict API error",
            extra=dict(
                kbid=kbid,
                url=str(resp.url),
                status_code=resp.status_code,
                detail=detail,
            ),
        )
        raise ProxiedPredictAPIError(status=resp.status_code, detail=detail)

    @backoff.on_exception(
        backoff.expo,
        RETRIABLE_EXCEPTIONS,
        jitter=backoff.random_jitter,
        max_tries=MAX_TRIES,
    )
    async def make_request(self, method: str, **request_args) -> httpx.Response:
        if not self.session:
            raise RuntimeError("PredictEngine session is not initialized")
        func = getattr(self.session, method.lower())
        return await func(**request_args)

    async def make_stream_request(self, method: str, **request_args) -> httpx.Response:
        """Open a streaming request and return the response without reading the body.
        The caller is responsible for consuming and closing the response."""
        if not self.session:
            raise RuntimeError("PredictEngine session is not initialized")
        request = self.session.build_request(method.upper(), **request_args)
        return await self.session.send(request, stream=True)

    @predict_observer.wrap({"type": "rephrase"})
    async def rephrase_query(self, kbid: str, item: RephraseModel) -> RephraseResponse:
        try:
            self.check_nua_key_is_configured_for_onprem()
        except NUAKeyMissingError:
            error = "Nuclia Service account is not defined so could not rephrase query"
            logger.warning(error)
            raise SendToPredictError(error)

        resp = await self.make_request(
            "POST",
            url=self.get_predict_url(REPHRASE, kbid),
            json=item.model_dump(),
            headers=self.get_predict_headers(kbid),
        )
        await self.check_response(kbid, resp, expected_status=200)
        return _parse_rephrase_response(resp)

    @predict_observer.wrap({"type": "chat_ndjson"})
    async def chat_query_ndjson(
        self, kbid: str, item: ChatModel, extra_headers: dict[str, str] | None = None
    ) -> tuple[str, str, AsyncGenerator[GenerativeChunk, None]]:
        """
        Chat query using the new stream format
        Format specs: https://github.com/ndjson/ndjson-spec
        """
        try:
            self.check_nua_key_is_configured_for_onprem()
        except NUAKeyMissingError:
            error = "Nuclia Service account is not defined so the chat operation could not be performed"
            logger.warning(error)
            raise SendToPredictError(error)

        # The ndjson format is triggered by the Accept header
        headers = self.get_predict_headers(kbid)
        headers["Accept"] = "application/x-ndjson"

        resp = await self.make_stream_request(
            "POST",
            url=self.get_predict_url(CHAT, kbid),
            json=item.model_dump(),
            headers={**headers, **(extra_headers or {})},
            timeout=httpx.Timeout(180.0, read=120.0),
        )
        await self.check_response(kbid, resp, expected_status=200)
        ident = resp.headers.get(NUCLIA_LEARNING_ID_HEADER) or "unknown"
        model = resp.headers.get(NUCLIA_LEARNING_MODEL_HEADER) or "unknown"
        return ident, model, get_chat_ndjson_generator(resp)

    @predict_observer.wrap({"type": "query"})
    async def query(
        self,
        kbid: str,
        item: QueryModel,
    ) -> QueryInfo:
        """
        Query endpoint: returns information to be used by NucliaDB at retrieval time, for instance:
        - The embeddings
        - The entities
        - The stop words
        - The semantic threshold
        - etc.

        :param kbid: KnowledgeBox ID
        :param sentence: The query sentence
        :param semantic_model: The semantic model to use to generate the embeddings
        :param generative_model: The generative model that will be used to generate the answer
        :param rephrase: If the query should be rephrased before calculating the embeddings for a better retrieval
        :param rephrase_prompt: Custom prompt to use for rephrasing
        """
        try:
            self.check_nua_key_is_configured_for_onprem()
        except NUAKeyMissingError:
            error = (
                "Nuclia Service account is not defined so could not ask query endpoint"
            )
            logger.warning(error)
            raise SendToPredictError(error)

        resp = await self.make_request(
            "POST",
            url=self.get_predict_url(QUERY, kbid),
            json=item.model_dump(),
            headers=self.get_predict_headers(kbid),
        )
        await self.check_response(kbid, resp, expected_status=200)
        data = resp.json()
        return QueryInfo(**data)

    @predict_observer.wrap({"type": "entities"})
    async def detect_entities(self, kbid: str, sentence: str) -> list[RelationNode]:
        try:
            self.check_nua_key_is_configured_for_onprem()
        except NUAKeyMissingError:
            logger.warning(
                "Nuclia Service account is not defined so could not retrieve entities from the query"
            )
            return []

        resp = await self.make_request(
            "GET",
            url=self.get_predict_url(TOKENS, kbid),
            params={"text": sentence},
            headers=self.get_predict_headers(kbid),
        )
        await self.check_response(kbid, resp, expected_status=200)
        data = resp.json()
        return convert_relations(data)

    @predict_observer.wrap({"type": "rerank"})
    async def rerank(self, kbid: str, item: RerankModel) -> RerankResponse:
        try:
            self.check_nua_key_is_configured_for_onprem()
        except NUAKeyMissingError:
            error = "Nuclia Service account is not defined. Rerank operation could not be performed"
            logger.warning(error)
            raise SendToPredictError(error)
        resp = await self.make_request(
            "POST",
            url=self.get_predict_url(RERANK, kbid),
            json=item.model_dump(),
            headers=self.get_predict_headers(kbid),
        )
        await self.check_response(kbid, resp, expected_status=200)
        data = resp.json()
        return RerankResponse.model_validate(data)


def get_chat_ndjson_generator(
    response: httpx.Response,
) -> AsyncGenerator[GenerativeChunk, None]:
    async def _parse_generative_chunks(response: httpx.Response):
        try:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    yield GenerativeChunk.model_validate_json(line.strip())
                except ValidationError as ex:
                    errors.capture_exception(ex)
                    logger.error(f"Invalid chunk received: {line}")
                    continue
        finally:
            await response.aclose()

    return _parse_generative_chunks(response)


def _parse_rephrase_response(
    resp: httpx.Response,
) -> RephraseResponse:
    """
    Predict api is returning a json payload that is a string with the following format:
    <rephrased_query><status_code>
    where status_code is "0" for success, "-1" for error and "-2" for no context
    it will raise an exception if the status code is not 0
    """
    content = resp.json()

    if content.endswith("0"):
        content = content[:-1]
    elif content.endswith("-1"):
        raise RephraseError(content[:-2])
    elif content.endswith("-2"):
        raise RephraseMissingContextError(content[:-2])

    use_chat_history = None
    if NUCLIA_LEARNING_CHAT_HISTORY_HEADER in resp.headers:
        use_chat_history = resp.headers[NUCLIA_LEARNING_CHAT_HISTORY_HEADER] == "true"
    return RephraseResponse(rephrased_query=content, use_chat_history=use_chat_history)
