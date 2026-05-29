import asyncio
import contextvars
import time
from datetime import datetime, timezone
from typing import cast

import backoff
import mmh3
import nats
from fastapi import Request
from hyperforge.feature_flag import Features, has_feature
from nucliadb_models.retrieval import RawQuery, RetrievalRequest
from nucliadb_models.search import (
    AskRequest,
    ChatContextMessage,
    NucliaDBClientType,
    PromptContext,
    PromptContextOrder,
)
from nucliadb_protos import audit_pb2, utils_pb2
from nucliadb_protos.audit_pb2 import (
    AuditRequest,
    ChatContext,
    RetrievedContext,
)
from nucliadb_telemetry.jetstream import get_traced_jetstream, get_traced_nats_client
from nucliadb_utils import logger
from nucliadb_utils.settings import audit_settings
from nucliadb_utils.utilities import Utility, clean_utility, get_utility, set_utility
from opentelemetry.trace import INVALID_SPAN, format_trace_id, get_current_span
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from nucliadb_agentic_api.ask.predict import AnswerStatusCode
from nucliadb_agentic_api.ask.utils.proto import client_type


class RequestContext:
    def __init__(self: "RequestContext"):
        self.audit_request: AuditRequest = AuditRequest()
        self.start_time: float = time.monotonic()
        self.path: str = ""


request_context_var = contextvars.ContextVar[RequestContext | None](
    "request_context", default=None
)


def get_trace_id() -> str | None:
    span = get_current_span()
    if span is INVALID_SPAN:
        return None
    return format_trace_id(span.get_span_context().trace_id)


def get_request_context() -> RequestContext | None:
    return request_context_var.get()


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        context = RequestContext()
        token = request_context_var.set(context)
        context.audit_request.time.FromDatetime(datetime.now(tz=timezone.utc))
        context.audit_request.trace_id = get_trace_id() or ""
        context.path = request.url.path

        response = await call_next(request)

        # This task will run when the response finishes streaming
        # When dealing with streaming responses, AND if we depend on any state that only will be available once
        # the request is fully finished, the response we have after the dispatch call_next is not enough, as
        # there, no iteration of the streaming response has been done yet.
        response.background = BackgroundTask(self.enqueue_pending, context)

        # It is safe to reset the context here since the asyncio task for generating the streaming response is
        # already running. If we want to spawn a different task during streaming and we want that task be able
        # to read the context_var, we need to manually pass the context into that task.
        request_context_var.reset(token)

        return response

    def enqueue_pending(self, context: RequestContext):
        if context.audit_request.kbid:
            # an audit request with no kbid makes no sense, we use this as an heuristic
            # mark that no audit has been set during this request

            context.audit_request.request_time = time.monotonic() - context.start_time
            audit = get_audit()
            if audit is not None:
                audit.send(context.audit_request)


class StreamAuditStorage:
    task: asyncio.Task | None
    initialized: bool
    queue: asyncio.Queue

    def __init__(
        self,
        nats_servers: list[str],
        nats_target: str,
        partitions: int,
        seed: int,
        nats_creds: str | None,
        service: str,
    ):
        self.nats_servers = nats_servers
        self.nats_creds = nats_creds
        self.nats_target = nats_target
        self.partitions = partitions
        self.seed = seed
        self.queue = asyncio.Queue()
        self.service = service
        self.task = None
        self.initialized = False

    def get_partition(self, kbid: str):
        return mmh3.hash(kbid, self.seed, signed=False) % self.partitions

    async def disconnected_cb(self):
        logger.info("Got disconnected from NATS!")

    async def reconnected_cb(self):
        # See who we are connected to on reconnect.
        logger.info(f"Got reconnected to NATS {self.nc.connected_url}")

    async def error_cb(self, e):
        logger.error(f"There was an error connecting to NATS audit: {e}", exc_info=True)

    async def closed_cb(self):
        logger.info("Connection is closed on NATS")

    async def initialize(self):
        options = {
            "error_cb": self.error_cb,
            "closed_cb": self.closed_cb,
            "reconnected_cb": self.reconnected_cb,
        }

        if self.nats_creds:
            options["user_credentials"] = self.nats_creds

        if len(self.nats_servers) > 0:
            options["servers"] = self.nats_servers

        nc = await nats.connect(**options)
        self.nc = get_traced_nats_client(nc, self.service)

        self.js = get_traced_jetstream(self.nc, self.service)
        self.task = asyncio.create_task(self.run())

        self.initialized = True

    async def finalize(self):
        if self.task is not None:
            self.task.cancel()
        if self.nc:
            await self.nc.flush()
            await self.nc.close()
            self.nc = None

    async def run(self):
        while True:
            item_dequeued = False
            try:
                audit = await self.queue.get()
                item_dequeued = True
                await self._send(audit)
            except (asyncio.CancelledError, KeyboardInterrupt, RuntimeError):
                return
            except Exception:  # pragma: no cover
                logger.exception("Could not send audit", stack_info=True)
            finally:
                if item_dequeued:
                    self.queue.task_done()

    def send(self, message: AuditRequest):
        self.queue.put_nowait(message)

    @backoff.on_exception(
        backoff.expo, (Exception,), jitter=backoff.random_jitter, max_tries=4
    )
    async def _send(self, message: AuditRequest):
        if self.js is None:  # pragma: no cover
            raise AttributeError()

        partition = self.get_partition(message.kbid)

        res = await self.js.publish(
            self.nats_target.format(partition=partition, type=message.type),
            message.SerializeToString(),
        )
        logger.debug(
            f"Pushed message to audit.  kb: {message.kbid}, resource: {message.rid}, partition: {partition}"
        )
        return res.seq

    def retrieve(
        self,
        retrieval_time: float,
        resources: int,
        retrieval_request: RetrievalRequest,
    ):
        context = get_request_context()
        if context is None:
            return

        auditrequest = context.audit_request

        auditrequest.retrieval_time = retrieval_time
        auditrequest.resources = resources

        auditrequest.search.result_per_page = retrieval_request.top_k

        if (
            isinstance(retrieval_request.query, RawQuery)
            and retrieval_request.query.keyword is not None
        ):
            auditrequest.search.body = retrieval_request.query.keyword.query
            auditrequest.search.min_score_bm25 = (
                retrieval_request.query.keyword.min_score
            )

        if (
            isinstance(retrieval_request.query, RawQuery)
            and retrieval_request.query.semantic is not None
        ):
            auditrequest.search.vector.extend(retrieval_request.query.semantic.query)
            auditrequest.search.min_score_bm25 = (
                retrieval_request.query.semantic.min_score
            )
            auditrequest.search.vectorset = retrieval_request.query.semantic.vectorset

        if retrieval_request.filters.filter_expression is not None:
            # NOTE: this filter is a dump of the API models. NucliaDB
            # implementation uses the filter expression proto in JSON format.
            # However, neither we have the proto nor we want to do a costly
            # conversion (we'd need to query nucliadb for a slug to rid
            # conversion in order to build the proto)
            auditrequest.search.filter = retrieval_request.filters.model_dump_json()

        if retrieval_request.filters.security is not None:
            security_pb = utils_pb2.Security()
            for group_id in retrieval_request.filters.security.groups:
                if group_id not in security_pb.access_groups:
                    security_pb.access_groups.append(group_id)
            auditrequest.search.security.CopyFrom(security_pb)

    def ask(
        self,
        kbid: str,
        user: str,
        client_type: int,
        origin: str,
        ask_request: AskRequest,
        question: str,
        rephrased_question: str | None,
        retrieval_rephrased_question: str | None,
        chat_context: list[ChatContext],
        retrieved_context: list[RetrievedContext],
        answer: str | None,
        reasoning: str | None,
        learning_id: str | None,
        status_code: int,
        model: str | None,
        rephrase_time: float | None = None,
        generative_answer_time: float | None = None,
        generative_answer_first_chunk_time: float | None = None,
        generative_reasoning_first_chunk_time: float | None = None,
    ):
        if not has_feature(Features.AUDIT_RAO_ASK_ENDPOINT):
            return

        rcontext = get_request_context()
        if rcontext is None:
            return

        audit_request = rcontext.audit_request

        audit_request.type = AuditRequest.AuditType.ASK
        audit_request.origin = origin
        audit_request.client_type = client_type  # type: ignore
        audit_request.userid = user
        audit_request.kbid = kbid
        audit_request.user_request = ask_request.model_dump_json(exclude_unset=True)
        if rephrase_time is not None:
            audit_request.rephrase_time = rephrase_time
        if generative_answer_time is not None:
            audit_request.generative_answer_time = generative_answer_time
        if generative_answer_first_chunk_time is not None:
            audit_request.generative_answer_first_chunk_time = (
                generative_answer_first_chunk_time
            )
        if generative_reasoning_first_chunk_time is not None:
            audit_request.generative_reasoning_first_chunk_time = (
                generative_reasoning_first_chunk_time
            )

        if retrieval_rephrased_question is not None:
            audit_request.retrieval_rephrased_question = retrieval_rephrased_question

        audit_request.chat.question = question
        audit_request.chat.chat_context.extend(chat_context)
        audit_request.chat.retrieved_context.extend(retrieved_context)
        if learning_id is not None:
            audit_request.chat.learning_id = learning_id
        if rephrased_question is not None:
            audit_request.chat.rephrased_question = rephrased_question
        if answer is not None:
            audit_request.chat.answer = answer
        if reasoning is not None:
            audit_request.chat.reasoning = reasoning

        audit_request.chat.status_code = status_code
        if model is not None:
            audit_request.chat.model = model


def get_audit() -> StreamAuditStorage | None:
    return get_utility(Utility.AUDIT)


async def start_audit_utility(service: str):
    audit_utility = StreamAuditStorage(
        nats_creds=audit_settings.audit_jetstream_auth,
        nats_servers=audit_settings.audit_jetstream_servers,
        nats_target=cast(str, audit_settings.audit_jetstream_target),
        partitions=audit_settings.audit_partitions,
        seed=audit_settings.audit_hash_seed,
        service=service,
    )
    await audit_utility.initialize()
    set_utility(Utility.AUDIT, audit_utility)
    return audit_utility


async def stop_audit_utility():
    audit_utility = get_utility(Utility.AUDIT)
    if audit_utility is None:
        return
    clean_utility(Utility.AUDIT)
    await audit_utility.finalize()


class ChatAuditor:
    def __init__(
        self,
        kbid: str,
        user_id: str,
        client_type: NucliaDBClientType,
        origin: str,
        ask_request: AskRequest,
        user_query: str,
        rephrased_query: str | None,
        retrieval_rephrased_query: str | None,
        chat_history: list[ChatContextMessage],
        learning_id: str | None,
        query_context: PromptContext,
        query_context_order: PromptContextOrder,
        model: str | None,
    ):
        self.kbid = kbid
        self.user_id = user_id
        self.client_type = client_type
        self.origin = origin
        self.ask_request = ask_request
        self.user_query = user_query
        self.rephrased_query = rephrased_query
        self.retrieval_rephrased_query = retrieval_rephrased_query
        self.chat_history = chat_history
        self.learning_id = learning_id
        self.query_context = query_context
        self.query_context_order = query_context_order
        self.model = model

    def audit(
        self,
        text_answer: bytes,
        text_reasoning: str | None,
        generative_answer_time: float,
        generative_answer_first_chunk_time: float,
        generative_reasoning_first_chunk_time: float | None,
        rephrase_time: float | None,
        status_code: AnswerStatusCode,
    ):
        audit = get_audit()
        if audit is None:
            return

        if (
            status_code == AnswerStatusCode.NO_CONTEXT
            or status_code == AnswerStatusCode.NO_RETRIEVAL_DATA
        ):  # We don't want to audit "Not enough context to answer this." and instead set a None.
            audit_answer = None
        else:
            audit_answer = text_answer.decode()

        # Append chat history
        chat_history_context = [
            audit_pb2.ChatContext(author=message.author, text=message.text)
            for message in self.chat_history
        ]

        # Append paragraphs retrieved on this chat
        chat_retrieved_context = [
            audit_pb2.RetrievedContext(text_block_id=paragraph_id, text=text)
            for paragraph_id, text in self.query_context.items()
        ]

        audit.ask(
            self.kbid,
            self.user_id,
            client_type(self.client_type),
            self.origin,
            ask_request=self.ask_request,
            question=self.user_query,
            generative_answer_time=generative_answer_time,
            generative_answer_first_chunk_time=generative_answer_first_chunk_time,
            generative_reasoning_first_chunk_time=generative_reasoning_first_chunk_time,
            rephrase_time=rephrase_time,
            rephrased_question=self.rephrased_query,
            retrieval_rephrased_question=self.retrieval_rephrased_query,
            chat_context=chat_history_context,
            retrieved_context=chat_retrieved_context,
            answer=audit_answer,
            reasoning=text_reasoning,
            learning_id=self.learning_id,
            status_code=int(status_code.value),
            model=self.model,
        )
