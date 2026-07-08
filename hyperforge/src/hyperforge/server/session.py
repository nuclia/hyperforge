import asyncio
import os
from asyncio import Task
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union

if TYPE_CHECKING:
    from hyperforge.standalone.agent import StaticAgentManager

import nucliadb_telemetry.context
import nucliadb_telemetry.metrics
import prometheus_client
from aiohttp.web import Server
from lru import LRU
from nucliadb_telemetry import errors
from nucliadb_telemetry.utils import get_telemetry
from opentelemetry import trace

from hyperforge.broker import Broker
from hyperforge.configure import load_all_configurations, scan
from hyperforge.db.agents import AgentManager
from hyperforge.engine import State, get_state
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
    ARAGException,
    Feedback,
    OAuthAuthenticateURL,
)
from hyperforge.memory.memory import QuestionMemory
from hyperforge.pubsub import (
    AgentAnswer,
    AgentDone,
    AgentMessage,
    AgentPing,
    AgentToUserRequest,
    OAuthRequest,
    StartInteraction,
    UserToAgentInteraction,
)
from hyperforge.server import SERVICE_NAME, logger
from hyperforge.server.cache import Cache
from hyperforge.server.settings import Settings
from hyperforge.server.utils import get_memory
from hyperforge.server.web import start_health_check

HOSTNAME = os.environ.get("HOSTNAME", "arag-server").encode()

answer_observer = nucliadb_telemetry.metrics.Observer("arag_answer")
activation_observer = nucliadb_telemetry.metrics.Observer("arag_activation")
answer_running = prometheus_client.Gauge(
    "arag_running_answers_count", "Number of answering processess currently running"
)


def tracer():
    provider = get_telemetry(SERVICE_NAME)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()


class SessionManager:
    server: Optional[Server] = None
    tasks: List[Task]
    hooks: Optional[Dict[str, List[Callable]]] = None

    def __init__(
        self,
        settings: Settings,
        broker: Broker,
        agent_manager: Union[AgentManager, "StaticAgentManager"],
        cache: Cache,
    ):
        self.settings = settings
        self.agent_manager = agent_manager
        self.broker = broker
        self.memory: LRU = LRU(800)
        self.activation_task: asyncio.Task | None = None
        self.tasks = []
        self.cache = cache

    async def activation_listener(self):
        import opentelemetry.propagate as otel_propagate
        from opentelemetry.context.context import Context

        async for msg, trace_headers in self.broker.subscribe_activations():
            try:
                context = (
                    otel_propagate.extract(trace_headers)
                    if trace_headers
                    else Context()
                )
                with tracer().start_as_current_span("Activate agent", context):
                    await self.activate(msg)
            except (asyncio.CancelledError, KeyboardInterrupt):
                logger.info("Activation listener cancelled, exiting...")
                break
            except Exception:
                logger.exception("Error processing activation message")
                errors.capture_exception()

    async def initialize(self, health_check: bool = True) -> None:

        for load_module in self.settings.load_modules:
            try:
                scan(load_module)
                load_all_configurations(load_module)
            except ImportError:
                logger.error(f"Module {load_module} could not be loaded")

        self.activation_task = asyncio.create_task(self.activation_listener())
        if health_check and self.settings.health_check_enabled:
            self.server = await start_health_check()

    async def finalize(self):

        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.activation_task and not self.activation_task.done():
            self.activation_task.cancel()
        if self.server is not None:
            await self.server.shutdown()
            self.server = None
        await self.broker.finalize()

        await self.agent_manager.finalize()

    def _remove_task(self, task: asyncio.Task):
        if task in self.tasks:
            self.tasks.remove(task)

    async def activate(self, message: StartInteraction):
        topic = None
        logger.info("Activation message received: %s", message)
        observation = activation_observer()
        observation.start()
        try:
            nucliadb_telemetry.context.add_context(
                {
                    "agent_id": message.agent_id,
                    "session_id": message.session,
                    "question_id": message.question_id,
                }
            )

            topic = self.question_topic(
                message.account,
                message.agent_id,
                message.session,
                message.question_id,
                message.workflow_id,
            )

            # Get or load session
            config = await self.agent_manager.get_agent_config(
                account=message.account,
                agent_id=message.agent_id,
                internal_nucliadb_url=self.settings.internal_nucliadb_url,
                workflow_id=message.workflow_id,
            )

            state = await get_state(
                agent_id=message.agent_id,
                config=config,
                internal_nua_api=self.settings.internal_nua_api,
                internal_nua=self.settings.internal_nua,
                local_openai=self.settings.local_openai,
                external_nua_api_key=self.settings.external_nua_api_key,
                account=message.account,
                kbid=None if self.settings.standalone else message.agent_id,
            )

            if message.session not in self.memory:
                memory = await get_memory(
                    settings=self.settings,
                    session=message.session,
                    cache=self.cache,
                    config=config.memory,
                    agent=message.agent_id,
                    workflow_id=message.workflow_id,
                    account_id=message.account,
                )
                # Ephemeral sessions are short-lived and should not be persisted in
                # the shared LRU cache across activations.
                if message.session != "ephemeral":
                    self.memory[message.session] = memory
            else:
                memory = self.memory[message.session]

            memory.rules = config.rules.rules

            question = memory.start_question(
                message.question,
                question_id=message.question_id,
                headers=message.headers,
                arguments=message.arguments,
                streaming=message.streaming,
                chat_history=message.chat_history,
            )

            task = asyncio.create_task(
                self.answer(
                    message.account,
                    message.agent_id,
                    message.workflow_id,
                    topic,
                    state,
                    question,
                )
            )
            task.add_done_callback(self._remove_task)
            self.tasks.append(task)

        except Exception as e:
            logger.exception("Activation exception")
            errors.capture_exception(e)
            observation.set_status("error")
            if topic:
                await self.callback(
                    topic,
                    AragAnswer(
                        exception=ARAGException(detail="Unable to start agent"),
                        operation=AnswerOperation.ERROR,
                    ),
                )
                await self.send_message(topic, AgentDone())

        observation.end()

    def question_topic(
        self,
        account: str,
        agent_id: str,
        session: str,
        question: str,
        workflow_id: str = "default",
    ):
        return self.settings.answers_subject.format(
            account=account,
            agent_id=agent_id,
            session=session,
            question=question,
            workflow_id=workflow_id,
        )

    async def send_message(
        self,
        topic: str,
        message: AgentMessage,
    ) -> None:
        try:
            await self.broker.publish(topic, message)

        except Exception as e:
            logger.exception("Error publishing answer to %s", topic)
            errors.capture_exception(e)

    async def oauth(self, topic: str, oauth: OAuthAuthenticateURL):
        await self.send_message(
            topic,
            OAuthRequest(oauth=oauth),
        )

    async def get_oauth_callback(
        self,
        account_id: str,
        agent_id: str,
        session_id: str,
        workflow_id: str,
        question_uuid: str,
        oauth_uuid: str,
        timeout_ms: int = 300000,
    ) -> str | None:
        subject = self.settings.oauth_subject.format(
            account=account_id,
            agent_id=agent_id,
            session=session_id,
            question=question_uuid,
            oauth_uuid=oauth_uuid,
            workflow_id=workflow_id,
        )
        # Cap the XREAD block time so it doesn't exceed the overall question
        # timeout.  We leave a 10 s margin so the caller can still handle the
        # None return before the outer asyncio.timeout fires.
        margin_ms = 10_000
        max_block_ms = self.settings.question_timeout_seconds * 1000 - margin_ms
        effective_timeout_ms = min(timeout_ms, max(max_block_ms, 0))

        logger.info(
            "Waiting for OAuth callback %s (timeout=%dms)",
            oauth_uuid,
            effective_timeout_ms,
        )
        try:
            payload = await self.broker.receive_reply(subject, effective_timeout_ms)
            if payload is None:
                return None

            logger.info("OAuth callback %s received successfully", oauth_uuid)
            return payload
        except Exception as e:
            logger.exception("Error receiving OAuth callback %s", oauth_uuid)
            errors.capture_exception(e)
        return None

    async def feedback(self, topic: str, feedback: Feedback):
        await self.send_message(
            topic,
            AgentToUserRequest(feedback=feedback),
        )

        try:
            payload = await self.broker.receive_reply(
                feedback.feedback_id, feedback.timeout_ms
            )
            if payload is None:
                return None
            return UserToAgentInteraction.model_validate_json(payload)
        except Exception as e:
            logger.exception("Error receiving feedback %s", topic)
            errors.capture_exception(e)
        return None

    async def callback(self, topic: str, message: AragAnswer):
        await self.send_message(topic, AgentAnswer(answer=message))

    async def keep_alive(self, topic: str):
        while True:
            await asyncio.sleep(self.broker.keepalive_seconds / 2)
            await self.send_message(topic, AgentPing())

    async def answer(
        self,
        account_id: str,
        agent_id: str,
        workflow_id: str,
        topic: str,
        state: State,
        question_memory: QuestionMemory,
    ):
        error = None

        keepalive = asyncio.create_task(self.keep_alive(topic))
        observation = answer_observer()
        observation.start()
        answer_running.inc()

        try:
            callback = partial(self.callback, topic)
            question_memory.set_callback_fn(callback)

            feedback = partial(self.feedback, topic)
            question_memory.set_feedback_fn(feedback)

            oauth = partial(self.oauth, topic)
            question_memory.set_oauth_fn(oauth)

            oauth_callback = partial(
                self.get_oauth_callback,
                account_id,
                agent_id,
                question_memory.session.id,
                workflow_id,
            )
            question_memory.set_oauth_callback_fn(oauth_callback)

            await self.callback(
                topic,
                AragAnswer(operation=AnswerOperation.START),
            )

            async with asyncio.timeout(self.settings.question_timeout_seconds):
                await state.agent(question_memory, state.manager)

        except Exception as e:
            logger.exception("Answering exception")
            errors.capture_exception(e)
            error = ARAGException(detail=str(e))
            observation.set_status("error")

        observation.end()
        answer_running.dec()
        keepalive.cancel()

        await self.callback(
            topic,
            AragAnswer(
                exception=error,
                answer=question_memory.final_answer,
                answer_citations=question_memory.final_answer_citations,
                answer_urls=question_memory.final_answer_urls,
                operation=AnswerOperation.ERROR
                if error is not None
                else AnswerOperation.ANSWER,
                data_visualizations=question_memory.data_visualizations
                if question_memory.data_visualizations
                else None,
            ),
        )
        await self.send_message(
            topic,
            AgentDone(),
        )

        try:
            await question_memory.save()
            self.process_event(
                "memory_saved",
                {"account_id": account_id, "question_memory": question_memory},
            )
        except Exception as e:
            # Log memory errors but don't report them to the user
            logger.exception("Error saving memory")
            errors.capture_exception(e)

    def process_event(self, event_name: str, data: dict):
        if self.hooks is not None and event_name in self.hooks:
            for hook in self.hooks[event_name]:
                try:
                    hook(**data)
                except Exception as e:
                    logger.exception("Error in hook for event %s", event_name)
                    errors.capture_exception(e)
