from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel

from .agents import published_agent_to_tools
from .clients import ModelClient, ReasoningEffort
from .context import format_context
from .execution import (
    current_tool_call_id,
    reset_current_tool_call_id,
    set_current_tool_call_id,
)
from .models import (
    HarnessConversation,
    HarnessEvent,
    HarnessEventType,
    HarnessInboxItem,
    HarnessMessage,
    HarnessToolCall,
)
from .storage import HarnessStorageProtocol, InMemoryHarnessStorage
from .tools import HarnessTool, ToolInheritancePolicy
from .tools.core import DictOutput, SendMessageInput, SpawnAgentInput, create_core_tools
from .usage import HarnessUsage, UsageLimitExceeded, UsageLimits

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an agent running in a tool loop. Use tools when they help. "
    "Incorporate tool results and new inbox messages. Return a direct answer when done."
)
DEFAULT_EVENT_QUEUE_SIZE = 1000
POST_TOOL_RETRY_PROMPT = "Use the tool results above to answer the user's request now."
EMPTY_RESPONSE_RETRY_PROMPT = (
    "Your previous response was empty. Continue the task now: use an appropriate "
    "tool when needed, otherwise answer the user's request directly."
)
MAX_EMPTY_RETRIES = 2


@dataclass
class AgentResult:
    text: str
    reasoning: str = ""
    tool_calls: list[HarnessToolCall] = field(default_factory=list)
    input_tokens: float = 0
    output_tokens: float = 0


@dataclass(frozen=True)
class HarnessConfig:
    model: str
    reasoning_effort: ReasoningEffort | None
    model_client: ModelClient
    tools: tuple[HarnessTool, ...]
    system_prompt: str
    title: str
    disabled_core_tools: frozenset[str]
    storage: HarnessStorageProtocol
    execution_context: Mapping[str, Any]
    category: str
    tags: tuple[str, ...]
    feedback_enabled: bool
    usage_limits: UsageLimits
    event_queue_size: int


@dataclass
class TurnLoopState:
    tool_result_pending: bool = False
    empty_post_tool_retries: int = 0
    empty_retries: int = 0

    def record_tool_calls(self) -> None:
        self.tool_result_pending = True
        self.empty_post_tool_retries = 0

    def retry_prompt(self, result: AgentResult) -> str | None:
        if result.text.strip():
            self.tool_result_pending = False
            return None
        if (
            self.tool_result_pending
            and self.empty_post_tool_retries < MAX_EMPTY_RETRIES
        ):
            self.empty_post_tool_retries += 1
            return POST_TOOL_RETRY_PROMPT
        if self.empty_retries < MAX_EMPTY_RETRIES:
            self.empty_retries += 1
            return EMPTY_RESPONSE_RETRY_PROMPT
        return None


class ChildAgent(NamedTuple):
    harness: AgentHarness
    task: asyncio.Task[str]


class LLMCallError(RuntimeError):
    """A model-call failure with the provider context needed for diagnostics."""

    def __init__(
        self,
        cause: Exception,
        *,
        call_id: str,
        trace_id: str | None,
        model: str,
    ) -> None:
        super().__init__(str(cause))
        self.cause = cause
        self.call_id = call_id
        self.trace_id = trace_id
        self.model = model


class ChildAgentManager:
    def __init__(
        self,
        owner: AgentHarness,
        active_agent_ids: set[str] | None = None,
    ) -> None:
        self.owner = owner
        self.children: dict[str, ChildAgent] = {}
        self.results: dict[str, DictOutput] = {}
        self.active_agent_ids = (
            active_agent_ids if active_agent_ids is not None else set()
        )

    def interrupt(self) -> None:
        for child in self.children.values():
            child.harness.interrupt()
            child.task.cancel()

    async def stop(self) -> None:
        children = list(self.children.items())
        tasks = [child.task for _, child in children if not child.task.done()]
        for child in self.children.values():
            child.harness.interrupt()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        for child_id, child in children:
            self.results.setdefault(child_id, self._result(child_id, child.task))
        self.children.clear()

    async def spawn(self, input_value: SpawnAgentInput) -> DictOutput:
        owner = self.owner
        if owner.spawn_depth >= owner.max_spawn_depth:
            raise ValueError(f"Maximum spawn depth reached: {owner.max_spawn_depth}")
        if len(self.active_agent_ids) >= owner.max_concurrent_agents - 1:
            return self._concurrency_limit_result()

        child_id = uuid.uuid4().hex
        child = owner._create_child(
            child_id, include_history=input_value.include_history
        )
        child._child_agents.active_agent_ids = self.active_agent_ids
        self.active_agent_ids.add(child_id)

        async def run_child() -> str:
            result = "interrupted"
            async for event in child.run(input_value.prompt):
                await owner._publish_event(event)
                if event.agent_id != child.agent_id:
                    continue
                if event.type == HarnessEventType.TURN_COMPLETED:
                    result = str(event.payload.get("text", ""))
                elif event.type == HarnessEventType.TURN_FAILED:
                    raise RuntimeError(str(event.payload.get("error", "Child failed")))
            return result

        task = asyncio.create_task(run_child())
        task.add_done_callback(lambda _: self.active_agent_ids.discard(child_id))
        self.children[child_id] = ChildAgent(child, task)
        await self._wait_until_started(child, task)
        await owner.emit(
            HarnessEventType.AGENT_STARTED,
            {
                "agent_id": child_id,
                "parent_agent_id": owner.agent_id,
                "prompt": input_value.prompt,
                "spawn_depth": child.spawn_depth,
                "max_spawn_depth": owner.max_spawn_depth,
            },
        )
        return DictOutput(value={"agent_id": child_id})

    async def send_message(self, input_value: SendMessageInput) -> DictOutput:
        child = self.children.get(input_value.agent_id)
        if child is None:
            raise ValueError("Agent not found")
        message_id = await child.harness.steer(input_value.message, sender="agent")
        return DictOutput(value={"message_id": message_id})

    async def wait(self, agent_id: str) -> DictOutput:
        completed = self.results.get(agent_id)
        if completed is not None:
            return completed
        child = self.children.get(agent_id)
        if child is None:
            raise ValueError("Agent not found")
        try:
            await child.task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
        except Exception:
            pass
        self.active_agent_ids.discard(agent_id)
        result = self._result(agent_id, child.task)
        self.results[agent_id] = result
        self.children.pop(agent_id, None)
        return result

    def _concurrency_limit_result(self) -> DictOutput:
        waitable_agent_ids = [
            agent_id
            for agent_id, child in self.children.items()
            if not child.task.done()
        ]
        if waitable_agent_ids:
            message = (
                "The conversation is at its concurrent agent limit. "
                "Call wait_agent for an existing child before trying to spawn another."
            )
        else:
            message = (
                "The conversation is at its concurrent agent limit, and this agent has no child "
                "it can wait for. Finish the current work so the parent can wait before spawning more."
            )
        return DictOutput(
            value={
                "status": "concurrency_limit_reached",
                "max_concurrent_agents": self.owner.max_concurrent_agents,
                "active_agents": len(self.active_agent_ids) + 1,
                "waitable_agent_ids": waitable_agent_ids,
                "message": message,
            }
        )

    @staticmethod
    async def _wait_until_started(child: AgentHarness, task: asyncio.Task[str]) -> None:
        if task.done():
            return
        started_task = asyncio.create_task(child._model_started.wait())
        await asyncio.wait({task, started_task}, return_when=asyncio.FIRST_COMPLETED)
        if not started_task.done():
            started_task.cancel()
        await asyncio.gather(started_task, return_exceptions=True)
        await asyncio.sleep(0)

    @staticmethod
    def _result(agent_id: str, task: asyncio.Task[str]) -> DictOutput:
        if task.cancelled():
            value: dict[str, Any] = {"agent_id": agent_id, "status": "cancelled"}
        elif (error := task.exception()) is not None:
            value = {"agent_id": agent_id, "status": "failed", "error": str(error)}
        else:
            result = task.result()
            if result == "interrupted":
                return DictOutput(value={"agent_id": agent_id, "status": "cancelled"})
            value = {
                "agent_id": agent_id,
                "status": "completed",
                "result": result,
            }
        return DictOutput(value=value)


class AgentHarness:
    def __init__(
        self,
        *,
        model: str,
        model_client: ModelClient,
        reasoning_effort: ReasoningEffort | None = None,
        tools: Iterable[HarnessTool | Any] = (),
        system_prompt: str = SYSTEM_PROMPT,
        title: str = "New conversation",
        conversation_id: str | None = None,
        disabled_core_tools: Iterable[str] = (),
        storage: HarnessStorageProtocol | None = None,
        execution_context: Mapping[str, Any] | None = None,
        category: str = "assistant",
        tags: Iterable[str] = (),
        feedback_enabled: bool = False,
        usage_limits: UsageLimits | None = None,
        spawn_depth: int = 0,
        parent_agent_id: str | None = None,
        event_queue_size: int = DEFAULT_EVENT_QUEUE_SIZE,
    ) -> None:
        if event_queue_size <= 0:
            raise ValueError("event_queue_size must be positive")
        normalized_tools = tuple(
            tool if isinstance(tool, HarnessTool) else tool.as_tool() for tool in tools
        )
        normalized_storage = storage or InMemoryHarnessStorage()
        normalized_context = dict(execution_context or {})
        normalized_tags = tuple(dict.fromkeys(tags))
        normalized_limits = usage_limits or UsageLimits()
        if spawn_depth < 0 or spawn_depth > normalized_limits.max_spawn_depth:
            raise ValueError("spawn_depth must be between zero and max_spawn_depth")
        self._config = HarnessConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            model_client=model_client,
            tools=normalized_tools,
            system_prompt=system_prompt,
            title=title,
            disabled_core_tools=frozenset(disabled_core_tools),
            storage=normalized_storage,
            execution_context=normalized_context,
            category=category,
            tags=normalized_tags,
            feedback_enabled=feedback_enabled,
            usage_limits=normalized_limits,
            event_queue_size=event_queue_size,
        )
        self.model = self._config.model
        self.reasoning_effort = self._config.reasoning_effort
        self.model_client = self._config.model_client
        self.system_prompt = self._config.system_prompt
        self.title = self._config.title
        self.conversation_id = conversation_id or uuid.uuid4().hex
        self.agent_id = uuid.uuid4().hex
        self.disabled_core_tools = set(self._config.disabled_core_tools)
        self.storage = self._config.storage
        self.execution_context = dict(self._config.execution_context)
        self.category = self._config.category
        self.tags = list(self._config.tags)
        self.usage_limits = self._config.usage_limits
        self.max_spawn_depth = self.usage_limits.max_spawn_depth
        self.max_concurrent_agents = self.usage_limits.max_concurrent_agents
        self.spawn_depth = spawn_depth
        self.parent_agent_id = parent_agent_id
        self.usage = HarnessUsage()
        self._usage_started_at: float | None = None
        self._owns_usage = True
        self._persist_events = True
        self.conversation: HarnessConversation | None = None
        self.messages: list[HarnessMessage] = []
        self.inbox: asyncio.Queue[HarnessInboxItem] = asyncio.Queue()
        self.event_queue_size = event_queue_size
        self._event_queue: asyncio.Queue[HarnessEvent] | None = None
        self._child_agents = ChildAgentManager(self)
        self.feedback_requests: dict[str, asyncio.Future[str]] = {}
        self._pending_messages: list[HarnessMessage] = []
        self._interrupted = asyncio.Event()
        self._model_started = asyncio.Event()
        self._run_lock = asyncio.Lock()
        self._active_task: asyncio.Task[Any] | None = None
        self._turn_id: str | None = None
        self._last_event_id: str | None = None
        self._external_tools = list(self._config.tools)
        self._tools = {tool.name: tool for tool in self._external_tools}
        self._active_lazy_tools: set[str] = set()
        if len(self._tools) != len(self._external_tools):
            raise ValueError("Tool names must be unique")
        self._install_core_tools(self._config.feedback_enabled)

    @property
    def children(self) -> dict[str, ChildAgent]:
        return self._child_agents.children

    @property
    def child_results(self) -> dict[str, DictOutput]:
        return self._child_agents.results

    async def load(self, *, create: bool = True) -> None:
        conversation = await self.storage.get_conversation(self.conversation_id)
        if conversation is None:
            if not create:
                raise ValueError("Conversation not found")
            conversation = HarnessConversation(
                id=self.conversation_id,
                title=self.title,
                category=self.category,
                tags=self.tags,
                metadata=self._persisted_metadata(),
            )
            await self.storage.create_conversation(conversation)
        self.conversation = conversation
        self.messages = []
        pending_tool_outputs: dict[str, HarnessMessage] = {}
        async for event in self.storage.iter_events(self.conversation_id):
            self._last_event_id = event.id
            if event.parent_agent_id is not None:
                continue
            if event.type in {
                HarnessEventType.MESSAGE_ADDED,
                HarnessEventType.MESSAGES_ADDED,
            }:
                raw_messages = (
                    [event.payload["message"]]
                    if event.type == HarnessEventType.MESSAGE_ADDED
                    else event.payload["messages"]
                )
                for raw_message in raw_messages:
                    message = HarnessMessage.model_validate(raw_message)
                    if message.role == "tool" and not any(
                        message.tool_call_id == call.id
                        for existing in self.messages
                        for call in existing.tool_calls
                    ):
                        if message.tool_call_id is not None:
                            pending_tool_outputs[message.tool_call_id] = message
                        continue
                    self.messages.append(message)
                    for call in message.tool_calls:
                        if call.id in pending_tool_outputs:
                            self.messages.append(pending_tool_outputs.pop(call.id))
            elif event.type == HarnessEventType.COMPACTED:
                self.messages = [
                    HarnessMessage.model_validate(item)
                    for item in event.payload["messages"]
                ]
                pending_tool_outputs.clear()
        if pending_tool_outputs:
            logger.warning(
                "Ignoring orphaned tool outputs while loading conversation: conversation=%s call_ids=%s",
                self.conversation_id,
                sorted(pending_tool_outputs),
                extra={
                    "conversation_id": self.conversation_id,
                    "call_ids": sorted(pending_tool_outputs),
                },
            )
        self.messages = _complete_tool_history(self.messages)

    def _persisted_metadata(self) -> dict[str, Any]:
        metadata = self.execution_context.get("conversation_metadata", {})
        return dict(metadata) if isinstance(metadata, Mapping) else {}

    def add_messages(self, messages: Iterable[HarnessMessage | dict[str, Any]]) -> None:
        self._pending_messages.extend(
            message
            if isinstance(message, HarnessMessage)
            else HarnessMessage.model_validate(message)
            for message in messages
        )

    @staticmethod
    def to_tools(
        agent: Any, *, namespace: str | None = None, manager: Any = None
    ) -> list[HarnessTool[Any, Any]]:
        """Expose a legacy agent's published functions as harness tools."""
        return published_agent_to_tools(agent, namespace=namespace, manager=manager)

    def iter_tools(self, *, include_inactive: bool = False) -> Iterable[HarnessTool]:
        """Iterate over registered tools available to this harness."""
        return (
            tool
            for tool in self._tools.values()
            if include_inactive
            or not tool.lazy_load
            or tool.name in self._active_lazy_tools
        )

    async def save_messages(self) -> None:
        await self._flush_pending_messages()

    async def steer(
        self, content: str, *, sender: Literal["user", "agent"] = "user"
    ) -> str:
        item = HarnessInboxItem(id=uuid.uuid4().hex, sender=sender, content=content)
        await self.inbox.put(item)
        await self.emit(
            HarnessEventType.INBOX_ADDED, {"item": item.model_dump(mode="json")}
        )
        return item.id

    def interrupt(self) -> None:
        self._interrupted.set()
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if self._active_task is not None and self._active_task is not current:
            self._active_task.cancel()
        self._child_agents.interrupt()

    def _check_limit(self, name: str, value: float) -> None:
        limit = getattr(self.usage_limits, name)
        if limit is not None and value > limit:
            raise UsageLimitExceeded(f"{name} limit exceeded: {value} > {limit}")

    def _start_usage(self) -> None:
        if self._owns_usage:
            self.usage = HarnessUsage()
            self._usage_started_at = time.monotonic()

    def _remaining_time(self) -> float | None:
        if self.usage_limits.max_time is None:
            return None
        if self._usage_started_at is None:
            self._usage_started_at = time.monotonic()
        remaining = self.usage_limits.max_time - (
            time.monotonic() - self._usage_started_at
        )
        if remaining <= 0:
            raise UsageLimitExceeded("max_time limit exceeded")
        return remaining

    async def _with_time_limit(self, awaitable: Any) -> Any:
        try:
            remaining = self._remaining_time()
        except BaseException:
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise
        if remaining is None:
            return await awaitable
        try:
            async with asyncio.timeout(remaining):
                return await awaitable
        except TimeoutError as exc:
            raise UsageLimitExceeded("max_time limit exceeded") from exc

    @property
    def turn_id(self) -> str | None:
        return self._turn_id

    async def request_feedback(
        self,
        *,
        question: str,
        response_schema: dict[str, Any],
        timeout_ms: int,
    ) -> str | None:
        request_id = uuid.uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.feedback_requests[request_id] = future
        await self.emit(
            HarnessEventType.FEEDBACK_REQUESTED,
            {
                "request_id": request_id,
                "question": question,
                "response_schema": response_schema,
                "timeout_ms": timeout_ms,
            },
        )
        try:
            return await asyncio.wait_for(future, timeout=timeout_ms / 1000)
        except TimeoutError:
            return None
        finally:
            self.feedback_requests.pop(request_id, None)

    async def respond_feedback(self, request_id: str, response: str) -> None:
        future = self.feedback_requests.get(request_id)
        if future is None or future.done():
            raise ValueError("Feedback request not found")
        future.set_result(response)
        await self.emit(
            HarnessEventType.FEEDBACK_RESOLVED,
            {"request_id": request_id, "response": response},
        )

    async def _publish_event(self, event: HarnessEvent) -> None:
        queue = self._event_queue
        if queue is None:
            return
        if (
            event.type
            in {
                HarnessEventType.TEXT_DELTA,
                HarnessEventType.REASONING_DELTA,
            }
            and queue.qsize() >= self.event_queue_size
        ):
            return
        queue.put_nowait(event)

    async def history(self) -> AsyncIterator[HarnessEvent]:
        async for event in self.storage.iter_events(self.conversation_id):
            yield event

    async def emit(
        self,
        event_type: HarnessEventType,
        payload: dict[str, Any],
        *,
        persist: bool = True,
    ) -> HarnessEvent:
        event = HarnessEvent(
            id=uuid.uuid4().hex,
            conversation_id=self.conversation_id,
            turn_id=self._turn_id,
            agent_id=self.agent_id,
            parent_agent_id=self.parent_agent_id,
            parent_call_id=current_tool_call_id(),
            category=self.category,
            tags=self.tags,
            metadata=self._persisted_metadata(),
            type=event_type,
            payload=payload,
        )
        if persist and self._persist_events:
            await self.storage.append_event(event)
            self._last_event_id = event.id
        await self._publish_event(event)
        return event

    async def llm(self, tools: Iterable[HarnessTool] | None = None) -> AgentResult:
        await self._flush_pending_messages()
        if tools is None:
            selected = list(self.iter_tools())
        else:
            selected = []
            seen: set[str] = set()
            for requested in tools:
                registered = self._tools.get(requested.name)
                if (
                    registered is not None
                    and registered.name not in seen
                    and (
                        not registered.lazy_load
                        or registered.name in self._active_lazy_tools
                    )
                ):
                    selected.append(registered)
                    seen.add(registered.name)
        self.usage.turns += 1
        self._check_limit("max_turns", self.usage.turns)
        result = AgentResult(text="")
        call_id = uuid.uuid4().hex
        history_head_event_id = self._last_event_id
        await self.emit(
            HarnessEventType.LLM_STARTED,
            {
                "call_id": call_id,
                "history_head_event_id": history_head_event_id,
                "message_count": len(self.messages),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
                "tools": [tool.name for tool in selected],
            },
        )
        trace_id: str | None = None
        resolved_model: str | None = None
        usage_recorded = False
        try:
            self._model_started.set()
            async for delta in self.model_client.stream(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
                messages=self.messages,
                tools=selected,
                execution_context=self.execution_context,
            ):
                result.text += delta.text
                result.reasoning += delta.reasoning
                result.tool_calls.extend(delta.tool_calls)
                result.input_tokens = max(result.input_tokens, delta.input_tokens)
                result.output_tokens = max(result.output_tokens, delta.output_tokens)
                trace_id = delta.trace_id or trace_id
                resolved_model = delta.model or resolved_model
                if delta.text:
                    await self.emit(
                        HarnessEventType.TEXT_DELTA, {"text": delta.text}, persist=False
                    )
                if delta.reasoning:
                    await self.emit(
                        HarnessEventType.REASONING_DELTA,
                        {"text": delta.reasoning},
                        persist=False,
                    )
            self.usage.input_tokens += result.input_tokens
            self.usage.output_tokens += result.output_tokens
            usage_recorded = True
            self._check_limit("max_input_tokens", self.usage.input_tokens)
            self._check_limit("max_output_tokens", self.usage.output_tokens)
        except asyncio.CancelledError:
            raise
        except UsageLimitExceeded as exc:
            await self._emit_llm_result(
                HarnessEventType.LLM_FAILED,
                call_id,
                history_head_event_id,
                result,
                trace_id,
                resolved_model,
                error=str(exc),
            )
            raise
        except Exception as exc:
            if not usage_recorded:
                self.usage.input_tokens += result.input_tokens
                self.usage.output_tokens += result.output_tokens
            await self._emit_llm_result(
                HarnessEventType.LLM_FAILED,
                call_id,
                history_head_event_id,
                result,
                trace_id,
                resolved_model,
                error=str(exc),
            )
            raise LLMCallError(
                exc,
                call_id=call_id,
                trace_id=trace_id,
                model=resolved_model or self.model,
            ) from exc
        await self._emit_llm_result(
            HarnessEventType.LLM_COMPLETED,
            call_id,
            history_head_event_id,
            result,
            trace_id,
            resolved_model,
        )
        return result

    async def _emit_llm_result(
        self,
        event_type: HarnessEventType,
        call_id: str,
        history_head_event_id: str | None,
        result: AgentResult,
        trace_id: str | None,
        resolved_model: str | None,
        *,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "call_id": call_id,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "history_head_event_id": history_head_event_id,
            "trace_id": trace_id,
            "model": resolved_model or self.model,
            "text": result.text,
            "reasoning": result.reasoning,
        }
        if error is not None:
            payload["error"] = error
        await self.emit(event_type, payload)

    async def __aenter__(self) -> AgentHarness:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.interrupt()
        task = self._active_task
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)

    async def run(self, prompt: str) -> AsyncIterator[HarnessEvent]:
        async with self._run_lock:
            if self._event_queue is not None:
                raise RuntimeError("A turn is already running")
            self._event_queue = asyncio.Queue()
            self._model_started.clear()
            task = asyncio.create_task(self._execute_turn(prompt))
            self._active_task = task
            try:
                terminal = False
                while not terminal or not task.done():
                    event_task = asyncio.create_task(self._event_queue.get())
                    done, _ = await asyncio.wait(
                        {task, event_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    if task in done and not event_task.done():
                        event_task.cancel()
                        await asyncio.gather(event_task, return_exceptions=True)
                        break
                    event = event_task.result()
                    yield event
                    terminal = (
                        event.agent_id == self.agent_id
                        and event.turn_id == self._turn_id
                        and event.type
                        in {
                            HarnessEventType.TURN_COMPLETED,
                            HarnessEventType.TURN_FAILED,
                            HarnessEventType.TURN_INTERRUPTED,
                        }
                    )
                await task
            finally:
                if not task.done():
                    self.interrupt()
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self._active_task = None
                self._event_queue = None
                self._interrupted.clear()

    async def _execute_turn(self, prompt: str) -> str:
        try:
            return await self._run(prompt)
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if self._interrupted.is_set() and not (
                current is not None and current.cancelling() > 1
            ):
                await self._stop_children()
                await self.emit(HarnessEventType.TURN_INTERRUPTED, {})
                return "interrupted"
            raise

    async def _run(self, prompt: str) -> str:
        if self.conversation is None:
            await self.load()
        self._start_usage()
        self._turn_id = uuid.uuid4().hex
        await self.emit(HarnessEventType.TURN_STARTED, {"prompt": prompt})
        latest_system = next(
            (
                message
                for message in reversed([*self.messages, *self._pending_messages])
                if message.role == "system"
            ),
            None,
        )
        if latest_system is None or latest_system.content != self.system_prompt:
            await self._append_message(
                HarnessMessage(role="system", content=self.system_prompt)
            )
        await self._flush_pending_messages()
        await self._append_message(HarnessMessage(role="user", content=prompt))
        loop_state = TurnLoopState()
        try:
            if self._interrupted.is_set():
                await self.emit(HarnessEventType.TURN_INTERRUPTED, {})
                return "interrupted"
            while not self._interrupted.is_set():
                await self._consume_inbox()
                result = await self._with_time_limit(self.llm())
                if result.tool_calls:
                    await self._with_time_limit(
                        self._execute_tool_calls(result.tool_calls, result.text)
                    )
                    loop_state.record_tool_calls()
                    continue
                if retry_prompt := loop_state.retry_prompt(result):
                    await self._append_message(
                        HarnessMessage(role="user", content=retry_prompt)
                    )
                    continue
                await self._append_message(
                    HarnessMessage(role="assistant", content=result.text)
                )
                if not self.inbox.empty():
                    continue
                await self.emit(HarnessEventType.TURN_COMPLETED, {"text": result.text})
                await self._stop_children()
                return result.text
            await self._stop_children()
            await self.emit(HarnessEventType.TURN_INTERRUPTED, {})
            return "interrupted"
        except Exception as exc:
            parent_agent_id = self.execution_context.get("parent_agent_id")
            log = logger.warning if parent_agent_id is not None else logger.exception
            failure = exc.cause if isinstance(exc, LLMCallError) else exc
            context = {
                "conversation_id": self.conversation_id,
                "turn_id": self._turn_id,
                "agent_id": self.agent_id,
                "parent_agent_id": parent_agent_id,
                "call_id": exc.call_id if isinstance(exc, LLMCallError) else None,
                "trace_id": exc.trace_id if isinstance(exc, LLMCallError) else None,
                "model": exc.model if isinstance(exc, LLMCallError) else self.model,
                "message_count": len(self.messages),
                "tool_count": len(self._tools),
                "error_type": type(failure).__name__,
                "error": str(failure),
            }
            if parent_agent_id is not None:
                log(
                    "Sub-agent turn failed: conversation=%s parent=%s agent=%s turn=%s call=%s model=%s error=%s",
                    self.conversation_id,
                    parent_agent_id,
                    self.agent_id,
                    self._turn_id,
                    context["call_id"],
                    context["model"],
                    failure,
                    extra=context,
                )
            else:
                log(
                    "Agent turn failed: conversation=%s agent=%s turn=%s call=%s model=%s error=%s",
                    self.conversation_id,
                    self.agent_id,
                    self._turn_id,
                    context["call_id"],
                    context["model"],
                    failure,
                    extra=context,
                )
            await self.emit(HarnessEventType.TURN_FAILED, {"error": str(failure)})
            await self._stop_children()
            raise

    async def _stop_children(self) -> None:
        await self._child_agents.stop()

    async def _append_message(self, message: HarnessMessage) -> None:
        self.messages.append(message)
        await self.emit(
            HarnessEventType.MESSAGE_ADDED, {"message": message.model_dump(mode="json")}
        )

    async def _append_messages(self, messages: list[HarnessMessage]) -> None:
        self.messages.extend(messages)
        await self.emit(
            HarnessEventType.MESSAGES_ADDED,
            {"messages": [message.model_dump(mode="json") for message in messages]},
        )

    async def _flush_pending_messages(self) -> None:
        pending, self._pending_messages = self._pending_messages, []
        for message in pending:
            await self._append_message(message)

    async def _consume_inbox(self) -> None:
        items: list[HarnessInboxItem] = []
        while not self.inbox.empty():
            items.append(self.inbox.get_nowait())
        if not items:
            return
        rendered = "New inbox messages:\n" + "\n".join(
            f"- {item.sender}: {item.content}" for item in items
        )
        await self._append_message(HarnessMessage(role="user", content=rendered))
        for item in items:
            await self.emit(HarnessEventType.INBOX_CONSUMED, {"id": item.id})

    async def _execute_tool_calls(
        self, calls: list[HarnessToolCall], text: str = ""
    ) -> None:
        self.usage.tool_calls += len(calls)
        self._check_limit("max_tool_calls", self.usage.tool_calls)
        calls = [
            call if call.id else call.model_copy(update={"id": uuid.uuid4().hex})
            for call in calls
        ]
        messages = await asyncio.gather(
            *(self._execute_tool_call(call) for call in calls)
        )
        await self._append_messages(
            [
                HarnessMessage(role="assistant", content=text, tool_calls=calls),
                *messages,
            ]
        )

    async def _execute_tool_call(self, call: HarnessToolCall) -> HarnessMessage:
        assert call.id is not None
        await self.emit(
            HarnessEventType.TOOL_REQUESTED, {"call": call.model_dump(mode="json")}
        )
        tool = self._tools.get(call.name)
        try:
            output: BaseModel | None = None
            if tool is None:
                raise ValueError(f"Unknown tool: {call.name}")
            elif tool.lazy_load and tool.name not in self._active_lazy_tools:
                raise ValueError(f"Tool is not active: {call.name}")
            else:
                token = set_current_tool_call_id(call.id)
                try:
                    output = await tool.execute(self, call.arguments)
                finally:
                    reset_current_tool_call_id(token)
                result = output.model_dump(mode="json")
                reference = tool.context(output)
                content = format_context(reference)
            await self.emit(
                HarnessEventType.TOOL_COMPLETED,
                {"call_id": call.id, "tool": call.name, "result": result},
            )
            return HarnessMessage(
                role="tool",
                tool_call_id=call.id,
                tool_name=call.name,
                content=content,
                context=reference,
            )
        except Exception as exc:
            logger.warning(
                "Agent tool execution failed: tool=%s call_id=%s error_type=%s error=%s",
                call.name,
                call.id,
                type(exc).__name__,
                exc,
                extra={
                    "conversation_id": self.conversation_id,
                    "turn_id": self._turn_id,
                    "agent_id": self.agent_id,
                    "call_id": call.id,
                    "tool": call.name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            result = {"error": str(exc)}
            await self.emit(
                HarnessEventType.TOOL_FAILED,
                {"call_id": call.id, "tool": call.name, "result": result},
            )
            return HarnessMessage(
                role="tool",
                tool_call_id=call.id,
                tool_name=call.name,
                content=json.dumps(result, default=str, separators=(",", ":")),
            )

    def _install_core_tools(self, feedback_enabled: bool) -> None:
        core_tools = create_core_tools(feedback_enabled=feedback_enabled)
        enabled_core_tools = {
            name: tool
            for name, tool in core_tools.items()
            if name not in self.disabled_core_tools
            and not (name == "spawn_agent" and self.spawn_depth >= self.max_spawn_depth)
        }
        conflicts = self._tools.keys() & enabled_core_tools.keys()
        if conflicts:
            raise ValueError(
                f"Tool names conflict with core tools: {', '.join(sorted(conflicts))}"
            )
        self._tools.update(enabled_core_tools)

    def _create_child(self, child_id: str, *, include_history: bool) -> AgentHarness:
        child_context = {
            **self.execution_context,
            "parent_agent_id": self.agent_id,
            "spawn_agent_id": child_id,
        }
        child = type(self)(
            model=self._config.model,
            model_client=self._config.model_client,
            reasoning_effort=self._config.reasoning_effort,
            tools=(
                tool
                for tool in self._config.tools
                if tool.inheritance == ToolInheritancePolicy.INHERIT
            ),
            system_prompt=self._config.system_prompt,
            title=self._config.title,
            conversation_id=self.conversation_id,
            disabled_core_tools=self._config.disabled_core_tools,
            storage=self._config.storage,
            execution_context=child_context,
            category=self._config.category,
            tags=self._config.tags,
            feedback_enabled=self._config.feedback_enabled,
            usage_limits=self._config.usage_limits,
            spawn_depth=self.spawn_depth + 1,
            parent_agent_id=self.agent_id,
            event_queue_size=self._config.event_queue_size,
        )
        child.agent_id = child_id
        child.usage = self.usage
        child._usage_started_at = self._usage_started_at
        child._owns_usage = False
        child.conversation = self.conversation
        if include_history:
            child.messages = list(self.messages)
        return child


def _complete_tool_history(messages: list[HarnessMessage]) -> list[HarnessMessage]:
    complete: list[HarnessMessage] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role != "assistant" or not message.tool_calls:
            complete.append(message)
            index += 1
            continue
        expected = {call.id for call in message.tool_calls if call.id is not None}
        outputs: list[HarnessMessage] = []
        cursor = index + 1
        while cursor < len(messages) and messages[cursor].role == "tool":
            outputs.append(messages[cursor])
            cursor += 1
        if expected and expected == {output.tool_call_id for output in outputs}:
            complete.extend([message, *outputs])
        else:
            logger.warning(
                "Ignoring incomplete tool exchange while loading conversation: call_ids=%s",
                sorted(expected),
            )
        index = cursor
    return complete


async def create_agent(**kwargs: Any) -> AgentHarness:
    create = bool(kwargs.pop("create", True))
    harness = AgentHarness(**kwargs)
    await harness.load(create=create)
    return harness
