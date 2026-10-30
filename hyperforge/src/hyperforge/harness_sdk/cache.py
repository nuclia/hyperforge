"""Decision cache: replay previously generated model decisions (tool requests).

The harness owns every safety decision: which history is cacheable, how keys are
derived and whether a stored entry may be replayed. Storage only persists and
retrieves entries. Tools always execute live; only the model's request to call
them is replayed.

A decision is cacheable only while the conversation history consists of system
prompts, the user question(s) and exchanges with tools whose results are stable
(``ToolCachePolicy.METADATA``). The rendered results of those tools are
fingerprinted into ``history_key``, so changed metadata invalidates descendants.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import HarnessMessage, HarnessToolCall

logger = logging.getLogger(__name__)

POLICY_VERSION = "decision-cache-v1"
MAX_DECISION_BYTES = 64_000


class ToolCachePolicy(StrEnum):
    """How a model decision that calls this tool may be cached and replayed."""

    NONE = "none"
    """Never replay a decision calling this tool (default; e.g. mutations)."""

    METADATA = "metadata"
    """Read-only, stable output. Replayable for similar questions, and the
    rendered result is fingerprinted so later decisions remain cacheable."""

    SIMILAR = "similar"
    """Replayable for similar questions, but the result is not stable evidence:
    decisions after it are not cached (e.g. spawning a sub-agent)."""

    EXACT = "exact"
    """Replayable only for the identical question and history, and decisions
    after its result are not cached (e.g. generating a data query)."""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CachedDecision(BaseModel):
    text: str = ""
    reasoning: str = ""
    tool_calls: list[HarnessToolCall]


class DecisionCacheQuery(BaseModel):
    namespace: str
    config_key: str
    history_key: str
    question_key: str
    question: str


class DecisionCacheEntry(BaseModel):
    id: str
    namespace: str
    config_key: str
    history_key: str
    question_key: str
    question: str
    similar: bool
    """True when the decision may be replayed for a different, similar question."""
    decision: CachedDecision
    created_datetime: datetime = Field(default_factory=utcnow)
    expires_datetime: datetime


class DecisionCacheStorage(Protocol):
    """Persistence for decision-cache entries.

    ``lookup`` must only return entries whose ``namespace``, ``config_key`` and
    ``history_key`` equal the query's, which have not expired, and where either
    ``question_key`` equals the query's or ``similar`` is true and the stored
    ``question`` is similar enough to the query's (e.g. by embedding). Exact
    question matches should be preferred. The harness re-validates every entry.
    """

    async def lookup(self, query: DecisionCacheQuery) -> DecisionCacheEntry | None: ...

    async def save(self, entry: DecisionCacheEntry) -> None: ...


class InMemoryDecisionCacheStorage:
    """Process-local, exact-question storage intended for tests."""

    def __init__(self) -> None:
        self.entries: dict[str, DecisionCacheEntry] = {}

    async def lookup(self, query: DecisionCacheQuery) -> DecisionCacheEntry | None:
        now = utcnow()
        for entry in self.entries.values():
            if (
                entry.namespace == query.namespace
                and entry.config_key == query.config_key
                and entry.history_key == query.history_key
                and entry.question_key == query.question_key
                and entry.expires_datetime > now
            ):
                return entry.model_copy(deep=True)
        return None

    async def save(self, entry: DecisionCacheEntry) -> None:
        self.entries[entry.id] = entry.model_copy(deep=True)


@dataclass(frozen=True)
class DecisionCache:
    """Enable decision caching on a harness.

    ``namespace`` isolates entries (e.g. account, project, user, agent) and
    ``key`` adds any other configuration that affects generation, such as
    model-client options. Both must be JSON-serializable.
    """

    storage: DecisionCacheStorage
    namespace: Any = ""
    key: Mapping[str, Any] = field(default_factory=dict)
    ttl: timedelta = timedelta(days=7)
    timeout: float = 2.0


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class _HistoryState:
    question: str
    question_key: str
    history_key: str


def _history_state(
    messages: Sequence[HarnessMessage],
    policy: Callable[[str], ToolCachePolicy],
    *,
    inherited_key: str | None = None,
) -> _HistoryState | None:
    """Return the cache identity of ``messages``, or ``None`` if uncacheable.

    ``inherited_key`` is the identity of context inherited from a parent agent;
    ``messages`` must then exclude that inherited prefix.
    """
    questions: list[str] = []
    transcript: list[object] = [["inherited", inherited_key]]
    pending: dict[str, str] = {}
    has_results = False
    for message in messages:
        if message.role == "system":
            transcript.append(["system", message.content])
        elif message.role == "user":
            # A user message after tool results is steering or a new turn; the
            # earlier results can no longer be certified.
            if pending or has_results:
                return None
            transcript.append(["user", len(questions)])
            questions.append(message.content)
        elif message.role == "assistant":
            if pending:
                return None
            calls = []
            for call in message.tool_calls:
                if (
                    not call.id
                    or call.id in pending
                    or policy(call.name) == ToolCachePolicy.NONE
                ):
                    return None
                pending[call.id] = call.name
                calls.append([call.name, call.arguments])
            transcript.append(["assistant", message.content, calls])
        else:
            name = pending.pop(message.tool_call_id or "", None)
            if name is None or policy(name) != ToolCachePolicy.METADATA:
                return None
            has_results = True
            transcript.append(["tool", name, message.content])
    if pending or not questions:
        return None
    return _HistoryState(
        question="\n\n".join(questions),
        question_key=digest(questions),
        history_key=digest(transcript),
    )


def _replay_policy(
    decision: CachedDecision, policy: Callable[[str], ToolCachePolicy]
) -> tuple[bool, bool]:
    """Return (replayable, replayable for similar questions)."""
    policies = [policy(call.name) for call in decision.tool_calls]
    replayable = bool(policies) and ToolCachePolicy.NONE not in policies
    similar = replayable and ToolCachePolicy.EXACT not in policies
    return replayable, similar


@dataclass
class DecisionCacheRuntime:
    """Per-agent decision-cache stack.

    Every spawned agent gets its own runtime. A child that inherits its parent's
    history builds on the parent's cache identity at spawn time; if that
    context was not cacheable, neither is anything the child decides on top of
    it. A fresh child starts an independent stack.
    """

    cache: DecisionCache
    lineage: tuple[str, ...] = ()
    inherited_key: str | None = None
    prefix_length: int = 0
    enabled: bool = True
    unavailable: bool = False
    _writes: set[asyncio.Task[None]] = field(default_factory=set)

    def child(
        self,
        *,
        include_history: bool,
        messages: Sequence[HarnessMessage],
        policy: Callable[[str], ToolCachePolicy],
    ) -> DecisionCacheRuntime:
        if not include_history:
            return DecisionCacheRuntime(self.cache, lineage=(*self.lineage, "fresh"))
        state = self._state(messages, policy) if self.enabled else None
        return DecisionCacheRuntime(
            self.cache,
            lineage=(*self.lineage, "history"),
            inherited_key=(
                digest([state.history_key, state.question_key]) if state else None
            ),
            prefix_length=len(messages),
            enabled=state is not None,
        )

    def _state(
        self,
        messages: Sequence[HarnessMessage],
        policy: Callable[[str], ToolCachePolicy],
    ) -> _HistoryState | None:
        return _history_state(
            messages[self.prefix_length :], policy, inherited_key=self.inherited_key
        )

    def query(
        self,
        *,
        model: str,
        reasoning_effort: str | None,
        messages: Sequence[HarnessMessage],
        tools: Sequence[Any],
        policy: Callable[[str], ToolCachePolicy],
        procedural: Mapping[str, Any] | None = None,
    ) -> DecisionCacheQuery | None:
        if self.unavailable or not self.enabled:
            return None
        state = self._state(messages, policy)
        if state is None:
            return None
        config_key = digest(
            {
                "policy": POLICY_VERSION,
                "key": self.cache.key,
                "lineage": self.lineage,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "procedural": procedural,
                "tools": [
                    [tool.name, tool.description, tool.parameters, tool.cache_policy]
                    for tool in tools
                ],
            }
        )
        return DecisionCacheQuery(
            namespace=digest(self.cache.namespace),
            config_key=config_key,
            history_key=state.history_key,
            question_key=state.question_key,
            question=state.question,
        )

    async def lookup(
        self,
        query: DecisionCacheQuery,
        *,
        tools: Mapping[str, Any],
        policy: Callable[[str], ToolCachePolicy],
    ) -> DecisionCacheEntry | None:
        try:
            async with asyncio.timeout(self.cache.timeout):
                entry = await self.cache.storage.lookup(query)
        except Exception:
            # Availability must never block live inference; stop retrying.
            self.unavailable = True
            logger.warning("Decision cache lookup failed; bypassing", exc_info=True)
            return None
        if entry is None or not self._valid(entry, query, tools, policy):
            return None
        exact = entry.question_key == query.question_key
        return entry.model_copy(
            update={
                "decision": CachedDecision(
                    # Prose can reference the original question's entities.
                    text=entry.decision.text if exact else "",
                    reasoning=entry.decision.reasoning if exact else "",
                    tool_calls=[
                        call.model_copy(update={"id": None})
                        for call in entry.decision.tool_calls
                    ],
                )
            }
        )

    def save(
        self,
        query: DecisionCacheQuery,
        decision: CachedDecision,
        policy: Callable[[str], ToolCachePolicy],
    ) -> None:
        replayable, similar = _replay_policy(decision, policy)
        if not replayable or len(decision.model_dump_json()) > MAX_DECISION_BYTES:
            return
        now = utcnow()
        entry = DecisionCacheEntry(
            id=digest(
                [
                    query.namespace,
                    query.config_key,
                    query.history_key,
                    query.question_key,
                ]
            ),
            namespace=query.namespace,
            config_key=query.config_key,
            history_key=query.history_key,
            question_key=query.question_key,
            question=query.question,
            similar=similar,
            decision=decision.model_copy(deep=True),
            created_datetime=now,
            expires_datetime=now + self.cache.ttl,
        )
        task = asyncio.create_task(self._write(entry))
        self._writes.add(task)
        task.add_done_callback(self._writes.discard)

    async def _write(self, entry: DecisionCacheEntry) -> None:
        try:
            async with asyncio.timeout(self.cache.timeout):
                await self.cache.storage.save(entry)
        except Exception:
            logger.warning("Decision cache write failed", exc_info=True)

    async def flush(self) -> None:
        if self._writes:
            await asyncio.gather(*self._writes, return_exceptions=True)

    @staticmethod
    def _valid(
        entry: DecisionCacheEntry,
        query: DecisionCacheQuery,
        tools: Mapping[str, Any],
        policy: Callable[[str], ToolCachePolicy],
    ) -> bool:
        if (
            entry.namespace != query.namespace
            or entry.config_key != query.config_key
            or entry.history_key != query.history_key
            or entry.expires_datetime <= utcnow()
        ):
            return False
        replayable, similar = _replay_policy(entry.decision, policy)
        if not replayable:
            return False
        if entry.question_key != query.question_key and not (entry.similar and similar):
            return False
        for call in entry.decision.tool_calls:
            tool = tools.get(call.name)
            if tool is None:
                return False
            try:
                tool.input_model.model_validate(call.arguments)
            except Exception:
                return False
        return True
