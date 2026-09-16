from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HarnessEventType(StrEnum):
    CONVERSATION_STARTED = "conversation.started"
    AGENT_STARTED = "agent.started"
    TURN_STARTED = "turn.started"
    MESSAGE_ADDED = "message.added"
    MESSAGES_ADDED = "messages.added"
    REASONING_DELTA = "reasoning.delta"
    TEXT_DELTA = "text.delta"
    TOOL_REQUESTED = "tool.requested"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    INBOX_ADDED = "inbox.added"
    INBOX_CONSUMED = "inbox.consumed"
    FEEDBACK_REQUESTED = "feedback.requested"
    FEEDBACK_RESOLVED = "feedback.resolved"
    MEMORY_REMEMBERED = "memory.remembered"
    MEMORY_FORGOTTEN = "memory.forgotten"
    COMPACTED = "context.compacted"
    LLM_STARTED = "llm.started"
    LLM_COMPLETED = "llm.completed"
    LLM_FAILED = "llm.failed"
    WORKFLOW_MESSAGE = "workflow.message"
    TURN_COMPLETED = "turn.completed"
    TURN_INTERRUPTED = "turn.interrupted"
    TURN_FAILED = "turn.failed"


class HarnessContextType(StrEnum):
    TOOL_RESULT = "tool_result"
    RETRIEVAL = "retrieval"
    STATUS = "status"
    ENTITY = "entity"
    COLLECTION = "collection"
    QUERY = "query"
    STRUCTURED = "structured"


class HarnessToolCall(BaseModel):
    id: str | None = None
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class HarnessContextReference(BaseModel):
    type: HarnessContextType
    content: dict[str, Any]


class HarnessMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[HarnessToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    tool_name: str | None = None
    context: HarnessContextReference | None = None


class HarnessConversation(BaseModel):
    id: str
    title: str = "New conversation"
    category: str = "assistant"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_datetime: datetime = Field(default_factory=utcnow)
    updated_datetime: datetime = Field(default_factory=utcnow)


class HarnessEvent(BaseModel):
    id: str
    conversation_id: str
    turn_id: str | None = None
    agent_id: str | None = None
    parent_agent_id: str | None = None
    parent_call_id: str | None = None
    category: str = "assistant"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    type: HarnessEventType
    created_datetime: datetime = Field(default_factory=utcnow)
    payload: dict[str, Any] = Field(default_factory=dict)


class HarnessMemory(BaseModel):
    id: str
    scope: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_datetime: datetime = Field(default_factory=utcnow)


class HarnessInboxItem(BaseModel):
    id: str
    sender: Literal["user", "agent"]
    content: str
    created_datetime: datetime = Field(default_factory=utcnow)
