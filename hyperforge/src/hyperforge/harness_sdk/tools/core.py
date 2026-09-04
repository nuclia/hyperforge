from __future__ import annotations

import uuid
from re import findall
from typing import Any

from pydantic import BaseModel

from ..models import HarnessEventType, HarnessMemory, HarnessMessage
from . import AgentContext, HarnessTool, tool


class RememberInput(BaseModel):
    text: str
    scope: str = "user_project"


class RecallInput(BaseModel):
    query: str = ""
    scope: str = "user_project"


class ForgetInput(BaseModel):
    id: str


class SpawnAgentInput(BaseModel):
    prompt: str
    include_history: bool = False


class AgentIdInput(BaseModel):
    agent_id: str


class SendMessageInput(AgentIdInput):
    message: str


class CompactInput(BaseModel):
    summary: str


class FeedbackInput(BaseModel):
    question: str


class SearchToolsInput(BaseModel):
    query: str
    limit: int = 10


class ActivateToolsInput(BaseModel):
    names: list[str]


class DictOutput(BaseModel):
    value: dict[str, Any]


class ListOutput(BaseModel):
    items: list[dict[str, Any]]


_SEARCH_STOP_WORDS = {"a", "an", "and", "for", "or", "the", "to"}


def _search_terms(value: str) -> set[str]:
    return {
        term
        for term in findall(r"[\w]+", value.casefold().replace("_", " "))
        if term not in _SEARCH_STOP_WORDS
    }


@tool(description="Search for additional tools that can be activated.")
async def search_tools(
    context: AgentContext, input_value: SearchToolsInput
) -> ListOutput:
    harness = context.harness
    terms = _search_terms(input_value.query)
    candidates: list[tuple[int, HarnessTool[Any, Any]]] = []
    for candidate in harness._external_tools:
        if not candidate.lazy_load or candidate.name in harness._active_lazy_tools:
            continue
        name_terms = _search_terms(candidate.name)
        searchable_terms = name_terms | _search_terms(candidate.description)
        matching_terms = terms & searchable_terms
        if terms and not matching_terms:
            continue
        score = sum(2 if term in name_terms else 1 for term in matching_terms)
        candidates.append((score, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    limit = max(1, min(input_value.limit, 50))
    return ListOutput(
        items=[
            {"name": candidate.name, "description": candidate.description}
            for _, candidate in candidates[:limit]
        ]
    )


@tool(description="Activate additional tools by their exact names.")
async def activate_tools(
    context: AgentContext, input_value: ActivateToolsInput
) -> DictOutput:
    harness = context.harness
    names = list(dict.fromkeys(input_value.names))
    unknown = [
        name
        for name in names
        if name not in harness._tools or not harness._tools[name].lazy_load
    ]
    if unknown:
        raise ValueError(f"Unknown lazy tools: {', '.join(unknown)}")
    harness._active_lazy_tools.update(names)
    return DictOutput(value={"activated": names})


@tool()
async def remember(context: AgentContext, input_value: RememberInput) -> DictOutput:
    harness = context.harness
    memory = HarnessMemory(
        id=uuid.uuid4().hex,
        text=input_value.text,
        scope=input_value.scope,
        metadata=harness._persisted_metadata(),
    )
    await harness.storage.remember(memory)
    await harness.emit(
        HarnessEventType.MEMORY_REMEMBERED,
        {"memory": memory.model_dump(mode="json")},
    )
    return DictOutput(value={"id": memory.id})


@tool()
async def recall(context: AgentContext, input_value: RecallInput) -> ListOutput:
    harness = context.harness
    memories = await harness.storage.recall(
        scope=input_value.scope,
        query=input_value.query,
    )
    return ListOutput(items=[memory.model_dump(mode="json") for memory in memories])


@tool()
async def forget(context: AgentContext, input_value: ForgetInput) -> DictOutput:
    harness = context.harness
    await harness.storage.forget(input_value.id)
    await harness.emit(HarnessEventType.MEMORY_FORGOTTEN, {"id": input_value.id})
    return DictOutput(value={"id": input_value.id})


@tool()
async def spawn_agent(
    context: AgentContext, input_value: SpawnAgentInput
) -> DictOutput:
    return await context.harness._child_agents.spawn(input_value)


@tool()
async def send_message(
    context: AgentContext, input_value: SendMessageInput
) -> DictOutput:
    return await context.harness._child_agents.send_message(input_value)


@tool()
async def wait_agent(context: AgentContext, input_value: AgentIdInput) -> DictOutput:
    return await context.harness._child_agents.wait(input_value.agent_id)


@tool()
async def compact(context: AgentContext, input_value: CompactInput) -> DictOutput:
    harness = context.harness
    harness.messages = [
        HarnessMessage(role="system", content=harness.system_prompt),
        HarnessMessage(
            role="user", content=f"Conversation summary:\n{input_value.summary}"
        ),
    ]
    await harness.emit(
        HarnessEventType.COMPACTED,
        {"messages": [message.model_dump(mode="json") for message in harness.messages]},
    )
    return DictOutput(value={"status": "compacted"})


@tool()
async def feedback(context: AgentContext, input_value: FeedbackInput) -> DictOutput:
    response = await context.harness.request_feedback(
        question=input_value.question,
        response_schema={
            "type": "object",
            "properties": {"response": {"type": "string"}},
        },
        timeout_ms=5 * 60 * 1000,
    )
    return DictOutput(value={"response": response})


def create_core_tools(*, feedback_enabled: bool) -> dict[str, HarnessTool[Any, Any]]:
    tools = {
        item.name: item
        for item in (
            remember,
            recall,
            forget,
            spawn_agent,
            send_message,
            wait_agent,
            compact,
            search_tools,
            activate_tools,
        )
    }
    if feedback_enabled:
        tools[feedback.name] = feedback
    return tools


__all__ = [
    "ActivateToolsInput",
    "AgentIdInput",
    "CompactInput",
    "DictOutput",
    "FeedbackInput",
    "ForgetInput",
    "ListOutput",
    "RecallInput",
    "RememberInput",
    "SearchToolsInput",
    "SendMessageInput",
    "SpawnAgentInput",
    "activate_tools",
    "compact",
    "create_core_tools",
    "feedback",
    "forget",
    "recall",
    "remember",
    "search_tools",
    "send_message",
    "spawn_agent",
    "wait_agent",
]
