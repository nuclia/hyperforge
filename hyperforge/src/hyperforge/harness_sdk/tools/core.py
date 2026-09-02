from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..models import HarnessEventType, HarnessMemory, HarnessMessage
from . import HarnessTool, tool

if TYPE_CHECKING:
    from ..harness import AgentHarness


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


@tool(description="Search for additional tools that can be activated.")
async def search_tools(
    harness: AgentHarness, input_value: SearchToolsInput
) -> ListOutput:
    terms = input_value.query.casefold().split()
    candidates: list[tuple[int, HarnessTool[Any, Any]]] = []
    for candidate in harness._external_tools:
        if not candidate.lazy_load or candidate.name in harness._active_lazy_tools:
            continue
        searchable = f"{candidate.name} {candidate.description}".casefold()
        if terms and not all(term in searchable for term in terms):
            continue
        score = sum(2 if term in candidate.name.casefold() else 1 for term in terms)
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
    harness: AgentHarness, input_value: ActivateToolsInput
) -> DictOutput:
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
async def remember(harness: AgentHarness, input_value: RememberInput) -> DictOutput:
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
async def recall(harness: AgentHarness, input_value: RecallInput) -> ListOutput:
    memories = await harness.storage.recall(
        scope=input_value.scope,
        query=input_value.query,
    )
    return ListOutput(items=[memory.model_dump(mode="json") for memory in memories])


@tool()
async def forget(harness: AgentHarness, input_value: ForgetInput) -> DictOutput:
    await harness.storage.forget(input_value.id)
    await harness.emit(HarnessEventType.MEMORY_FORGOTTEN, {"id": input_value.id})
    return DictOutput(value={"id": input_value.id})


@tool()
async def spawn_agent(
    harness: AgentHarness, input_value: SpawnAgentInput
) -> DictOutput:
    return await harness._child_agents.spawn(input_value)


@tool()
async def send_message(
    harness: AgentHarness, input_value: SendMessageInput
) -> DictOutput:
    return await harness._child_agents.send_message(input_value)


@tool()
async def wait_agent(harness: AgentHarness, input_value: AgentIdInput) -> DictOutput:
    return await harness._child_agents.wait(input_value.agent_id)


@tool()
async def compact(harness: AgentHarness, input_value: CompactInput) -> DictOutput:
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
async def feedback(harness: AgentHarness, input_value: FeedbackInput) -> DictOutput:
    response = await harness.request_feedback(
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
