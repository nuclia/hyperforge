from __future__ import annotations

import uuid
from re import findall
from typing import Any, Literal

from pydantic import BaseModel

from ..models import HarnessEventType, HarnessMemory, HarnessMessage
from . import HarnessTool, ToolCallContext, tool


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


class CallToolInput(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


class ToolSearchResult(BaseModel):
    name: str
    description: str
    active: bool


class ActivatedTool(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


class ActivateToolsOutput(BaseModel):
    activated: list[ActivatedTool]


class IdOutput(BaseModel):
    id: str


class AgentSpawnedOutput(BaseModel):
    agent_id: str


class AgentConcurrencyLimitOutput(BaseModel):
    status: Literal["concurrency_limit_reached"] = "concurrency_limit_reached"
    max_concurrent_agents: int
    active_agents: int
    waitable_agent_ids: list[str]
    message: str


class SendMessageOutput(BaseModel):
    message_id: str


class AgentCancelledOutput(BaseModel):
    agent_id: str
    status: Literal["cancelled"] = "cancelled"


class AgentCompletedOutput(BaseModel):
    agent_id: str
    status: Literal["completed"] = "completed"
    result: str


class AgentFailedOutput(BaseModel):
    agent_id: str
    status: Literal["failed"] = "failed"
    error: str


type AgentResultOutput = (
    AgentCancelledOutput | AgentCompletedOutput | AgentFailedOutput
)


class CompactOutput(BaseModel):
    status: Literal["compacted"] = "compacted"


class FeedbackOutput(BaseModel):
    response: str | None


_SEARCH_STOP_WORDS = {"a", "an", "and", "for", "or", "the", "to"}


def _search_terms(value: str) -> set[str]:
    return {
        term
        for term in findall(r"[\w]+", value.casefold().replace("_", " "))
        if term not in _SEARCH_STOP_WORDS
    }


@tool(description="Search for additional tools that can be activated.")
async def search_tools(
    context: ToolCallContext, input_value: SearchToolsInput
) -> list[ToolSearchResult]:
    harness = context.harness
    terms = _search_terms(input_value.query)
    candidates: list[tuple[int, HarnessTool[Any, Any]]] = []
    for candidate in harness.iter_lazy_tools():
        name_terms = _search_terms(candidate.name)
        searchable_terms = name_terms | _search_terms(candidate.description)
        matching_terms = terms & searchable_terms
        if terms and not matching_terms:
            continue
        score = sum(2 if term in name_terms else 1 for term in matching_terms)
        candidates.append((score, candidate))
    candidates.sort(key=lambda item: (-item[0], item[1].name))
    limit = max(1, min(input_value.limit, 50))
    return [
        ToolSearchResult(
            name=candidate.name,
            description=candidate.description,
            active=harness.is_tool_active(candidate.name),
        )
        for _, candidate in candidates[:limit]
    ]


@tool(description="Activate additional tools by their exact names.")
async def activate_tools(
    context: ToolCallContext, input_value: ActivateToolsInput
) -> ActivateToolsOutput:
    tools = await context.harness.activate_tools(input_value.names)
    return ActivateToolsOutput(
        activated=[
            ActivatedTool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
            for tool in tools
        ]
    )


@tool(
    description=(
        "Execute an activated tool. Use the exact tool_name and arguments schema "
        "returned by activate_tools."
    )
)
async def call_tool(context: ToolCallContext, input_value: CallToolInput) -> Any:
    output = await context.harness.call_tool(
        input_value.tool_name,
        input_value.arguments,
        call_id=context.id,
    )
    tool = next(
        tool
        for tool in context.harness.iter_lazy_tools()
        if tool.name == input_value.tool_name
    )
    return tool.dump_output(output)


@tool()
async def remember(context: ToolCallContext, input_value: RememberInput) -> IdOutput:
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
    return IdOutput(id=memory.id)


@tool()
async def recall(
    context: ToolCallContext, input_value: RecallInput
) -> list[HarnessMemory]:
    harness = context.harness
    memories = await harness.storage.recall(
        scope=input_value.scope,
        query=input_value.query,
    )
    return memories


@tool()
async def forget(context: ToolCallContext, input_value: ForgetInput) -> IdOutput:
    harness = context.harness
    await harness.storage.forget(input_value.id)
    await harness.emit(HarnessEventType.MEMORY_FORGOTTEN, {"id": input_value.id})
    return IdOutput(id=input_value.id)


@tool()
async def spawn_agent(
    context: ToolCallContext, input_value: SpawnAgentInput
) -> AgentSpawnedOutput | AgentConcurrencyLimitOutput:
    return await context.harness._child_agents.spawn(input_value)


@tool()
async def send_message(
    context: ToolCallContext, input_value: SendMessageInput
) -> SendMessageOutput:
    return await context.harness._child_agents.send_message(input_value)


@tool()
async def wait_agent(
    context: ToolCallContext, input_value: AgentIdInput
) -> AgentResultOutput:
    return await context.harness._child_agents.wait(input_value.agent_id)


@tool()
async def compact(context: ToolCallContext, input_value: CompactInput) -> CompactOutput:
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
    return CompactOutput()


@tool()
async def feedback(
    context: ToolCallContext, input_value: FeedbackInput
) -> FeedbackOutput:
    response = await context.harness.request_feedback(
        question=input_value.question,
        response_schema={
            "type": "object",
            "properties": {"response": {"type": "string"}},
        },
        timeout_ms=5 * 60 * 1000,
    )
    return FeedbackOutput(response=response)


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
            call_tool,
        )
    }
    if feedback_enabled:
        tools[feedback.name] = feedback
    return tools


__all__ = [
    "ActivateToolsInput",
    "ActivateToolsOutput",
    "ActivatedTool",
    "AgentCancelledOutput",
    "AgentCompletedOutput",
    "AgentConcurrencyLimitOutput",
    "AgentFailedOutput",
    "AgentIdInput",
    "AgentResultOutput",
    "AgentSpawnedOutput",
    "CallToolInput",
    "CompactInput",
    "CompactOutput",
    "FeedbackInput",
    "FeedbackOutput",
    "ForgetInput",
    "IdOutput",
    "RecallInput",
    "RememberInput",
    "SearchToolsInput",
    "SendMessageInput",
    "SendMessageOutput",
    "SpawnAgentInput",
    "ToolSearchResult",
    "activate_tools",
    "call_tool",
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
