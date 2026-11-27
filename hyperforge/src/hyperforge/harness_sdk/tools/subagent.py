from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from hyperforge.procedural.guidance import ProceduralGuidanceConfig

from ..models import HarnessEventType
from .utils import HarnessTool, ToolCallContext

if TYPE_CHECKING:
    from ..clients import ModelClient, ReasoningEffort
    from ..skills import HarnessSkill


class SubAgentInput(BaseModel):
    prompt: str = Field(
        description="Self-contained task or question for the sub-agent to solve."
    )


class SubAgentOutput(BaseModel):
    text: str
    agent_id: str


def create_subagent_tool(
    *,
    name: str,
    description: str,
    system_prompt: str | None = None,
    tools: Iterable[HarnessTool[Any, Any]] = (),
    skills: Iterable[HarnessSkill] = (),
    model: str | None = None,
    model_client: ModelClient | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    disabled_core_tools: Iterable[str] = (
        "remember",
        "recall",
        "forget",
        "spawn_agent",
        "send_message",
        "wait_agent",
    ),
    feedback_enabled: bool = False,
    procedural_guidance: ProceduralGuidanceConfig | None = None,
    lazy_load: bool = False,
) -> HarnessTool[SubAgentInput, SubAgentOutput]:
    """Wrap a scoped `AgentHarness` loop as a single callable `HarnessTool`.

    Because the returned object is an external `HarnessTool` (not a core tool),
    it can be invoked both by an `AgentHarness` solver and as an ordinary
    synchronous function inside Code Mode while sharing usage limits and event
    publication with the parent harness.
    """
    captured_tools = tuple(tools)
    captured_skills = tuple(skills)
    captured_disabled = tuple(disabled_core_tools)

    async def execute(
        context: ToolCallContext, input_value: SubAgentInput
    ) -> SubAgentOutput:
        from ..harness import SYSTEM_PROMPT, AgentHarness
        from ..storage import InMemoryHarnessStorage

        parent = context.harness
        if parent.spawn_depth >= parent.max_spawn_depth:
            raise ValueError(f"Maximum spawn depth reached: {parent.max_spawn_depth}")

        child_id = uuid.uuid4().hex
        child_context = {
            **parent.execution_context,
            "parent_agent_id": parent.agent_id,
            "spawn_agent_id": child_id,
        }
        child = AgentHarness(
            model=model or parent.model,
            model_client=model_client or parent.model_client,
            reasoning_effort=(
                reasoning_effort
                if reasoning_effort is not None
                else parent.reasoning_effort
            ),
            tools=captured_tools,
            skills=captured_skills,
            system_prompt=system_prompt or SYSTEM_PROMPT,
            conversation_id=f"{parent.conversation_id}:{child_id}",
            disabled_core_tools=captured_disabled,
            storage=InMemoryHarnessStorage(),
            execution_context=child_context,
            category=parent.category,
            tags=parent.tags,
            feedback_enabled=feedback_enabled,
            usage_limits=parent.usage_limits,
            spawn_depth=parent.spawn_depth + 1,
            parent_agent_id=parent.agent_id,
            event_queue_size=parent.event_queue_size,
            procedural_guidance=procedural_guidance,
        )
        child.agent_id = child_id
        child.usage = parent.usage
        child._usage_started_at = parent._usage_started_at
        child._owns_usage = False
        child._persist_events = False

        await parent.emit(
            HarnessEventType.AGENT_STARTED,
            {
                "agent_id": child_id,
                "parent_agent_id": parent.agent_id,
                "prompt": input_value.prompt,
                "spawn_depth": child.spawn_depth,
                "max_spawn_depth": parent.max_spawn_depth,
                "tool": name,
            },
            parent_call_id=context.id,
        )

        final_text = ""
        async with child:
            async for event in child.run(input_value.prompt):
                await parent._publish_event(event)
                if event.agent_id != child.agent_id:
                    continue
                if event.type == HarnessEventType.TURN_COMPLETED:
                    final_text = str(event.payload.get("text", ""))
                elif event.type == HarnessEventType.TURN_FAILED:
                    raise RuntimeError(
                        str(event.payload.get("error", f"Sub-agent {name!r} failed"))
                    )
                elif event.type == HarnessEventType.FEEDBACK_REQUESTED:
                    response = await parent.request_feedback(
                        question=str(event.payload.get("question", "")),
                        response_schema=dict(
                            event.payload.get("response_schema") or {}
                        ),
                        timeout_ms=int(event.payload.get("timeout_ms", 60_000)),
                    )
                    if response is not None:
                        await child.respond_feedback(
                            str(event.payload["request_id"]), response
                        )

        return SubAgentOutput(text=final_text, agent_id=child_id)

    execute.__name__ = name
    return HarnessTool(
        name=name,
        description=description,
        handler=execute,
        lazy_load=lazy_load,
    )


__all__ = [
    "SubAgentInput",
    "SubAgentOutput",
    "create_subagent_tool",
]
