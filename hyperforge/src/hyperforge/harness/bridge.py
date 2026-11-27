from __future__ import annotations

import json
import time
from typing import Any

from hyperforge.harness_sdk import AgentHarness, HarnessEvent, HarnessEventType
from hyperforge.interaction import Feedback, OAuthAuthenticateURL, StreamingChunk
from hyperforge.memory.memory import QuestionMemory
from hyperforge.models import AnswerCitations, Chunk, Context, Image, Visualization


class HarnessEventBridge:
    """Translate live `AgentHarness` events onto `QuestionMemory` for API/A2A/MCP parity."""

    def __init__(
        self,
        harness: AgentHarness,
        memory: QuestionMemory,
        *,
        flow_id: str = "harness",
    ) -> None:
        self.harness = harness
        self.memory = memory
        self.flow_id = flow_id
        self._tool_starts: dict[str, tuple[str, float]] = {}
        self._tool_emitted_step: set[str] = set()
        self.final_text = ""
        self.final_citations: AnswerCitations | None = None
        self.final_visualizations: list[Visualization] = []
        self.rich_answer: dict[str, Any] | None = None

    async def handle_event(self, event: HarnessEvent) -> None:
        is_root = event.agent_id == self.harness.agent_id
        event_type = event.type

        if event_type == HarnessEventType.TEXT_DELTA and is_root:
            await self.memory.emit_streaming_chunk(
                StreamingChunk(text=str(event.payload.get("text", "")))
            )
            return

        if event_type == HarnessEventType.REASONING_DELTA and is_root:
            await self.memory.emit_streaming_chunk(
                StreamingChunk(text=str(event.payload.get("text", ""))),
                reasoning=True,
            )
            return

        if event_type == HarnessEventType.FEEDBACK_REQUESTED and is_root:
            response = await self.memory.send_feedback(
                Feedback(
                    request_id=str(event.payload["request_id"]),
                    question=str(event.payload["question"]),
                    module=str(event.payload.get("module", "harness")),
                    agent_id=self.memory.get_agent_id(),
                    data=event.payload.get("data", {}),
                    timeout_ms=int(event.payload.get("timeout_ms", 60_000)),
                    response_schema=event.payload.get("response_schema", {}),
                    get_credentials=event.payload.get("get_credentials"),
                    credentials=event.payload.get("credentials"),
                )
            )
            if response is not None:
                await self.harness.respond_feedback(
                    str(event.payload["request_id"]), response.response
                )
            return

        if event_type == HarnessEventType.OAUTH_REQUESTED and is_root:
            oauth_url = event.payload.get("oauth_url")
            if isinstance(oauth_url, str) and oauth_url:
                await self.memory.send_oauth(OAuthAuthenticateURL(oauth_url=oauth_url))
            return

        if event_type == HarnessEventType.TOOL_REQUESTED:
            call = event.payload.get("call")
            if isinstance(call, dict):
                call_id = call.get("id")
                tool_name = call.get("name")
                if isinstance(call_id, str) and isinstance(tool_name, str):
                    self._tool_starts[call_id] = (tool_name, time.monotonic())
            return

        if event_type == HarnessEventType.TOOL_COMPLETED:
            await self._record_tool_completed(event)
            return

        if event_type == HarnessEventType.TOOL_FAILED:
            await self._record_tool_failed(event)
            return

        if event_type == HarnessEventType.SKILLS_LOADED and is_root:
            ids = event.payload.get("ids", [])
            if isinstance(ids, list) and ids:
                await self.memory.add_step(
                    step_module="harness",
                    step_title="Skills loaded",
                    step_agent_path="/harness/skills",
                    timeit=0.0,
                    step_value=", ".join(str(item) for item in ids),
                )
            return

        if event_type == HarnessEventType.PROCEDURAL_GUIDANCE and is_root:
            status = str(event.payload.get("status", "completed"))
            active_node = event.payload.get("active_node")
            await self.memory.add_step(
                step_module="procedural",
                step_title=f"Procedural guidance ({active_node or 'full'})",
                step_agent_path="/harness/procedural",
                timeit=float(event.payload.get("latency_seconds", 0.0)),
                input_nuclia_tokens=float(
                    event.payload.get("nuclia_input_tokens", 0.0)
                ),
                output_nuclia_tokens=float(
                    event.payload.get("nuclia_output_tokens", 0.0)
                ),
                step_value=str(event.payload.get("text", "")),
                step_reason=f"status={status}",
                error=(
                    str(event.payload["error"])
                    if event.payload.get("error") is not None
                    else None
                ),
            )
            return

        if event_type == HarnessEventType.WORKFLOW_STEP:
            for active_call_id in self._tool_starts:
                self._tool_emitted_step.add(active_call_id)
            await self.memory.add_step(
                step_module=str(event.payload.get("module", "workflow")),
                step_title=str(event.payload.get("title", "Workflow step")),
                step_agent_path=str(
                    event.payload.get("agent_path", "/harness/workflow")
                ),
                timeit=float(event.payload.get("timeit", 0.0)),
                input_nuclia_tokens=(
                    float(event.payload["input_nuclia_tokens"])
                    if event.payload.get("input_nuclia_tokens") is not None
                    else None
                ),
                output_nuclia_tokens=(
                    float(event.payload["output_nuclia_tokens"])
                    if event.payload.get("output_nuclia_tokens") is not None
                    else None
                ),
                step_value=(
                    str(event.payload["value"])
                    if event.payload.get("value") is not None
                    else None
                ),
                step_reason=(
                    str(event.payload["reason"])
                    if event.payload.get("reason") is not None
                    else None
                ),
                error=(
                    str(event.payload["error"])
                    if event.payload.get("error") is not None
                    else None
                ),
            )
            return

        if event_type == HarnessEventType.TURN_COMPLETED and is_root:
            self.final_text = str(event.payload.get("text", ""))
            self.memory.is_answered = bool(
                event.payload.get("is_answered", bool(self.final_text))
            )
            raw_citations = event.payload.get("citations")
            if isinstance(raw_citations, dict):
                self.final_citations = AnswerCitations.model_validate(raw_citations)
            raw_contexts = event.payload.get("contexts")
            if isinstance(raw_contexts, list):
                await self._extract_contexts({"contexts": raw_contexts}, "workflow")
            raw_related = event.payload.get("related_questions")
            if isinstance(raw_related, list):
                self.memory.add_future_questions(
                    [
                        str(item)
                        for item in raw_related
                        if isinstance(item, str) and item
                    ]
                )
            raw_visualizations = event.payload.get("visualizations")
            if isinstance(raw_visualizations, list):
                self.final_visualizations = [
                    Visualization.model_validate(item)
                    for item in raw_visualizations
                    if isinstance(item, dict)
                ]
            raw_meta = event.payload.get("metadata")
            if isinstance(raw_meta, dict) and isinstance(
                raw_meta.get("rich_answer"), dict
            ):
                self.rich_answer = raw_meta["rich_answer"]

    async def finalize(self) -> None:
        await self.memory.emit_streaming_chunk(StreamingChunk(text="", last=True))
        primary_vis = (
            self.final_visualizations[0] if self.final_visualizations else None
        )
        rich_kwargs: dict[str, Any] = {}
        if self.rich_answer is not None:
            rich_kwargs["chunks"] = [
                Chunk.model_validate(c)
                for c in (self.rich_answer.get("chunks") or [])
                if isinstance(c, dict)
            ]
            rich_kwargs["structured"] = list(self.rich_answer.get("structured") or [])
            rich_kwargs["images"] = {
                k: Image.model_validate(v)
                for k, v in (self.rich_answer.get("images") or {}).items()
                if isinstance(v, dict)
            }
            rich_kwargs["image_urls"] = list(self.rich_answer.get("image_urls") or [])
        await self.memory.add_answer(
            self.final_text,
            "harness",
            "/harness",
            citations=self.final_citations,
            visualization=primary_vis,
            **rich_kwargs,
        )
        if len(self.final_visualizations) > 1:
            self.memory.data_visualizations.extend(self.final_visualizations[1:])
        await self.memory.add_final_answer()

    async def _record_tool_completed(self, event: HarnessEvent) -> None:
        call_id = str(event.payload.get("call_id", ""))
        tool_name = str(event.payload.get("tool", "tool"))
        started_entry = self._tool_starts.pop(call_id, None)
        elapsed = (
            time.monotonic() - started_entry[1]
            if started_entry is not None
            else float(event.payload.get("duration_ms", 0.0)) / 1000.0
        )
        raw_result = event.payload.get("result")
        await self._extract_contexts(raw_result, tool_name)
        emitted_custom = call_id in self._tool_emitted_step
        self._tool_emitted_step.discard(call_id)
        if event.payload.get("code_workflow") or emitted_custom:
            return

        preview = _summarize_result(raw_result)
        await self.memory.add_step(
            step_module="harness_tool",
            step_title=f"Tool: {tool_name}",
            step_agent_path=f"/harness/tool/{tool_name}",
            timeit=elapsed,
            step_value=preview,
        )

    async def _record_tool_failed(self, event: HarnessEvent) -> None:
        call_id = str(event.payload.get("call_id", ""))
        tool_name = str(event.payload.get("tool", "tool"))
        started_entry = self._tool_starts.pop(call_id, None)
        elapsed = (
            time.monotonic() - started_entry[1]
            if started_entry is not None
            else float(event.payload.get("duration_ms", 0.0)) / 1000.0
        )
        error_payload = event.payload.get("result", {})
        error_msg = (
            str(error_payload.get("error", "Tool execution failed"))
            if isinstance(error_payload, dict)
            else str(error_payload)
        )
        await self.memory.add_step(
            step_module="harness_tool",
            step_title=f"Tool failed: {tool_name}",
            step_agent_path=f"/harness/tool/{tool_name}",
            timeit=elapsed,
            error=error_msg,
        )

    async def _extract_contexts(self, raw_result: Any, tool_name: str) -> None:
        candidates: list[dict[str, Any]] = []
        if isinstance(raw_result, dict):
            unwrapped = raw_result.get("value", raw_result)
            if isinstance(unwrapped, dict):
                if "chunks" in unwrapped and "source" in unwrapped:
                    candidates.append(unwrapped)
                elif isinstance(unwrapped.get("contexts"), list):
                    candidates.extend(
                        item
                        for item in unwrapped["contexts"]
                        if isinstance(item, dict)
                        and "chunks" in item
                        and "source" in item
                    )
            elif isinstance(unwrapped, list):
                candidates.extend(
                    item
                    for item in unwrapped
                    if isinstance(item, dict) and "chunks" in item and "source" in item
                )

        existing_ids = {item.id for item in self.memory.contexts}
        for candidate in candidates:
            normalized = {
                "original_question_uuid": self.memory.original_question_uuid,
                "actual_question_uuid": self.memory.actual_question_uuid,
                **candidate,
            }
            try:
                ctx = Context.model_validate(normalized)
            except Exception:
                continue
            if ctx.id in existing_ids:
                continue
            existing_ids.add(ctx.id)
            await self.memory.save_context(
                flow_id=self.flow_id,
                context=ctx,
                agent_id=ctx.agent_id or tool_name,
            )
            if ctx.image_urls:
                await self.memory.save_image_urls(list(ctx.image_urls))


def _summarize_result(raw_result: Any, max_chars: int = 500) -> str:
    if raw_result is None:
        return ""
    try:
        rendered = json.dumps(raw_result, default=str, ensure_ascii=False)
    except Exception:
        rendered = str(raw_result)
    if len(rendered) > max_chars:
        return rendered[:max_chars] + "..."
    return rendered


__all__ = ["HarnessEventBridge"]
