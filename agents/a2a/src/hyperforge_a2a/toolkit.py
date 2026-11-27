from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from a2a.types import a2a_pb2
from hyperforge.configure import toolkit
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessContextType,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    tool,
)
from hyperforge.models import Chunk, Context
from hyperforge.toolkits.common import (
    RetrievalOutput,
    resolve_manager,
    retrieval_context_factory,
)
from pydantic import BaseModel, Field

from .client import (
    ResponseTextAccumulator,
    build_feedback_request,
    build_send_request,
    extract_feedback_request,
    raise_for_terminal_task_error,
)
from .driver import A2ADriver


class A2ASendInput(BaseModel):
    question: str = Field(
        description="Self-contained question or task to send to the remote A2A agent."
    )
    extra_metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Optional per-call scalar metadata merged into A2A routing metadata.",
    )


class A2AToolkitConfig(ToolkitConfig):
    toolkit: Literal["a2a"] = "a2a"
    source: str = Field(
        ...,
        title="A2A driver identifier",
        description="Identifier of the configured A2A gRPC driver.",
    )
    remote_workflow_id: str = Field(default="default")
    extra_metadata: dict[str, str] = Field(default_factory=dict)
    valid_headers: list[str] = Field(default_factory=list)


@toolkit(
    id="a2a",
    title="Agent2Agent (A2A) Client Toolkit",
    description="Delegate questions to external A2A gRPC agents with interactive feedback forwarding.",
    config_schema=A2AToolkitConfig,
)
class A2AToolkit(BaseToolkit[A2AToolkitConfig]):
    toolkit_id = "a2a"

    def _extract_headers(self, call_context: ToolCallContext) -> dict[str, str]:
        exec_ctx = call_context.harness.execution_context
        raw: dict[str, str] = {}
        mem = exec_ctx.get("memory")
        if mem is not None and hasattr(mem, "headers"):
            raw.update(mem.headers)
        ctx_headers = exec_ctx.get("headers")
        if isinstance(ctx_headers, dict):
            raw.update(ctx_headers)
        return raw

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="a2a_send",
            description="Send a question to the configured external A2A agent and return its grounded response.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def a2a_send(
            call_context: ToolCallContext, value: A2ASendInput
        ) -> RetrievalOutput:
            manager = resolve_manager(call_context)
            drv = manager.drivers.get(cfg.source)
            if drv is None:
                raise ValueError(f"A2A driver {cfg.source!r} is not configured")
            driver = cast(A2ADriver, drv)

            all_headers = self._extract_headers(call_context)
            authorization = (
                all_headers.get("authorization")
                if "authorization" in cfg.valid_headers
                else None
            )
            forwarded_headers = {
                h: all_headers[h]
                for h in cfg.valid_headers
                if h != "authorization" and h in all_headers
            }
            metadata: dict[str, Any] = {"workflow_id": cfg.remote_workflow_id}
            if forwarded_headers:
                metadata["headers"] = forwarded_headers
            metadata.update(cfg.extra_metadata)
            metadata.update(value.extra_metadata)

            accumulator = ResponseTextAccumulator()
            client = await driver.client(authorization=authorization)
            try:
                async with asyncio.timeout(driver.config.read_timeout_seconds):
                    request: a2a_pb2.SendMessageRequest | None = build_send_request(
                        value.question, metadata
                    )
                    while request is not None:
                        continuation: a2a_pb2.SendMessageRequest | None = None
                        async for response in client.send_message(request):
                            raise_for_terminal_task_error(response)
                            remote_fb = extract_feedback_request(response)
                            if remote_fb is not None:
                                user_reply = (
                                    await call_context.harness.request_feedback(
                                        question=remote_fb.question,
                                        response_schema=remote_fb.response_schema,
                                        timeout_ms=driver.config.read_timeout_seconds
                                        * 1000,
                                    )
                                )
                                if user_reply is None or not user_reply.strip():
                                    raise ValueError(
                                        "A2A remote feedback request was not answered"
                                    )
                                continuation = build_feedback_request(
                                    user_reply, remote_fb
                                )
                                break
                            accumulator.add(response)
                        request = continuation
            except TimeoutError as exc:
                raise ValueError(
                    f"A2A request timed out after {driver.config.read_timeout_seconds} seconds"
                ) from exc
            finally:
                await client.close()

            answer = "\n".join(accumulator.texts())
            ctx = Context(
                agent_id=cfg.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=value.question,
                title=cfg.title or "A2A remote agent",
                source="a2a",
                agent="a2a",
                summary=answer,
                chunks=(
                    [
                        Chunk(
                            chunk_id=uuid4().hex,
                            text=answer,
                            origin_agent="a2a",
                        )
                    ]
                    if answer
                    else []
                ),
            )
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        return [a2a_send]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="a2a_delegation_guide",
                name=f"A2A Remote Delegation ({self.config.source})",
                description="How to formulate self-contained tasks for the external A2A agent.",
                instructions=(
                    "The remote A2A agent `{namespace}` does not share your local tool "
                    "state unless included in `question`. Pass complete, self-contained "
                    "questions to `{namespace_prefix}a2a_send` with relevant entity names and constraints."
                ),
                recommended_tools=("a2a_send",),
            )
        ]


__all__ = [
    "A2ASendInput",
    "A2AToolkit",
    "A2AToolkitConfig",
]
