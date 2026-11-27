from __future__ import annotations

import inspect
import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Literal

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.configure import toolkit
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessEventType,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    tool,
)
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.models import Context
from hyperforge.toolkits.common import resolve_manager
from hyperforge.utils import http as http_utils
from hyperforge.utils import sync_dns_validation
from pydantic import BaseModel, Field, field_validator

from .config import Method

_EXTRACT_WEBHOOK_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Collect parameters matching the schema from the question, context, and answer:
{% if prompt -%}
{{ prompt }}
{% endif -%}
[START OF CONTEXT]
{{ context }}
[END OF CONTEXT]
MAIN QUESTION: {{ question }}
MAIN ANSWER: {{ answer }}
"""
)


class WebhookSendInput(BaseModel):
    payload: dict[str, Any] | list[Any] = Field(
        description="JSON payload to send directly to the configured webhook URL."
    )


class WebhookExtractInput(BaseModel):
    question: str
    answer: str = ""
    context: str = ""
    contexts: list[dict[str, Any]] = Field(default_factory=list)


class WebhookOutput(BaseModel):
    status_code: int
    response_text: str


class WebhookToolkitConfig(ToolkitConfig):
    toolkit: Literal["external"] = "external"
    url: str
    method: Method = Method.POST
    headers: dict[str, str] = Field(default_factory=dict)
    prompt: str | None = None
    description: str | None = None
    call_schema: dict[str, Any] | None = None
    call_obj: dict[str, Any] | None = None
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.reasoning),
        title="Parameter extraction model",
    )
    timeout: int = Field(default=30, ge=1, le=300)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1, le=10 * 1024 * 1024)

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        return sync_dns_validation(value)


@toolkit(
    id="external",
    title="External Webhook & API Call Toolkit",
    description="Send structured webhook payloads directly or extract parameters via LLM schema.",
    config_schema=WebhookToolkitConfig,
)
class WebhookToolkit(BaseToolkit[WebhookToolkitConfig]):
    toolkit_id = "external"

    async def _dispatch_json(self, body: Any) -> WebhookOutput:
        from . import agent as legacy_ext_agent

        cfg = self.config
        client_factory = (
            legacy_ext_agent.safe_http_client
            if legacy_ext_agent.safe_http_client is not http_utils.safe_http_client
            else http_utils.safe_http_client
        )
        async with client_factory(timeout=cfg.timeout) as client:
            built: Any = client.build_request(
                method=cfg.method.value,
                url=cfg.url,
                json=body,
                headers=cfg.headers,
            )
            req = await built if inspect.isawaitable(built) else built
            resp = await client.send(req, stream=True)
            try:
                raw = await http_utils.read_limited_response(
                    resp, cfg.max_response_bytes
                )
            finally:
                await resp.aclose()
        if resp.status_code != 200:
            raise RuntimeError(
                f"External webhook returned non-200 status: {resp.status_code}"
            )
        return WebhookOutput(
            status_code=resp.status_code,
            response_text=raw.decode("utf-8", errors="replace"),
        )

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="webhook_send",
            description="Send an explicit JSON payload to the configured external webhook endpoint.",
        )
        async def webhook_send(
            _call_context: ToolCallContext, value: WebhookSendInput
        ) -> WebhookOutput:
            return await self._dispatch_json(value.payload)

        @tool(
            name="webhook_extract_and_send",
            description="Build or extract a JSON payload from question/answer/context and send it to the webhook.",
        )
        async def webhook_extract_and_send(
            call_context: ToolCallContext, value: WebhookExtractInput
        ) -> WebhookOutput:
            ctx_text = value.context
            if not ctx_text and value.contexts:
                parsed_ctxs = [
                    Context.model_validate(
                        {
                            "original_question_uuid": None,
                            "actual_question_uuid": None,
                            **raw,
                        }
                    )
                    for raw in value.contexts
                ]
                ctx_text = "\n\n".join(c.context_markdown() for c in parsed_ctxs)
            if cfg.call_schema:
                manager = resolve_manager(call_context)
                rendered = _EXTRACT_WEBHOOK_PROMPT.render(
                    question=value.question,
                    context=ctx_text,
                    answer=value.answer,
                    prompt=cfg.prompt,
                )
                payload, _, _ = await manager.execute_json(
                    user_id="external_webhook_extract",
                    model=cfg.model,
                    schema={
                        "title": "external_call",
                        "description": cfg.description
                        or "Extract parameters for external webhook",
                        "parameters": cfg.call_schema,
                    },
                    prompt=rendered,
                )
            else:
                payload = deepcopy(cfg.call_obj) if cfg.call_obj is not None else {}
                payload["question"] = value.question
                payload["answer"] = value.answer
            out = await self._dispatch_json(payload)
            await call_context.harness.emit(
                HarnessEventType.WORKFLOW_STEP,
                {
                    "module": "external",
                    "title": "External call: Webhook dispatched",
                    "agent_path": "/postprocess/external",
                    "value": json.dumps(payload),
                    "reason": out.response_text,
                },
            )
            return out

        return [webhook_send, webhook_extract_and_send]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="webhook_integration_guide",
                name="External Webhook Integration Guide",
                description="When to pass explicit JSON payloads vs schema-extracted webhook calls.",
                instructions=(
                    "In Code Mode or when structured fields are already known, call "
                    "`{namespace_prefix}webhook_send(payload={...})` directly to avoid "
                    "an extra LLM extraction call. Use `{namespace_prefix}webhook_extract_and_send` "
                    "when extracting parameters from free-form text via `call_schema`."
                ),
                recommended_tools=("webhook_send", "webhook_extract_and_send"),
            )
        ]


__all__ = [
    "WebhookExtractInput",
    "WebhookOutput",
    "WebhookSendInput",
    "WebhookToolkit",
    "WebhookToolkitConfig",
]
