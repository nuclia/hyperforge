from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal
from uuid import uuid4

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
from hyperforge.toolkits.common import RetrievalOutput, retrieval_context_factory
from hyperforge.utils import http as http_utils
from hyperforge.utils import sync_dns_validation
from pydantic import BaseModel, Field, field_validator


class HTTPFetchInput(BaseModel):
    question: str = Field(
        default="",
        description="Optional query value passed via configured query/form parameter.",
    )


class HTTPToolkitConfig(ToolkitConfig):
    toolkit: Literal["http"] = "http"
    method: Literal["GET", "POST"] = "GET"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    question_query_param: str | None = None
    question_post_field: str | None = None
    timeout: int = Field(default=10, ge=1, le=300)
    max_response_bytes: int = Field(default=1024 * 1024, ge=1, le=10 * 1024 * 1024)

    @field_validator("url")
    @classmethod
    def check_url(cls, value: str) -> str:
        return sync_dns_validation(value)


@toolkit(
    id="http",
    title="HTTP Endpoint Retrieval Toolkit",
    description="Fetch text or JSON payloads from a configured public HTTP endpoint.",
    config_schema=HTTPToolkitConfig,
)
class HTTPToolkit(BaseToolkit[HTTPToolkitConfig]):
    toolkit_id = "http"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="http_fetch",
            description=f"Execute HTTP {cfg.method} against the configured endpoint and return response content.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def http_fetch(
            _call_context: ToolCallContext, value: HTTPFetchInput
        ) -> RetrievalOutput:
            params = (
                {cfg.question_query_param: value.question}
                if cfg.question_query_param
                else None
            )
            data = (
                {cfg.question_post_field: value.question}
                if cfg.method == "POST" and cfg.question_post_field
                else None
            )
            async with http_utils.safe_http_client(timeout=cfg.timeout) as client:
                req = client.build_request(
                    cfg.method,
                    cfg.url,
                    headers=cfg.headers,
                    params=params,
                    data=data,
                )
                resp = await client.send(req, stream=True)
                try:
                    raw = await http_utils.read_limited_response(
                        resp, cfg.max_response_bytes
                    )
                finally:
                    await resp.aclose()

            if resp.status_code != 200:
                raise RuntimeError(
                    f"HTTP {cfg.method} request failed with status {resp.status_code}"
                )
            body = raw.decode("utf-8", errors="replace")
            ctx = Context(
                agent_id=cfg.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=value.question,
                source="http",
                agent="http",
                title=cfg.title or "HTTP Call",
                chunks=[
                    Chunk(
                        chunk_id=uuid4().hex,
                        text=body,
                        origin_url=cfg.url,
                        origin_agent="http",
                    )
                ],
            )
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        return [http_fetch]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="http_connector_guide",
                name="HTTP Connector Guide",
                description=f"How to retrieve data from `{self.config.url}`.",
                instructions=(
                    "Call `{namespace_prefix}http_fetch` with `question` to retrieve live "
                    "payloads from the configured HTTP endpoint."
                ),
                recommended_tools=("http_fetch",),
            )
        ]


__all__ = [
    "HTTPFetchInput",
    "HTTPToolkit",
    "HTTPToolkitConfig",
]
