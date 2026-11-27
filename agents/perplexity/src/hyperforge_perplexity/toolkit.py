from __future__ import annotations

from base64 import b64encode
from collections.abc import Sequence
from typing import Any, Literal, cast
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
from hyperforge.toolkits.common import (
    RetrievalOutput,
    resolve_manager,
    retrieval_context_factory,
)
from hyperforge.utils.http import safe_http_client
from nuclia.lib.nua_responses import Image
from perplexity.types.shared.chat_message_output import ChatMessageOutput
from perplexity.types.shared_params.chat_message_input import ChatMessageInput
from perplexity.types.shared_params.web_search_options import WebSearchOptions
from pydantic import BaseModel, Field

from .driver import PerplexityDriver

_DEFAULT_SYSTEM_PROMPT = "Be precise and concise"


class PerplexityQueryInput(BaseModel):
    question: str = Field(description="Web search query or question.")


class PerplexityToolkitConfig(ToolkitConfig):
    toolkit: Literal["perplexity"] = "perplexity"
    source: str = "perplexity"
    domain: list[str] = Field(default_factory=list)
    search_context_size: Literal["low", "medium", "high"] = "low"
    images: bool = False
    prompt: str | None = None
    max_results: int = Field(default=10, ge=1, le=50)
    max_tokens_per_page: int = Field(default=4096, ge=128)


@toolkit(
    id="perplexity",
    title="Perplexity Web Search & Answer Toolkit",
    description="Grounded web synthesis (`perplexity_answer`) and raw web snippet search (`perplexity_search`).",
    config_schema=PerplexityToolkitConfig,
)
class PerplexityToolkit(BaseToolkit[PerplexityToolkitConfig]):
    toolkit_id = "perplexity"

    def _driver(self, call_context: ToolCallContext) -> PerplexityDriver:
        manager = resolve_manager(call_context)
        drv = manager.drivers.get(self.config.source)
        if drv is None:
            raise ValueError(
                f"Perplexity driver {self.config.source!r} is not configured"
            )
        return cast(PerplexityDriver, drv)

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="perplexity_answer",
            description="Run Perplexity sonar-pro grounded web search and synthesis with citations.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def perplexity_answer(
            call_context: ToolCallContext, value: PerplexityQueryInput
        ) -> RetrievalOutput:
            driver = self._driver(call_context)
            messages = [
                ChatMessageInput(
                    role="system", content=cfg.prompt or _DEFAULT_SYSTEM_PROMPT
                ),
                ChatMessageInput(role="user", content=value.question),
            ]
            response = await driver.client.chat.completions.create(
                messages=messages,
                model="sonar-pro",
                return_images=cfg.images,
                return_related_questions=False,
                search_domain_filter=cfg.domain,
                web_search_options=WebSearchOptions(
                    search_context_size=cfg.search_context_size
                ),
            )
            ctx = Context(
                agent_id=cfg.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=value.question,
                title=cfg.title or "Internet search with Perplexity",
                source="perplexity",
                agent="perplexity",
            )
            text: str | None = None
            for choice in response.choices:
                if (
                    choice.message is not None
                    and choice.message.content is not None
                    and isinstance(choice.message, ChatMessageOutput)
                ):
                    text = (
                        choice.message.content
                        if isinstance(choice.message.content, str)
                        else str(choice.message.content)
                    )
            raw_images = getattr(response, "images", None)
            if cfg.images and isinstance(raw_images, list):
                async with safe_http_client() as session:
                    for raw_img in raw_images:
                        img_dict = cast(dict[str, Any], raw_img)
                        resp = await session.get(img_dict["image_url"])
                        ctype = (
                            (resp.headers.get("Content-Type") or "")
                            .split(";", 1)[0]
                            .strip()
                        )
                        if ctype:
                            ctx.images[img_dict["origin_url"]] = Image(
                                content_type=ctype,
                                b64encoded=b64encode(resp.content).decode(),
                            )
            citations = list(getattr(response, "citations", None) or [])
            if text is not None:
                ctx.chunks.append(
                    Chunk(
                        chunk_id=uuid4().hex,
                        text=text,
                        url=citations,
                        origin_url=citations[0] if citations else None,
                        origin_agent="perplexity",
                    )
                )
            ctx.summary = text or ""
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        @tool(
            name="perplexity_search",
            description="Perform raw Perplexity web search returning ranked page snippets and URLs without synthesis.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def perplexity_search(
            call_context: ToolCallContext, value: PerplexityQueryInput
        ) -> RetrievalOutput:
            driver = self._driver(call_context)
            response = await driver.client.search.create(
                query=value.question,
                search_domain_filter=cfg.domain,
                max_results=cfg.max_results,
                max_tokens_per_page=cfg.max_tokens_per_page,
            )
            chunks: list[Chunk] = []
            for item in response.results or []:
                url = item.url or ""
                chunks.append(
                    Chunk(
                        chunk_id=uuid4().hex,
                        text=item.snippet or "",
                        title=item.title or url or "Web Result",
                        url=[url] if url else [],
                        origin_url=url or None,
                        origin_agent="perplexity",
                    )
                )
            ctx = Context(
                agent_id=cfg.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=value.question,
                title=cfg.title or "Perplexity Search",
                source="perplexity",
                agent="perplexity_search",
                chunks=chunks,
            )
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        return [perplexity_answer, perplexity_search]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="perplexity_search_guide",
                name="Perplexity Web Search & Synthesis Guide",
                description="Choosing between raw web snippet search and grounded sonar-pro answers.",
                instructions=(
                    "Prefer `{namespace_prefix}perplexity_search` when you want raw page "
                    "snippets and URLs to synthesize yourself without intermediate summarization. "
                    "Use `{namespace_prefix}perplexity_answer` when you want Perplexity `sonar-pro` "
                    "to synthesize a multi-page web answer directly."
                ),
                recommended_tools=("perplexity_search", "perplexity_answer"),
            )
        ]


__all__ = [
    "PerplexityQueryInput",
    "PerplexityToolkit",
    "PerplexityToolkitConfig",
]
