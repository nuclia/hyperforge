from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast
from uuid import uuid4

from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    ThinkingConfig,
)
from google.genai.types import (
    Tool as GenAITool,
)
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
from pydantic import BaseModel, Field

from .driver import GoogleDriver


class GoogleSearchInput(BaseModel):
    question: str = Field(description="Web search query or question.")


class GoogleSearchToolkitConfig(ToolkitConfig):
    toolkit: Literal["google"] = "google"
    source: str = "google"
    gen_model_id: str = "gemini-3.6-flash"


@toolkit(
    id="google",
    title="Google Grounded Web Search Toolkit",
    description="Grounded internet search via Google Gemini with resolved source URLs.",
    config_schema=GoogleSearchToolkitConfig,
)
class GoogleSearchToolkit(BaseToolkit[GoogleSearchToolkitConfig]):
    toolkit_id = "google"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config
        search_tool = GenAITool(google_search=GoogleSearch())

        @tool(
            name="internet_search",
            description="Perform a grounded web search using Google Gemini and return cited chunks.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def internet_search(
            call_context: ToolCallContext, value: GoogleSearchInput
        ) -> RetrievalOutput:
            manager = resolve_manager(call_context)
            drv = manager.drivers.get(cfg.source)
            if drv is None:
                raise ValueError(f"Google driver {cfg.source!r} is not configured")
            driver = cast(GoogleDriver, drv)

            response = await driver.client.aio.models.generate_content(
                model=cfg.gen_model_id,
                contents=value.question,
                config=GenerateContentConfig(
                    tools=[search_tool],
                    response_modalities=["TEXT"],
                    thinking_config=ThinkingConfig(include_thoughts=True),
                ),
            )

            ctx = Context(
                agent_id=cfg.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question=value.question,
                source="google",
                agent="google",
                title=cfg.title or "Internet search with Google",
            )
            answer = ""
            resolved_urls: dict[int, str] = {}
            if (
                response
                and response.candidates
                and response.candidates[0].content
                and response.candidates[0].content.parts
            ):
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.thought is not True and part.text is not None:
                            answer += part.text
                if (
                    candidate.grounding_metadata
                    and candidate.grounding_metadata.grounding_chunks
                ):
                    async with safe_http_client() as http_client:
                        for idx, g_chunk in enumerate(
                            candidate.grounding_metadata.grounding_chunks
                        ):
                            if g_chunk.web is not None and g_chunk.web.uri is not None:
                                resp = await http_client.get(g_chunk.web.uri)
                                resolved_urls[idx] = resp.headers.get(
                                    "location", g_chunk.web.uri
                                )
                if (
                    candidate.grounding_metadata
                    and candidate.grounding_metadata.grounding_supports
                ):
                    for ground in candidate.grounding_metadata.grounding_supports:
                        urls = [
                            resolved_urls[i]
                            for i in (ground.grounding_chunk_indices or [])
                            if i in resolved_urls
                        ]
                        seg_text = (
                            ground.segment.text
                            if ground.segment and ground.segment.text
                            else ""
                        )
                        ctx.chunks.append(
                            Chunk(
                                chunk_id=uuid4().hex,
                                text=seg_text,
                                url=urls,
                                origin_url=urls[0] if urls else None,
                                origin_agent="google",
                            )
                        )
            ctx.summary = answer
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        return [internet_search]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="google_search_guide",
                name="Google Grounded Web Search Guide",
                description="When and how to query current public web information via Google Search.",
                instructions=(
                    "Use `{namespace_prefix}internet_search` for current events, public "
                    "market facts, or external documentation not present in internal Knowledge Boxes. "
                    "Formulate concise, unambiguous search queries and preserve returned origin URLs."
                ),
                recommended_tools=("internet_search",),
            )
        ]


__all__ = [
    "GoogleSearchInput",
    "GoogleSearchToolkit",
    "GoogleSearchToolkitConfig",
]
