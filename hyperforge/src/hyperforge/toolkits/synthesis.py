from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Literal

from nuclia.lib.nua_responses import ChatModel, UserPrompt
from pydantic import BaseModel, Field

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.configure import toolkit
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    tool,
)
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.manager import build_reasoning
from hyperforge.models import AnswerCitations, CitationMetadata, Context

from .common import resolve_manager, resolve_workflow_history, resolve_workflow_rules
from .context_ops import generate_block_id

DEFAULT_SYNTHESIS_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Provide accurate, clear, and well-structured "
    "answers strictly from the provided context. Do not use outside knowledge or "
    "assumptions. Acknowledge clearly when information is insufficient."
)

_CITATION_RULES_SUFFIX = """
You are given source blocks with IDs like `block-AA` or `block-AA-0`.
1. Cite sources in the body only with bracketed Arabic numerals in order of first use: [1], [2], etc.
2. Reuse the same number when citing the same block again. Separate adjacent citations with spaces: [1] [2].
3. After a blank line at the end, list ONLY the unique mappings (`[1]: block-AA`), one per line, with no heading.
"""

_SYNTHESIZE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    """{% if rules -%}
# Generation Rules
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}
## Question
{{ question }}

## Provided Context
[START OF CONTEXT]
{{ context_text }}
[END OF CONTEXT]

## Answering Guidelines
- Treat existing context summaries and chunks as supporting evidence.
- Combine complementary facts across contexts when the question has multiple parts.
{% if conversational -%}
- If the context is incomplete, provide a partial answer from supported facts and invite clarification.
{% else -%}
- If the context is insufficient to answer, state: "Not enough data to answer this."
{% endif -%}
{% if prompt -%}
## Additional Instructions
{{ prompt }}
{% endif -%}
{% if chat_history -%}
## Conversation History
{{ chat_history }}
{% endif -%}
"""
)

_GENERATE_TEXT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    """Generate the requested output strictly from the provided context.
If the context is insufficient, respond with: "Not enough data to generate this."
{% if rules -%}
# Rules
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}
{% if prompt -%}
{{ prompt }}
{% endif -%}

{{ context_text }}

MAIN QUESTION: {{ question }}
"""
)

_FOOTNOTE_DEF = re.compile(r"^\[(\d+)\]:\s*(block-\S+)\s*$", re.MULTILINE)
_FOOTNOTE_MARKER = re.compile(r"\[(\d+)\]")


def _parse_citation_id(citation_id: str) -> tuple[str, int | None]:
    if citation_id.count("-") >= 2:
        prefix, tail = citation_id.rsplit("-", 1)
        return prefix, int(tail)
    return citation_id, None


def split_markdown_footnotes(answer: str) -> tuple[str, list[tuple[str, str]]]:
    defs = list(_FOOTNOTE_DEF.finditer(answer))
    if not defs:
        return answer, []
    return (
        answer[: defs[0].start()].rstrip(),
        [(m.group(1), m.group(2)) for m in defs],
    )


def normalize_chunk_citations(answer: str, contexts: list[Context]) -> str:
    body, footnotes = split_markdown_footnotes(answer)
    if not footnotes:
        return answer
    by_id = {c.citations_id: c for c in contexts if c.citations_id is not None}
    targets_by_marker: dict[str, list[str]] = {}
    changed = False
    for marker, cid in footnotes:
        try:
            ctx_cid, chunk_idx = _parse_citation_id(cid)
        except ValueError:
            targets_by_marker[marker] = [cid]
            continue
        ctx = by_id.get(ctx_cid)
        if chunk_idx is None and ctx is not None and ctx.chunks:
            indices = [
                idx
                for idx, ch in enumerate(ctx.chunks)
                if ctx.citations is None or ch.chunk_id in ctx.citations
            ]
            if indices:
                targets_by_marker[marker] = [f"{cid}-{idx}" for idx in indices]
                changed = True
                continue
        targets_by_marker[marker] = [cid]
    if not changed:
        return answer
    ordered = list(
        dict.fromkeys(
            target for marker, _ in footnotes for target in targets_by_marker[marker]
        )
    )
    numbers = {cid: idx for idx, cid in enumerate(ordered, start=1)}
    new_body = _FOOTNOTE_MARKER.sub(
        lambda m: (
            " ".join(f"[{numbers[t]}]" for t in targets_by_marker[m.group(1)])
            if m.group(1) in targets_by_marker
            else m.group(0)
        ),
        body,
    )
    refs = "\n".join(f"[{idx}]: {cid}" for idx, cid in enumerate(ordered, start=1))
    return f"{new_body}\n\n{refs}"


def build_citations(answer: str, contexts: list[Context]) -> AnswerCitations:
    result = AnswerCitations()
    by_id = {c.citations_id: c for c in contexts if c.citations_id is not None}
    _, footnotes = split_markdown_footnotes(answer)
    for cid in dict.fromkeys(c for _, c in footnotes):
        try:
            ctx_cid, chunk_idx = _parse_citation_id(cid)
        except ValueError:
            continue
        ctx = by_id.get(ctx_cid)
        if ctx is None:
            continue
        urls: list[str] = []
        if chunk_idx is not None:
            if 0 <= chunk_idx < len(ctx.chunks) and ctx.chunks[chunk_idx].origin_url:
                urls.append(ctx.chunks[chunk_idx].origin_url or "")
        else:
            for ch in ctx.chunks:
                if (
                    ctx.citations is None or ch.chunk_id in ctx.citations
                ) and ch.origin_url:
                    urls.append(ch.origin_url)
        result.metadata[cid] = CitationMetadata(
            context_id=ctx.id,
            origin_urls=list(dict.fromkeys(u for u in urls if u)),
            chunk_index=chunk_idx,
        )
    return result


def _render_contexts(
    contexts: list[Context], *, full: bool = False, include_summaries: bool = False
) -> str:
    parts: list[str] = []
    for ctx in contexts:
        header = (
            f"## [{ctx.citations_id}] {ctx.title}\n\n"
            if ctx.citations_id and ctx.title
            else f"## [{ctx.citations_id}]\n\n"
            if ctx.citations_id
            else f"## {ctx.title}\n\n"
            if ctx.title
            else ""
        )
        if not full and ctx.summary.strip():
            body = ctx.answer_summary_markdown()
        else:
            body = ""
            if include_summaries and ctx.summary.strip():
                body += f"### Existing context summary\n\n{ctx.summary}\n\n"
            body += ctx.context_markdown()
        parts.append(header + body)
    return "\n\n".join(parts)


class SynthesizeInput(BaseModel):
    question: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None
    citations: bool | None = None
    conversational: bool | None = None
    chat_history: str = ""
    rules: list[str] = Field(default_factory=list)


class SynthesizeOutput(BaseModel):
    answer: str
    citations: dict[str, Any] | None = None
    is_answered: bool = True


class GenerateTextInput(BaseModel):
    question: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None
    rules: list[str] = Field(default_factory=list)


class PassthroughInput(BaseModel):
    contexts: list[dict[str, Any]] = Field(default_factory=list)


class GenerateTextOutput(BaseModel):
    text: str
    rich_answer: dict[str, Any] | None = None


class SynthesisToolkitConfig(ToolkitConfig):
    toolkit: Literal["synthesis"] = "synthesis"
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
    )
    system_prompt: str | None = None
    prompt: str | None = None
    citations: bool = False
    force_chunk_level_citations: bool = False
    conversational: bool = False
    rich_context: bool = False
    rules: list[str] = Field(default_factory=list)


@toolkit(
    id="synthesis",
    title="Answer Synthesis & Generation",
    description="Synthesize retrieved contexts into grounded answers with optional markdown footnote citations.",
    config_schema=SynthesisToolkitConfig,
)
class SynthesisToolkit(BaseToolkit[SynthesisToolkitConfig]):
    toolkit_id = "synthesis"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="synthesize_answer",
            description=(
                "Synthesize retrieved contexts into a final grounded answer, optionally "
                "producing two-level markdown footnote citations."
            ),
        )
        async def synthesize_answer(
            context: ToolCallContext, value: SynthesizeInput
        ) -> SynthesizeOutput:
            manager = resolve_manager(context)
            parsed = [
                Context.model_validate(
                    {
                        "original_question_uuid": None,
                        "actual_question_uuid": None,
                        **raw,
                    }
                )
                for raw in value.contexts
            ]
            citations_on = (
                value.citations if value.citations is not None else cfg.citations
            )
            conversational = (
                value.conversational
                if value.conversational is not None
                else cfg.conversational
            )
            if citations_on:
                for idx, ctx in enumerate(parsed):
                    ctx.citations_id = generate_block_id(idx)

            effective_rules = [
                *resolve_workflow_rules(context),
                *cfg.rules,
                *value.rules,
            ]
            effective_history = value.chat_history
            if conversational and not effective_history:
                effective_history = "\n".join(resolve_workflow_history(context))
            use_full = citations_on and cfg.force_chunk_level_citations
            rendered_prompt = _SYNTHESIZE_PROMPT_TEMPLATE.render(
                question=value.question,
                context_text=_render_contexts(
                    parsed, full=use_full, include_summaries=use_full
                ),
                prompt=value.prompt or cfg.prompt,
                conversational=conversational,
                chat_history=effective_history,
                rules=effective_rules,
            )
            if citations_on:
                rendered_prompt += _CITATION_RULES_SUFFIX

            chat_model = ChatModel(
                user_id="summarize",
                question="",
                user_prompt=UserPrompt(prompt=rendered_prompt),
                system=cfg.system_prompt or DEFAULT_SYNTHESIS_SYSTEM_PROMPT,
                format_prompt=False,
                generative_model=cfg.model.model_id,
                reasoning=build_reasoning(cfg.model),
                max_tokens=5000,
            )
            resp, _, _ = await manager.execute_raw(chat_model)
            answer = resp.answer or ""
            end_code = getattr(resp, "code", "0")

            if end_code == "-2" and not use_full:
                fallback_prompt = _SYNTHESIZE_PROMPT_TEMPLATE.render(
                    question=value.question,
                    context_text=_render_contexts(
                        parsed, full=True, include_summaries=True
                    ),
                    prompt=value.prompt or cfg.prompt,
                    conversational=conversational,
                    chat_history=effective_history,
                    rules=effective_rules,
                )
                if citations_on:
                    fallback_prompt += _CITATION_RULES_SUFFIX
                chat_model.user_prompt = UserPrompt(prompt=fallback_prompt)
                resp, _, _ = await manager.execute_raw(chat_model)
                answer = resp.answer or ""
                end_code = resp.code

            if citations_on and cfg.force_chunk_level_citations:
                answer = normalize_chunk_citations(answer, parsed)

            citations_dict = (
                build_citations(answer, parsed).model_dump(mode="json")
                if citations_on
                else None
            )
            return SynthesizeOutput(
                answer=answer,
                citations=citations_dict,
                is_answered=end_code != "-2",
            )

        @tool(
            name="generate_text",
            description="Generate custom structured or free-form text from retrieved contexts.",
        )
        async def generate_text(
            context: ToolCallContext, value: GenerateTextInput
        ) -> GenerateTextOutput:
            manager = resolve_manager(context)
            parsed = [
                Context.model_validate(
                    {
                        "original_question_uuid": None,
                        "actual_question_uuid": None,
                        **raw,
                    }
                )
                for raw in value.contexts
            ]
            rendered = _GENERATE_TEXT_TEMPLATE.render(
                question=value.question,
                context_text=_render_contexts(parsed, full=False),
                prompt=value.prompt or cfg.prompt,
                rules=[*cfg.rules, *value.rules],
            )
            chat_model = ChatModel(
                user_id="synthesis_generate",
                question="",
                user_prompt=UserPrompt(prompt=rendered),
                format_prompt=False,
                generative_model=cfg.model.model_id,
                reasoning=build_reasoning(cfg.model),
                max_tokens=2000,
            )
            resp, _, _ = await manager.execute_raw(chat_model)
            return GenerateTextOutput(text=resp.answer or "")

        @tool(
            name="passthrough_contexts",
            description="Concatenate retrieved context chunks directly without an LLM call.",
        )
        async def passthrough_contexts(
            _context: ToolCallContext, value: PassthroughInput
        ) -> GenerateTextOutput:
            parsed = [
                Context.model_validate(
                    {
                        "original_question_uuid": None,
                        "actual_question_uuid": None,
                        **raw,
                    }
                )
                for raw in value.contexts
            ]
            if cfg.rich_context:
                chunks = [
                    ch.model_dump(mode="json")
                    for ctx in parsed
                    for ch in ctx.chunks
                    if ch.text
                ]
                structured_data = [s for ctx in parsed for s in ctx.structured if s]
                images = {
                    k: img.model_dump(mode="json")
                    for ctx in parsed
                    for k, img in (ctx.images or {}).items()
                }
                image_urls = [url for ctx in parsed for url in ctx.image_urls if url]
                return GenerateTextOutput(
                    text="",
                    rich_answer={
                        "chunks": chunks,
                        "structured": structured_data,
                        "images": images,
                        "image_urls": image_urls,
                    },
                )
            parts = [
                item
                for ctx in parsed
                for item in (
                    *(ch.text for ch in ctx.chunks if ch.text),
                    *(s for s in ctx.structured if s),
                )
            ]
            return GenerateTextOutput(
                text="\n\n".join(parts) if parts else "(no context retrieved)"
            )

        return [synthesize_answer, generate_text, passthrough_contexts]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="synthesis_guide",
                name="Grounded Synthesis & Citations",
                description="Synthesize multi-source contexts with verified footnote citations.",
                instructions=(
                    "In Code Mode workflows, call `{namespace_prefix}synthesize_answer` "
                    "to produce a final answer with `citations=True`, and forward `answer`, "
                    "`citations`, and `is_answered` into `output({...})`."
                ),
                recommended_tools=("synthesize_answer", "generate_text"),
                code_examples=(
                    "synth = {namespace_prefix}synthesize_answer(question=question, contexts=contexts, citations=True)\n"
                    'output({"answer": synth["answer"], "citations": synth["citations"], "contexts": contexts})',
                ),
            )
        ]


__all__ = [
    "GenerateTextInput",
    "GenerateTextOutput",
    "PassthroughInput",
    "SynthesisToolkit",
    "SynthesisToolkitConfig",
    "SynthesizeInput",
    "SynthesizeOutput",
    "build_citations",
    "normalize_chunk_citations",
    "split_markdown_footnotes",
]
