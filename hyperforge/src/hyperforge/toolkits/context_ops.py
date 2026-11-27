from __future__ import annotations

import string
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from hyperforge.configure import toolkit
from hyperforge.harness_sdk import (
    BaseToolkit,
    HarnessContextType,
    HarnessEventType,
    HarnessSkill,
    HarnessTool,
    ToolCallContext,
    ToolkitConfig,
    tool,
)
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.models import Chunk, Context
from hyperforge.prompts import (
    NEXT_REPHRASE_JSON_SCHEMA,
    NEXT_REPHRASE_PROMPT_SYSTEM,
    NEXT_REPHRASE_PROMPT_TEMPLATE,
    VALIDATE_MULTIPLE_CONTEXTS_JSON_SCHEMA,
    VALIDATE_MULTIPLE_CONTEXTS_PROMPT_TEMPLATE,
)

from .common import RetrievalOutput, resolve_manager, retrieval_context_factory


@lru_cache(maxsize=128)
def generate_block_id(n: int) -> str:
    """Generate deterministic block IDs (`block-AA` .. `block-ZZ`) for two-level citations."""
    if n < 0:
        raise ValueError("Block index must be non-negative")
    idx = n % (26 * 26)
    letters = string.ascii_uppercase
    return f"block-{letters[idx // 26]}{letters[idx % 26]}"


class ValidateContextsInput(BaseModel):
    question: str = Field(description="Question used to evaluate context relevance.")
    contexts: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved context dictionaries to validate and prune.",
    )
    prune: bool = Field(
        default=True,
        description="Prune uncited chunks from relevant contexts.",
    )


class ValidateContextsOutput(BaseModel):
    useful: bool
    contexts: list[dict[str, Any]]
    missing_query: str | None = None


class RephraseNextInput(BaseModel):
    question: str
    summaries: list[str] = Field(default_factory=list)
    agent_description: str = "Retrieval tool providing domain context."
    extra_prompt: str | None = None


class RephraseNextOutput(BaseModel):
    question: str
    needed: bool
    reason: str = ""


class EmptyInput(BaseModel):
    pass


class ContextOpsToolkitConfig(ToolkitConfig):
    toolkit: Literal["context_ops"] = "context_ops"
    validation_model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Context validation model",
    )
    rephrase_model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Next-step rephrase model",
    )


class StaticContextToolkitConfig(ToolkitConfig):
    toolkit: Literal["static"] = "static"
    context: str = Field(default="", description="Static text chunk to return.")
    structured: str | None = Field(
        default=None, description="Optional structured payload string."
    )


@toolkit(
    id="context_ops",
    title="Context Validation & Chaining",
    description="Validate retrieved contexts, prune uncited chunks, and rephrase follow-up queries.",
    config_schema=ContextOpsToolkitConfig,
)
class ContextOpsToolkit(BaseToolkit[ContextOpsToolkitConfig]):
    toolkit_id = "context_ops"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="validate_contexts",
            description=(
                "Assess relevance of retrieved contexts for a question, extract block "
                "citations, optionally prune uncited chunks, and report any missing_query."
            ),
        )
        async def validate_contexts(
            context: ToolCallContext, value: ValidateContextsInput
        ) -> ValidateContextsOutput:
            if not value.contexts:
                return ValidateContextsOutput(
                    useful=False,
                    contexts=[],
                    missing_query=value.question,
                )

            parsed_contexts = [
                Context.model_validate(
                    {
                        "original_question_uuid": None,
                        "actual_question_uuid": None,
                        **raw,
                    }
                )
                for raw in value.contexts
            ]
            source_complete = all(
                ctx.summary.strip() and ctx.citations is not None
                for ctx in parsed_contexts
            )
            if source_complete:
                retained: list[dict[str, Any]] = []
                for ctx in parsed_contexts:
                    if value.prune and ctx.citations:
                        ctx.prune_to_citations()
                    retained.append(ctx.model_dump(mode="json"))
                return ValidateContextsOutput(
                    useful=True,
                    contexts=retained,
                    missing_query=None,
                )

            manager = resolve_manager(context)
            context_by_id = {ctx.id: ctx for ctx in parsed_contexts}
            existing_summaries = {ctx.id: ctx.summary for ctx in parsed_contexts}
            existing_citations = {ctx.id: ctx.citations for ctx in parsed_contexts}
            block_targets: dict[str, tuple[str, str]] = {}
            prompt_contexts: list[dict[str, Any]] = []
            block_index = 0

            for ctx in parsed_contexts:
                ctx.citations = None
                blocks: dict[str, str] = {}
                for chunk in ctx.chunks:
                    block_id = generate_block_id(block_index)
                    block_index += 1
                    blocks[block_id] = chunk.render()
                    block_targets[block_id] = (ctx.id, chunk.chunk_id)
                for s_idx, structured in enumerate(ctx.structured):
                    block_id = generate_block_id(block_index)
                    block_index += 1
                    blocks[block_id] = structured
                    block_targets[block_id] = (ctx.id, f"structured-{s_idx}")
                prompt_contexts.append(
                    {
                        "context_id": ctx.id,
                        "title": ctx.title,
                        "summary": ctx.summary,
                        "blocks": blocks,
                    }
                )

            prompt = VALIDATE_MULTIPLE_CONTEXTS_PROMPT_TEMPLATE.render(
                question=value.question,
                contexts=prompt_contexts,
            )
            try:
                resp, _, _ = await manager.execute_json(
                    user_id="context_ops_validate",
                    prompt=prompt,
                    schema=VALIDATE_MULTIPLE_CONTEXTS_JSON_SCHEMA,
                    model=cfg.validation_model,
                )
            except Exception:
                return ValidateContextsOutput(
                    useful=True,
                    contexts=[ctx.model_dump(mode="json") for ctx in parsed_contexts],
                    missing_query=None,
                )

            aggregated: dict[str, tuple[list[str], list[str]]] = {}
            for ctx_res in resp.get("contexts", []):
                cid = ctx_res.get("context_id")
                if cid not in context_by_id:
                    continue
                answers, cited = aggregated.setdefault(cid, ([], []))
                ans = str(ctx_res.get("answer", ""))
                if ans and ans not in answers:
                    answers.append(ans)
                for block_id in dict.fromkeys(ctx_res.get("citations", [])):
                    target = block_targets.get(block_id)
                    if (
                        target is not None
                        and target[0] == cid
                        and target[1] not in cited
                    ):
                        cited.append(target[1])

            selected: list[dict[str, Any]] = []
            for ctx in parsed_contexts:
                answers, cited = aggregated.get(ctx.id, ([], []))
                prev_summary = existing_summaries[ctx.id]
                if not answers and not cited and not prev_summary.strip():
                    continue
                if prev_summary.strip():
                    ctx.summary = prev_summary
                    ctx.citations = existing_citations[ctx.id] or cited or []
                elif answers:
                    ctx.summary = "\n".join(answers)
                    ctx.citations = cited or existing_citations[ctx.id] or []
                else:
                    ctx.citations = cited or existing_citations[ctx.id] or []
                    ctx.summary = ""

                if value.prune:
                    if ctx.citations == []:
                        if not ctx.summary:
                            continue
                        ctx.chunks = []
                        ctx.structured = []
                    elif ctx.citations is not None:
                        ctx.prune_to_citations()
                selected.append(ctx.model_dump(mode="json"))

            raw_missing = str(resp.get("missing_info_query") or "").strip()
            await context.harness.emit(
                HarnessEventType.WORKFLOW_STEP,
                {
                    "module": "context_ops",
                    "title": "Context validation",
                    "agent_path": "/context/validate",
                    "value": raw_missing or "No missing information",
                    "reason": str(resp.get("reason") or ""),
                },
            )
            return ValidateContextsOutput(
                useful=bool(selected),
                contexts=selected,
                missing_query=raw_missing or None,
            )

        @tool(
            name="rephrase_for_next",
            description="Rephrase a question for a chained downstream retrieval tool using prior summaries.",
        )
        async def rephrase_for_next(
            context: ToolCallContext, value: RephraseNextInput
        ) -> RephraseNextOutput:
            manager = resolve_manager(context)
            prompt = NEXT_REPHRASE_PROMPT_TEMPLATE.render(
                question=value.question,
                contexts=value.summaries,
                info=value.agent_description,
                extra_info=value.extra_prompt,
            )
            try:
                resp, _, _ = await manager.execute_json(
                    user_id="context_ops_next_rephrase",
                    system=NEXT_REPHRASE_PROMPT_SYSTEM,
                    prompt=prompt,
                    schema=NEXT_REPHRASE_JSON_SCHEMA,
                    model=cfg.rephrase_model,
                )
            except Exception:
                return RephraseNextOutput(
                    question=value.question,
                    needed=True,
                    reason="fallback_on_error",
                )
            candidate = str(resp.get("rephrased_question", "")).strip()
            needed = bool(resp.get("needed", True)) and bool(candidate)
            if "not enough data to answer this" in candidate.lower():
                needed = False
            return RephraseNextOutput(
                question=candidate if needed else value.question,
                needed=needed,
                reason=str(resp.get("reason", "")),
            )

        return [validate_contexts, rephrase_for_next]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="context_chaining_guide",
                name="Context Validation & Multi-Hop Chaining",
                description="How to prune noisy retrieval chunks, trigger fallbacks, and chain retrieval steps.",
                instructions=(
                    "Use `{namespace_prefix}validate_contexts` after broad retrieval to "
                    "keep only cited chunks and detect `missing_query` for fallback sources. "
                    "When chaining a second source that depends on facts discovered in the "
                    "first source, call `{namespace_prefix}rephrase_for_next` with the prior summaries."
                ),
                recommended_tools=("validate_contexts", "rephrase_for_next"),
                code_examples=(
                    "checked = {namespace_prefix}validate_contexts(question=question, contexts=raw_contexts)\n"
                    'if checked["missing_query"]:\n'
                    '    extra = fallback__search(question=checked["missing_query"])',
                ),
            )
        ]


@toolkit(
    id="static",
    title="Static Context",
    description="Provide static text or structured reference data as retrieval context.",
    config_schema=StaticContextToolkitConfig,
)
class StaticContextToolkit(BaseToolkit[StaticContextToolkitConfig]):
    toolkit_id = "static"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="get_static_context",
            description="Return the configured static text and structured context.",
            context_type=HarnessContextType.RETRIEVAL,
            context_factory=retrieval_context_factory,
        )
        async def get_static_context(
            _context: ToolCallContext, _value: EmptyInput
        ) -> RetrievalOutput:
            ctx = Context(
                agent_id=cfg.id,
                original_question_uuid=None,
                actual_question_uuid=None,
                question="",
                source="static",
                agent="static",
                title=cfg.title or "Static Context",
                chunks=(
                    [
                        Chunk(
                            chunk_id=uuid4().hex,
                            text=cfg.context,
                            origin_agent="static",
                        )
                    ]
                    if cfg.context
                    else []
                ),
                structured=[cfg.structured] if cfg.structured else [],
            )
            return RetrievalOutput(contexts=[ctx.model_dump(mode="json")])

        return [get_static_context]


__all__ = [
    "ContextOpsToolkit",
    "ContextOpsToolkitConfig",
    "EmptyInput",
    "RephraseNextInput",
    "RephraseNextOutput",
    "StaticContextToolkit",
    "StaticContextToolkitConfig",
    "ValidateContextsInput",
    "ValidateContextsOutput",
    "generate_block_id",
]
