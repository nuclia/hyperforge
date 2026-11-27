from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Literal

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
from hyperforge.models import Chunk, Context

from .common import resolve_manager

_MAX_CONTEXTS_REMI = 60

_RELATED_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Provide follow-up questions related to the original question that are not yet answered by the context.
Return an empty list if none are needed.
{% if prompt -%}
{{ prompt }}
{% endif -%}
[START OF CONTEXT]
{{ context_text }}
[END OF CONTEXT]
MAIN QUESTION: {{ question }}
"""
)

_RELATED_SCHEMA: dict[str, Any] = {
    "title": "related_questions",
    "description": "Related follow-up questions",
    "parameters": {
        "type": "object",
        "properties": {
            "related": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["related"],
    },
}

_MISSING_INFO_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Review the main question and retrieved context to identify any missing information needed to answer completely.
- If the context is sufficient, set `answered=true` and `questions=[]`.
- Otherwise set `answered=false` and return only concise, actionable sub-questions for the missing facts.

# Main Question
{{ question }}

# Context
{{ context_text }}
"""
)

_MISSING_INFO_SCHEMA: dict[str, Any] = {
    "title": "missing_questions",
    "description": "Gap analysis for iterative retrieval",
    "type": "object",
    "properties": {
        "answered": {"type": "boolean"},
        "questions": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["answered", "questions"],
}


class RemiEvalInput(BaseModel):
    question: str
    answer: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    is_answered: bool = True


class RemiEvalOutput(BaseModel):
    answer_relevance: float | None = None
    groundedness: float | None = None
    summary: str = ""
    error: str | None = None
    contexts: list[dict[str, Any]] = Field(default_factory=list)


class RelatedInput(BaseModel):
    question: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)
    prompt: str | None = None


class RelatedOutput(BaseModel):
    questions: list[str] = Field(default_factory=list)


class MissingInfoInput(BaseModel):
    question: str
    contexts: list[dict[str, Any]] = Field(default_factory=list)


class MissingInfoOutput(BaseModel):
    answered: bool
    questions: list[str] = Field(default_factory=list)
    reason: str = ""


class EvaluationToolkitConfig(ToolkitConfig):
    toolkit: Literal["evaluation"] = "evaluation"
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Evaluation & gap-analysis model",
    )
    context_granularity: Literal["full", "partial_answers"] = "full"
    related_prompt: str | None = None


def _contexts_to_strings(
    raw_contexts: list[dict[str, Any]], *, granularity: str
) -> list[str]:
    parsed = [
        Context.model_validate(
            {"original_question_uuid": None, "actual_question_uuid": None, **raw}
        )
        for raw in raw_contexts
    ]
    pieces: list[str] = []
    if granularity == "partial_answers":
        for ctx in parsed:
            pieces.append(
                ctx.answer_summary_markdown()
                if ctx.summary.strip()
                else ctx.context_markdown()
            )
    else:
        for ctx in parsed:
            for chunk in ctx.chunks:
                pieces.append(f"## {chunk.title or ''}\n\n{chunk.text}")
    return pieces


@toolkit(
    id="evaluation",
    title="REMi Evaluation, Gap Analysis & Related Questions",
    description="Evaluate groundedness/relevance with REMi, detect missing evidence for retries, and suggest follow-ups.",
    config_schema=EvaluationToolkitConfig,
)
class EvaluationToolkit(BaseToolkit[EvaluationToolkitConfig]):
    toolkit_id = "evaluation"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="evaluate_remi",
            description="Evaluate answer relevance and context groundedness (0-5) using Nuclia REMi.",
        )
        async def evaluate_remi(
            context: ToolCallContext, value: RemiEvalInput
        ) -> RemiEvalOutput:
            manager = resolve_manager(context)
            pieces = _contexts_to_strings(
                value.contexts, granularity=cfg.context_granularity
            )
            error_msg: str | None = None
            if len(pieces) > _MAX_CONTEXTS_REMI:
                pieces = pieces[:_MAX_CONTEXTS_REMI]
                error_msg = f"Truncated contexts to {_MAX_CONTEXTS_REMI} for REMi."

            tasks = [
                manager.remi(
                    question=value.question, answer=value.answer, contexts=None
                )
            ]
            if value.is_answered and pieces:
                tasks.append(
                    manager.remi(question=None, answer=value.answer, contexts=pieces)
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)

            rel_score: int | float | None = None
            ground_score: int | float | None = None
            groundedness_list: list[int | None] | None = None
            rel_res = results[0]
            if isinstance(rel_res, BaseException) or rel_res.answer_relevance is None:
                err = "Error evaluating answer relevance."
                error_msg = f"{error_msg} {err}".strip() if error_msg else err
            else:
                rel_score = rel_res.answer_relevance.score

            if not value.is_answered:
                groundedness_list = [0] * max(len(pieces), 1)
                ground_score = 0
            elif len(results) > 1:
                g_res = results[1]
                if isinstance(g_res, BaseException) or g_res.groundedness is None:
                    err = "Error evaluating groundedness."
                    error_msg = f"{error_msg} {err}".strip() if error_msg else err
                else:
                    groundedness_list = list(g_res.groundedness)
                    scores = [s for s in groundedness_list if s is not None]
                    ground_score = max(scores) if scores else 0

            summary_parts = []
            if rel_score is not None:
                summary_parts.append(f"Answer relevance: {rel_score}/5.")
            if ground_score is not None:
                summary_parts.append(f"Answer groundedness: {ground_score}/5.")
            summary_str = " ".join(summary_parts)

            remi_contexts_out: list[dict[str, Any]] = []
            if groundedness_list is not None:
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
                remi_chunks: list[Chunk] = []
                if cfg.context_granularity != "partial_answers":
                    c_idx = 0
                    for p_ctx in parsed_ctxs:
                        agent_name = p_ctx.agent_id or p_ctx.agent
                        for chunk in p_ctx.chunks:
                            if c_idx >= min(_MAX_CONTEXTS_REMI, len(groundedness_list)):
                                break
                            sc = groundedness_list[c_idx]
                            g_sc = sc if sc is not None else 0
                            remi_chunks.append(
                                Chunk(
                                    chunk_id=chunk.chunk_id or f"remi_chunk_{c_idx}",
                                    title=f"Groundedness {g_sc}/5 - [{agent_name}] {chunk.title or 'Untitled'}",
                                    text=f"**Groundedness: {g_sc}/5**\n\n{chunk.text}",
                                    origin_agent="remi",
                                )
                            )
                            c_idx += 1
                else:
                    for i, p_ctx in enumerate(parsed_ctxs[:_MAX_CONTEXTS_REMI]):
                        if i < len(groundedness_list):
                            sc = groundedness_list[i]
                            g_sc = sc if sc is not None else 0
                            agent_name = p_ctx.agent_id or p_ctx.agent
                            eval_text = (
                                p_ctx.answer_summary_markdown()
                                if p_ctx.summary.strip()
                                else p_ctx.context_markdown()
                            )
                            remi_chunks.append(
                                Chunk(
                                    chunk_id=p_ctx.id or f"remi_{i}",
                                    title=f"Groundedness {g_sc}/5 - [{agent_name}] {p_ctx.title or 'Untitled'}",
                                    text=f"**Groundedness: {g_sc}/5**\n\n{eval_text}",
                                    origin_agent="remi",
                                )
                            )
                if remi_chunks:
                    remi_ctx = Context(
                        original_question_uuid=None,
                        actual_question_uuid=None,
                        agent="remi",
                        agent_id="remi",
                        title="REMi Evaluation Breakdown",
                        summary=summary_str
                        or f"Evaluated {len(remi_chunks)} contexts.",
                        question=value.question,
                        source="remi",
                        chunks=remi_chunks,
                    )
                    remi_contexts_out.append(remi_ctx.model_dump(mode="json"))

            return RemiEvalOutput(
                answer_relevance=(float(rel_score) if rel_score is not None else None),
                groundedness=(
                    float(ground_score) if ground_score is not None else None
                ),
                summary=summary_str,
                error=error_msg,
                contexts=remi_contexts_out,
            )

        @tool(
            name="suggest_related_questions",
            description="Generate unanswered related follow-up questions from context.",
        )
        async def suggest_related_questions(
            context: ToolCallContext, value: RelatedInput
        ) -> RelatedOutput:
            manager = resolve_manager(context)
            pieces = _contexts_to_strings(value.contexts, granularity="partial_answers")
            prompt = _RELATED_PROMPT.render(
                question=value.question,
                context_text="\n\n".join(pieces),
                prompt=value.prompt or cfg.related_prompt,
            )
            resp, _, _ = await manager.execute_json(
                prompt=prompt,
                user_id="evaluation_related",
                model=cfg.model,
                schema=_RELATED_SCHEMA,
            )
            questions = [str(q) for q in (resp.get("related") or []) if str(q).strip()]
            return RelatedOutput(questions=questions)

        @tool(
            name="detect_missing_info",
            description="Check if retrieved contexts answer the question or return sub-questions for missing facts.",
        )
        async def detect_missing_info(
            context: ToolCallContext, value: MissingInfoInput
        ) -> MissingInfoOutput:
            manager = resolve_manager(context)
            pieces = _contexts_to_strings(value.contexts, granularity="partial_answers")
            prompt = _MISSING_INFO_PROMPT.render(
                question=value.question,
                context_text="\n\n".join(pieces),
            )
            resp, _, _ = await manager.execute_json(
                prompt=prompt,
                user_id="evaluation_missing",
                model=cfg.model,
                schema=_MISSING_INFO_SCHEMA,
            )
            questions = [
                str(q) for q in (resp.get("questions") or []) if str(q).strip()
            ]
            answered = bool(resp.get("answered", False)) or not questions
            return MissingInfoOutput(
                answered=answered,
                questions=[] if answered else questions,
                reason=str(resp.get("reason", "")),
            )

        return [evaluate_remi, suggest_related_questions, detect_missing_info]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="evaluation_and_retry_guide",
                name="Groundedness Evaluation & Gap Recovery",
                description="Verify answer groundedness with REMi and iterate when evidence is incomplete.",
                instructions=(
                    "Call `{namespace_prefix}detect_missing_info` to check whether retrieved "
                    "contexts fully answer a multi-part question before finalizing, and "
                    "`{namespace_prefix}evaluate_remi` to audit groundedness and relevance."
                ),
                recommended_tools=(
                    "detect_missing_info",
                    "evaluate_remi",
                    "suggest_related_questions",
                ),
            )
        ]


__all__ = [
    "EvaluationToolkit",
    "EvaluationToolkitConfig",
    "MissingInfoInput",
    "MissingInfoOutput",
    "RelatedInput",
    "RelatedOutput",
    "RemiEvalInput",
    "RemiEvalOutput",
]
