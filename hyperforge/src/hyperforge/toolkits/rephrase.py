from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

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

from .common import resolve_manager, resolve_workflow_history, resolve_workflow_rules

_SINGLE_REPHRASE_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """You are an expert at rephrasing complex questions for a retrieval system.
Review the main question and any provided context, then rephrase the question to maximize clarity and focus.
- Only use information present in the provided context; never invent facts.
- Keep key names, dates, identifiers, and domain terms intact.
{% if rules -%}
# Rules
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}
{% if context -%}
# Context
{% for info in context -%}
- {{ info }}
{% endfor -%}
{% endif -%}
# Main Question
{{ question }}
"""
)

_SPLIT_REPHRASE_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """You are an expert at decomposing multi-part questions into atomic retrieval sub-questions.
- Split when the question asks about distinct entities or independent sub-questions.
- Keep as a single question when already atomic and focused.
{% if rules -%}
# Rules
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}
{% if context -%}
# Context
{% for info in context -%}
- {{ info }}
{% endfor -%}
{% endif -%}
# Main Question
{{ question }}
"""
)

_SINGLE_SCHEMA: dict[str, Any] = {
    "title": "rephrase",
    "description": "Rephrase question for retrieval",
    "type": "object",
    "properties": {
        "rephrased_question": {"type": "string"},
        "rules": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["rephrased_question"],
}

_SPLIT_SCHEMA: dict[str, Any] = {
    "title": "questions",
    "description": "Split complex question into sub-questions",
    "type": "object",
    "properties": {
        "questions": {"type": "array", "items": {"type": "string"}},
        "rules": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
    },
    "required": ["questions"],
}


class RephraseInput(BaseModel):
    question: str = Field(description="Question to clarify or decompose for retrieval.")
    extra_context: list[str] = Field(
        default_factory=list,
        description="Optional conversation or domain context strings.",
    )
    extra_rule: str | None = Field(
        default=None,
        description="Optional additional constraint for rephrasing.",
    )


class RephraseOutput(BaseModel):
    questions: list[str]
    rules: list[str] = Field(default_factory=list)
    reason: str = ""


class RephraseToolkitConfig(ToolkitConfig):
    toolkit: Literal["rephrase"] = "rephrase"
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
        description="Model used to rephrase or split queries.",
    )
    rules: list[str] = Field(default_factory=list)
    provided_synonyms: dict[str, list[str]] = Field(default_factory=dict)
    history: bool = True


@toolkit(
    id="rephrase",
    title="Query Rephrase & Decomposition",
    description="Rephrase ambiguous questions or split multi-part queries into atomic sub-questions.",
    config_schema=RephraseToolkitConfig,
)
class RephraseToolkit(BaseToolkit[RephraseToolkitConfig]):
    toolkit_id = "rephrase"

    def _expand_synonyms(self, question: str) -> str:
        expanded = question
        for term, synonyms in self.config.provided_synonyms.items():
            if term.lower() in expanded.lower() and synonyms:
                expanded = f"{expanded} {' '.join(synonyms)}"
        return expanded

    def _effective_rules(self, extra_rule: str | None) -> list[str]:
        rules = list(self.config.rules)
        if extra_rule:
            rules.append(extra_rule)
        return rules

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="rephrase_query",
            description="Clarify and focus a question using optional context and synonym rules.",
        )
        async def rephrase_query(
            context: ToolCallContext, value: RephraseInput
        ) -> RephraseOutput:
            manager = resolve_manager(context)
            expanded_q = self._expand_synonyms(value.question)
            ctx_items = list(value.extra_context)
            if cfg.history and not ctx_items:
                ctx_items.extend(resolve_workflow_history(context))
            merged_rules = [
                *resolve_workflow_rules(context),
                *self._effective_rules(value.extra_rule),
            ]
            await context.harness.emit(
                HarnessEventType.WORKFLOW_STEP,
                {
                    "module": "rephrase",
                    "title": "Rephrase: Context prepared",
                    "agent_path": "/preprocess/rephrase",
                    "value": "\n".join(ctx_items),
                },
            )
            prompt = _SINGLE_REPHRASE_PROMPT.render(
                question=expanded_q,
                context=ctx_items,
                rules=merged_rules,
            )
            resp, in_tok, out_tok = await manager.execute_json(
                model=cfg.model,
                user_id="rephrase",
                prompt=prompt,
                schema=_SINGLE_SCHEMA,
            )
            rephrased = str(resp.get("rephrased_question") or value.question)
            reason = str(resp.get("reason", ""))
            await context.harness.emit(
                HarnessEventType.WORKFLOW_STEP,
                {
                    "module": "rephrase",
                    "title": "Rephrase: Question rephrased",
                    "agent_path": "/preprocess/rephrase",
                    "value": rephrased,
                    "reason": reason,
                    "input_nuclia_tokens": in_tok,
                    "output_nuclia_tokens": out_tok,
                },
            )
            return RephraseOutput(
                questions=[rephrased],
                rules=[str(r) for r in resp.get("rules", [])],
                reason=reason,
            )

        @tool(
            name="split_query",
            description="Decompose a multi-part question into independent atomic sub-questions.",
        )
        async def split_query(
            context: ToolCallContext, value: RephraseInput
        ) -> RephraseOutput:
            manager = resolve_manager(context)
            expanded_q = self._expand_synonyms(value.question)
            ctx_items = list(value.extra_context)
            if cfg.history and not ctx_items:
                ctx_items.extend(resolve_workflow_history(context))
            merged_rules = [
                *resolve_workflow_rules(context),
                *self._effective_rules(value.extra_rule),
            ]
            await context.harness.emit(
                HarnessEventType.WORKFLOW_STEP,
                {
                    "module": "rephrase",
                    "title": "Rephrase: Context prepared",
                    "agent_path": "/preprocess/rephrase",
                    "value": "\n".join(ctx_items),
                },
            )
            prompt = _SPLIT_REPHRASE_PROMPT.render(
                question=expanded_q,
                context=ctx_items,
                rules=merged_rules,
            )
            resp, in_tok, out_tok = await manager.execute_json(
                model=cfg.model,
                user_id="rephrase",
                prompt=prompt,
                schema=_SPLIT_SCHEMA,
            )
            questions = [str(q) for q in resp.get("questions", []) if str(q).strip()]
            final_qs = questions or [value.question]
            reason = str(resp.get("reason", ""))
            await context.harness.emit(
                HarnessEventType.WORKFLOW_STEP,
                {
                    "module": "rephrase",
                    "title": "Rephrase: Question split",
                    "agent_path": "/preprocess/rephrase",
                    "value": "\n".join(final_qs),
                    "reason": reason,
                    "input_nuclia_tokens": in_tok,
                    "output_nuclia_tokens": out_tok,
                },
            )
            return RephraseOutput(
                questions=final_qs,
                rules=[str(r) for r in resp.get("rules", [])],
                reason=reason,
            )

        return [rephrase_query, split_query]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="query_decomposition_guide",
                name="Query Rephrasing & Decomposition",
                description="When and how to clarify or split complex user questions before retrieval.",
                instructions=(
                    "Use `{namespace_prefix}rephrase_query` when follow-up questions rely on "
                    "conversational pronouns or domain synonyms. Use `{namespace_prefix}split_query` "
                    "when a question compares multiple entities or combines independent sub-questions, "
                    "then retrieve evidence for each returned sub-question concurrently."
                ),
                recommended_tools=("rephrase_query", "split_query"),
                code_examples=(
                    "prep = {namespace_prefix}split_query(question=question)\n"
                    'queries = prep["questions"]',
                ),
            )
        ]


__all__ = [
    "RephraseInput",
    "RephraseOutput",
    "RephraseToolkit",
    "RephraseToolkitConfig",
]
