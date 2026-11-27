from __future__ import annotations

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

from .common import resolve_manager

_CONDITION_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Assess whether the given text fulfills the specified condition.
# CONDITION
{{ condition_prompt }}

# TEXT
{{ text }}
{% if similarity -%}
# SIMILAR EXAMPLES
{% for ex in similarity -%}
- {{ ex }}
{% endfor -%}
{% endif -%}
"""
)

_CONDITION_SCHEMA: dict[str, Any] = {
    "title": "yes_no",
    "description": "Evaluate whether condition holds",
    "parameters": {
        "type": "object",
        "properties": {
            "yes": {"type": "boolean"},
            "reason": {"type": "string"},
        },
        "required": ["yes"],
    },
}

_CHOOSE_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Select the single best option matching the user's question.
# QUESTION
{{ question }}

# OPTIONS
{% for key, desc in options.items() -%}
- {{ key }}: {{ desc }}
{% endfor -%}
{% if extra_info -%}
# EXTRA INSTRUCTIONS
{{ extra_info }}
{% endif -%}
"""
)

_EXTRACT_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Extract structured fields matching the schema from the request below.
{% if labels -%}
Allowed labels: {{ labels | join(', ') }}
{% endif -%}
# REQUEST
{{ request }}
"""
)

_PLANNER_PROMPT = PROMPT_ENVIRONMENT.from_string(
    """Draft a concise step-by-step retrieval plan to answer the question, or mark status='done' if history already suffices.
# QUESTION
{{ question }}

# AVAILABLE TOOLS
{{ tools_description }}

{% if history_summary -%}
# PROGRESS SO FAR
{{ history_summary }}
{% endif -%}
"""
)

_PLANNER_SCHEMA: dict[str, Any] = {
    "title": "plan",
    "description": "Structured retrieval plan",
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["continue", "done"]},
        "reasoning": {"type": "string"},
        "summary": {"type": "string"},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["description"],
            },
        },
    },
    "required": ["status", "steps"],
}


class ConditionInput(BaseModel):
    text: str = Field(
        description="Text to evaluate (question, context summary, or answer)."
    )
    condition_prompt: str | None = Field(
        default=None, description="Condition prompt; falls back to toolkit default."
    )
    similarity: list[str] = Field(default_factory=list)


class ConditionOutput(BaseModel):
    matched: bool
    reason: str = ""


class ChooseInput(BaseModel):
    question: str
    options: dict[str, str] = Field(description="Mapping of option_key -> description.")
    extra_info: str | None = None


class ChooseOutput(BaseModel):
    selected: str
    reason: str = ""


class ExtractInput(BaseModel):
    request: str
    schema_def: dict[str, Any] = Field(
        alias="schema", description="JSON Schema describing the object to extract."
    )
    labels: list[str] = Field(default_factory=list)

    model_config = {"populate_by_name": True}


class ExtractOutput(BaseModel):
    extracted: dict[str, Any]


class TransformInput(BaseModel):
    question: str
    context: str


class TransformOutput(BaseModel):
    text: str


class PlanInput(BaseModel):
    question: str
    tools_description: str
    history_summary: str = ""


class PlanOutput(BaseModel):
    status: Literal["continue", "done"]
    reasoning: str = ""
    summary: str = ""
    steps: list[dict[str, str]] = Field(default_factory=list)


class DecisionToolkitConfig(ToolkitConfig):
    toolkit: Literal["decision"] = "decision"
    model: LLMField = Field(
        default=LLMConfig(model_id=llm_defaults.reasoning),
        title="Decision & routing model",
    )
    default_condition_prompt: str | None = None


@toolkit(
    id="decision",
    title="Decision, Routing, Extraction & Planning",
    description="Evaluate boolean conditions, classify intents, extract JSON, and draft plans.",
    config_schema=DecisionToolkitConfig,
)
class DecisionToolkit(BaseToolkit[DecisionToolkitConfig]):
    toolkit_id = "decision"

    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        cfg = self.config

        @tool(
            name="check_condition",
            description="Evaluate whether a condition holds on a question, context, or answer.",
        )
        async def check_condition(
            context: ToolCallContext, value: ConditionInput
        ) -> ConditionOutput:
            manager = resolve_manager(context)
            cond = value.condition_prompt or cfg.default_condition_prompt or ""
            prompt = _CONDITION_PROMPT.render(
                condition_prompt=cond,
                text=value.text,
                similarity=value.similarity,
            )
            resp, _, _ = await manager.execute_json(
                prompt=prompt,
                schema=_CONDITION_SCHEMA,
                user_id="decision_condition",
                model=cfg.model,
            )
            return ConditionOutput(
                matched=bool(resp.get("yes", False)),
                reason=str(resp.get("reason", "")),
            )

        @tool(
            name="choose_option",
            description="Select the best option key from a dictionary of named options.",
        )
        async def choose_option(
            context: ToolCallContext, value: ChooseInput
        ) -> ChooseOutput:
            if not value.options:
                raise ValueError("options must not be empty")
            manager = resolve_manager(context)
            schema = {
                "title": "choose_option",
                "description": "Select the best matching option",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected": {
                            "type": "string",
                            "enum": [*value.options.keys(), "else"],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["selected"],
                },
            }
            prompt = _CHOOSE_PROMPT.render(
                question=value.question,
                options=value.options,
                extra_info=value.extra_info,
            )
            resp, _, _ = await manager.execute_json(
                prompt=prompt,
                schema=schema,
                user_id="decision_choose",
                model=cfg.model,
            )
            return ChooseOutput(
                selected=str(resp.get("selected", "else")),
                reason=str(resp.get("reason", "")),
            )

        @tool(
            name="extract_structured",
            description="Extract a JSON object conforming to the supplied schema from text.",
        )
        async def extract_structured(
            context: ToolCallContext, value: ExtractInput
        ) -> ExtractOutput:
            manager = resolve_manager(context)
            prompt = _EXTRACT_PROMPT.render(request=value.request, labels=value.labels)
            extracted, _, _ = await manager.execute_json(
                prompt=prompt,
                schema=value.schema_def,
                user_id="decision_extract",
                model=cfg.model,
            )
            return ExtractOutput(extracted=dict(extracted))

        @tool(
            name="transform_text",
            description="Rewrite or refine a question given intermediate context.",
        )
        async def transform_text(
            context: ToolCallContext, value: TransformInput
        ) -> TransformOutput:
            manager = resolve_manager(context)
            prompt = (
                f"Given the context below, rewrite or refine the question clearly.\n\n"
                f"Context:\n{value.context}\n\nQuestion:\n{value.question}"
            )
            text, _, _, _ = await manager.execute(
                prompt=prompt,
                user_id="decision_transform",
                model=cfg.model,
            )
            return TransformOutput(text=text)

        @tool(
            name="generate_plan",
            description="Draft a structured retrieval plan or signal completion.",
        )
        async def generate_plan(
            context: ToolCallContext, value: PlanInput
        ) -> PlanOutput:
            manager = resolve_manager(context)
            prompt = _PLANNER_PROMPT.render(
                question=value.question,
                tools_description=value.tools_description,
                history_summary=value.history_summary,
            )
            resp, _, _ = await manager.execute_json(
                prompt=prompt,
                schema=_PLANNER_SCHEMA,
                user_id="decision_plan",
                model=cfg.model,
            )
            raw_status = resp.get("status", "done")
            status: Literal["continue", "done"] = (
                "continue" if raw_status == "continue" else "done"
            )
            return PlanOutput(
                status=status,
                reasoning=str(resp.get("reasoning", "")),
                summary=str(resp.get("summary", "")),
                steps=[
                    {
                        "description": str(s.get("description", "")),
                        "reason": str(s.get("reason", "")),
                    }
                    for s in resp.get("steps", [])
                    if isinstance(s, dict)
                ],
            )

        return [
            check_condition,
            choose_option,
            extract_structured,
            transform_text,
            generate_plan,
        ]

    def build_skills(self) -> Sequence[HarnessSkill]:
        return [
            HarnessSkill(
                id="decision_routing_guide",
                name="Conditional Routing & Structured Extraction",
                description="Branch workflows with boolean conditions, intent selection, and schema extraction.",
                instructions=(
                    "Use `{namespace_prefix}check_condition` or `{namespace_prefix}choose_option` "
                    "inside Code Mode `if`/`elif` blocks to route queries across connectors, "
                    "and `{namespace_prefix}extract_structured` to turn unstructured text into typed parameters."
                ),
                recommended_tools=(
                    "check_condition",
                    "choose_option",
                    "extract_structured",
                ),
                code_examples=(
                    'cond = {namespace_prefix}check_condition(text=question, condition_prompt="Is this a billing query?")\n'
                    'if cond["matched"]:\n'
                    "    res = billing__ask(question=question)",
                ),
            )
        ]


__all__ = [
    "ChooseInput",
    "ChooseOutput",
    "ConditionInput",
    "ConditionOutput",
    "DecisionToolkit",
    "DecisionToolkitConfig",
    "ExtractInput",
    "ExtractOutput",
    "PlanInput",
    "PlanOutput",
    "TransformInput",
    "TransformOutput",
]
