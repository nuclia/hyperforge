from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from .tools import HarnessTool


class HarnessSkill(BaseModel):
    """Reusable domain instructions, tool hints, and Code Mode examples for an agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    instructions: str = Field(min_length=1)
    recommended_tools: tuple[str, ...] = ()
    code_examples: tuple[str, ...] = ()
    preload: bool = False

    @field_validator("id", "name", "description", "instructions")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Skill fields must not be blank")
        return value

    @field_validator("recommended_tools")
    @classmethod
    def validate_recommended_tools(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        cleaned: list[str] = []
        for item in value:
            if not item.strip():
                raise ValueError("recommended_tools entries must not be blank")
            if item not in cleaned:
                cleaned.append(item)
        return tuple(cleaned)

    @field_validator("code_examples")
    @classmethod
    def validate_code_examples(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item.strip():
                raise ValueError("code_examples entries must not be blank")
        return value


def _interpolate_namespace(text: str, namespace: str | None) -> str:
    prefix = f"{namespace}__" if namespace else ""
    ns = namespace or ""
    return text.replace("{namespace_prefix}", prefix).replace("{namespace}", ns)


def namespace_skill(skill: HarnessSkill, namespace: str | None) -> HarnessSkill:
    """Scope a toolkit-provided skill to a toolkit namespace safely.

    Uses literal placeholder replacement for ``{namespace}`` and
    ``{namespace_prefix}`` rather than ``str.format`` so JSON schemas and Python
    dictionary literals in skill instructions remain untouched.
    """
    if not namespace:
        return skill.model_copy(
            update={
                "instructions": _interpolate_namespace(skill.instructions, None),
                "code_examples": tuple(
                    _interpolate_namespace(example, None)
                    for example in skill.code_examples
                ),
            }
        )

    prefix = f"{namespace}__"
    scoped_id = skill.id if skill.id.startswith(prefix) else f"{prefix}{skill.id}"
    scoped_tools = tuple(
        tool_name if "__" in tool_name else f"{prefix}{tool_name}"
        for tool_name in skill.recommended_tools
    )
    return skill.model_copy(
        update={
            "id": scoped_id,
            "instructions": _interpolate_namespace(skill.instructions, namespace),
            "recommended_tools": scoped_tools,
            "code_examples": tuple(
                _interpolate_namespace(example, namespace)
                for example in skill.code_examples
            ),
        }
    )


def format_skill_instructions(
    skill: HarnessSkill, *, include_code_examples: bool = False
) -> str:
    """Format a single skill's full body, optionally appending Code Mode examples."""
    parts = [f"### Skill: {skill.name} (`{skill.id}`)\n{skill.instructions.strip()}"]
    if skill.recommended_tools:
        tools_list = ", ".join(f"`{name}`" for name in skill.recommended_tools)
        parts.append(f"Recommended tools: {tools_list}")
    if include_code_examples and skill.code_examples:
        examples_rendered = "\n\n".join(
            f"```python\n{example.strip()}\n```" for example in skill.code_examples
        )
        parts.append(f"Code Mode examples:\n{examples_rendered}")
    return "\n\n".join(parts)


def render_skills_system_prompt(
    base_prompt: str,
    skills: Sequence[HarnessSkill],
    *,
    include_code_examples: bool = False,
) -> str:
    """Append preloaded skill bodies and the on-demand skill catalog to a system prompt."""
    if not skills:
        return base_prompt

    sections = [base_prompt]
    preloaded = [skill for skill in skills if skill.preload]
    catalog = [skill for skill in skills if not skill.preload]

    if preloaded:
        rendered_preloaded = "\n\n".join(
            format_skill_instructions(
                skill, include_code_examples=include_code_examples
            )
            for skill in preloaded
        )
        sections.append(f"## Active Skills\n\n{rendered_preloaded}")

    if catalog:
        entries = "\n".join(
            f"  <skill>\n"
            f"    <id>{skill.id}</id>\n"
            f"    <name>{skill.name}</name>\n"
            f"    <description>{skill.description}</description>\n"
            f"  </skill>"
            for skill in catalog
        )
        sections.append(
            "Skills provide specialized instructions and workflows for specific tasks "
            "and connectors.\n"
            "Call `load_skill` when a task matches an available skill's description.\n"
            f"<available_skills>\n{entries}\n</available_skills>"
        )

    return "\n\n".join(sections)


def format_loaded_skill_payload(
    skill: HarnessSkill,
    *,
    include_code_examples: bool = False,
    activated_tools: Sequence[HarnessTool[Any, Any]] = (),
) -> dict[str, Any]:
    """Build the structured result returned by the `load_skill` core tool."""
    payload: dict[str, Any] = {
        "id": skill.id,
        "name": skill.name,
        "description": skill.description,
        "instructions": skill.instructions,
    }
    if skill.recommended_tools:
        payload["recommended_tools"] = list(skill.recommended_tools)
    if include_code_examples and skill.code_examples:
        payload["code_examples"] = list(skill.code_examples)
    if activated_tools:
        payload["activated_tools"] = [
            {
                "name": item.name,
                "description": item.description,
                "parameters": item.parameters,
            }
            for item in activated_tools
        ]
    return payload


__all__ = [
    "HarnessSkill",
    "format_loaded_skill_payload",
    "format_skill_instructions",
    "namespace_skill",
    "render_skills_system_prompt",
]
