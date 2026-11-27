from __future__ import annotations

from hyperforge.configure import skill
from hyperforge.harness_sdk import HarnessSkill

PLAN_AND_EXECUTE_SKILL = HarnessSkill(
    id="plan_and_execute",
    name="Plan and Execute",
    description=(
        "Structured multi-step planning and iterative verification for complex "
        "questions spanning multiple data sources or tools."
    ),
    instructions=(
        "Follow an explicit Plan-and-Execute loop for complex or multi-part requests:\n"
        "1. Decompose the user's goal into minimal, atomic information needs.\n"
        "2. Prefer running independent read-only lookups concurrently (in one tool batch "
        "or via `parallel([call(...), ...])` when Code Mode is enabled).\n"
        "3. After each tool batch, inspect returned evidence for gaps or contradictions "
        "before issuing follow-up calls. Never repeat an identical tool call that "
        "already returned an empty or failed result.\n"
        "4. Stop calling tools as soon as the gathered evidence suffices, and synthesize "
        "a concise, well-grounded answer."
    ),
    preload=False,
)

CITATION_GROUNDED_ANSWER_SKILL = HarnessSkill(
    id="citation_grounded_answer",
    name="Citation-Grounded Answering",
    description=(
        "Strict source-grounding and two-level markdown footnote citation formatting "
        "([1] -> [1]: block-AA) for retrieved context blocks."
    ),
    instructions=(
        "Base every factual claim strictly on retrieved tool contexts and cite sources "
        "with two-level markdown footnotes when block IDs (such as `block-AA` or "
        "`block-AA-0`) are present:\n"
        "1. In the answer body, cite using bracketed Arabic numerals in order of first "
        "appearance: `[1]`, `[2]`, etc. Separate multiple citations with a space: `[1] [2]`.\n"
        "2. Reuse the same number whenever citing the same block again.\n"
        "3. End the response after a blank line with one mapping per line (`[1]: block-AA`), "
        "with no heading and no unreferenced blocks.\n"
        "4. State clearly if the retrieved evidence is insufficient rather than guessing."
    ),
    preload=False,
)


@skill(id="plan_and_execute")
def get_plan_and_execute_skill() -> HarnessSkill:
    return PLAN_AND_EXECUTE_SKILL


@skill(id="citation_grounded_answer")
def get_citation_grounded_answer_skill() -> HarnessSkill:
    return CITATION_GROUNDED_ANSWER_SKILL


__all__ = [
    "CITATION_GROUNDED_ANSWER_SKILL",
    "PLAN_AND_EXECUTE_SKILL",
    "get_citation_grounded_answer_skill",
    "get_plan_and_execute_skill",
]
