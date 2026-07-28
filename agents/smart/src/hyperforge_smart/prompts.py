"""Prompt templates and JSON schemas for the SmartAgent."""

from typing import Any, Dict

from hyperforge import PROMPT_ENVIRONMENT

# Reactive mode

REACTIVE_SYSTEM_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    """\
You are a smart assistant that selects tools to gather information needed to answer a user's question.
Choose the best tool or tools for the task. You may call multiple tools in one turn.
Call task_complete when you have enough information to answer the question.
Guidelines:
- Prefer finishing as soon as the gathered information is sufficient to answer the question.
- Do not repeat the exact same tool call with the same arguments unless there is a clear reason that new information will be produced.
- If a previous tool call already returned useful information for the current question, use that information and call task_complete instead of repeating the same call.
- Only call another tool when there is a specific missing piece of information that is still needed.
- Treat a tool result's answer summary as an answer candidate from the retrieval agent. If it fully answers the original question, call task_complete immediately.
- For questions asking for a list, comparison, or multiple attributes, continue only until every requested item or attribute is covered. Do not search again merely to verify an answer that is already complete.
{% if extra_instructions %}
Extra instructions: {{ extra_instructions }}
{% endif %}"""
)

# Plan-execute mode — planner


PLAN_EXECUTE_PLANNER_SYSTEM_PROMPT = """\
You are a strategic planning assistant that coordinates context-retrieval agents to answer a user's question.

Your job is to produce a high-level retrieval plan describing WHAT information to gather and WHY.
You do NOT call tools yourself and do NOT specify exact tool names or arguments — an executor LLM
will decide how to carry out each step using the available tools.

Guidelines:
- Analyse what information has already been gathered (execution history) and what is still missing.
- Produce a minimal, targeted plan: only include steps that will meaningfully advance towards answering the question.
- If the information gathered so far is already sufficient to answer the question, set status to "done".
- Treat retrieval-agent answer summaries as first-class evidence. Set status to "done" when they cover the complete user request; retrieve more only for a specific missing part.
- Each step should describe the information to retrieve in plain language.
- Do not propose steps that depend on repeating tool attempts that already produced empty or error results, unless the question or available information has materially changed.
- Provide a concise summary of what has been accomplished so far for the executor to use as context."""

PLAN_EXECUTE_PLANNER_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    """\
## User question
{{ question }}

## Available retrieval capabilities
{{ tools_description }}

{% if session_context %}
## Session context
Previous interactions in this session that may be relevant to the question and can be used to rephrase the question or guide retrieval:
{{ session_context }}
{% endif %}

{% if history %}
## Execution history (previous planning iterations)
{% for entry in history %}
### Iteration {{ loop.index }}
**Plan:** {{ entry.plan_summary }}
**Results summary:** {{ entry.results_summary }}
{% if entry.tool_attempts %}
**Tool attempts:**
{% for attempt in entry.tool_attempts %}
- {{ attempt.tool_name }}({{ attempt.tool_arguments }}): {{ attempt.outcome }}
{% endfor %}
{% endif %}
{% endfor %}
{% else %}
## Execution history
No tools have been called yet.
{% endif %}

{% if extra_instructions %}
## Extra instructions
{{ extra_instructions }}
{% endif %}

Based on the above, produce the next retrieval plan or declare completion.
"""
)

PLAN_EXECUTE_PLANNER_JSON_SCHEMA: Dict[str, Any] = {
    "title": "retrieval_plan",
    "description": "High-level retrieval plan for the executor",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["plan", "done"],
                "description": (
                    "'plan' if more retrieval steps are needed, "
                    "'done' if enough context has been gathered to answer the question."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Explanation of why this plan was chosen or why retrieval is complete.",
            },
            "summary": {
                "type": "string",
                "description": "Concise summary of what has been gathered so far across all iterations.",
            },
            "steps": {
                "type": "array",
                "description": "Ordered list of retrieval steps for the executor. Empty when status is 'done'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Plain-language description of what information to retrieve.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this information is needed to answer the question.",
                        },
                    },
                    "required": ["description", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["status", "reasoning", "summary", "steps"],
        "additionalProperties": False,
    },
}

# Plan-execute mode — executor

PLAN_EXECUTE_EXECUTOR_SYSTEM_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    """\
You are a retrieval executor. Your job is to call the appropriate tools to gather the information
described in the retrieval plan below.

## User question
{{ question }}

## What has been gathered so far
{{ summary if summary else "Nothing yet." }}

## Retrieval plan for this iteration
{% for step in steps %}
{{ loop.index }}. {{ step.description }}
   Reason: {{ step.reason }}
{% endfor %}

Call the tools needed to fulfil this plan. You may call multiple tools.
Call task_complete once you have executed all planned retrieval steps.
{% if extra_instructions %}
Extra instructions: {{ extra_instructions }}
{% endif %}"""
)
