from pydantic import BaseModel

from hyperforge import PROMPT_ENVIRONMENT


class PromptArgument(BaseModel):
    """An argument for a prompt template."""

    name: str
    """The name of the argument."""
    description: str | None = None
    """A human-readable description of the argument."""
    required: bool | None = None
    """Whether this argument must be provided."""


class PromptConfig(BaseModel):
    name: str
    description: str
    prompt: str
    arguments: list[PromptArgument] | None = None
    icons: dict[str, str] | None = None
    meta: dict[str, str] | None = None
    prompt_id: str | None = None


VALIDATE_OR_ANSWER_PROMPT = """
Based on the provided context and user question, perform the following tasks:

1. Select only information directly relevant to the question.
2. Break down compound sentences into simple, single-idea statements. Preserve original phrasing when possible.
3. For any named entity with descriptive details, separate those details into distinct propositions.
4. Ensure clarity by replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full names of the entities they reference, and add necessary modifiers to clarify meaning.
5. The context may be delimited by tags such as <START OF CONTEXT> and <END OF CONTEXT>. Treat everything between these tags as context.
6. Assess whether the context sufficiently answers the question. If it answers it partially, provide the answer; if it does not answer it fully, specify what information is missing to answer the question.
7. If the context does not answer the question at all, just return the original question as the missing information.
8. The `citations` field consists ONLY in a list of block IDs that are relevant to the answer, following these rules:
  - Use the format: block-AB
  - Just mention the block IDs, do NOT include any other text.
  - Just mention the blocks actually relevant and that contain information used in the answer, do NOT include blocks that are not relevant.
  - No duplicates.
9. Do NOT hallucinate block IDs. Only use those provided in the context.
10. Your output must be a JSON object with the following fields:
    - "reason": Explain your reasoning for the answer or validation.
    - "answer": Provide a partial or complete answer to the user query strictly from the information in the context. If there isn't enough information to even provide a partial answer, leave 'answer' empty.
    - "missing_info_query": If the context is insufficient, specify what information is missing in a query shape; otherwise, leave it empty. Just return the query needed to retrieve the missing information.
    - "useful": Indicate if the context is useful to answer the question ("yes" or "no").
    - "citations": List the IDs of the blocks relevant to the answer, if any (e.g., ["block-AB", "block-CD"]).
11. **IMPORTANT** If any extra instructions are provided, you MUST follow them carefully when generating the answer field. These instructions may specify the format, style, tools to use, or other requirements for the answer.

{% if extra_prompts -%}
<ADDITIONAL INSTRUCTIONS>
These are extra instructions about how to generate the answer. They may include information about tools, style, or specific requirements.
**IMPORTANT** You MUST use these instructions when generating the answer, and follow any specific requirements they contain.
{% for prompt in extra_prompts -%}
- {{ prompt }}
{% endfor %}
<END OF ADDITIONAL INSTRUCTIONS>
{% endif -%}

<QUESTION>
{{question}}

<START OF CONTEXT>
{% if contexts %}
Context:
{% for ident, chunk in contexts.items() %}
**{{ident}}**\n\n{{chunk}}\n\n---"

{% endfor %}
{% endif %}

<END OF CONTEXT>


"""

VALIDATE_OR_ANSWER_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    VALIDATE_OR_ANSWER_PROMPT
)


VALIDATE_JSON_SCHEMA = {
    "title": "validate_or_answer",
    "description": "Validate or answer",
    "parameters": {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Reasoning for the answer or validation",
            },
            "answer": {
                "type": "string",
                "description": "Partial or complete answer to the user query from the information in the context.",
            },
            "missing_info_query": {
                "type": "string",
                "description": "Query needed to retrieve the missing information in case the context is not enough to answer the question. If the context does not answer the question at all, just return the original question.",
            },
            "useful": {
                "type": "string",
                "description": "Is the context useful to answer the question?",
                "enum": ["yes", "no"],
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "Block ID cited in the answer, e.g. block-AB",
                },
                "description": "List of block IDs cited in the answer, if any",
            },
        },
        "required": ["reason", "answer", "missing_info_query", "useful", "citations"],
    },
}


VALIDATE_MULTIPLE_CONTEXTS_PROMPT = """
Based on the provided contexts and user question, decide what each context can contribute to the answer.

For every relevant context:
- Return its exact `context_id`.
- Write an `answer` containing the partial or complete answer supported by that context.
- Return only the block IDs from that context that contain relevant supporting information in `citations`.

The existing summary is first-class evidence, just like the blocks:
- Always consider the existing summary when deciding whether a context is relevant.
- If the existing summary answers the question but none of the blocks do, include the context, generate its `answer` from the summary, and return an empty `citations` list.
- Do not report information as missing when it is present in an existing summary.
- Citations refer only to blocks. An empty citations list is valid when the answer is supported only by the existing summary.

Omit irrelevant contexts from the `contexts` list. Do not invent context IDs or block IDs.

`missing_info_query` is global across all contexts. Leave it empty when the contexts together contain enough information. Otherwise, return the query needed to retrieve the missing information.

<QUESTION>
{{ question }}

<CONTEXTS>
{% for context in contexts %}
<CONTEXT id="{{ context.context_id }}">
{% if context.title %}Title: {{ context.title }}{% endif %}
{% if context.summary %}
Existing summary:
{{ context.summary }}
{% endif %}
{% for block_id, block in context.blocks.items() %}
**{{ block_id }}**
{{ block }}
{% endfor %}
</CONTEXT>
{% endfor %}
</CONTEXTS>
"""

VALIDATE_MULTIPLE_CONTEXTS_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    VALIDATE_MULTIPLE_CONTEXTS_PROMPT
)

VALIDATE_MULTIPLE_CONTEXTS_JSON_SCHEMA = {
    "title": "validate_multiple_contexts",
    "description": "Validate several contexts and summarize each relevant one",
    "parameters": {
        "type": "object",
        "properties": {
            "missing_info_query": {
                "type": "string",
                "description": "Query needed to retrieve information missing across all contexts, or an empty string.",
            },
            "contexts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "context_id": {
                            "type": "string",
                            "description": "Exact ID of a relevant input context.",
                        },
                        "answer": {
                            "type": "string",
                            "description": "Partial or complete answer supported by this context.",
                        },
                        "citations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Relevant block IDs belonging to this context.",
                        },
                    },
                    "required": [
                        "context_id",
                        "answer",
                        "citations",
                    ],
                },
            },
        },
        "required": ["missing_info_query", "contexts"],
    },
}


NEXT_REPHRASE_JSON_SCHEMA = {
    "title": "next_rephrase",
    "description": "Rephrase the question if needed given info about previous contexts and the current agent",
    "parameters": {
        "type": "object",
        "properties": {
            "rephrased_question": {
                "type": "string",
                "description": "Rephrased question if needed. If the question does not need to be rephrased, return an empty string.",
            },
            "needed": {
                "type": "boolean",
                "description": "Is rephrasing needed?",
            },
            "reason": {
                "type": "string",
                "description": "Reasoning for rephrasing or not the question",
            },
        },
        "required": ["rephrased_question", "needed", "reason"],
    },
}

NEXT_REPHRASE_PROMPT_SYSTEM = """
You  decide if a question needs to be rephrased to be answered by the current agent, given the context provided by previous agents and the description of the current agent."""

NEXT_REPHRASE_PROMPT = """
You  decide if a question needs to be rephrased to be answered by the current agent, given the context provided by previous agents and the description of the current agent.

Guidelines:
1. If the question can be answered by the current agent without rephrasing, return an empty string as the rephrased question and "False" as needed.
2. If the question needs to be rephrased to be answered by the current agent, return the rephrased question and "True" as needed.
3. Provide a clear reasoning for your decision in the reason field.

Return only a JSON object with the following fields:
- "rephrased_question": The rephrased question if needed. If the question does not need to be rephrased, return an empty string.
- "needed": "True" if rephrasing is needed, "False" otherwise.
- "reason": Explain your reasoning for rephrasing or not the question.

Here you have the question, the contexts from previous agents and the description of the current agent:


<CURRENT AGENT DESCRIPTION>
{{info}}

{%if extra_info %}
<ADDITIONAL INSTRUCTIONS>
These are extra instructions about how to rephrase the question. They may include information about format, style, tools to use, or specific requirements.
**IMPORTANT** You MUST use these instructions when rephrasing the question, and follow any specific requirements they contain.
{{extra_info}}
<END OF ADDITIONAL INSTRUCTIONS>
{% endif %}


<PREVIOUS CONTEXTS>
{% if contexts %}
Context:
{% for context in contexts %}
- {{context}}
{% endfor %}
{% else %}
No previous context.
{% endif %}

<QUESTION>
{{question}}

"""
NEXT_REPHRASE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(NEXT_REPHRASE_PROMPT)
