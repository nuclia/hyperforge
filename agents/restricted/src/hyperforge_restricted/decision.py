from hyperforge import PROMPT_ENVIRONMENT

EXTRACT_TEMPLATE = """
Given a schema and a request of information, extract the relevant information from the context.

# Request:
{{ request }}

{% if labels %}
# Available labels:
{% for labelset, labels_values in labels.items() %}
{% for label in labels_values %}
- {{labelset}}/{{label}}
{% endfor %}
{% endfor -%}
{% endif %}

"""

EXTRACT_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(EXTRACT_TEMPLATE)


CHOOSE_AGENT = """
Given a series of cases with a descriptions and a question, you must choose the most appropriate case to answer the question.
You can only select one of the provided cases.

{% if extra_info %}
# Additional information:
{{ extra_info }}
{% endif -%}

# Question:
{{ question }}

# Cases:
{% for case_id, case_description in options.items() %}
- ID: {{case_id}}
  Description: {{case_description}}
{% endfor -%}
if none of the cases is appropriate, select "else".

# Output definition:

Your output should always follow the provided json schema and only contain the fields defined there.
- "selected": The ID of the selected option.
- "reason": The reason for selecting this option.
"""

CHOOSE_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(CHOOSE_AGENT)


CHOOSE_SCHEMA = {
    "title": "Case selection",
    "description": "Choose the id of the most appropriate option and explain the reason for selecting it.",
    "parameters": {
        "type": "object",
        "properties": {
            "selected": {
                "type": "string",
                "enum": ["else"],
                "description": "The ID of the selected option",
            },
            "reason": {
                "type": "string",
                "description": "The reason for selecting this option",
            },
        },
        "required": ["selected", "reason"],
        "additionalProperties": False,
    },
}


TRANSFORM_REPHRASE = """
Given a question and a context, rephrase the question to make it more specific to the context

# Question:
{question}
# Context:
{context}

"""
