from time import time

from hyperforge.agent import Agent
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.trace import trace_agent

from agents.related.src.hyperforge_related.config import RelatedAgentConfig
from hyperforge import PROMPT_ENVIRONMENT

ASK_JSON_SCHEMA = {
    "title": "related_questions",
    "description": "Related questions to main question based on context information",
    "parameters": {
        "type": "object",
        "properties": {
            "related": {
                "type": "array",
                "items": {
                    "type": "string",
                    "description": "related question to the main question",
                },
            },
        },
    },
}


RELATED_PROMPT = """
Provice a list of questions related to the original question that are not answered in the context.

{{prompt}}

[START OF CONTEXT]

{% for con in context -%}
Text:
{% for chunk in con.chunks %}
## {{chunk.title}}
Labels: {% for label in chunk.labels %} {{label}} {% endfor -%}
URL: {% for url in chunk.url%}{{url}}{% endfor -%}

{{chunk.text}}
{% endfor -%}

Structured info:
{% for structured in con.structured %}
{{structured}}
{% endfor -%}

Answer summary: {{con.summary}}

{% endfor -%}
[END OF CONTEXT]

MAIN QUESTION: {{question}}

# Notes
- Use the context provided without being overly selective.
- If there is no need to add a new question return an empty list

"""


RELATED_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(RELATED_PROMPT)


class RelatedAgent(Agent[RelatedAgentConfig]):
    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        # For each context in memory add context query, summary and answer onto a text and the initial question
        t0 = time()

        related_prompt = RELATED_PROMPT_TEMPLATE.render(
            question=memory.original_question,
            context=memory.contexts,
            prompt=self.config.prompt,
        )

        related, input, output = await manager.execute_json(
            prompt=related_prompt,
            user_id="related",
            model=self.config.model,
            schema=ASK_JSON_SCHEMA,
            tracking=memory.get_tracking_info(),
        )

        if related is not None:
            for related_question in related.get("related", []):
                memory.add_future_questions(related_question)
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Related questions"),
                step_value=str(related),
                step_reason="",
                step_agent_path=f"/postprocess/{self.config.id if self.config.id else 'default'}",
                timeit=time() - t0,
                input_nuclia_tokens=input,
                output_nuclia_tokens=output,
            )
