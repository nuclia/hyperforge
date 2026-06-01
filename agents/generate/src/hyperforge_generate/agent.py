from time import time
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.trace import trace_agent
from nuclia.lib.nua_responses import ChatModel, UserPrompt

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge_generate.config import GenerateAgentConfig

GENERATE_PROMPT = """
Generate the following request using **only** the provided context. Refrain from incorporating any outside knowledge. If the context is insufficient to answer the question comprehensively, respond with: "Not enough data to generate this."


{% if rules -%}
# Generation Rules
{% for rule in rules -%}
- {{rule}}
{% endfor -%}
{% endif -%}

{{prompt}}

{{context}}

MAIN QUESTION: {{question}}

# Notes
- Use the context provided without being overly selective.
- Please try to generate if possible, even if it requires to make a bit of a deduction.
- If they are images attached, look through them carefully.
- For rules related to charts or images, pay extra attention to extracting details and interpreting data presented visually. Think about it carefully before giving inaccurate interpretations

"""

GENERATE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(GENERATE_PROMPT)


@agent(
    id="generate",
    agent_type="generation",
    title="Generate Answer",
    description="Generate answers based on provided context.",
    config_schema=GenerateAgentConfig,
)
class GenerateAgent(Agent[GenerateAgentConfig]):
    __root_agent__ = True

    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        # For each context in memory add context query, summary and answer onto a text and the initial question
        prompt = GENERATE_PROMPT_TEMPLATE.render(
            question=memory.original_question,
            context=memory.contexts_minimal(),
            prompt=self.config.prompt,
            rules=memory.generation_rules,
        )
        t0 = time()

        chat_model = ChatModel(
            user_id="generate",
            question="",
            user_prompt=UserPrompt(prompt=prompt),
            format_prompt=False,
            generative_model=self.config.model,
            max_tokens=2000,
            tracking=memory.get_tracking_info(),
        )

        agent_path = f"/generation/{self.config.id if self.config.id else 'default'}"
        resp, input_tokens, output_tokens = await manager.execute_raw(
            chat_model, memory=memory, module="generate", agent_path=agent_path
        )

        generated_text = resp.answer or ""

        memory.generated_texts[uuid4().hex] = generated_text

        await memory.add_generated_text(
            generated_text, self.config.id if self.config.id else "default"
        )
        # We assume that JSON generation always produces an answer
        memory.is_answered = True

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Generate"),
            step_value="",
            step_reason="",
            step_agent_path=agent_path,
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )
