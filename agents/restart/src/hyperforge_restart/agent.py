from time import time

from hyperforge import PROMPT_ENVIRONMENT, logger
from hyperforge.agent import Agent
from hyperforge.exceptions import MaxRetries
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.trace import trace_agent

from hyperforge_restart.config import RestartAgentConfig

REPHRASE_PROMPT = """
You are an expert assistant helping to answer complex questions using the provided context and any partial answers already available.
Your goal is to identify only the missing information needed to fully answer the main question.

Instructions:
1. Carefully review the main question and the context, which may include partial answers or relevant details.
2. If the main question is not fully answered by the context, generate only the specific sub-questions or clarifications needed to obtain the missing information.
3. Briefly explain why each new question is necessary.
4. If the context is sufficient to answer the main question, simply state that no additional questions are needed. Fill the questions list with an empty string. And answer true in the answered parameter.
Be concise and focus on actionable, relevant questions that address only the gaps in information.

# Main Question:
{{question}}

#Context (may include partial answers or supporting information):
{{context}}
"""
JSON_SCHEMA = {
    "title": "questions",
    "description": (
        "An object containing a list of specific sub-questions or clarifications needed "
        "to fully answer the main question, and a brief explanation for each. "
        "Focus only on actionable, relevant questions that address gaps in information."
    ),
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "string",
                "description": "A concise, actionable sub-question or clarification needed to fill a gap in the answer.",
            },
            "description": "List of missing sub-questions or clarifications required to fully answer the main question.",
        },
        "reason": {
            "type": "string",
            "description": "A brief explanation of why these questions are necessary to complete the answer.",
        },
        "answered": {
            "type": "boolean",
            "description": "Indicates whether the main question can be answered with the provided context.",
        },
    },
    "required": ["questions"],
}

REPHRASE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(REPHRASE_PROMPT)


class RestartAgent(Agent[RestartAgentConfig]):
    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        # check how many restart options

        retry = 0
        for step in memory.steps:
            if step.module == "restart":
                retry += 1

        question = memory.original_question

        prompt = REPHRASE_PROMPT_TEMPLATE.render(
            context=memory.contexts_minimal(), question=question
        )
        t0 = time()
        # Ask the LLM to define which information is needed to answer.
        information, input, output = await manager.execute_json(
            model=self.config.model,
            user_id="rephrase",
            prompt=prompt,
            schema=JSON_SCHEMA,
            tracking=memory.get_tracking_info(),
        )

        # retrieve all paragraphs and ask to rephrase on the NUA API
        new_questions = information.get("questions", [])
        reason = information.get("reason", "")
        answered = information.get("answered", False)

        if retry >= self.config.retries and not answered:
            error_message = (
                f"Maximum retries reached ({self.config.retries}). "
                "Unable to generate new questions to answer the original question."
            )
            logger.error(error_message)
            raise MaxRetries(error_message)
        if answered:
            logger.info("No additional questions needed to answer the main question.")
            memory.restart = False
            return
        await memory.add_step(
            step_module="restart",
            step_title=self.step_title("Rephrase"),
            step_value=f"New questions: {', '.join(new_questions)}",
            step_reason=reason,
            timeit=time() - t0,
            step_agent_path=f"/postprocess/{self.config.id if self.config.id else 'default'}",
            input_nuclia_tokens=input,
            output_nuclia_tokens=output,
        )

        memory.add_context_questions(new_questions)
        memory.restart = True
