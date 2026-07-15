from time import time
from typing import Any, Dict, Optional, cast

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.models import TrackingInfo
from hyperforge.trace import trace_agent
from hyperforge_nucliadb.driver import NucliaDBDriver

from hyperforge_rephrase.config import RephraseAgentConfig

REPHRASE_PROMPT = """
You are an expert at rephrasing complex questions for an agentic RAG system.

Your task is to review the main question and any provided context, then rephrase the question to maximize clarity and focus. Follow these steps:

1. Carefully analyze the main question and all context provided, including sources, previous questions and answers, and any other relevant information.
2. I present, assess whether previous questions and answers (history) are necessary for rephrasing. Only use history if it is relevant and improves the clarity or specificity of the main question; otherwise, ignore it.
3. If the question can be made clearer or more specific, rephrase it accordingly. If it is already clear and focused, return it unchanged.
4. Only use information present in the provided context. Do not introduce external knowledge or assumptions.
5. Return a JSON object with the following fields:
    - "rephrased_question": The rephrased version of the main question, keep the same question if no rephrasing is needed or possible.
    - "rules": Any rules or guidelines that should be followed when generating the answer.
    - "reason": Explain why the rephrasing was necessary or beneficial.
{% if rules -%}
# IMPORTANT CONSIDERATIONS FOR REPHRASING
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}
Return only the JSON object as your response.
{% if context %}
Additional context to assist with rephrasing:
# CONTEXT
{% for info in context %}
- {{ info }}
{% endfor %}
{% endif %}

{% if arguments %}
Arguments that might be useful for rephrasing or splitting the question:
# ARGUMENTS
{% for key, value in arguments.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

# MAIN QUESTION:
{{question}}
"""

MULTI_REPHRASE_PROMPT = """
You are an expert at clarifying and rephrasing questions for an agentic RAG system. Your goal is to identify wheteher a question needs rephrasing or breaking down into sub-questions to ensure it can be answered effectively.

Given the main question and any provided context, your task is to identify whether splitting into sub-questions would improve the answer quality.

Guidelines for splitting:
- Split when the question asks about multiple distinct topics or entities that would benefit from separate retrieval
- Split when there are multiple independent questions that could be answered more thoroughly separately
- A good sub-question should be atomic. A good rule of thumb is that if the question contains a logical operator, it probably should be split further.

Keep as a single question when:
- It asks about a single topic, even if complex
- The question is already clear and focused

Steps:
1. Analyze the main question and all context provided, including sources, previous questions and answers, and any other relevant information.
2. Identify if answering just that could provide a valid answer.
3. If it can't and the question can be broken down into smaller, more specific sub-questions, do so. Each sub-question should be clear, focused, and do not depend on the other sub-questions to be answered.
4. If the main question is already clear and does not require further breakdown, return it as is.
5. Provide a reason for each sub-question explaining why it is necessary to answer the main question.
6. If there are any rules or guidelines that should be followed when generating the answer, include them in the response.
7. Keep key information such as names, titles, dates, and specific terms unchanged to preserve the original intent.
8. Return a JSON object with the following fields:
    - "questions": A list of the minimum set of sub-questions needed to answer the main question. If no sub-questions are needed, return just the main question.
    - "rules": A list of rules or guidelines to follow when generating the answer.
    - "reason": A reason for each sub-question explaining why it is necessary to answer the main question.

{% if rules -%}
# IMPORTANT CONSIDERATIONS FOR REPHRASING
{% for rule in rules -%}
- {{ rule }}
{% endfor -%}
{% endif -%}

{% if context %}
Additional context to assist with rephrasing:
# CONTEXT
{% for info in context %}
- {{ info }}
{% endfor %}
{% endif %}

{% if arguments %}
Arguments that might be useful for rephrasing or splitting the question:
# ARGUMENTS
{% for key, value in arguments.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

# MAIN QUESTION:
{{question}}


"""
MULTI_REPHRASE_JSON_SCHEMA = {
    "title": "questions",
    "description": "",
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of strictly minimum set of sub-questions needed to answer the main question. If not needed, return just the main question - rephrased if necessary.",
        },
        "rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Rules or guidelines to follow when generating the answer.",
        },
        "reason": {
            "type": "string",
            "description": "Reason for each sub-question explaining why it is necessary.",
        },
    },
    "required": ["questions"],
}
REPHRASE_JSON_SCHEMA = {
    "title": "rephrase",
    "description": "",
    "type": "object",
    "properties": {
        "rephrased_question": {
            "type": "string",
            "description": "Rephrased version of the main question.",
        },
        "rules": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Rules or guidelines to follow when generating the answer.",
        },
        "reason": {
            "type": "string",
            "description": "Reason for rephrasing explaining why it is necessary.",
        },
    },
    "required": ["rephrased_question"],
}
REPHRASE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(REPHRASE_PROMPT)
MULTI_REPHRASE_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(MULTI_REPHRASE_PROMPT)

DEFINE_ACTIONS_PROMPT = """
Define the actions needed to transform the retrieved information to user needs

{{context}}
"""
DEFINE_ACTIONS_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(DEFINE_ACTIONS_PROMPT)


@agent(
    id="rephrase",
    agent_type="preprocess",
    title="Rephrase Query",
    description="Agent that rephrases a given question.",
    config_schema=RephraseAgentConfig,
)
class RephraseAgent(Agent[RephraseAgentConfig]):
    async def multi_rephrase(
        self,
        question: str,
        context: list[str],
        manager: Manager,
        extra_rule: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        tracking: TrackingInfo | None = None,
    ) -> tuple[list[str], list[str], str, str, float, float]:
        if extra_rule is not None and self.config.rules is not None:
            rules = self.config.rules.copy() + [extra_rule]
        elif extra_rule is not None:
            rules = [extra_rule]
        else:
            rules = self.config.rules if self.config.rules is not None else []
        prompt = MULTI_REPHRASE_PROMPT_TEMPLATE.render(
            context=context,
            question=question,
            rules=rules,
            arguments=arguments,
        )
        # Ask the LLM to define which information is needed to answer.
        information, input_tokens, output_tokens = await manager.execute_json(
            model=self.config.model,
            user_id="rephrase",
            prompt=prompt,
            schema=MULTI_REPHRASE_JSON_SCHEMA,
            tracking=tracking,
        )

        # retrieve all paragraphs and ask to rephrase on the NUA API
        new_questions = information.get("questions", [])
        list_rules = information.get("rules", [])
        reason = information.get("reason", "")
        step_value = (
            f"{len(new_questions)} questions.\nQuestions: {', '.join(new_questions)}"
        )

        return (
            new_questions,
            list_rules,
            reason,
            step_value,
            input_tokens,
            output_tokens,
        )

    async def rephrase(
        self,
        question: str,
        context: list[str],
        manager: Manager,
        extra_rule: Optional[str] = None,
        arguments: Optional[Dict[str, Any]] = None,
        tracking: TrackingInfo | None = None,
    ) -> tuple[list[str], list[str], str, str, float, float]:
        if extra_rule is not None and self.config.rules is not None:
            rules = self.config.rules.copy() + [extra_rule]
        elif extra_rule is not None:
            rules = [extra_rule]
        else:
            rules = self.config.rules if self.config.rules is not None else []

        prompt = REPHRASE_PROMPT_TEMPLATE.render(
            context=context,
            question=question,
            rules=rules,
            arguments=arguments,
        )
        # Ask the LLM to define which information is needed to answer.
        information, input_tokens, output_tokens = await manager.execute_json(
            model=self.config.model,
            user_id="rephrase",
            prompt=prompt,
            schema=REPHRASE_JSON_SCHEMA,
            tracking=tracking,
        )
        # retrieve all paragraphs and ask to rephrase on the NUA API
        rephrased_question = [information.get("rephrased_question", question)]
        list_rules = information.get("rules", [])
        reason = information.get("reason", "")
        step_value = (
            (
                f"Rephrased question: {rephrased_question[0]}\nRules: {', '.join(list_rules)}\n"
            )
            if rephrased_question[0] != question
            else "No rephrasing considered necessary"
        )

        return (
            rephrased_question,
            list_rules,
            reason,
            step_value,
            input_tokens,
            output_tokens,
        )

    async def inner_rephrase(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        extra_rule: Optional[str] = None,
    ) -> list[str]:
        context = []
        if self.config.kb is not None:
            # Search by BM25 of the words on the actual query (Synonyms)
            nucliadb_driver: Optional[NucliaDBDriver] = cast(
                NucliaDBDriver, manager.drivers.get(self.config.kb)
            )
            if nucliadb_driver is None:
                raise Exception(f"No KnowledgeBox source {self.config.kb} found")

            if self.config.synonyms:
                question = await nucliadb_driver.synonyms(question)

            if self.config.provided_synonyms:
                for key in self.config.provided_synonyms:
                    if key.lower() in question.lower():
                        question += " ".join(self.config.provided_synonyms[key])

            if self.config.extend:
                extra = []
                # Only BM25
                find_result = await nucliadb_driver.find(
                    question, filters=self.config.labels, rids=self.config.rids
                )
                for resource in find_result.resources.values():
                    for field in resource.fields.values():
                        for paragraph in field.paragraphs.values():
                            extra.append(paragraph.text)
                if extra:
                    extra_paragraph = "\n".join(extra)
                    if extra_paragraph:
                        context.append(
                            f"## Extra information to append to the question:\n{extra_paragraph}"
                        )

        if self.config.session_info:
            # Add session information
            context.append(f"## Session information:\n{memory.context_user_info()}")

        if self.config.history:
            qa_history, interactions = await memory.context_history()
            await memory.add_step(
                step_module="rephrase",
                step_title=self.step_title("History check"),
                step_value="Included {} interactions of Q&A history".format(
                    interactions,
                ),
                step_reason="",
                timeit=0,
                step_agent_path=f"/preprocess/{self.config.id if self.config.id else 'default'}",
                input_nuclia_tokens=0.0,
                output_nuclia_tokens=0.0,
            )

            context.append(
                f"## Previous questions and answers in this session:\n{qa_history}"
            )

        t0 = time()
        if self.config.split_question:
            (
                new_questions,
                list_rules,
                reason,
                step_value,
                input_tokens,
                output_tokens,
            ) = await self.multi_rephrase(
                question,
                context,
                manager,
                extra_rule,
                memory.arguments,
                tracking=memory.get_tracking_info(),
            )
            step_title = self.step_title("Sub-questions")
        else:
            (
                new_questions,
                list_rules,
                reason,
                step_value,
                input_tokens,
                output_tokens,
            ) = await self.rephrase(
                question,
                context,
                manager,
                extra_rule,
                memory.arguments,
                tracking=memory.get_tracking_info(),
            )
            step_title = self.step_title("Rephrase")

        await memory.add_step(
            step_module="rephrase",
            step_title=step_title,
            step_value=step_value,
            step_reason=reason,
            timeit=time() - t0,
            step_agent_path=f"/preprocess/{self.config.id if self.config.id else 'default'}",
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
        )
        return new_questions

    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        question = memory.original_question
        if question is None:
            raise Exception("No question")
        new_questions = await self.inner_rephrase(
            question=question,
            memory=memory,
            manager=manager,
        )
        memory.add_context_questions(new_questions)
        # Disabling rules since they do not seem to be helping atm
        # memory.add_generation_rules(list_rules)
