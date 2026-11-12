from typing import Any, Dict, Literal, Optional, cast
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent, trace_agent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.manager import Manager
from hyperforge.memory.memory import Chunk, Context, QuestionMemory
from pydantic import Field
from pydantic.config import ConfigDict

from hyperforge_conditional.conditional import (
    Conditional,
    ConditionalAgentConfig,
)


class ContextConditionalAgentConfig(ConditionalAgentConfig, ContextAgentConfig):
    model_config = ConfigDict(title="Condition")
    module: Literal["context_conditional"] = "context_conditional"
    on: Literal["QUESTION", "CONTEXT"] = Field(
        default="QUESTION",
        title="Evaluate condition on",
        description="Source to evaluate the condition on. CONTEXT is only valid when this agent is used as a next agent in a chain.",
    )


@agent(
    id="context_conditional",
    agent_type="context",
    title="Context Conditional",
    description="Use Context Conditional to get information from the internet to answer questions.",
    config_schema=ContextConditionalAgentConfig,
)
class ContextConditional(
    ContextAgent, Conditional, Agent[ContextConditionalAgentConfig]
):
    arag_step: str = "context"

    async def inner_from_config(
        self, config: ContextConditionalAgentConfig, agent_id: Optional[str] = None
    ):
        # Build then and else branches
        await self.context_from_config(config)
        await self.conditional_from_config(config)

    async def __call__(self, memory: QuestionMemory, manager: Manager):
        await Conditional.__call__(self, memory, manager)

    @trace_agent
    async def get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ):
        self.config: ContextConditionalAgentConfig
        if extra_context is not None:
            question, question_uuid = await self.rephrase(
                memory=memory,
                manager=manager,
                question_uuid=question_uuid,
                question=question,
                contexts=extra_context,
                model=self.config.rephrase_model,
                module=self.config.module,
                user_id="next_rephrase",
                ident=self.agent_id,
            )

        conditional_subject = question
        # Previous context only used when selected and extra context is provided - that is, the conditional comes after a next agent in the flow
        if self.config.on == "CONTEXT" and extra_context is not None:
            conditional_subject = "\n".join(extra_context.values())

        # condition checking and branching
        condition = await self.make_decision(
            memory=memory,
            manager=manager,
            question=conditional_subject,
        )
        selected_agents: list[Agent] = [
            agent for agent in (self.then if condition else self.else_) or []
        ]
        for selected_agent in selected_agents:
            await cast(ContextAgent, selected_agent).get_question_context(
                memory, manager, question_uuid, question, flow_id=flow_id
            )
        # If fallback is configured, we check if there are missing questions taking into account the executed agents contexts/summaries
        if self.fallback is not None:
            conditional_context = Context(
                agent_id=self.agent_id,
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=question_uuid,
                question=question,
                agent="Conditional",
                title="Summarizing Conditional Agents Contexts",
                source="conditional",
                chunks=[],
            )

            for agent in selected_agents:
                conditional_context.chunks.extend(
                    [
                        Chunk(
                            chunk_id=agent.agent_id,
                            text=summary,
                            origin_agent=self.config.module,
                        )
                        for summary in memory.get_agent_answer_summaries(
                            flow_id=flow_id, agent_id=agent.agent_id
                        )
                    ]
                )
            (
                _,
                missing_question,
                _,
            ) = await self.validate_ctx_and_answer(
                memory,
                manager,
                conditional_context,
                question=question,
            )
            if missing_question is not None and missing_question.strip() != "":
                await self.fallback.get_question_context(
                    memory,
                    manager,
                    question_uuid=uuid4().hex,
                    question=missing_question,
                    flow_id=flow_id,
                )

        if self.next_agent is not None:
            extra_context = extra_context or {}
            for nagent in (
                selected_agents + [self.fallback]
                if self.fallback is not None
                else selected_agents
            ):
                if nagent:
                    answer_summaries = memory.get_agent_answer_summaries(
                        flow_id=flow_id, agent_id=nagent.agent_id
                    )
                    if answer_summaries:
                        extra_context[nagent.agent_id] = "\n".join(answer_summaries)

            await self.next_agent.get_question_context(
                memory,
                manager,
                question_uuid,
                question,
                extra_context=extra_context,
                flow_id=flow_id,
            )
