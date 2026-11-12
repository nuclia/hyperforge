import asyncio

from hyperforge import logger
from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory
from hyperforge.trace import trace_agent
from nuclia_models.predict.remi import RemiResponse

from hyperforge_remi.config import (
    ContextGranularity,
    RemiAgentConfig,
)

MAX_CONTEXTS_REMI = 60


@agent(
    id="remi",
    agent_type="postprocess",
    title="REMI Evaluation",
    description="Agent that performs REMI evaluation.",
    config_schema=RemiAgentConfig,
)
class RemiAgent(Agent[RemiAgentConfig]):
    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        error = None
        # For each context in memory add context query, summary and answer onto a text and the initial question
        if memory.final_answer is None:
            raise Exception("No final answer")
        if memory.original_question is None:
            raise Exception("No original question")

        # Only answer relevance and groundedness for now
        # If we also did context relevance, we would combine into a single remi call
        tasks = [
            manager.remi(
                question=memory.original_question,
                answer=memory.final_answer,
                contexts=None,
            )
        ]
        contexts = (
            memory.list_contexts_minimal()
            if self.config.context_granularity == ContextGranularity.PARTIAL_ANSWERS
            else memory.list_chunks_markdown()
        )
        if len(contexts) > MAX_CONTEXTS_REMI:
            contexts = contexts[:MAX_CONTEXTS_REMI]
            error = f"Too many contexts for groundedness evaluation, truncated to {MAX_CONTEXTS_REMI}. "

        if memory.is_answered is True:
            tasks.append(
                manager.remi(
                    question=None,
                    answer=memory.final_answer,
                    contexts=contexts,
                )
            )
        results = await asyncio.gather(*tasks, return_exceptions=True)
        ev_answer_rel: RemiResponse | BaseException = results[0]
        if memory.is_answered is True:
            ev_groundedness: RemiResponse | BaseException | None = results[1]
        else:
            logger.info(
                "Forcing REMi groundedness to 0 on question flagged as not answered"
            )
            ev_groundedness = RemiResponse(
                groundedness=[0] * max(len(contexts), 1), time=0
            )

        response_str = ""
        if (
            isinstance(ev_answer_rel, BaseException)
            or not ev_answer_rel
            or not ev_answer_rel.answer_relevance
        ):
            msg = "Error evaluating answer relevance. "
            logger.warning(
                msg + str(ev_answer_rel)
                if isinstance(ev_answer_rel, BaseException)
                else ""
            )
            error = error + msg if error else msg
        else:
            response_str += (
                f"Answer relevance: {ev_answer_rel.answer_relevance.score}/5. "
            )

        # Max aggregation for groundedness, could be configurable
        if (
            isinstance(ev_groundedness, BaseException)
            or not ev_groundedness
            or not ev_groundedness.groundedness
        ):
            msg = "Error evaluating answer groundedness. "
            logger.warning(
                msg + str(ev_groundedness)
                if isinstance(ev_groundedness, BaseException)
                else ""
            )
            error = error + msg if error else msg
        else:
            groundedness = max(
                [g if g is not None else 0 for g in ev_groundedness.groundedness]
            )
            response_str += f"Answer groundedness: {groundedness}/5."

            remi_chunks = []

            # Use chunks directly if context granularity is chunk (markdown)
            if self.config.context_granularity != ContextGranularity.PARTIAL_ANSWERS:
                chunk_idx = 0
                for context in memory.contexts:
                    agent_name = context.agent_id if context.agent_id else context.agent
                    if context.chunks:
                        for chunk in context.chunks:
                            if chunk_idx >= min(
                                MAX_CONTEXTS_REMI, len(ev_groundedness.groundedness)
                            ):
                                break
                            score = ev_groundedness.groundedness[chunk_idx]
                            g_score = score if score is not None else 0

                            c = Chunk(
                                chunk_id=chunk.chunk_id or f"remi_chunk_{chunk_idx}",
                                title=f"Groundedness {g_score}/5 - [{agent_name}] {chunk.title or 'Untitled'}",
                                text=f"**Groundedness: {g_score}/5**\n\n{chunk.text}",
                                origin_agent=self.config.module,
                            )
                            remi_chunks.append(c)
                            chunk_idx += 1
            else:
                for i, context in enumerate(memory.contexts[:MAX_CONTEXTS_REMI]):
                    if i < len(ev_groundedness.groundedness):
                        score = ev_groundedness.groundedness[i]
                        g_score = score if score is not None else 0
                        agent_name = (
                            context.agent_id if context.agent_id else context.agent
                        )

                        evaluated_text = (
                            context.answer_summary_markdown()
                            if context.summary.strip()
                            else context.context_markdown()
                        )
                        c = Chunk(
                            chunk_id=context.id or f"remi_{i}",
                            title=f"Groundedness {g_score}/5 - [{agent_name}] {context.title or 'Untitled'}",
                            text=f"**Groundedness: {g_score}/5**\n\n{evaluated_text}",
                            origin_agent=self.config.module,
                        )
                        remi_chunks.append(c)

            if remi_chunks:
                remi_context = Context(
                    original_question_uuid=memory.original_question_uuid,
                    actual_question_uuid=memory.actual_question_uuid,
                    agent="remi",
                    agent_id="remi",
                    title="REMi Evaluation Breakdown",
                    summary=response_str.strip()
                    if response_str.strip()
                    else f"Evaluated {len(remi_chunks)} contexts for groundedness.",
                    question=memory.original_question or "",
                    source="remi",
                    chunks=remi_chunks,
                )
                if error:
                    remi_context.summary += f"\nErrors: {error}"
                await memory.save_context("postprocess", remi_context)
