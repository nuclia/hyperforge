import asyncio
from functools import wraps
from typing import Callable, List, Literal, Optional
from uuid import uuid4

from sentry_sdk import capture_exception

from hyperforge import logger
from hyperforge.agent import Agent
from hyperforge.configure import get_agent_klass
from hyperforge.context.agent import ContextAgent
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from hyperforge.retrieval.config import RetrievalAgentConfig


def handle_stage_error(stage_name: str):
    """Decorator to handle errors in retrieval agent stages"""

    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(self, memory: QuestionMemory, *args, **kwargs):
            try:
                return await func(self, memory, *args, **kwargs)
            except Exception as e:
                capture_exception(e)
                logger.exception(f"Error in {stage_name}")
                memory.final_answer = f"Error in {stage_name}"
                raise  # Re-raise to stop execution

        return wrapper

    return decorator


class RetrievalAgent(Agent):
    module: Literal["retrieval"] = "retrieval"
    debug: bool = False
    preprocess: Optional[list[Agent]] = None
    context: Optional[list[ContextAgent]] = None
    generation: Optional[list[Agent]] = None
    postprocess: Optional[list[Agent]] = None

    def __init__(
        self,
        debug: bool = False,
        preprocess: Optional[list[Agent]] = None,
        context: Optional[list[ContextAgent]] = None,
        generation: Optional[list[Agent]] = None,
        postprocess: Optional[list[Agent]] = None,
    ):
        self.debug = debug
        self.preprocess = preprocess or []
        self.context = context or []
        self.generation = generation or []
        self.postprocess = postprocess or []

    async def inner_from_config(self, config: object, agent_id: object = None) -> None:  # type: ignore[override]
        """No-op: RetrievalAgent is constructed via from_config_class, not from_config."""
        pass

    @classmethod
    async def from_config_class(cls, config: RetrievalAgentConfig):
        preprocess = []

        for agent_obj in config.preprocess:
            agent_class = get_agent_klass(agent_obj.module)
            preprocess.append(await agent_class.from_config(agent_obj))

        context: list[ContextAgent] = []
        for context_agent_obj in config.context:
            agent_class = get_agent_klass(context_agent_obj.module)
            context.append(await agent_class.from_config(context_agent_obj))  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]

        generation = []
        for generation_agent_obj in config.generation:
            agent_class = get_agent_klass(generation_agent_obj.module)
            generation.append(await agent_class.from_config(generation_agent_obj))

        postprocess = []
        for post_agent_obj in config.postprocess:
            agent_class = get_agent_klass(post_agent_obj.module)
            postprocess.append(await agent_class.from_config(post_agent_obj))

        return cls(
            preprocess=preprocess,
            context=context,
            postprocess=postprocess,
            generation=generation,
        )

    @handle_stage_error("preprocess")
    async def _run_preprocess(self, memory: QuestionMemory, manager: Manager):
        if self.preprocess:
            await asyncio.gather(
                *[
                    preprocess(memory=memory, manager=manager)
                    for preprocess in self.preprocess
                ]
            )

    @handle_stage_error("context")
    async def _run_context(self, memory: QuestionMemory, manager: Manager):
        if self.context and self.debug is False:
            questions: List[tuple[str, str]] = memory.get_questions()
            # Launch all context agents for all questions in parallel
            tasks = [
                generation.get_question_context(
                    memory,
                    manager,
                    question_uuid=question_uuid,
                    question=question,
                    flow_id=str(uuid4()),
                )
                for question_uuid, question in questions
                for generation in self.context
            ]
            await asyncio.gather(*tasks)
        elif self.context and self.debug is True:
            questions = memory.get_questions()
            for generation in self.context:
                for question_uuid, question in questions:
                    await generation.get_question_context(
                        memory,
                        manager,
                        question_uuid=question_uuid,
                        question=question,
                        flow_id=str(uuid4()),
                    )

    @handle_stage_error("generation")
    async def _run_generation(self, memory: QuestionMemory, manager: Manager):
        if self.generation:
            await asyncio.gather(
                *[agent(memory, manager) for agent in self.generation],
            )

    @handle_stage_error("postprocess")
    async def _run_postprocess(self, memory: QuestionMemory, manager: Manager):
        if self.postprocess:
            await asyncio.gather(
                *[postprocess(memory, manager) for postprocess in self.postprocess],
            )

    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        while memory.restart:
            memory.restart = False

            await self._run_preprocess(memory=memory, manager=manager)
            if memory.secure is False:
                memory.final_answer = "Insecure query"
                break

            await self._run_context(memory, manager)

            if memory.final_answer is None:
                await self._run_generation(memory, manager)

            if memory.restart is False:
                await memory.add_final_answer()
                await self._run_postprocess(memory, manager)

            if memory.secure is False:
                memory.final_answer = "Insecure context retrieved"
                break
