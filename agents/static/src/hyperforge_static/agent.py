from time import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory.memory import Chunk, Context, QuestionMemory

from hyperforge_static.config import StaticAgentConfig


@agent(
    id="static",
    agent_type="context",
    title="Static Context",
    description="Provide static context to answer questions.",
    config_schema=StaticAgentConfig,
)
class StaticAgent(ContextAgent, Agent[StaticAgentConfig]):
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "static_context": FunctionDefinition(
            name="static_context",
            description="Provide static context to answer questions.",
            parameters={},
        )
    }

    async def static_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: Optional[str] = "",
        question_uuid: Optional[str] = None,
    ) -> Context:
        if question_uuid is None:
            question_uuid = uuid4().hex

        context = Context(
            agent_id=self.config.id if self.config.id else "static_context",
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source="static_context",
            agent="static_context",
            title=self.config.title if self.config.title else "Static Context",
        )
        if self.config.context:
            context.chunks.append(
                Chunk(
                    chunk_id=uuid4().hex,
                    text=self.config.context,
                    origin_agent=self.config.module,
                )
            )
        if self.config.structured:
            context.structured.append(self.config.structured)
        return context

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, str]]:
        t0 = time()
        error = None

        context = await self.static_context(
            memory=memory,
            manager=manager,
        )

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Static context"),
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_value=" Static context retrieval",
            timeit=time() - t0,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
            error=error,
        )
        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return [missing] if missing is not None else []
