from time import time
from typing import Any, ClassVar, Dict, Optional

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory
from pydantic import Field


class StaticStringAgentConfig(ContextAgentConfig):
    context: str = Field(description="Data to add to the context")


@agent(
    id="static_string",
    agent_type="context",
    title="Static String",
    description="Use a static string to provide context for answering questions.",
    config_schema=StaticStringAgentConfig,
)
class StaticStringAgent(ContextAgent, Agent[StaticStringAgentConfig]):
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "static_string": FunctionDefinition(
            name="static_string",
            description="Returns a static string to provide context for answering questions.",
            parameters={},
        )
    }

    def static_string(self) -> str:
        return self.config.context

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> list[tuple[str, str]]:
        error = None
        t0 = time()
        missing = await self.save_ctx_and_return_missing(
            memory=memory,
            context=Context(
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=question_uuid,
                question=question,
                chunks=[
                    Chunk(
                        chunk_id="static_string",
                        text=self.static_string(),
                        origin_agent=self.config.module,
                    )
                ],
                source=f"/context/{self.agent_id}",
                agent=self.config.module,
            ),
            question=question,
            manager=manager,
            flow_id=flow_id,
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Search results"),
            step_agent_path=f"/context/{self.agent_id}",
            step_value="String done",
            timeit=time() - t0,
            input_nuclia_tokens=0.5,
            output_nuclia_tokens=0.5,
            error=error,
        )
        return [missing] if missing is not None else []
