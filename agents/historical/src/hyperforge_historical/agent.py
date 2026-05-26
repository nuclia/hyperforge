from time import time

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.trace import trace_agent

from hyperforge_historical.config import HistoricalAgentConfig


@agent(
    id="historical",
    agent_type="preprocess",
    title="Historical Context",
    description="Agent that provides historical context for a given question.",
    config_schema=HistoricalAgentConfig,
)
class HistoricalAgent(Agent[HistoricalAgentConfig]):
    @trace_agent
    async def __call__(
        self,
        memory: QuestionMemory,
        manager: Manager,
    ):
        t0 = time()
        if memory.original_question is not None:
            result = await memory.search_in_questions(
                memory.original_question, self.config.all
            )
            if result.total > 0:
                # Summarize the fins results

                # Check if its answers

                answer = ""
                await memory.add_answer(
                    answer=answer,
                    module="historical",
                    agent_path=f"/preprocess/{self.agent_id}",
                )
        await memory.add_step(
            step_module="historical",
            step_title=self.step_title("Historical context"),
            step_agent_path=f"/preprocess/{self.agent_id}",
            timeit=time() - t0,
        )
