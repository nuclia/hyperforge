from time import time
from typing import Any, ClassVar, Dict, List, Optional, cast
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory
from hyperforge_perplexity.driver import PerplexityDriver

from hyperforge_perplexity_search.config import PerplexitySearchAgentConfig

SYSTEM_PROMPT = "Be precise and concise"


@agent(
    id="perplexity_search",
    agent_type="context",
    title="Perplexity Search",
    description="Use Perplexity to get information from the internet.",
    config_schema=PerplexitySearchAgentConfig,
)
class PerplexitySearchAgent(ContextAgent, Agent[PerplexitySearchAgentConfig]):
    driver: Optional[PerplexityDriver] = None
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "internet_search": FunctionDefinition(
            name="internet_search",
            description="Performs an internet search using Perplexity and returns the results as context to answer questions. Does not generate an answer, only search and return the results as context.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to search for on the internet.",
                },
            },
        )
    }

    async def search(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: Optional[str] = None,
    ) -> Context:
        t0 = time()

        if self.driver is None:
            self.driver: Optional[PerplexityDriver] = cast(
                Optional[PerplexityDriver], manager.drivers.get(self.config.source)
            )
        if self.driver is None:
            raise Exception("Perplexity source does not exist")

        response = await self.driver.client.search.create(
            query=question,
            search_domain_filter=self.config.domain,
            max_results=self.config.max_results,
            max_tokens_per_page=self.config.max_tokens_per_page,
        )
        chunks = []
        if response.results is not None:
            for result in response.results:
                text = result.snippet if result.snippet is not None else ""
                title = (
                    result.title
                    if result.title is not None
                    else result.url
                    if result.url is not None
                    else "No title"
                )
                chunk = Chunk(
                    chunk_id=uuid4().hex,
                    text=text,
                    title=title,
                    labels=[],
                    url=[result.url] if result.url is not None else [],
                    origin_url=result.url if result.url is not None else "",
                    origin_agent=self.config.module,
                )
                chunks.append(chunk)
        context = Context(
            agent_id=self.config.id or self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            title=self.config.title
            if self.config.title
            else "Internet search with Perplexity",
            source="perplexity",
            agent=self.config.module,
            chunks=chunks,
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Search results"),
            step_reason="",
            step_agent_path=f"/context/{self.config.id or self.agent_id}",
            step_value=f"{len(chunks)} results found with Perplexity"
            if chunks
            else "No results found",
            timeit=time() - t0,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
        )
        return context

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        if self.driver is None:
            self.driver: Optional[PerplexityDriver] = cast(
                Optional[PerplexityDriver], manager.drivers.get(self.config.source)
            )

        if self.driver is None:
            raise Exception("Perplexity source does not exist")

        context = await self.search(
            question,
            memory,
            manager,
            question_uuid=question_uuid,
        )

        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return [missing] if missing is not None else []
