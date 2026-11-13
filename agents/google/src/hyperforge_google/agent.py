# GOOD REFERENCE: https://github.com/philschmid/gemini-samples/blob/main/examples/gemini-google-search.ipynb

from time import time
from typing import Any, ClassVar, Dict, List, Optional, Tuple, cast
from uuid import uuid4

from google.genai.types import (
    GenerateContentConfig,
    GoogleSearch,
    ThinkingConfig,
    Tool,
)
from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory.memory import Chunk, Context, QuestionMemory
from hyperforge.models import ExternalUsage, ExternalUsageOperation
from hyperforge.utils.http import safe_http_client

from hyperforge_google.config import GoogleAgentConfig
from hyperforge_google.driver import GoogleDriver


@agent(
    id="google",
    agent_type="context",
    title="Google Search",
    description="Use Google Search to get information from the internet to answer questions.",
    config_schema=GoogleAgentConfig,
)
class GoogleAgent(ContextAgent, Agent[GoogleAgentConfig]):
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "internet_search": FunctionDefinition(
            name="internet_search",
            description="Performs an internet search using Google Search to get context to answer questions.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to search for on the internet.",
                },
            },
        )
    }
    google_search_tool: Optional[Tool] = None

    async def inner_from_config(
        self, config: GoogleAgentConfig, agent_id: Optional[str] = None
    ):
        await self.context_from_config(config)
        self.google_search_tool = Tool(google_search=GoogleSearch())

    def driver(self, manager: Manager) -> GoogleDriver:
        return cast(GoogleDriver, manager.drivers[self.config.source])

    async def internet_search(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: Optional[str] = None,
        flow_id: Optional[str] = None,
    ) -> Context:
        if question_uuid is None:
            question_uuid = uuid4().hex

        driver = self.driver(manager)
        t0 = time()
        assert self.google_search_tool is not None, "google_search_tool not initialized"
        response = await driver.client.aio.models.generate_content(
            model=self.config.gen_model_id,
            contents=question,
            config=GenerateContentConfig(
                tools=[self.google_search_tool],
                response_modalities=["TEXT"],
                thinking_config=ThinkingConfig(include_thoughts=True),
            ),
        )
        usage = response.usage_metadata if response is not None else None
        context = Context(
            agent_id=self.config.id or self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source="google",
            agent="google",
            title=self.config.title
            if self.config.title
            else "Internet search with Google",
        )
        http_client = safe_http_client()
        chunks: Dict[int, str] = {}
        answer = ""
        reasoning = ""
        count = 0
        if (
            response
            and response.candidates
            and len(response.candidates) > 0
            and response.candidates[0].content
            and response.candidates[0].content.parts
        ):
            candidate = response.candidates[0]
            for each in response.candidates[0].content.parts:
                if each.thought is True and each.text is not None:
                    reasoning += each.text
                elif each.text is not None:
                    answer += each.text
            if (
                candidate.grounding_metadata
                and candidate.grounding_metadata.grounding_chunks
            ):
                for chunk_id, chunk in enumerate(
                    candidate.grounding_metadata.grounding_chunks
                ):
                    if chunk.web is not None and chunk.web.uri is not None:
                        resp = await http_client.get(chunk.web.uri)
                        chunks[chunk_id] = resp.headers.get("location", "")

            if (
                candidate.grounding_metadata
                and candidate.grounding_metadata.grounding_supports
            ):
                for ground in candidate.grounding_metadata.grounding_supports:
                    if ground.grounding_chunk_indices:
                        url = [
                            chunks[indice] for indice in ground.grounding_chunk_indices
                        ]
                    else:
                        url = []
                    if ground.segment and ground.segment.text:
                        text = ground.segment.text
                    else:
                        text = ""
                    count += 1
                    context.chunks.append(
                        Chunk(
                            chunk_id=uuid4().hex,
                            text=text,
                            labels=[],
                            url=url,
                            origin_agent=self.config.module,
                        )
                    )
                # if candidate.grounding_metadata.web_search_queries:
                #     for query in candidate.grounding_metadata.web_search_queries:
                #         # Queries
        await memory.add_answer(
            answer,
            module=self.config.module,
            agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
        )
        context.summary = answer
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Search results"),
            step_agent_path=f"/context/{self.config.id if self.config.id else 'default'}",
            step_reason=reasoning,
            step_value=f"{count} results found",
            timeit=time() - t0,
            external_usage=[
                ExternalUsage(
                    operation=ExternalUsageOperation.INTERNET_SEARCH,
                    provider="google",
                    model=(response.model_version if response is not None else None)
                    or self.config.gen_model_id,
                    input_tokens=(usage.prompt_token_count or 0) if usage else 0,
                    output_tokens=(
                        (usage.candidates_token_count or 0)
                        + (usage.thoughts_token_count or 0)
                        + (usage.tool_use_prompt_token_count or 0)
                        if usage
                        else 0
                    ),
                )
            ],
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
    ) -> List[Tuple[str, str]]:
        context = await self.internet_search(
            question=question,
            memory=memory,
            manager=manager,
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
