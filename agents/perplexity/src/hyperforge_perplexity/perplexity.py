from base64 import b64encode
from time import time
from typing import Any, ClassVar, Dict, List, Optional, cast
from uuid import uuid4

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory
from hyperforge.utils.http import safe_http_client
from nuclia.lib.nua_responses import Image
from perplexity.types import ChatMessageInput
from perplexity.types.shared.chat_message_output import ChatMessageOutput
from perplexity.types.shared_params.web_search_options import WebSearchOptions

from hyperforge_perplexity.config import PerplexityAgentConfig
from hyperforge_perplexity.driver import PerplexityDriver

SYSTEM_PROMPT = "Be precise and concise"


@agent(
    id="perplexity",
    agent_type="context",
    title="Perplexity Answers",
    description="Use Perplexity to get information from the internet.",
    config_schema=PerplexityAgentConfig,
)
class PerplexityAgent(ContextAgent, Agent[PerplexityAgentConfig]):
    driver: Optional[PerplexityDriver] = None
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "internet_search": FunctionDefinition(
            name="internet_search",
            description="Performs an internet search using Perplexity Search to get context to answer questions.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to search for on the internet.",
                },
            },
        )
    }

    def build_messages(self, question: str) -> list:
        """Builds the messages list for the chat completion."""
        system_prompt = (
            self.config.prompt if self.config.prompt is not None else SYSTEM_PROMPT
        )
        return [
            ChatMessageInput(role="system", content=system_prompt),
            ChatMessageInput(role="user", content=question),
        ]

    async def internet_search(
        self,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: Optional[str] = None,
    ) -> Context:
        messages = self.build_messages(question)

        t0 = time()

        if self.driver is None:
            self.driver: Optional[PerplexityDriver] = cast(
                Optional[PerplexityDriver], manager.drivers.get(self.config.source)
            )
        if self.driver is None:
            raise Exception("Perplexity driver does not exist")

        web_search_options = WebSearchOptions(
            search_context_size=self.config.search_context_size,
        )
        response = await self.driver.client.chat.completions.create(
            messages=messages,
            model="sonar-pro",
            return_images=self.config.images,
            return_related_questions=self.config.related_questions,
            search_domain_filter=self.config.domain,
            web_search_options=web_search_options,
        )
        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            title=self.config.title
            if self.config.title
            else "Internet search with Perplexity",
            source="perplexity",
            agent=self.config.module,
        )
        text = None
        for choice in response.choices:
            if (
                choice.message is not None
                and choice.message.content is not None
                and isinstance(choice.message, ChatMessageOutput)
            ):
                text = (
                    choice.message.content
                    if isinstance(choice.message.content, str)
                    else str(choice.message.content)
                )

        if (
            hasattr(response, "images")
            and response.images is not None
            and isinstance(response.images, list)
        ):
            for image in response.images:
                image_dict = cast(Dict[str, Any], image)
                async with safe_http_client() as session:
                    resp = await session.get(image_dict["image_url"])
                content_type = resp.headers.get("Content-Type")
                if content_type:
                    # Remove parameters like "; charset=utf-8"
                    mime = content_type.split(";", 1)[0].strip()
                    if mime:
                        context.images[image_dict["origin_url"]] = Image(
                            content_type=mime,
                            b64encoded=b64encode(resp.content).decode(),
                        )

        if text is not None:
            chunk = Chunk(
                chunk_id=uuid4().hex,
                text=text,
                labels=[],
                url=response.citations,  # type: ignore
                origin_agent=self.config.module,
            )
            context.chunks.append(chunk)

        # for chunk in response add to context
        input_nuclia_tokens = 0
        context.summary = text if text is not None else ""

        if (
            self.config.related_questions
            and hasattr(response, "related_questions")
            and response.related_questions is not None
        ):
            questions: List[str] = cast(List[str], response.related_questions)
            memory.add_future_questions(questions)

        # TODO: Report nuclia tokens correctly
        if response.usage is not None and response.usage.total_tokens is not None:
            input_nuclia_tokens += response.usage.total_tokens

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Search results"),
            step_reason="",
            step_agent_path=f"/context/{self.config.id or self.agent_id}",
            step_value=text if text is not None else "",
            timeit=time() - t0,
            input_nuclia_tokens=input_nuclia_tokens,
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
            raise Exception("Perplexity driver does not exist")

        context = await self.internet_search(
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
