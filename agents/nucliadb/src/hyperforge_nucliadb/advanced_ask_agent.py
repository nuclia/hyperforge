import asyncio
from time import time
from typing import Any, Dict, List, Optional, cast

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory, Source
from nucliadb_models import filters as ndb_filters
from nucliadb_models.search import AskRequest

from hyperforge import logger
from hyperforge_nucliadb.advanced_ask_config import (
    AdvancedAskAgentConfig,
)
from hyperforge_nucliadb.ask.multi import choose_source
from hyperforge_nucliadb.ask_utils import (
    combine_filter_expressions,
    get_chunk_text,
    to_field_filter_expression,
)
from hyperforge_nucliadb.driver import (
    NucliaDBConnection,
    NucliaDBDriver,
)


@agent(
    id="advanced_ask",
    agent_type="context",
    title="Knowledge Box Advanced Ask",
    description="Ask a question to the knowledge box and retrieve relevant information",
    config_schema=AdvancedAskAgentConfig,
)
class AdvancedAskAgent(ContextAgent, Agent[AdvancedAskAgentConfig]):
    agent_description: str = "Agent that queries a NucliaDB Knowledge Box to get context to answer questions. It is a retrieval agent, so questions should be in a format that makes sense for retrieval."

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        sources = self.config.sources
        # Choose sources based on the question/s and the in
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.config.id if self.config.id else "default",
            step_title=self.step_title("Choose sources"),
        )

        missing: list[tuple[str, str] | None] = await asyncio.gather(
            *[
                self.rag(
                    memory, manager, source, question_uuid, question, flow_id=flow_id
                )
                for source in chosen_sources
            ]
        )
        # We only want to fallback if all sources failed
        if any(m is None for m in missing):
            return []
        else:
            # XXX: We might be duplicating some results here if multiple sources return the same missing question
            return [r for r in missing if r is not None]

    async def rag(
        self,
        memory: QuestionMemory,
        manager: Manager,
        source_obj: Source,
        question_uuid: str,
        question: str,
        flow_id: str,
    ) -> tuple[str, str] | None:
        source = source_obj.id

        nucliadb_driver = get_ndb_driver(manager, source)

        context = Context(
            agent_id=self.config.id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source=source,
            agent="advanced_ask",
            title=self.config.title
            if self.config.title
            else f"Retrieval on {source} Knowledge Box",
        )
        t0 = time()
        ask_request = build_ask_request(self.config, nucliadb_driver.config, question)
        paragraphs = await nucliadb_driver.ask(ask_request)
        answer = None
        if paragraphs is not None:
            input_tokens = (
                paragraphs.metadata.tokens.input_nuclia
                if paragraphs.metadata and paragraphs.metadata.tokens
                else 0
            )
            output_tokens = (
                paragraphs.metadata.tokens.output_nuclia
                if paragraphs.metadata and paragraphs.metadata.tokens
                else 0
            )
            context.chunks = []
            answer = (
                paragraphs.answer
                if paragraphs.answer
                and "not enough data to answer this" not in paragraphs.answer
                else ""
            )
            if paragraphs.citations != {}:
                result_chunks = list(paragraphs.citations.keys())
            else:
                result_chunks = paragraphs.retrieval_results.best_matches
            for chunk_id in result_chunks:
                resource_id = chunk_id.split("/")[0]
                resource = paragraphs.retrieval_results.resources[resource_id]
                text = get_chunk_text(paragraphs, chunk_id)
                context.chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        title=resource.title,
                        text=text,
                        source=source,
                        origin_agent=self.config.module,
                    )
                )
                # TODO: Save citations properly

        # XXX: This answer will be overriden by any call to save_ctx_and_return_missing below
        if answer:
            context.summary = answer
        if self.fallback is None:
            if answer is not None and answer != "":
                missing = await self.save_ctx_and_return_missing(
                    context=context,
                    question=question,
                    memory=memory,
                    manager=manager,
                    flow_id=flow_id,
                )
            else:
                missing = (question_uuid, question)
                logger.info(
                    f"No context found for question {question} in source {source}, skipping"
                )
            ident = self.config.id if self.config.id else "default"
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("RAG retrieval"),
                step_reason="Got answer" if answer else "No answer",
                step_value=answer if answer else "No answer",
                timeit=time() - t0,
                input_nuclia_tokens=input_tokens if input_tokens else 0,
                output_nuclia_tokens=output_tokens if output_tokens else 0,
                step_agent_path=f"/context/{ident}",
            )
            return missing
        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return missing


def get_ndb_driver(manager: Manager, source: str) -> NucliaDBDriver:
    driver = manager.drivers.get(source)
    if driver is None:
        raise Exception("No NDB available")
    return cast(NucliaDBDriver, driver)


def build_ask_request(
    agent: AdvancedAskAgentConfig,
    driver: NucliaDBConnection,
    question: str,
) -> AskRequest:
    filter_expression = parse_filter_expression(agent, driver)
    ask_request = AskRequest(
        query=question,
        generative_model=agent.generative_model,
        citations=True,
        filters=driver.filters,
    )
    if filter_expression is not None:
        # filters and filter_expression are mutually exclusive
        ask_request.filter_expression = filter_expression
        ask_request.filters = []

    # Fields to copy from agent to ask_request
    COPYABLE_FIELDS = [
        "answer_json_schema",
        "citation_threshold",
        "citations",
        "extra_context_images",
        "extra_context",
        "features",
        "generate_answer",
        "max_tokens",
        "min_score",
        "prompt",
        "query_image",
        "rag_images_strategies",
        "rag_strategies",
        "rank_fusion",
        "rephrase",
        "reranker",
        "resource_filters",
        "security",
        "show_hidden",
        "top_k",
        "vectorset",
        "search_configuration",
    ]
    for field in COPYABLE_FIELDS:
        value = getattr(agent, field, None)
        if value is not None:
            setattr(ask_request, field, value)
    return ask_request


def parse_filter_expression(
    agent: AdvancedAskAgentConfig,
    driver: NucliaDBConnection,
) -> Optional[ndb_filters.FilterExpression]:
    """
    Make sure to merge agent and driver filter expressions properly if both are present.
    """
    driver_filter_expression: Optional[ndb_filters.FilterExpression] = None
    if len(driver.filters) > 0:
        operands: list[ndb_filters.FieldFilterExpression] = []
        for filter in driver.filters:
            field_filter = to_field_filter_expression(filter)
            if field_filter is not None:
                operands.append(field_filter)
        if len(operands) == 0:
            driver_filter_expression = None
        elif len(operands) == 1:
            driver_filter_expression = operands[0]  # type: ignore
        else:
            driver_filter_expression = ndb_filters.FilterExpression(
                field=ndb_filters.And(operands=operands)
            )
    elif driver.filter_expression is not None:
        driver_filter_expression = driver.filter_expression

    if agent.filter_expression is None:
        return driver_filter_expression
    else:
        if driver_filter_expression is None:
            return agent.filter_expression
        else:
            return combine_filter_expressions(
                [agent.filter_expression, driver_filter_expression], operator="and"
            )
