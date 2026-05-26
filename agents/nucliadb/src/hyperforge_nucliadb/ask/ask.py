import asyncio
from time import time
from typing import Any, ClassVar, Dict, List, Optional, cast

from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent, trace_agent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory.memory import Context, QuestionMemory, Source

from hyperforge import logger
from hyperforge_nucliadb.ask.analysis import question_analysis
from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_nucliadb.ask.hydrate import hydrate
from hyperforge_nucliadb.ask.knowledge_scan import knowledge_scan
from hyperforge_nucliadb.ask.models import Analysis
from hyperforge_nucliadb.ask.multi import choose_source
from hyperforge_nucliadb.ask.nucliadb import query_ndb, standard_query_ndb
from hyperforge_nucliadb.ask.rerank import rerank
from hyperforge_nucliadb.driver import NucliaDBDriver


@agent(
    id="ask",
    agent_type="context",
    title="Knowledge Box Ask",
    description="Ask a question to the knowledge box and retrieve relevant information",
    config_schema=AskAgentConfig,
)
class AskAgent(ContextAgent, Agent[AskAgentConfig]):
    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "search_by_title": FunctionDefinition(
            name="search_by_title",
            description="Search for context in the Knowledge Box by title. Useful for specific queries where the title is known.",
            parameters={
                "title": {
                    "type": "string",
                    "description": "The title to search for in the Knowledge Box.",
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filters to apply when searching in the Knowledge Box.",
                },
            },
        ),
        "ask_analysis_query": FunctionDefinition(
            name="ask_analysis_query",
            description="Generate filters to be used when querying the Knowledge Box. Useful to refine search results based on the question.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question based on which to generate filters.",
                },
                "kbid": {
                    "type": "string",
                    "description": "The Knowledge Box ID to use for generating the analysis. If not provided, the default sources will be used.",
                },
            },
        ),
        "ask_agent": FunctionDefinition(
            name="ask_agent",
            description="Search for context in the Knowledge Box by title. Useful for specific queries where the title is known.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to search for in the Knowledge Box.",
                },
            },
        ),
    }

    fallback: Optional[ContextAgent] = None

    async def ask_analysis_query(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        **kwargs: Any,
    ) -> List[Analysis]:
        sources = await self.get_source(question, memory, manager)
        result = []
        for source in sources:
            nucliadb_driver: Optional[NucliaDBDriver] = cast(
                NucliaDBDriver, manager.drivers.get(source.id)
            )
            if nucliadb_driver is None:
                raise Exception("No NDB available")
            analysis = await question_analysis(
                memory,
                manager,
                nucliadb_driver,
                source,
                config=self.config,
                question=question,
                step_title=self.step_title("Choose parameters"),
            )  # type: ignore
            result.append(analysis)
        return result

    async def work(
        self,
        memory: QuestionMemory,
        manager: Manager,
        context: Context,
        nucliadb_driver: NucliaDBDriver,
        source: Source,
        question: str,
    ):
        # Generate semantic questions required to answer
        analysis: Optional[Analysis] = None
        if self.config.ai_parameter_search:
            try:
                analysis = await question_analysis(
                    memory,
                    manager,
                    nucliadb_driver,
                    source,
                    config=self.config,
                    question=question,
                    step_title=self.step_title("Choose parameters"),
                )
            except Exception as e:
                logger.exception("Error analyzing question")
                raise e

            try:
                if analysis.knowledge_graph is not None:
                    search_result = await knowledge_scan(
                        memory,
                        manager,
                        analysis,
                        nucliadb_driver,
                        config=self.config,
                    )
                else:
                    search_result = await query_ndb(
                        memory,
                        manager,
                        analysis,
                        nucliadb_driver,
                        config=self.config,
                    )

            except Exception as e:
                logger.exception("Error querying NDB")
                raise e
        else:
            search_result = await standard_query_ndb(
                memory,
                manager,
                nucliadb_driver,
                config=self.config,
                question=question,
            )

        await hydrate(
            context,
            search_result,
            nucliadb_driver=nucliadb_driver,
            vllm=self.config.vllm,
            after=self.config.after,
            before=self.config.before,
            full_resource=self.config.full_resource,
            visual=analysis.visual if analysis is not None else False,
            link=analysis.link if analysis is not None else False,
            module=self.config.module,
        )
        await rerank(context, manager=manager, top_k=analysis.top_k if analysis else 20)

    async def get_source(
        self, question: str, memory: QuestionMemory, manager: Manager
    ) -> List[Source]:
        if len(self.config.sources) == 0:
            # Do we really want to do this without telling the user?
            sources = manager.nucliadbs()
        else:
            sources = self.config.sources

        # Only makes sense if we have more than one source
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.config.id if self.config.id else "default",
            step_title=self.step_title("Choose sources"),
        )
        return chosen_sources

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
        if len(self.config.sources) == 0:
            # Do we really want to do this without telling the user?
            sources = manager.nucliadbs()
        else:
            sources = self.config.sources

        # Only makes sense if we have more than one source
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.config.id if self.config.id else "default",
            step_title=self.step_title("Choose sources"),
        )

        await asyncio.gather(
            *[
                self.rag(
                    memory, manager, source, question_uuid, question, flow_id=flow_id
                )
                for source in chosen_sources
            ]
        )

    async def rag(
        self,
        memory: QuestionMemory,
        manager: Manager,
        source_obj: Source,
        question_uuid: str,
        question: str,
        flow_id: str,
    ):
        source = source_obj.id
        nucliadb_driver: Optional[NucliaDBDriver] = cast(
            NucliaDBDriver, manager.drivers.get(source)
        )
        if nucliadb_driver is None:
            raise Exception("No NDB available")

        context = Context(
            agent_id=self.config.id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source=source,
            agent="ask",
            title=self.config.title
            if self.config.title
            else f"Retrieval on {source} Knowledge Box",
        )

        await self.work(memory, manager, context, nucliadb_driver, source_obj, question)
        ident = self.config.id if self.config.id else "default"
        # If no chunks were found, we can skip the summarization
        if context.chunks is not None and len(context.chunks) != 0:
            # These missing questions are not used as of now
            missing_question = await self.save_ctx_and_return_missing(
                context=context,
                question=question,
                memory=memory,
                manager=manager,
                flow_id=flow_id,
            )
        else:
            missing_question = (question_uuid, question)
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("NucliaDB context"),
                step_reason="No context found",
                step_value="Unsuccessful retrieval",
                timeit=time() - time(),
                step_agent_path=f"/context/{ident}",
            )
        # Query NDB
        if missing_question is not None:
            missing_uuid, missing = missing_question
            # This is always failing back to another nucliadb call in case of missing information
            # TODO: decide if we want to do this or not
            # implicit fallback to another NDB call may introduce unnecessary delay
            # for now I'm only adding it if we have no fallback agent configured
            if self.config.fallback is None:
                context = Context(
                    agent_id=self.config.id,
                    original_question_uuid=memory.original_question_uuid,
                    actual_question_uuid=missing_uuid,
                    question=missing,
                    source=source,
                    agent="ask",
                    title=self.config.title
                    if self.config.title
                    else f"Retrieval on {source} Knowledge Box",
                )

                await self.work(
                    memory,
                    manager,
                    context,
                    nucliadb_driver,
                    source_obj,
                    question,
                )
                new_missing_questions = await self.save_ctx_and_return_missing(
                    context=context,
                    question=missing,
                    memory=memory,
                    manager=manager,
                    flow_id=flow_id,
                )
                if new_missing_questions is not None:
                    # TODO: add proper structure to notify the user that the answer is missing
                    logger.warning(
                        f"Missing information after fallback: {[m[1] for m in new_missing_questions]}. No fallback agent configured."
                    )
            elif self.fallback is not None:
                # TODO: Add next agent
                await self.fallback.get_question_context(
                    memory,
                    manager,
                    question=missing,
                    question_uuid=missing_uuid,
                    flow_id=flow_id,
                )
