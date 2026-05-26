from asyncio import gather
from typing import Awaitable, List, Optional, cast

from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from nucliadb_models.filters import And, Or
from nucliadb_models.graph.requests import (
    GraphPathQuery,
)
from nucliadb_models.resource import ExtractedDataTypeName
from nucliadb_models.search import (
    FindOptions,
    FindRequest,
    KnowledgeboxFindResults,
    MinScore,
    RerankerName,
    ResourceProperties,
)
from nucliadb_models.security import RequestSecurity

from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_nucliadb.ask.models import Analysis, SearchResults
from hyperforge_nucliadb.ask.utils import (
    empty,
    get_nodes,
    get_relations,
)
from hyperforge_nucliadb.driver import NucliaDBDriver


def query_relations(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
    relations: List[GraphPathQuery],
    nodes: List[GraphPathQuery],
) -> Awaitable[KnowledgeboxFindResults]:
    """Query relations in NucliaDB based on the provided analysis."""
    find_request = FindRequest(
        features=[FindOptions.GRAPH],
        filters=analysis.labels,
        graph_query=And(
            operands=[
                Or(operands=relations),
                Or(operands=nodes),
            ]
        ),
        security=RequestSecurity(groups=config.security_groups),
        reranker=RerankerName.NOOP,
        show=[
            ResourceProperties.BASIC,
            ResourceProperties.ORIGIN,
            ResourceProperties.EXTRA,
            ResourceProperties.VALUES,
            ResourceProperties.RELATIONS,
        ],
    )
    return nucliadb_driver.find_raw(find_request)


def query_semantic(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
) -> Awaitable[KnowledgeboxFindResults]:
    """Semantic search in NucliaDB based on the provided analysis."""

    find_request = FindRequest(
        features=[FindOptions.SEMANTIC],
        query=analysis.semantic_query,
        min_score=MinScore(semantic=analysis.semantic_config.threshold),
        vectorset=analysis.semantic_model,
        filters=analysis.labels,
        security=RequestSecurity(groups=config.security_groups),
        reranker=RerankerName.NOOP,
        show=[
            ResourceProperties.BASIC,
            ResourceProperties.ORIGIN,
            ResourceProperties.EXTRA,
            ResourceProperties.EXTRACTED,
            ResourceProperties.VALUES,
            ResourceProperties.RELATIONS,
        ],
        extracted=[
            ExtractedDataTypeName.TEXT,
            ExtractedDataTypeName.METADATA,
            ExtractedDataTypeName.FILE,
            ExtractedDataTypeName.LINK,
        ],
    )

    return nucliadb_driver.find_raw(find_request)


def query_lexical(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
) -> Awaitable[KnowledgeboxFindResults]:
    """Lexical search (aka keyword search) in NucliaDB based on the provided analysis."""

    find_request = FindRequest(
        features=[FindOptions.KEYWORD],
        query=analysis.lexical_query,
        filters=analysis.labels,
        keyword_filters=analysis.keywords_filter,
        security=RequestSecurity(groups=config.security_groups),
        reranker=RerankerName.NOOP,
        show=[
            ResourceProperties.BASIC,
            ResourceProperties.ORIGIN,
            ResourceProperties.EXTRA,
            ResourceProperties.EXTRACTED,
            ResourceProperties.VALUES,
            ResourceProperties.RELATIONS,
        ],
        extracted=[
            ExtractedDataTypeName.TEXT,
            ExtractedDataTypeName.METADATA,
            ExtractedDataTypeName.FILE,
            ExtractedDataTypeName.LINK,
        ],
    )

    return nucliadb_driver.find_raw(find_request)


async def query_ndb(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
) -> SearchResults:
    semantic_paragraphs = query_semantic(
        memory, manager, analysis, nucliadb_driver, config
    )
    lexical_paragraphs = query_lexical(
        memory, manager, analysis, nucliadb_driver, config
    )
    nodes = cast(list[GraphPathQuery], get_nodes(analysis))
    relations = cast(list[GraphPathQuery], get_relations(analysis))

    if len(relations) > 0 and len(nodes) > 0:
        relations_paragraphs: Awaitable[Optional[KnowledgeboxFindResults]] = (
            query_relations(
                memory,
                manager,
                analysis,
                nucliadb_driver,
                config,
                relations=cast(list[GraphPathQuery], relations),
                nodes=cast(list[GraphPathQuery], nodes),
            )
        )

    else:
        relations_paragraphs = empty()

    (
        lexical_find_results,
        semantic_find_results,
        graph_find_results,
    ) = await gather(
        lexical_paragraphs,
        semantic_paragraphs,
        relations_paragraphs,
    )

    if (
        len(lexical_find_results.resources) == 0
        and analysis.keywords_filter is not None
        and len(analysis.keywords_filter) > 0
    ):
        # XXX: This is repeating the lexical query with the same parameters no?

        # If no lexical results, we can try to find some using the keywords filter
        lexical_find_results = await query_lexical(
            memory, manager, analysis, nucliadb_driver, config
        )

    result = SearchResults(
        lexical_find_results=lexical_find_results,
        semantic_find_results=semantic_find_results,
        graph_find_results=graph_find_results,
        graph_results=None,
        nodes_results=None,
        relations_results=None,
    )
    return result


async def standard_query_ndb(
    memory: QuestionMemory,
    manager: Manager,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
    question: str,
) -> SearchResults:
    find_request = FindRequest(
        features=[FindOptions.SEMANTIC, FindOptions.KEYWORD],
        query=question,
        security=RequestSecurity(groups=config.security_groups),
        reranker=RerankerName.NOOP,
        show=[
            ResourceProperties.BASIC,
            ResourceProperties.ORIGIN,
            ResourceProperties.EXTRA,
            ResourceProperties.EXTRACTED,
            ResourceProperties.VALUES,
            ResourceProperties.RELATIONS,
        ],
        extracted=[
            ExtractedDataTypeName.TEXT,
            ExtractedDataTypeName.METADATA,
            ExtractedDataTypeName.FILE,
            ExtractedDataTypeName.LINK,
        ],
        filters=nucliadb_driver.config.filters,
    )

    paragraphs = await nucliadb_driver.find_raw(find_request)

    result = SearchResults(
        lexical_find_results=None,
        semantic_find_results=paragraphs,
        graph_find_results=None,
        graph_results=None,
        nodes_results=None,
        relations_results=None,
    )
    return result
