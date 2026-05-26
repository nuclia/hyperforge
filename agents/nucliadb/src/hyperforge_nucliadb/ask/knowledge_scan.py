from asyncio import gather
from typing import Awaitable, List, Optional, cast

from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory
from nucliadb_models import RequestSecurity
from nucliadb_models.filters import And, Or
from nucliadb_models.graph.requests import (
    GraphNodesQuery,
    GraphNodesSearchRequest,
    GraphPathQuery,
    GraphRelationsQuery,
    GraphRelationsSearchRequest,
    GraphSearchRequest,
)
from nucliadb_models.graph.responses import (
    GraphNodesSearchResponse,
    GraphRelationsSearchResponse,
    GraphSearchResponse,
)

from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_nucliadb.ask.models import Analysis, SearchResults
from hyperforge_nucliadb.ask.utils import (
    empty,
    get_nodes,
    get_relations,
)
from hyperforge_nucliadb.driver import NucliaDBDriver


def query_graph_search(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
    relations: List[GraphPathQuery],
    nodes: List[GraphPathQuery],
) -> Awaitable[GraphSearchResponse]:
    """Query relations in NucliaDB based on the provided analysis."""

    find_request = GraphSearchRequest(
        top_k=50,
        security=RequestSecurity(groups=config.security_groups),
        query=And(
            operands=[
                Or(operands=relations),
                Or(operands=nodes),
            ]
        ),
    )
    return nucliadb_driver.graph_search(find_request)


def query_graph_nodes(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
    nodes: List[GraphNodesQuery],
) -> Awaitable[GraphNodesSearchResponse]:
    """Query relations in NucliaDB based on the provided analysis."""

    find_request = GraphNodesSearchRequest(
        top_k=50,
        security=RequestSecurity(groups=config.security_groups),
        query=Or(operands=nodes),
    )

    return nucliadb_driver.graph_nodes(find_request)


def query_graph_relations(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
    relations: List[GraphRelationsQuery],
) -> Awaitable[GraphRelationsSearchResponse]:
    """Query relations in NucliaDB based on the provided analysis."""
    find_request = GraphRelationsSearchRequest(
        top_k=50,
        security=RequestSecurity(groups=config.security_groups),
        query=Or(operands=relations),
    )
    return nucliadb_driver.graph_relations(find_request)


async def knowledge_scan(
    memory: QuestionMemory,
    manager: Manager,
    analysis: Analysis,
    nucliadb_driver: NucliaDBDriver,
    config: AskAgentConfig,
) -> SearchResults:
    nodes = cast(list[GraphPathQuery], get_nodes(analysis))
    relations = cast(list[GraphPathQuery], get_relations(analysis))

    graph: Awaitable[Optional[GraphSearchResponse]] = query_graph_search(
        memory,
        manager,
        analysis,
        nucliadb_driver,
        config=config,
        relations=relations,
        nodes=nodes,
    )
    if len(nodes) > 0:
        nodes_search: Awaitable[Optional[GraphNodesSearchResponse]] = query_graph_nodes(
            memory,
            manager,
            analysis,
            nucliadb_driver,
            config=config,
            nodes=cast(list[GraphNodesQuery], nodes),
        )
    else:
        nodes_search = empty()

    if len(relations) > 0:
        relations_search: Awaitable[Optional[GraphRelationsSearchResponse]] = (
            query_graph_relations(
                memory,
                manager,
                analysis,
                nucliadb_driver,
                config=config,
                relations=cast(list[GraphRelationsQuery], relations),
            )
        )
    else:
        relations_search = empty()

    nodes_results, relations_results, graph_results = await gather(
        nodes_search,
        relations_search,
        graph,
    )

    return SearchResults(
        lexical_find_results=None,
        semantic_find_results=None,
        graph_find_results=None,
        graph_results=graph_results,
        nodes_results=nodes_results,
        relations_results=relations_results,
    )
