from dataclasses import dataclass
from typing import Any, List, Optional

from hyperforge.models import FieldTypes
from nuclia.lib.nua_responses import SemanticConfig
from nuclia_models.worker.triggers import Relation
from nucliadb_models.graph.responses import (
    GraphNodesSearchResponse,
    GraphRelationsSearchResponse,
    GraphSearchResponse,
)
from nucliadb_models.search import Filter, KnowledgeboxFindResults, KnowledgeGraphEntity


@dataclass
class NDBChunk:
    chunk_id: str
    text: str
    page_with_visual: bool
    reference: Optional[str]
    start: Optional[int]
    end: Optional[int]
    page: Optional[int]
    labels: List[str]
    field: Optional[FieldTypes]
    resource_labels: List[str]
    link: Optional[str]


@dataclass
class Analysis:
    semantic_query: str
    lexical_query: str
    visual: bool
    keywords_filter: List[str]
    reason: Optional[str]
    input_tokens: Optional[float]
    output_tokens: Optional[float]
    labels: List[Filter]
    semantic_model: Optional[str]
    entities: List[KnowledgeGraphEntity]
    pre_queries: List[str]
    relations: List[Relation]
    new_text_fields: List[str]
    new_json_fields: dict[str, Any]
    semantic_config: SemanticConfig
    top_k: int
    link: bool
    knowledge_graph: Optional[str]


@dataclass
class SearchResults:
    lexical_find_results: Optional[KnowledgeboxFindResults]
    semantic_find_results: Optional[KnowledgeboxFindResults]
    graph_find_results: Optional[KnowledgeboxFindResults]
    graph_results: Optional[GraphSearchResponse]
    nodes_results: Optional[GraphNodesSearchResponse]
    relations_results: Optional[GraphRelationsSearchResponse]
