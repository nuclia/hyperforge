from collections.abc import Sequence
from typing import List

from nucliadb_models.graph.requests import (
    AnyNode,
    NodeMatchKindName,
    Relation,
)

from hyperforge_nucliadb.ask.models import Analysis


async def empty() -> None:
    """A placeholder to call."""
    pass


def get_nodes(analysis: Analysis) -> List[AnyNode]:
    """Extract nodes from the analysis."""
    nodes: List[AnyNode] = []
    for entity in analysis.entities:
        if entity.type == "PERSON":
            nodes.append(AnyNode(value=entity.name, match=NodeMatchKindName.FUZZY))
        elif entity.type == "LOCATION" or entity.type == "ORGANIZATION":
            nodes.append(AnyNode(value=entity.name, match=NodeMatchKindName.EXACT))
        else:
            nodes.append(AnyNode(value=entity.name, match=NodeMatchKindName.FUZZY))
    return nodes


def get_relations(analysis: Analysis) -> Sequence[Relation]:
    """Extract relations from the analysis."""
    relations: List[Relation] = []
    for relation in analysis.relations:
        relations.append(Relation(label=relation.label))
    return relations
