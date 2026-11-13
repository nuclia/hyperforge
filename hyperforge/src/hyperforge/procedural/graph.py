"""Immutable procedural graphs with explicit cycle policies and atomic edits.

Cycles are never repaired: ``allow`` permits cycles with a path to a terminal,
while ``reject`` requires an acyclic graph.
"""

import hashlib
import json
from collections import deque
from collections.abc import Iterable
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ProcedureNode(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    type: Literal["ACTION", "REASONING", "STATUS"]
    description: str = ""
    procedure_name: str | None = None

    @field_validator("id", "procedure_name")
    @classmethod
    def validate_nonblank_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Identifiers and procedure names must not be blank")
        return value


class ProcedureEdge(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str
    relation: Literal["LEADS_TO", "TRIGGERS", "PROVIDES_INPUT_FOR", "CONVERGES_TO"] = (
        "LEADS_TO"
    )
    condition: str | None = None
    guidance: str = Field(min_length=1)
    pitfalls: str = ""


class ProceduralGraph(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    nodes: tuple[ProcedureNode, ...] = Field(max_length=1000)
    edges: tuple[ProcedureEdge, ...] = Field(max_length=10000)
    cycle_policy: Literal["allow", "reject"] = "allow"

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        ids = {node.id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("Duplicate node ids")
        if "Start" not in ids:
            raise ValueError("Graph must contain Start")
        if any(node.id == "Start" and node.type != "STATUS" for node in self.nodes):
            raise ValueError("Start must be a STATUS node")
        bindings = [
            node.procedure_name or node.id
            for node in self.nodes
            if node.type == "ACTION"
        ]
        if "Start" in bindings:
            raise ValueError(
                "Start is reserved and cannot be an ACTION procedure binding"
            )
        if len(set(bindings)) != len(bindings):
            raise ValueError("Duplicate ACTION procedure bindings")

        outgoing: dict[str, list[str]] = {node_id: [] for node_id in ids}
        incoming: dict[str, list[str]] = {node_id: [] for node_id in ids}
        triplets = set()
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids:
                raise ValueError(
                    f"Unknown edge endpoint: {edge.source!r} -> {edge.target!r}"
                )
            triplet = (edge.source, edge.target, edge.relation)
            if triplet in triplets:
                raise ValueError(f"Duplicate edge triplet: {triplet!r}")
            triplets.add(triplet)
            outgoing[edge.source].append(edge.target)
            incoming[edge.target].append(edge.source)

        # Reverse traversal from all sinks also detects closed cyclic components.
        reaches_terminal = {node_id for node_id in ids if not outgoing[node_id]}
        pending = deque(reaches_terminal)
        while pending:
            for source in incoming[pending.popleft()]:
                if source not in reaches_terminal:
                    reaches_terminal.add(source)
                    pending.append(source)
        if reaches_terminal != ids:
            raise ValueError(
                "Every node must reach a zero-outdegree terminal; unreachable: "
                f"{sorted(ids - reaches_terminal)!r}"
            )

        if self.cycle_policy == "reject":
            indegree = {node_id: len(incoming[node_id]) for node_id in ids}
            pending = deque(node_id for node_id in ids if indegree[node_id] == 0)
            visited = 0
            while pending:
                visited += 1
                for target in outgoing[pending.popleft()]:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        pending.append(target)
            if visited != len(ids):
                raise ValueError("Cycles are forbidden by cycle_policy='reject'")
        return self

    def check_tools(self, available: Iterable[str]) -> None:
        """Require each exact ACTION binding to name an available tool."""
        available = set(available)
        unknown = {
            node.procedure_name or node.id
            for node in self.nodes
            if node.type == "ACTION"
        } - available
        if unknown:
            raise ValueError(f"Unknown ACTION procedure bindings: {sorted(unknown)!r}")

    @property
    def fingerprint(self) -> str:
        """SHA256 of canonical graph JSON, independent of node/edge input order."""
        data = self.model_dump(mode="json")
        data["nodes"].sort(key=lambda node: node["id"])
        data["edges"].sort(
            key=lambda edge: (edge["source"], edge["target"], edge["relation"])
        )
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def locate(self, procedure: str) -> str | None:
        """Resolve Start or an exact ACTION binding, never a fuzzy node match."""
        if procedure == "Start":
            return "Start"
        for node in self.nodes:
            if node.type == "ACTION" and (node.procedure_name or node.id) == procedure:
                return node.id
        return None

    def neighborhood(self, procedure: str, hops: int = 2) -> dict:
        """Include edges traversed within ``hops`` outgoing steps, or all on miss.

        The active node is included even at zero hops or when it is a terminal.
        Edges leaving nodes at the hop boundary are not expanded.
        """
        if hops < 0:
            raise ValueError("hops must be nonnegative")
        active = self.locate(procedure)
        nodes = self.nodes
        edges = self.edges
        if active is not None:
            outgoing: dict[str, list[ProcedureEdge]] = {}
            for edge in self.edges:
                outgoing.setdefault(edge.source, []).append(edge)
            seen = {active}
            pending = deque([(active, 0)])
            selected_edges = []
            while pending:
                source, depth = pending.popleft()
                if depth >= hops:
                    continue
                for edge in outgoing.get(source, ()):
                    selected_edges.append(edge)
                    if edge.target not in seen:
                        seen.add(edge.target)
                        pending.append((edge.target, depth + 1))
            nodes = tuple(node for node in self.nodes if node.id in seen)
            edges = tuple(selected_edges)
        return {
            "active_node": active,
            "scope": "local" if active is not None else "full",
            "nodes": [node.model_dump(mode="json") for node in nodes],
            "edges": [edge.model_dump(mode="json") for edge in edges],
        }

    @classmethod
    def skeleton(cls) -> Self:
        """Create a minimal graph with generic guidance and no tool bindings."""
        return cls(
            nodes=(
                ProcedureNode(id="Start", type="STATUS"),
                ProcedureNode(id="End", type="STATUS"),
            ),
            edges=(
                ProcedureEdge(
                    source="Start",
                    target="End",
                    guidance="Complete the requested task, then summarize the outcome.",
                ),
            ),
        )


class EdgeEndpoints(BaseModel):
    """An edge deletion matches every relation between these directed endpoints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    target: str


class GraphEdits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    add_nodes: tuple[ProcedureNode, ...] = ()
    delete_nodes: tuple[str, ...] = ()
    add_edges: tuple[ProcedureEdge, ...] = ()
    delete_edges: tuple[EdgeEndpoints, ...] = ()


def prepare_candidate(
    graph: ProceduralGraph,
    edits: GraphEdits,
    available_tools: Iterable[str] | None = None,
) -> ProceduralGraph:
    """Apply deletions then additions to a new validated graph, without repair.

    Unknown or repeated deletions and duplicate additions are errors. Revise
    attributes by explicitly deleting and readding a node or edge. Deleting a
    node also removes its incident edges. The original graph is never changed.
    """
    nodes = {node.id: node for node in graph.nodes}
    edges = {(edge.source, edge.target, edge.relation): edge for edge in graph.edges}
    endpoints = {(edge.source, edge.target) for edge in graph.edges}
    deleted_endpoints = set()
    for deletion in edits.delete_edges:
        pair = (deletion.source, deletion.target)
        if pair in deleted_endpoints:
            raise ValueError(f"Duplicate edge deletion: {pair!r}")
        if pair not in endpoints:
            raise ValueError(f"Unknown edge deletion: {pair!r}")
        deleted_endpoints.add(pair)
    edges = {
        key: edge for key, edge in edges.items() if key[:2] not in deleted_endpoints
    }
    deleted_nodes = set()
    for node_id in edits.delete_nodes:
        if node_id in deleted_nodes:
            raise ValueError(f"Duplicate node deletion: {node_id!r}")
        if node_id not in nodes:
            raise ValueError(f"Unknown node deletion: {node_id!r}")
        deleted_nodes.add(node_id)
        del nodes[node_id]
    edges = {
        key: edge
        for key, edge in edges.items()
        if edge.source not in deleted_nodes and edge.target not in deleted_nodes
    }
    for node in edits.add_nodes:
        if node.id in nodes:
            raise ValueError(f"Duplicate node addition: {node.id!r}")
        nodes[node.id] = node
    for edge in edits.add_edges:
        key = (edge.source, edge.target, edge.relation)
        if key in edges:
            raise ValueError(f"Duplicate edge addition: {key!r}")
        edges[key] = edge
    candidate = ProceduralGraph(
        schema_version=graph.schema_version,
        nodes=tuple(nodes.values()),
        edges=tuple(edges.values()),
        cycle_policy=graph.cycle_policy,
    )
    if available_tools is not None:
        candidate.check_tools(available_tools)
    return candidate
