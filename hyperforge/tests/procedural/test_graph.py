import hashlib
import json

import pytest
from pydantic import ValidationError

from hyperforge.procedural.graph import (
    EdgeEndpoints,
    GraphEdits,
    ProceduralGraph,
    ProcedureEdge,
    ProcedureNode,
    prepare_candidate,
)


def edge(source, target, **kwargs):
    return ProcedureEdge(source=source, target=target, guidance="Continue", **kwargs)


@pytest.fixture
def graph():
    return ProceduralGraph(
        nodes=(
            ProcedureNode(id="Start", type="STATUS"),
            ProcedureNode(id="Search", type="ACTION", procedure_name="search"),
            ProcedureNode(id="Think", type="REASONING"),
            ProcedureNode(id="answer", type="ACTION"),
            ProcedureNode(id="End", type="STATUS"),
            ProcedureNode(id="Other", type="STATUS"),
        ),
        edges=(
            edge("Start", "Search"),
            edge(
                "Search",
                "Think",
                relation="TRIGGERS",
                condition="found",
                pitfalls="Do not guess",
            ),
            edge("Think", "answer", relation="PROVIDES_INPUT_FOR"),
            edge("answer", "End", relation="CONVERGES_TO"),
            edge("Other", "Search"),
        ),
    )


def test_skeleton():
    graph = ProceduralGraph.skeleton()
    assert [(node.id, node.type) for node in graph.nodes] == [
        ("Start", "STATUS"),
        ("End", "STATUS"),
    ]
    assert len(graph.edges) == 1
    assert graph.edges[0].source == "Start"
    assert graph.edges[0].target == "End"
    assert graph.edges[0].guidance
    assert graph.schema_version == 1
    assert graph.cycle_policy == "allow"
    graph.check_tools(())


@pytest.mark.parametrize(
    "model,data",
    [
        (ProcedureNode, {"id": "", "type": "STATUS"}),
        (ProcedureNode, {"id": "x", "type": "OTHER"}),
        (ProcedureNode, {"id": "x", "type": "STATUS", "extra": True}),
        (ProcedureEdge, {"source": "a", "target": "b", "guidance": ""}),
        (ProcedureEdge, {"source": "a", "target": "b"}),
        (
            ProcedureEdge,
            {"source": "a", "target": "b", "guidance": "go", "relation": "OTHER"},
        ),
        (
            ProcedureEdge,
            {"source": "a", "target": "b", "guidance": "go", "extra": True},
        ),
        (EdgeEndpoints, {"source": "a", "target": "b", "relation": "LEADS_TO"}),
        (GraphEdits, {"extra": True}),
    ],
)
def test_invalid_models(model, data):
    with pytest.raises(ValidationError):
        model.model_validate(data)


def test_defaults_and_json_roundtrip(graph):
    node = ProcedureNode(id="x", type="ACTION")
    assert node.description == ""
    assert node.procedure_name is None
    assert edge("a", "b").model_dump() == {
        "source": "a",
        "target": "b",
        "relation": "LEADS_TO",
        "condition": None,
        "guidance": "Continue",
        "pitfalls": "",
    }
    assert GraphEdits().model_dump() == {
        "add_nodes": (),
        "delete_nodes": (),
        "add_edges": (),
        "delete_edges": (),
    }
    assert ProceduralGraph.model_validate_json(graph.model_dump_json()) == graph


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"nodes": ()}, "Start"),
        ({"schema_version": 2}, "schema_version"),
        ({"cycle_policy": "repair"}, "cycle_policy"),
        ({"extra": True}, "Extra inputs"),
        ({"edges": (edge("Missing", "End"),)}, "Unknown edge endpoint"),
        ({"edges": (edge("Start", "Missing"),)}, "Unknown edge endpoint"),
        (
            {
                "edges": (
                    edge("Start", "End"),
                    edge("Start", "End", condition="different"),
                )
            },
            "Duplicate edge triplet",
        ),
    ],
)
def test_invalid_graph(changes, match):
    data = ProceduralGraph.skeleton().model_dump()
    data.update(changes)
    with pytest.raises(ValidationError, match=match):
        ProceduralGraph.model_validate(data)


def test_duplicate_node_ids(graph):
    with pytest.raises(ValidationError, match="Duplicate node ids"):
        ProceduralGraph(nodes=(*graph.nodes, graph.nodes[0]), edges=graph.edges)


def test_unique_action_bindings(graph):
    with pytest.raises(ValidationError, match="Duplicate ACTION procedure bindings"):
        ProceduralGraph(
            nodes=(*graph.nodes, ProcedureNode(id="search", type="ACTION")),
            edges=graph.edges,
        )
    # Bindings are exact, not case-normalized, and non-ACTION metadata is ignored.
    other = ProceduralGraph(
        nodes=(
            *graph.nodes,
            ProcedureNode(id="SEARCH", type="ACTION"),
            ProcedureNode(id="metadata", type="STATUS", procedure_name="search"),
        ),
        edges=graph.edges,
    )
    other.check_tools(iter(["search", "SEARCH", "answer"]))


@pytest.mark.parametrize(
    "procedure,expected",
    [
        ("Start", "Start"),
        ("search", "Search"),
        ("answer", "answer"),
        ("Search", None),
        ("SEARCH", None),
        (" search", None),
        ("search()", None),
        ("Think", None),
        ("End", None),
        ("missing", None),
    ],
)
def test_locate_exact_bindings(graph, procedure, expected):
    assert graph.locate(procedure) == expected


@pytest.mark.parametrize("node_type", ["ACTION", "REASONING"])
def test_start_must_be_status(node_type):
    with pytest.raises(ValidationError, match="Start must be a STATUS node"):
        ProceduralGraph(nodes=(ProcedureNode(id="Start", type=node_type),), edges=())


def test_start_binding_is_reserved():
    graph = ProceduralGraph.skeleton()
    alias = ProcedureNode(id="alias", type="ACTION", procedure_name="Start")
    with pytest.raises(ValidationError, match="Start is reserved"):
        ProceduralGraph(nodes=(*graph.nodes, alias), edges=graph.edges)
    with pytest.raises(ValueError, match="Start is reserved"):
        prepare_candidate(graph, GraphEdits(add_nodes=(alias,)))
    assert graph == ProceduralGraph.skeleton()
    assert graph.locate("Start") == "Start"


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
@pytest.mark.parametrize("field", ["id", "procedure_name"])
@pytest.mark.parametrize("node_type", ["ACTION", "REASONING", "STATUS"])
def test_node_names_must_not_be_blank(value, field, node_type):
    data = {"id": "node", "type": node_type, field: value}
    with pytest.raises(ValidationError):
        ProcedureNode.model_validate(data)


def test_nonblank_names_are_preserved_exactly_and_none_binding_falls_back():
    graph = ProceduralGraph(
        nodes=(
            ProcedureNode(id="Start", type="STATUS"),
            ProcedureNode(id=" alias ", type="ACTION", procedure_name=" search "),
            ProcedureNode(id="fallback ", type="ACTION", procedure_name=None),
        ),
        edges=(),
    )
    assert graph.locate(" search ") == " alias "
    assert graph.locate("search") is None
    assert graph.locate("fallback ") == "fallback "
    assert graph.locate("fallback") is None
    graph.check_tools([" search ", "fallback "])
    with pytest.raises(ValueError, match="Unknown ACTION"):
        graph.check_tools(["search", "fallback"])


def test_check_tools(graph):
    assert graph.check_tools(iter(["search", "answer", "unused"])) is None
    with pytest.raises(ValueError, match="Unknown ACTION.*search"):
        graph.check_tools(["Search", "answer"])
    with pytest.raises(ValueError, match="Unknown ACTION"):
        graph.check_tools([])


def test_directed_two_hop_neighborhood(graph):
    result = graph.neighborhood("search")
    assert result["active_node"] == "Search"
    assert result["scope"] == "local"
    assert {node["id"] for node in result["nodes"]} == {"Search", "Think", "answer"}
    assert result["edges"] == [graph.edges[1].model_dump(), graph.edges[2].model_dump()]
    assert json.loads(json.dumps(result)) == result
    assert set(result) == {"active_node", "scope", "nodes", "edges"}


def test_hop_boundaries_and_fallback(graph):
    zero = graph.neighborhood("search", hops=0)
    assert [node["id"] for node in zero["nodes"]] == ["Search"]
    assert zero["edges"] == []
    one = graph.neighborhood("search", hops=1)
    assert {node["id"] for node in one["nodes"]} == {"Search", "Think"}
    assert len(one["edges"]) == 1
    full = graph.neighborhood("Search", hops=0)
    assert full == {
        "active_node": None,
        "scope": "full",
        "nodes": graph.model_dump(mode="json")["nodes"],
        "edges": graph.model_dump(mode="json")["edges"],
    }
    with pytest.raises(ValueError, match="hops"):
        graph.neighborhood("search", hops=-1)


def test_sink_neighborhood():
    graph = ProceduralGraph(nodes=(ProcedureNode(id="Start", type="STATUS"),), edges=())
    assert graph.neighborhood("Start") == {
        "active_node": "Start",
        "scope": "local",
        "nodes": [graph.nodes[0].model_dump()],
        "edges": [],
    }


@pytest.mark.parametrize("policy", ["allow", "reject"])
def test_closed_cycle_rejected_even_with_unrelated_terminal(policy):
    with pytest.raises(ValidationError, match="zero-outdegree terminal"):
        ProceduralGraph(
            nodes=ProceduralGraph.skeleton().nodes,
            edges=(edge("Start", "Start"),),
            cycle_policy=policy,
        )


def test_cycle_with_exit_and_explicit_rejection(graph):
    cyclic_edges = (*graph.edges, edge("Think", "Search"))
    cyclic = ProceduralGraph(nodes=graph.nodes, edges=cyclic_edges)
    result = cyclic.neighborhood("search", hops=10000)
    assert len(result["edges"]) == 4
    assert {node["id"] for node in result["nodes"]} == {
        "Search",
        "Think",
        "answer",
        "End",
    }
    assert edge("Think", "Search").model_dump() in result["edges"]
    with pytest.raises(ValidationError, match="Cycles are forbidden"):
        ProceduralGraph(nodes=graph.nodes, edges=cyclic_edges, cycle_policy="reject")


def test_self_loop_with_exit():
    skeleton = ProceduralGraph.skeleton()
    graph = ProceduralGraph(
        nodes=skeleton.nodes, edges=(*skeleton.edges, edge("Start", "Start"))
    )
    assert len(graph.neighborhood("Start", hops=1)["edges"]) == 2
    with pytest.raises(ValidationError, match="Cycles are forbidden"):
        ProceduralGraph(nodes=graph.nodes, edges=graph.edges, cycle_policy="reject")


def test_disconnected_closed_component_rejected():
    skeleton = ProceduralGraph.skeleton()
    with pytest.raises(ValidationError, match="zero-outdegree terminal"):
        ProceduralGraph(
            nodes=(*skeleton.nodes, ProcedureNode(id="isolated", type="STATUS")),
            edges=(*skeleton.edges, edge("isolated", "isolated")),
        )


def test_iterative_validation_at_node_limit():
    nodes = (ProcedureNode(id="Start", type="STATUS"),) + tuple(
        ProcedureNode(id=str(index), type="STATUS") for index in range(999)
    )
    edges = tuple(
        edge(source.id, target.id) for source, target in zip(nodes, nodes[1:])
    )
    graph = ProceduralGraph(nodes=nodes, edges=edges, cycle_policy="reject")
    assert len(graph.neighborhood("Start", hops=1000)["nodes"]) == 1000
    with pytest.raises(ValidationError, match="1000"):
        ProceduralGraph(
            nodes=(*nodes, ProcedureNode(id="overflow", type="STATUS")), edges=edges
        )


def test_edge_limit():
    nodes = (ProcedureNode(id="Start", type="STATUS"),) + tuple(
        ProcedureNode(id=str(index), type="STATUS") for index in range(143)
    )
    edges = tuple(
        edge(source.id, target.id)
        for index, source in enumerate(nodes)
        for target in nodes[index + 1 :]
    )
    assert (
        len(
            ProceduralGraph(
                nodes=nodes, edges=edges[:10000], cycle_policy="reject"
            ).edges
        )
        == 10000
    )
    with pytest.raises(ValidationError, match="10000"):
        ProceduralGraph(nodes=nodes, edges=edges[:10001])


def test_models_are_frozen(graph):
    edits = GraphEdits(
        add_nodes=graph.nodes,
        add_edges=graph.edges,
        delete_edges=(EdgeEndpoints(source="Start", target="Search"),),
    )
    for model, field, value in [
        (graph, "nodes", ()),
        (graph.nodes[0], "description", "changed"),
        (graph.edges[0], "guidance", "changed"),
        (edits, "add_nodes", ()),
        (edits.delete_edges[0], "target", "changed"),
    ]:
        with pytest.raises(ValidationError, match="frozen"):
            setattr(model, field, value)
    assert isinstance(graph.nodes, tuple)
    assert isinstance(graph.edges, tuple)
    with pytest.raises(TypeError):
        graph.nodes[0] = graph.nodes[1]
    result = graph.neighborhood("Start")
    result["nodes"][0]["description"] = "changed"
    assert graph.nodes[0].description == ""


def test_fingerprint_canonical_and_order_independent(graph):
    reordered = ProceduralGraph(
        nodes=tuple(reversed(graph.nodes)), edges=tuple(reversed(graph.edges))
    )
    assert reordered.fingerprint == graph.fingerprint
    data = graph.model_dump(mode="json")
    data["nodes"].sort(key=lambda node: node["id"])
    data["edges"].sort(
        key=lambda edge: (edge["source"], edge["target"], edge["relation"])
    )
    assert (
        graph.fingerprint
        == hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert (
        ProceduralGraph.model_validate_json(graph.model_dump_json()).fingerprint
        == graph.fingerprint
    )


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("nodes", "description", "new description"),
        ("edges", "guidance", "new guidance"),
        ("edges", "condition", "new condition"),
        ("edges", "pitfalls", "new pitfalls"),
        ("edges", "relation", "TRIGGERS"),
    ],
)
def test_fingerprint_includes_attributes(graph, section, field, value):
    data = graph.model_dump(mode="json")
    data[section][0][field] = value
    assert ProceduralGraph.model_validate(data).fingerprint != graph.fingerprint
    assert (
        ProceduralGraph(
            nodes=graph.nodes, edges=graph.edges, cycle_policy="reject"
        ).fingerprint
        != graph.fingerprint
    )


def test_multirelation_edges_and_endpoint_deletion():
    skeleton = ProceduralGraph.skeleton()
    graph = ProceduralGraph(
        nodes=skeleton.nodes,
        edges=(
            *skeleton.edges,
            edge("Start", "End", relation="TRIGGERS"),
            edge("Start", "End", relation="PROVIDES_INPUT_FOR"),
            edge("Start", "End", relation="CONVERGES_TO"),
        ),
        cycle_policy="reject",
    )
    assert len(graph.neighborhood("Start")["edges"]) == 4
    candidate = prepare_candidate(
        graph,
        GraphEdits(
            delete_edges=(EdgeEndpoints(source="Start", target="End"),),
            add_edges=(edge("Start", "End", relation="TRIGGERS", condition="revised"),),
        ),
    )
    assert len(candidate.edges) == 1
    assert candidate.edges[0].condition == "revised"
    assert candidate.cycle_policy == "reject"
    assert len(graph.edges) == 4


def test_delete_readd_node_and_edge_attributes(graph):
    original = graph.model_dump_json()
    candidate = prepare_candidate(
        graph,
        GraphEdits(
            delete_edges=(EdgeEndpoints(source="Search", target="Think"),),
            delete_nodes=("Search",),
            add_nodes=(
                ProcedureNode(
                    id="Search",
                    type="ACTION",
                    procedure_name="new_search",
                    description="Revised",
                ),
            ),
            add_edges=(
                edge("Start", "Search"),
                edge("Search", "End", pitfalls="Check output"),
            ),
        ),
        available_tools=iter(["new_search", "answer"]),
    )
    assert candidate.locate("new_search") == "Search"
    assert candidate.locate("search") is None
    assert not any(edge.source == "Other" for edge in candidate.edges)
    assert candidate.nodes[-1].description == "Revised"
    assert candidate.edges[-1].pitfalls == "Check output"
    assert graph.model_dump_json() == original
    assert candidate.fingerprint != graph.fingerprint


@pytest.mark.parametrize(
    "edits,match",
    [
        (GraphEdits(delete_nodes=("missing",)), "Unknown node deletion"),
        (GraphEdits(delete_nodes=("End", "End")), "Duplicate node deletion"),
        (
            GraphEdits(delete_edges=(EdgeEndpoints(source="End", target="Start"),)),
            "Unknown edge deletion",
        ),
        (
            GraphEdits(delete_edges=(EdgeEndpoints(source="Start", target="End"),) * 2),
            "Duplicate edge deletion",
        ),
        (
            GraphEdits(add_nodes=(ProcedureNode(id="End", type="STATUS"),)),
            "Duplicate node addition",
        ),
        (
            GraphEdits(add_nodes=(ProcedureNode(id="new", type="STATUS"),) * 2),
            "Duplicate node addition",
        ),
        (
            GraphEdits(add_edges=(edge("Start", "End", condition="revised"),)),
            "Duplicate edge addition",
        ),
        (GraphEdits(add_edges=(edge("End", "Start"),) * 2), "Duplicate edge addition"),
        (GraphEdits(add_edges=(edge("End", "missing"),)), "Unknown edge endpoint"),
        (GraphEdits(delete_nodes=("Start",)), "Start"),
        (GraphEdits(add_edges=(edge("End", "Start"),)), "zero-outdegree terminal"),
    ],
)
def test_failed_edits_leave_original_unchanged(edits, match):
    graph = ProceduralGraph.skeleton()
    original = graph.model_dump_json()
    fingerprint = graph.fingerprint
    with pytest.raises(ValueError, match=match):
        prepare_candidate(graph, edits)
    assert graph.model_dump_json() == original
    assert graph.fingerprint == fingerprint


def test_candidate_tool_validation_is_optional_and_atomic():
    graph = ProceduralGraph.skeleton()
    edits = GraphEdits(add_nodes=(ProcedureNode(id="unknown", type="ACTION"),))
    candidate = prepare_candidate(graph, edits)
    assert candidate.locate("unknown") == "unknown"
    assert prepare_candidate(graph, edits, iter(["unknown"])) == candidate
    with pytest.raises(ValueError, match="Unknown ACTION"):
        prepare_candidate(graph, edits, [])
    assert graph == ProceduralGraph.skeleton()
    assert prepare_candidate(graph, GraphEdits()) == graph
    assert prepare_candidate(graph, GraphEdits()) is not graph


def test_edits_preserve_cycle_policy_without_repair():
    skeleton = ProceduralGraph.skeleton()
    edits = GraphEdits(add_edges=(edge("Start", "Start"),))
    assert len(prepare_candidate(skeleton, edits).edges) == 2
    strict = ProceduralGraph(
        nodes=skeleton.nodes, edges=skeleton.edges, cycle_policy="reject"
    )
    with pytest.raises(ValueError, match="Cycles are forbidden"):
        prepare_candidate(strict, edits)
    assert strict.edges == skeleton.edges
