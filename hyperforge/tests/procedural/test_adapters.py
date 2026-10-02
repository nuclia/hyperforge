import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from hyperforge.procedural.adapters import HarnessRollout, ManagerRefiner
from hyperforge.procedural.evolution import EvaluationTask
from hyperforge.procedural.graph import GraphEdits, ProceduralGraph
from hyperforge.procedural.guidance import ProceduralGuidanceConfig


@pytest.mark.asyncio
async def test_manager_refiner_schema_prompt_and_configuration():
    manager = SimpleNamespace(execute_json=AsyncMock(return_value=({}, 10, 20)))
    model = object()
    tracking = object()
    refiner = ManagerRefiner(
        manager,
        model=model,
        tracking=tracking,
        available_tools=["lookup"],
        task_description="Research and answer",
        user_id="offline-test",
        max_tokens=1234,
    )
    graph = ProceduralGraph.skeleton()
    rejections = [{"edits": {"delete_nodes": ["Start"]}, "reason": "proposal_error"}]
    assert await refiner(graph, "query and scored traces", rejections) == GraphEdits()
    args = manager.execute_json.call_args.kwargs
    assert args["schema"] == GraphEdits.model_json_schema()
    assert args["model"] is model
    assert args["tracking"] is tracking
    assert args["max_tokens"] == 1234
    assert args["user_id"] == "offline-test"
    prompt = json.loads(args["prompt"])
    assert prompt["graph"] == graph.model_dump(mode="json")
    assert prompt["available_tools"] == ["lookup"]
    assert prompt["task_description"] == "Research and answer"
    assert prompt["training_context"] == "query and scored traces"
    assert prompt["rejected_proposals"] == rejections
    for rule in [
        "Preserve existing node IDs",
        "exactly match",
        "generalizable",
        "conditional",
        "pitfalls",
        "ALL relations",
        "cycle_policy",
        "zero-outdegree terminal",
    ]:
        assert rule in args["system"]


@pytest.mark.asyncio
async def test_manager_refiner_rejects_malformed_and_propagates_infrastructure_errors():
    manager = SimpleNamespace(
        execute_json=AsyncMock(return_value=({"unknown": []}, 0, 0))
    )
    refiner = ManagerRefiner(
        manager, model="fake", available_tools=[], task_description="task"
    )
    with pytest.raises(ValidationError):
        await refiner(ProceduralGraph.skeleton(), "", [])
    manager.execute_json.side_effect = OSError("offline")
    with pytest.raises(OSError, match="offline"):
        await refiner(ProceduralGraph.skeleton(), "", [])


def event(kind, payload=None, *, child=False):
    return SimpleNamespace(
        type=kind,
        payload=payload or {},
        agent_id="child" if child else "root",
        parent_agent_id="root" if child else None,
    )


class FakeHarness:
    def __init__(self, events):
        self.events = events
        self.agent_id = "root"
        self.system_prompt = "Base system prompt"
        self.procedural_guidance = ProceduralGuidanceConfig(
            graph=ProceduralGraph.skeleton()
        )
        self.procedural_trajectory = ()
        self.messages = []
        self._pending_messages = []
        self.conversation = None
        self.conversation_id = "fresh-conversation"
        self.storage = SimpleNamespace(get_conversation=AsyncMock(return_value=None))
        self.usage = SimpleNamespace(
            tool_calls=1, turns=2, input_tokens=3, output_tokens=4
        )
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def run(self, query):
        self.query = query
        yield event(
            "procedural.guidance",
            {
                "graph_version": self.procedural_guidance.graph.fingerprint,
                "status": "completed",
                "text": "Public guidance",
                "decision_id": "decision1",
            },
        )
        for item in self.events:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.mark.asyncio
async def test_harness_rollout_pins_graph_pairs_root_tools_and_excludes_reasoning():
    graph = ProceduralGraph.skeleton()
    task = EvaluationTask(
        id="one", query="User objective", expected="PRIVATE GOLD", metadata={"limit": 3}
    )
    harness = FakeHarness(
        [
            event("reasoning.delta", {"text": "PRIVATE REASONING"}),
            event("message.added", {"message": "PRIVATE REASONING"}),
            event("turn.completed", {"text": "CHILD OUTPUT"}, child=True),
            event(
                "tool.requested",
                {
                    "call": {
                        "id": "call1",
                        "name": "lookup",
                        "arguments": {"query": "term"},
                    },
                    "reasoning": "PRIVATE REASONING",
                },
            ),
            event(
                "tool.completed",
                {"call_id": "call1", "tool": "lookup", "result": {"answer": 2}},
            ),
            event(
                "procedural.step",
                {
                    "graph_version": graph.fingerprint,
                    "decision_id": "decision1",
                    "call_id": "call1",
                    "procedure_name": "lookup",
                    "arguments": "{}",
                    "observation": "answer 2",
                    "status": "completed",
                },
            ),
            event("turn.completed", {"text": "Final output"}),
        ]
    )
    factory = AsyncMock(return_value=harness)
    scorer = AsyncMock(return_value=0.75)
    result = await HarnessRollout(factory, scorer)(graph, task)
    factory.assert_awaited_once_with(graph, task.query, {"limit": 3})
    assert "PRIVATE GOLD" not in repr(factory.call_args)
    assert harness.query == task.query
    assert harness.system_prompt == "Base system prompt"
    guidance = result.trajectory[0]
    assert guidance["type"] == "procedural.guidance"
    assert guidance["payload"]["graph_version"] == graph.fingerprint
    assert guidance["payload"]["text"] == "Public guidance"
    assert "graph" not in guidance["payload"]
    assert result.trajectory[1] == {
        "type": "tool",
        "call_id": "call1",
        "tool": "lookup",
        "arguments": {"query": "term"},
        "result": {"answer": 2},
        "status": "completed",
    }
    assert "PRIVATE" not in result.model_dump_json()
    assert "CHILD" not in result.model_dump_json()
    assert result.output == "Final output"
    assert result.score == 0.75
    assert result.trajectory[2]["type"] == "procedural.step"
    assert result.trajectory[3] == {
        "type": "turn.completed",
        "payload": {"text": "Final output"},
    }
    assert result.metrics["wall_time_seconds"] >= 0
    assert {
        key: value
        for key, value in result.metrics.items()
        if key != "wall_time_seconds"
    } == {
        "tool_calls": 1,
        "turns": 2,
        "input_tokens": 3,
        "output_tokens": 4,
    }
    scorer.assert_awaited_once_with(task, "Final output", result.trajectory)
    assert harness.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["turn.failed", "turn.interrupted", "llm.failed"])
async def test_harness_failure_events_never_become_zero_scores(failure):
    harness = FakeHarness([event(failure)])
    scorer = AsyncMock(return_value=0)
    with pytest.raises(RuntimeError, match="Harness rollout failed"):
        await HarnessRollout(AsyncMock(return_value=harness), scorer)(
            ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q")
        )
    scorer.assert_not_awaited()
    assert harness.closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "events, error",
    [
        ([], RuntimeError),
        ([OSError("network")], OSError),
        (
            [
                event(
                    "tool.completed",
                    {"call_id": "absent", "tool": "lookup", "result": {}},
                )
            ],
            ValueError,
        ),
        (
            [
                event(
                    "tool.requested",
                    {"call": {"id": "a", "name": "lookup", "arguments": {}}},
                ),
                event("turn.completed", {"text": "done"}),
            ],
            RuntimeError,
        ),
    ],
)
async def test_incomplete_or_broken_harness_fails(events, error):
    harness = FakeHarness(events)
    scorer = AsyncMock(return_value=0)
    with pytest.raises(error):
        await HarnessRollout(AsyncMock(return_value=harness), scorer)(
            ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q")
        )
    scorer.assert_not_awaited()
    assert harness.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [float("nan"), 1.1, -1])
async def test_harness_score_validated(score):
    harness = FakeHarness([event("turn.completed", {"text": "done"})])
    with pytest.raises(ValidationError):
        await HarnessRollout(
            AsyncMock(return_value=harness), AsyncMock(return_value=score)
        )(ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q"))


@pytest.mark.asyncio
async def test_scorer_infrastructure_error_propagates():
    harness = FakeHarness([event("turn.completed", {"text": "done"})])
    with pytest.raises(OSError, match="scorer down"):
        await HarnessRollout(
            AsyncMock(return_value=harness),
            AsyncMock(side_effect=OSError("scorer down")),
        )(ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q"))


@pytest.mark.asyncio
async def test_graph_guidance_cannot_change_during_rollout():
    class MutatingHarness(FakeHarness):
        async def run(self, query):
            self.procedural_guidance = None
            yield event("turn.completed", {"text": "done"})

    harness = MutatingHarness([])
    with pytest.raises(ValueError, match="guidance configuration"):
        await HarnessRollout(
            AsyncMock(return_value=harness), AsyncMock(return_value=1)
        )(ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q"))


@pytest.mark.asyncio
async def test_actual_agent_harness_with_fake_model():
    from hyperforge.harness_sdk import AgentHarness, ModelDelta

    graph = ProceduralGraph.skeleton()
    calls = []
    guidance_calls = []
    harnesses = []

    class Model:
        async def stream(self, **kwargs):
            calls.append(kwargs)
            yield ModelDelta(reasoning="PRIVATE REASONING")
            yield ModelDelta(text="answer", input_tokens=3, output_tokens=1)

    class GuidanceModel:
        async def stream(self, **kwargs):
            guidance_calls.append(kwargs)
            yield ModelDelta(reasoning="PRIVATE GUIDANCE REASONING")
            yield ModelDelta(
                text="Summarize your findings.", input_tokens=2, output_tokens=1
            )

    async def factory(graph, query, metadata):
        harness = AgentHarness(
            model="fake",
            model_client=Model(),
            guidance_client=GuidanceModel(),
            procedural_guidance=ProceduralGuidanceConfig(
                graph=graph, model="fake-guidance"
            ),
        )
        harnesses.append(harness)
        return harness

    scorer = AsyncMock(return_value=1)
    result = await HarnessRollout(factory, scorer)(
        graph, EvaluationTask(id="one", query="objective", expected="GOLD")
    )
    assert result.output == "answer"
    assert result.score == 1
    assert "PRIVATE" not in result.model_dump_json()
    assert "GOLD" not in repr(calls)
    assert graph.fingerprint not in repr(calls)
    assert "graph_context" not in repr(calls)
    assert "Summarize your findings." in repr(calls)
    assert "graph_context" in repr(guidance_calls)
    assert guidance_calls[0]["model"] == "fake-guidance"
    assert "GOLD" not in repr(guidance_calls)
    assert result.trajectory[0]["type"] == "procedural.guidance"
    assert result.trajectory[0]["payload"]["graph_version"] == graph.fingerprint
    assert result.metrics["wall_time_seconds"] > 0
    assert "Summarize your findings." not in harnesses[0].system_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid", ["disabled", "wrong_graph", "conversation", "messages", "trajectory"]
)
async def test_factory_requires_fresh_matching_guided_harness(invalid):
    harness = FakeHarness([])
    if invalid == "disabled":
        harness.procedural_guidance = None
    elif invalid == "wrong_graph":
        graph = ProceduralGraph.skeleton().model_copy(update={"cycle_policy": "reject"})
        harness.procedural_guidance = ProceduralGuidanceConfig(graph=graph)
    elif invalid == "conversation":
        harness.conversation = object()
    elif invalid == "messages":
        harness.messages = [object()]
    else:
        harness.procedural_trajectory = (object(),)
    scorer = AsyncMock(return_value=1)
    with pytest.raises(ValueError):
        await HarnessRollout(AsyncMock(return_value=harness), scorer)(
            ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q")
        )
    scorer.assert_not_awaited()
    assert not hasattr(harness, "query")
    assert harness.closed


@pytest.mark.asyncio
async def test_failed_tool_is_paired_and_solver_can_recover():
    graph = ProceduralGraph.skeleton()
    harness = FakeHarness(
        [
            event(
                "tool.requested",
                {"call": {"id": "failed-call", "name": "lookup", "arguments": {}}},
            ),
            event(
                "tool.failed",
                {
                    "call_id": "failed-call",
                    "tool": "lookup",
                    "result": {"error": "not found"},
                },
            ),
            event(
                "procedural.step",
                {
                    "graph_version": graph.fingerprint,
                    "call_id": "failed-call",
                    "status": "failed",
                },
            ),
            event(
                "tool.requested",
                {
                    "call": {
                        "id": "retry-call",
                        "name": "lookup",
                        "arguments": {"retry": True},
                    }
                },
            ),
            event(
                "tool.completed",
                {"call_id": "retry-call", "tool": "lookup", "result": {"answer": 2}},
            ),
            event("turn.completed", {"text": "recovered"}),
        ]
    )
    scorer = AsyncMock(return_value=1)
    result = await HarnessRollout(AsyncMock(return_value=harness), scorer)(
        graph, EvaluationTask(id="one", query="q")
    )
    pairs = [item for item in result.trajectory if item["type"] == "tool"]
    assert [pair["status"] for pair in pairs] == ["failed", "completed"]
    assert pairs[0]["result"] == {"error": "not found"}
    assert result.output == "recovered"
    assert result.score == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["procedural.guidance", "procedural.step"])
async def test_procedural_events_must_match_graph(kind):
    harness = FakeHarness(
        [event(kind, {"graph_version": "stale", "status": "completed"})]
    )
    scorer = AsyncMock(return_value=1)
    with pytest.raises(ValueError, match="does not match"):
        await HarnessRollout(AsyncMock(return_value=harness), scorer)(
            ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q")
        )
    scorer.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_harness_unguided_fallback_is_rejected_and_reuse_denied():
    from hyperforge.harness_sdk import AgentHarness, ModelDelta

    class Solver:
        async def stream(self, **kwargs):
            yield ModelDelta(text="fallback answer")

    class FailingGuidance:
        async def stream(self, **kwargs):
            raise OSError("guidance unavailable")
            yield  # Make this a streaming fake.

    graph = ProceduralGraph.skeleton()
    harness = AgentHarness(
        model="fake",
        model_client=Solver(),
        guidance_client=FailingGuidance(),
        procedural_guidance=ProceduralGuidanceConfig(
            graph=graph, failure_policy="unguided"
        ),
    )
    scorer = AsyncMock(return_value=0)
    runner = HarnessRollout(AsyncMock(return_value=harness), scorer)
    with pytest.raises(RuntimeError, match="unguided fallback"):
        await runner(graph, EvaluationTask(id="one", query="q"))
    scorer.assert_not_awaited()
    with pytest.raises(ValueError, match="fresh"):
        await runner(graph, EvaluationTask(id="two", query="q"))


@pytest.mark.asyncio
async def test_no_guidance_events_is_not_an_evaluable_run():
    class UnguidedHarness(FakeHarness):
        async def run(self, query):
            yield event("turn.completed", {"text": "unguided"})

    scorer = AsyncMock(return_value=1)
    with pytest.raises(RuntimeError, match="completed guidance"):
        await HarnessRollout(AsyncMock(return_value=UnguidedHarness([])), scorer)(
            ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q")
        )
    scorer.assert_not_awaited()


@pytest.mark.asyncio
async def test_actual_harness_records_failed_and_recovered_procedural_steps():
    from pydantic import BaseModel

    from hyperforge.harness_sdk import (
        AgentHarness,
        HarnessTool,
        HarnessToolCall,
        ModelDelta,
        ToolCallContext,
    )

    class Input(BaseModel):
        retry: bool

    class Output(BaseModel):
        value: str

    async def lookup(context: ToolCallContext, value: Input) -> Output:
        if not value.retry:
            raise ValueError("Try a different lookup")
        return Output(value="found")

    class Solver:
        def __init__(self):
            self.calls = 0

        async def stream(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                yield ModelDelta(
                    tool_calls=[
                        HarnessToolCall(
                            id=f"call-{self.calls}",
                            name="lookup",
                            arguments={"retry": self.calls == 2},
                        )
                    ]
                )
            else:
                yield ModelDelta(text="recovered answer")

    class Guidance:
        async def stream(self, **kwargs):
            yield ModelDelta(
                text="Use the observations to decide whether to retry or finish."
            )

    graph = ProceduralGraph.skeleton()
    harness = AgentHarness(
        model="fake",
        model_client=Solver(),
        guidance_client=Guidance(),
        tools=[HarnessTool("lookup", lookup)],
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
    )
    result = await HarnessRollout(
        AsyncMock(return_value=harness), AsyncMock(return_value=1)
    )(graph, EvaluationTask(id="one", query="Find the answer"))
    pairs = [item for item in result.trajectory if item["type"] == "tool"]
    steps = [
        item["payload"]
        for item in result.trajectory
        if item["type"] == "procedural.step"
    ]
    guidance = [
        item for item in result.trajectory if item["type"] == "procedural.guidance"
    ]
    assert len(guidance) == 3
    assert [pair["status"] for pair in pairs] == ["failed", "completed"]
    assert [step["status"] for step in steps] == ["failed", "completed"]
    assert [step["call_id"] for step in steps] == [pair["call_id"] for pair in pairs]
    assert steps == [step.model_dump() for step in harness.procedural_trajectory]
    assert result.output == "recovered answer"


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["queued_messages", "stored_conversation"])
async def test_queued_gold_or_seeded_storage_rejected_before_solver(source):
    from hyperforge.harness_sdk import AgentHarness
    from hyperforge.harness_sdk.models import HarnessConversation

    graph = ProceduralGraph.skeleton()
    solver = SimpleNamespace(stream=AsyncMock())
    guidance = SimpleNamespace(stream=AsyncMock())
    harness = AgentHarness(
        model="fake",
        model_client=solver,
        guidance_client=guidance,
        procedural_guidance=ProceduralGuidanceConfig(graph=graph),
    )
    if source == "queued_messages":
        harness.add_messages([{"role": "assistant", "content": "GOLD ANSWER"}])
    else:
        await harness.storage.create_conversation(
            HarnessConversation(
                id=harness.conversation_id, title="Seeded gold conversation"
            )
        )
    assert harness.conversation is None
    assert harness.messages == []
    assert harness.procedural_trajectory == ()
    scorer = AsyncMock(return_value=1)
    with pytest.raises(ValueError, match="fresh"):
        await HarnessRollout(AsyncMock(return_value=harness), scorer)(
            graph, EvaluationTask(id="one", query="q", expected="GOLD ANSWER")
        )
    solver.stream.assert_not_called()
    guidance.stream.assert_not_called()
    scorer.assert_not_awaited()


@pytest.mark.asyncio
async def test_freshness_storage_error_propagates_without_execution():
    harness = FakeHarness([])
    harness.storage.get_conversation.side_effect = OSError("storage unavailable")
    scorer = AsyncMock(return_value=1)
    with pytest.raises(OSError, match="storage unavailable"):
        await HarnessRollout(AsyncMock(return_value=harness), scorer)(
            ProceduralGraph.skeleton(), EvaluationTask(id="one", query="q")
        )
    harness.storage.get_conversation.assert_awaited_once_with(harness.conversation_id)
    assert not hasattr(harness, "query")
    scorer.assert_not_awaited()
    assert harness.closed
