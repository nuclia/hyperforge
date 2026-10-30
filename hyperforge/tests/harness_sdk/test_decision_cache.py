from collections.abc import AsyncIterator
from uuid import uuid4

from pydantic import BaseModel

from hyperforge.harness_sdk import (
    AgentHarness,
    DecisionCache,
    DecisionCacheEntry,
    DecisionCacheQuery,
    HarnessEventType,
    HarnessToolCall,
    InMemoryDecisionCacheStorage,
    ModelDelta,
    ToolCachePolicy,
    tool,
)
from hyperforge.procedural.graph import ProceduralGraph
from hyperforge.procedural.guidance import ProceduralGuidanceConfig


class NameInput(BaseModel):
    name: str = ""


class Output(BaseModel):
    value: str


schema_version = {"value": "v1"}
executed: list[str] = []


@tool(cache_policy=ToolCachePolicy.METADATA)
async def describe(context, input_value: NameInput) -> Output:
    executed.append("describe")
    return Output(value=schema_version["value"])


@tool(cache_policy=ToolCachePolicy.EXACT)
async def query(context, input_value: NameInput) -> Output:
    executed.append("query")
    return Output(value="rows")


@tool()
async def mutate(context, input_value: NameInput) -> Output:
    executed.append("mutate")
    return Output(value="ok")


class ScriptedModel:
    """Chooses the next action from how many tool results are in history."""

    def __init__(self, plan: list[str]) -> None:
        self.plan = plan
        self.calls = 0

    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        self.calls += 1
        step = sum(1 for message in kwargs["messages"] if message.role == "tool")
        if step < len(self.plan):
            name = self.plan[step]
            yield ModelDelta(
                tool_calls=[HarnessToolCall(id=f"c{step}", name=name, arguments={})],
                input_tokens=10,
            )
        else:
            yield ModelDelta(text="done", input_tokens=10)


async def run(storage, model, prompt="count orders") -> list:
    harness = AgentHarness(
        model="m",
        model_client=model,
        tools=[describe, query, mutate],
        decision_cache=DecisionCache(storage=storage, namespace=["acct", "proj"]),
    )
    events = [event async for event in harness.run(prompt)]
    assert harness._decision_cache is not None
    await harness._decision_cache.flush()
    return [
        event.payload.get("decision_cache", {}).get("status")
        for event in events
        if event.type == HarnessEventType.LLM_COMPLETED
    ]


async def test_replays_decisions_and_executes_tools_live() -> None:
    storage = InMemoryDecisionCacheStorage()
    schema_version["value"] = "v1"
    executed.clear()
    plan = ["describe", "describe", "query"]

    cold = ScriptedModel(plan)
    assert await run(storage, cold) == ["miss", "miss", "miss", None]
    assert cold.calls == 4

    warm = ScriptedModel(plan)
    assert await run(storage, warm) == ["hit", "hit", "hit", None]
    assert warm.calls == 1
    assert executed[3:] == ["describe", "describe", "query"]

    # Changed metadata invalidates every decision after it.
    schema_version["value"] = "v2"
    changed = ScriptedModel(plan)
    assert await run(storage, changed) == ["hit", "miss", "miss", None]


async def test_does_not_cache_unsafe_tools_or_different_questions() -> None:
    storage = InMemoryDecisionCacheStorage()
    await run(storage, ScriptedModel(["mutate"]))
    assert storage.entries == {}

    await run(storage, ScriptedModel(["query"]))
    assert len(storage.entries) == 1
    assert not next(iter(storage.entries.values())).similar

    class SimilarStorage(InMemoryDecisionCacheStorage):
        """Pretends every stored question is similar to the query."""

        async def lookup(self, query: DecisionCacheQuery) -> DecisionCacheEntry | None:
            return next(iter(self.entries.values()), None)

    similar = SimilarStorage()
    similar.entries = storage.entries
    model = ScriptedModel(["query"])
    # EXACT decisions are rejected by the harness for a different question.
    assert await run(similar, model, prompt="other") == ["miss", None]


async def test_storage_failure_falls_back_to_live_inference() -> None:
    class Broken(InMemoryDecisionCacheStorage):
        lookups = 0

        async def lookup(self, query: DecisionCacheQuery) -> DecisionCacheEntry | None:
            self.lookups += 1
            raise RuntimeError("down")

    storage = Broken()
    model = ScriptedModel(["describe"])
    assert await run(storage, model) == ["miss", None]
    assert model.calls == 2
    assert storage.lookups == 1


class DelegatingModel:
    """Parent describes, spawns a child, waits; the child describes and answers."""

    def __init__(self, include_history: bool) -> None:
        self.include_history = include_history
        self.calls = {"parent": 0, "child": 0}

    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        messages = kwargs["messages"]
        is_child = bool(kwargs["execution_context"].get("parent_agent_id"))
        self.calls["child" if is_child else "parent"] += 1
        results = [message for message in messages if message.role == "tool"]
        if is_child:
            own = [m for m in results if m.tool_name == "describe"]
            if len(own) < (2 if self.include_history else 1):
                call = HarnessToolCall(name="describe", arguments={})
            else:
                yield ModelDelta(text="child done")
                return
        elif not results:
            call = HarnessToolCall(name="describe", arguments={})
        elif results[-1].tool_name == "describe":
            call = HarnessToolCall(
                name="spawn_agent",
                arguments={
                    "prompt": "inspect",
                    "include_history": self.include_history,
                },
            )
        elif results[-1].tool_name == "spawn_agent":
            agent_id = results[-1].content.split('"agent_id":"')[1].split('"')[0]
            call = HarnessToolCall(name="wait_agent", arguments={"agent_id": agent_id})
        else:
            yield ModelDelta(text="done")
            return
        yield ModelDelta(tool_calls=[call.model_copy(update={"id": uuid4().hex})])


async def test_children_keep_their_own_cache_stack() -> None:
    for include_history in (False, True):
        storage = InMemoryDecisionCacheStorage()
        schema_version["value"] = "v1"
        cold = DelegatingModel(include_history)
        await run(storage, cold)
        warm = DelegatingModel(include_history)
        await run(storage, warm)
        # Parent: describe + spawn replay; wait and answer follow a live result.
        assert warm.calls["parent"] == cold.calls["parent"] - 2
        # Child: its own describe decision replays; the answer is live.
        assert warm.calls["child"] == 1


class GuidanceModel:
    def __init__(self) -> None:
        self.calls = 0

    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        self.calls += 1
        yield ModelDelta(text=f"guidance {uuid4().hex}")


async def test_procedural_guidance_is_keyed_by_graph() -> None:
    storage = InMemoryDecisionCacheStorage()
    schema_version["value"] = "v1"
    skeleton = ProceduralGraph.skeleton()
    changed = skeleton.model_copy(
        update={
            "edges": tuple(
                edge.model_copy(update={"guidance": "Describe first."})
                for edge in skeleton.edges
            )
        }
    )

    async def run_guided(graph: ProceduralGraph) -> tuple[list, int, int]:
        model, guidance = ScriptedModel(["describe"]), GuidanceModel()
        harness = AgentHarness(
            model="m",
            model_client=model,
            tools=[describe],
            procedural_guidance=ProceduralGuidanceConfig(graph=graph),
            guidance_client=guidance,
            decision_cache=DecisionCache(storage=storage),
        )
        events = [event async for event in harness.run("count orders")]
        assert harness._decision_cache is not None
        await harness._decision_cache.flush()
        statuses = [
            event.payload.get("decision_cache", {}).get("status")
            for event in events
            if event.type == HarnessEventType.LLM_COMPLETED
        ]
        return statuses, model.calls, guidance.calls

    assert await run_guided(skeleton) == (["miss", "miss"], 2, 2)
    # Same graph: the hit skips both guidance and inference despite new guidance text.
    assert await run_guided(skeleton) == (["hit", "miss"], 1, 1)
    assert await run_guided(changed) == (["miss", "miss"], 2, 2)
