# Procedural Graphs

Procedural graphs add opt-in, advisory guidance to the [Harness SDK](harness-sdk.md)
and support offline, validation-gated graph evolution. They are not a context
agent, a tool permission system, or an executable workflow: the solver still
chooses tools and answers through the normal `AgentHarness` loop.

## Start Small

Use an existing `ModelClient` and a minimal graph first. This asynchronous example
needs a configured client and a model ID supported by that client:

```python
from hyperforge.harness_sdk import AgentHarness, HarnessEventType
from hyperforge.procedural import ProceduralGraph, ProceduralGuidanceConfig


async def answer(model_client, model: str, question: str) -> str:
    agent = AgentHarness(
        model=model,
        model_client=model_client,
        procedural_guidance=ProceduralGuidanceConfig(
            graph=ProceduralGraph.skeleton(),
        ),
    )
    async with agent:
        async for event in agent.run(question):
            if event.type == HarnessEventType.TURN_COMPLETED:
                return event.payload["text"]
    raise RuntimeError("Turn did not complete")
```

`skeleton()` contains `Start -> End` with generic task-completion guidance and
no tool bindings. The harness's ordinary core tools remain available. Add your
application tools as described in the SDK guide; the graph never registers tools.
Client initialization and shutdown remain application responsibilities.

`ProceduralGraph`, its nodes and edges, and `ProceduralGuidanceConfig` are frozen
Pydantic models with unknown fields forbidden. The harness revalidates a snapshot
at construction. Inspect it through `agent.procedural_guidance`; create a new
harness to change configuration, and a new conversation to change the graph.

## Optional Expert Graph

An expert-authored search graph is optional, not a prerequisite for evolution.
Construct graphs with the real Pydantic types:

```python
from hyperforge.procedural import ProceduralGraph, ProcedureEdge, ProcedureNode

search_graph = ProceduralGraph(
    cycle_policy="reject",
    nodes=(
        ProcedureNode(id="Start", type="STATUS"),
        ProcedureNode(id="Search", type="ACTION", procedure_name="search"),
        ProcedureNode(id="Assess", type="REASONING"),
        ProcedureNode(id="End", type="STATUS"),
    ),
    edges=(
        ProcedureEdge(
            source="Start", target="Search",
            condition="External evidence is needed",
            guidance="Search for evidence relevant to the user's question.",
        ),
        ProcedureEdge(
            source="Search", target="Assess", relation="PROVIDES_INPUT_FOR",
            guidance="Check whether the returned evidence supports the answer.",
            pitfalls="Do not treat a search result as proof of an unsupported claim.",
        ),
        ProcedureEdge(
            source="Assess", target="End",
            guidance="Answer from supported evidence; state any remaining gaps.",
        ),
    ),
)
search_graph.check_tools(["search"])
```

Supply a real `HarnessTool` named exactly `search` in `AgentHarness(tools=...)`
when using this graph. `Search` is a node ID, not its tool name. An `ACTION`
binds to `procedure_name`, or to its `id` when that field is absent; matching is
exact and case-sensitive. For legacy published tools, use the namespaced name,
such as `docs__ask_agent`, rather than the original method name.

Validation requires unique, nonblank node IDs and ACTION bindings, a `Start`
node of type `STATUS`, known edge endpoints, and unique
`(source, target, relation)` triplets. `Start` is reserved: no ACTION may bind to
it, including through `procedure_name`. Relations are
`LEADS_TO`, `TRIGGERS`, `PROVIDES_INPUT_FOR`, or `CONVERGES_TO`; guidance must be
nonempty. Every node must reach a zero-outdegree terminal. `cycle_policy="allow"`
(the default) permits cycles only with such an exit; `"reject"` forbids cycles.
Invalid graphs are rejected, never silently repaired. Limits are 1,000 nodes and
10,000 edges.

The harness checks bindings against its registered tool catalog, including
inactive lazy tools, before any model call. Guidance does not activate lazy tools,
change permissions, or enforce edge conditions. REASONING and STATUS nodes
provide advice; they are not executed steps.

## Runtime Behavior

Before every solver decision, a separate guidance request receives the current
query, latest user-message observation, graph neighborhood, recent tool steps,
and currently visible tool names. It receives no executable tools. By default it
uses the solver's client and model; set `guidance_client=` on `AgentHarness` and/or
`model=` on `ProceduralGuidanceConfig` to override these independently.

- Defaults are `hops=2` and `window=3`. Localization starts at
  `Start`, then uses the most recent recorded tool action's exact binding. A
  localization miss supplies the full graph, not a fuzzy match or an empty graph.
- Guidance is a temporary system message for that solver request, not factual
  evidence. It is not appended to conversation messages or streamed as answer
  text; its text is recorded in procedural events.
- The graph fingerprint is pinned for the harness lifetime; the trajectory
  accumulates across successive `run()` turns and the query updates each turn.
  Loading an existing conversation restores its root procedural steps when the
  pinned graph version matches; see [Conversation Reload](#conversation-reload).
  The config-dictionary bridge constructs a new harness per call with fresh
  default storage, so chat history alone does not restore procedural state.
- Spawned children do not inherit the graph, guidance client, or trajectory,
  even with history inclusion. There is no automatic graph inheritance into
  Code Mode capabilities or other internal agent loops either.
- Normal parallel tool execution is unchanged. Steps are recorded in requested
  order, not completion order; the next anchor is the **last requested action in
  the executed batch**, including a failed tool result. All steps in that batch
  share a decision ID.
- With `single_action=True`, a response requesting multiple calls executes
  **none** of them. The solver is reprompted, with at most two consecutive retries
  before a third multi-call response raises. It does not silently choose one call.

The default `timeout_seconds=30` and `failure_policy="unguided"` let ordinary
runs continue without advice after guidance timeouts or errors. Failed refreshes
do not reuse old advice. Empty guidance, tool requests from the advisor, and
oversized guidance are errors too. Set `failure_policy="raise"` to fail instead;
usage-limit violations still propagate under either policy.

`max_guidance_chars=8000` bounds advisor output; `max_observation_chars=8000`
truncates each recorded argument string and observation. `max_graph_chars=100000`
checks the serialized **full graph context** at configuration time, so fallback
cannot bypass the cap. Reduce an oversized graph rather than expecting automatic
truncation. These are character limits, not token limits or a total prompt budget.

### Conversation Reload

Creating a guided conversation stores its graph fingerprint in conversation
metadata as `procedural_graph_version`. To continue it, construct a harness with
the same graph, storage, and `conversation_id`, then call
`await agent.load(create=False)` (or let `run()` load it). Loading restores
`procedural_trajectory` from persisted root `procedural.step` events, ignoring
child events, so the next decision uses the restored action anchor and recent
steps. Supply the graph explicitly; it is not reconstructed from event history.

With guidance enabled, an existing conversation whose version is missing or
differs from the supplied graph raises: **start a new conversation**. Persisted
steps with a different graph version also raise. This supports conversation
continuation, not resuming an in-flight tool call or an offline evolution run.

### Events and Cost

Inspect `agent.procedural_trajectory` for a tuple of `ProcedureStep` records and
consume `HarnessEventType.PROCEDURAL_GUIDANCE` (`procedural.guidance`) and
`PROCEDURAL_STEP` (`procedural.step`) events through `run()` or persisted history.
Guidance events include graph version, decision ID, active node/scope, status,
text/error, model/trace ID, token usage, and latency. Steps include call and
decision IDs, exact procedure name, arguments, observation, status, and graph
version.

Guidance tokens, including reported partial usage on failed requests, are added
to total `agent.usage` alongside solver usage. Reported model-token and Nuclia
accounting fields are included too. Guidance events provide the separate advisor
breakdown; `usage.turns` counts solver decisions, not an additional turn per
advisor call. Account for advisor latency and cost when comparing runs. This
implementation does not establish live benchmark gains or complete reproduction
of a research result.

## Config Dictionary

For the harness-based engine workflow, use top-level `procedural_guidance` with
an inline graph, not a graph filename or a context-stage agent. The `agents` key
selects `HarnessAgentConfig`, even when the list is empty:

```python
from hyperforge.procedural import ProceduralGraph, ProceduralGuidanceConfig

graph = ProceduralGraph.skeleton()
config = {
    "model": "your-model",
    "agents": [],
    "drivers": [],
    "rules": {"rules": ["Be concise"]},
    "memory": {},
    "procedural_guidance": ProceduralGuidanceConfig(
        graph=graph, hops=2, window=3,
    ).model_dump(mode="json"),
}
```

Pass this dictionary to `hyperforge.engine.main(config=config, ...)` as in the
[legacy-agent example](harness-sdk.md#using-legacy-agents). Existing agent/driver
entries can remain in place, provided all ACTION bindings match the resulting
namespaced tools. Export just a graph with `graph.model_dump_json(indent=2)`;
load trusted graph JSON with `ProceduralGraph.model_validate_json(graph_json)`.
After offline review, explicitly embed the retained graph's
`model_dump(mode="json")` under `config["procedural_guidance"]["graph"]`.

## Offline Evolution

`evolve()` runs real, sequential rollouts, requests `GraphEdits`, validates a new
candidate, and retains it only when its mean validation score is not worse.
`ManagerRefiner` uses an explicitly supplied manager's `execute_json()` with the
`GraphEdits` schema. Neither adapter initializes clients on import.

The following is complete orchestration code, but a **conceptual integration**,
not a standalone benchmark. It requires existing model clients, a configured
`Manager`, a read-only `search` tool over an isolated corpus, and real task splits.
Here the task is source-ID selection: each task's `expected` is a nonempty list
of correct source IDs. Replace this scorer with your task's actual evaluator,
not a constant success score. Clients, corpus setup, and cleanup are caller-owned.

```python
import json

from hyperforge.harness_sdk import AgentHarness, UsageLimits
from hyperforge.procedural import ProceduralGraph, ProceduralGuidanceConfig
from hyperforge.procedural.adapters import HarnessRollout, ManagerRefiner
from hyperforge.procedural.evolution import EvaluationTask, evolve


async def evolve_sources(
    *, model_client, guidance_client, manager, model: str, refiner_model: str,
    search_tool, train_tasks, validation_tasks, test_tasks, output,
):
    if search_tool.name != "search" or search_tool.lazy_load:
        raise ValueError("Provide an immediately visible tool named search")

    async def factory(graph, query, metadata):
        # Fresh default storage and conversation ID for every rollout.
        return AgentHarness(
            model=model,
            model_client=model_client,
            guidance_client=guidance_client,
            tools=[search_tool],
            disabled_core_tools={
                "remember", "recall", "forget", "compact", "spawn_agent",
                "send_message", "wait_agent", "search_tools", "activate_tools",
            },
            system_prompt=(
                "Search the corpus for sources answering the user's question. "
                'Return only JSON: {"source_ids": ["id", ...]}. '
                "Use only IDs from search results."
            ),
            usage_limits=UsageLimits(max_turns=12, max_tool_calls=10, max_time=120),
            procedural_guidance=ProceduralGuidanceConfig(
                graph=graph, single_action=True, failure_policy="raise",
            ),
        )

    async def scorer(task: EvaluationTask, output: str, events: list[dict]) -> float:
        gold = task.expected
        if not isinstance(gold, list) or not gold or not all(
            isinstance(item, str) and item for item in gold
        ):
            raise ValueError("Expected a nonempty list of gold source IDs")
        try:
            prediction = json.loads(output)
        except json.JSONDecodeError:
            return 0.0  # A completed solver answer with invalid task output.
        ids = prediction.get("source_ids") if isinstance(prediction, dict) else None
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            return 0.0
        return float(set(ids) == set(gold))

    tools = ["search"]
    return await evolve(
        ProceduralGraph.skeleton(),
        output=output,
        train_tasks=train_tasks,
        validation_tasks=validation_tasks,
        test_tasks=test_tasks,
        runner=HarnessRollout(factory, scorer),
        refiner=ManagerRefiner(
            manager, model=refiner_model, available_tools=tools,
            task_description="Find the exact set of supporting source IDs.",
        ),
        available_tools=tools,
        provenance={
            "model": {"solver": model, "guidance": model, "refiner": refiner_model},
            "evaluator": "source-id-set-exact-match-v1",
            "tools": tools,
        },
        rounds=3,
        batch_size=2,
    )
```

The async factory receives only `(graph, query, metadata)`, never the task object
or gold answer. Only the async scorer receives the full task plus output and
public root-agent events, including paired tool results but not reasoning
streams. Keep gold out of queries, metadata, prompts, and tool configuration.
The factory owns tool/environment isolation and side effects; fresh harness
storage alone does not reset external systems. Record actual model versions,
corpus/tool versions, evaluator settings, and other execution controls in
JSON-serializable provenance when running an experiment.

`HarnessRollout` requires a fresh, unloaded harness with the supplied graph and
no loaded or queued messages (`_pending_messages`) or procedural trajectory.
It also rejects a `conversation_id` already present in storage, even if not yet
loaded. Use isolated storage and a fresh conversation ID for each rollout.
It evaluates **one user turn**.
For multi-turn environments, supply your own `runner(graph, task)` returning a
`ScoredRollout`, with a task-specific scorer and explicit reset/isolation rules.
Tool failures may be recovered by the solver, but failed guidance rejects the
rollout even with `failure_policy="unguided"`: fallback samples are not comparable
scientific evaluations. Infrastructure/scorer errors, failed/interrupted turns,
and missing completion or guidance are errors, never zero scores.

### Selection Rules

- Train and validation splits must be nonempty; all task IDs must be nonblank,
  unique, and disjoint across train/validation/test. The manifest stores IDs and
  hashes of ordered task JSON, including expected answers and metadata, not the
  task bodies. Test tasks are recorded only, never run or sent to the refiner.
- Initial validation is evaluated once and cached. Each candidate is compared
  against the retained score, not a freshly rerun baseline; **ties are accepted**.
  This is not a statistical significance test. Initial validation failure logs
  an error and aborts; later training/refiner/proposal/validation errors reject
  that round without changing the retained graph or assigning a zero score.
- Training visits consecutive batches and cycles at the end. Refiner context is
  the **tail of the current batch's** scored trajectories and queries, not the
  entire run or gold answers. Default `max_context_chars=16000` counts characters.
  This explicitly differs from token-bounded protocols. For a token limit, pass
  `max_context_tokens=N` and a `TokenCodec` implementing `tokenize(text)` and
  `decode(tokens)` for the refiner model; no tokenizer is assumed.
- Rejection memory defaults to at most 10 entries and 16,000 serialized
  characters. It includes edits, fingerprints, aggregate scores, and proposal
  structural diagnostics, not validation examples or execution errors.
- `prepare_candidate()` applies deletions then additions atomically to a new
  graph and checks invariants and the supplied tool catalog. Revise attributes
  by deleting and re-adding; deleting a node removes incident edges, and deleting
  edge endpoints removes all relations for that directed pair. Unknown deletions
  and duplicate additions fail. No cycle repair or tool installation occurs.

## Local CLI and Artifacts

The CLI imports **trusted local Python code**, not a remote adapter or a config
file. Supply `MODULE:FACTORY`, where a synchronous zero-argument factory returns
an `evolve()` keyword-argument dictionary containing `initial_graph`, an isolated
async `runner`, async `refiner`, splits, tool names, and provenance. Arrange async
client initialization inside your adapter's async callables when necessary.

```bash
uv run python -m hyperforge.procedural.cli \
  --adapter my_experiment:build_kwargs \
  --output ./runs/source-search-001 \
  --rounds 3
```

`my_experiment` must be your importable local module. CLI `--output` and
`--rounds` override the returned dictionary. The output directory must **not
exist**, even for zero rounds; there is no experiment resume or overwrite mode.

Artifacts include `manifest.json`, `baseline.json`, fingerprint-addressed
`graphs/`, per-round `rounds/`, `rejections/`, and an atomically updated
`retained.json` pointer. New directories use mode `0700`; artifact files are
exclusively created at `0600`, flushed, then made read-only at `0400`.
`retained.json` is the mutable pointer, not an immutable artifact. Existing parent
directory permissions are not tightened. These are private, immutable-by-convention
local records, not tamper-proof storage or production versions. Traces may contain
sensitive queries, tool data, and model output; choose a suitably private path.
File I/O failures propagate rather than pretending a record was committed.

There is no dedicated Smart adapter, graph UI, database graph persistence,
automatic production promotion, or offline evolution resume. Review and
evaluate retained graphs before explicitly installing them into new harnesses.
