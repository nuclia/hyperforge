# Harness SDK

The Hyperforge Harness SDK provides an asynchronous agent loop for applications
that need streaming model responses, validated tools, conversation persistence,
live events, usage limits, feedback, memory, and sub-agent delegation.

Import the framework from `hyperforge.harness_sdk`:

```python
from hyperforge.harness_sdk import AgentHarness
```

## Agent Loop

An `AgentHarness` run:

1. Loads or creates its conversation.
2. Adds the system prompt and user prompt to message history.
3. Streams a response from the configured `ModelClient`.
4. Executes any requested tools and returns their validated results to the model.
5. Repeats until the model returns a final text response.
6. Persists messages and lifecycle events through the configured storage.

Calls to `run()` on one harness are serialized. A harness instance can therefore
be reused for multiple turns in the same conversation.

## Creating an Agent

Hyperforge includes `NucliaModelClient`, which connects the harness to Nuclia's
chat-completions endpoint through Hyperforge's existing asynchronous NUA client:

```python
from hyperforge.harness_sdk import AgentHarness, HarnessEventType, NucliaModelClient


async def build_agent() -> AgentHarness:
    model_client = await NucliaModelClient.from_api_key("your-api-key")
    return AgentHarness(
        model="your-model",
        model_client=model_client,
        reasoning_effort="high",
        system_prompt="You are a concise support assistant.",
    )


async def answer(agent: AgentHarness, prompt: str) -> str:
    result = ""
    async for event in agent.run(prompt):
        if event.type == HarnessEventType.TURN_COMPLETED:
            result = event.payload["text"]
    return result
```

Close a `NucliaModelClient` created by the application during application
shutdown:

```python
await model_client.aclose()
```

Applications can provide another model integration by implementing
`ModelClient.stream()`. It receives the model name, complete message history,
registered tools, and execution context, and yields `ModelDelta` objects:

```python
from collections.abc import AsyncIterator

from hyperforge.harness_sdk import ModelDelta


class ApplicationModelClient:
    async def stream(
        self, *, model, reasoning_effort, messages, tools, execution_context
    ) -> AsyncIterator[ModelDelta]:
        # Translate messages and tools to the provider's API here.
        yield ModelDelta(text="Hello", input_tokens=10, output_tokens=1)
```

## Using Legacy Agents

The legacy engine can use the harness as its root orchestrator. Configure an
`agents` list instead of the retrieval pipeline stages; each configured legacy
agent contributes its published functions as namespaced harness tools:

```python
config = {
    "model": "your-model",
    "reasoning_effort": "high",
    "agents": [
        {"id": "docs", "module": "basic_ask", "sources": ["nuclia-docs"]},
    ],
    "drivers": [
        {
            "name": "Nuclia docs",
            "provider": "nucliadb",
            "identifier": "nuclia-docs",
            "config": {"url": "...", "kbid": "...", "key": "..."},
        }
    ],
    "rules": {"rules": ["Be concise"]},
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": None,
        "parameters": {},
    },
}

memory = await hyperforge.engine.main(
    config=config,
    question="How do I use the API?",
    loaded_modules=["hyperforge_nucliadb"],
    external_nua_api_key="...",
)
```

For example, an agent with ID `docs` that publishes `ask_agent` is exposed to
the model as `docs__ask_agent`.

Token values in streaming deltas are cumulative for one model request. The
harness records the largest reported values and aggregates them across the run.

## Defining Tools

A tool uses Pydantic models for its arguments and return value. Field
descriptions are included in the JSON Schema sent to the model.

```python
from pydantic import BaseModel, Field

from hyperforge.harness_sdk import AgentHarness, tool


class WeatherInput(BaseModel):
    city: str = Field(description="City for the weather lookup")


class WeatherOutput(BaseModel):
    summary: str


@tool(description="Get the current weather for a city.")
async def weather(
    harness: AgentHarness,
    input_value: WeatherInput,
) -> WeatherOutput:
    return WeatherOutput(summary=f"Sunny in {input_value.city}")


agent = AgentHarness(
    model="your-model",
    model_client=model_client,
    tools=[weather],
)
```

The `@tool` decorator creates a `HarnessTool` and infers its input and output
models from the handler annotations. The decorated tool remains callable, so
application code can use `await weather(harness, WeatherInput(city="Boston"))`
as it would call the original function. Direct `HarnessTool(...)` construction
is available when a decorator is not appropriate.
Nested JSON Schema references are flattened before schemas are sent to the model;
the emitted schema does not contain `$ref`, `$defs`, or `definitions`.
The harness validates model-provided arguments before calling the handler and
validates the result against its annotated return model. Missing or non-Pydantic
input and return annotations fail when the tool is constructed. Tool failures are
returned to the model as failed tool results so it can recover or answer differently.
Multiple tool calls from one model response run concurrently; their result
messages retain the model's original call order.

Tools marked `lazy_load=True` are registered for execution but omitted from the
model's initial tool list. The always-available `search_tools` and
`activate_tools` tools let the model discover matching lazy tools and expose
their schemas on subsequent model calls. Lazy activation remains active for the
life of the harness instance.

```python
@tool(description="Get the current weather for a city.", lazy_load=True)
async def weather(harness: AgentHarness, input_value: WeatherInput) -> WeatherOutput:
    return WeatherOutput(summary=f"Sunny in {input_value.city}")
```

`lazy_load` can also be passed directly to `HarnessTool`. Legacy published agent
functions support the same option on `FunctionDefinition`.

For a tool without arguments, define and use an empty Pydantic input model. Tool
handlers always receive a validated model instance; they never receive `None`.

Tools can attach typed context to their result by setting `context_type` and, if
needed, registering a schema and formatter with `register_context()`.

## Usage Limits

All limits are disabled by default. Configure only the limits needed by the
application:

```python
from hyperforge.harness_sdk import AgentHarness, UsageLimits

agent = AgentHarness(
    model="your-model",
    model_client=model_client,
    usage_limits=UsageLimits(
        max_tool_calls=20,
        max_turns=10,
        max_input_tokens=100_000,
        max_output_tokens=20_000,
        max_time=120,
    ),
)
```

`max_turns` counts model requests in one `run()` call. `max_time` is in seconds
and covers model and tool work. A parent agent and its sub-agents share usage and
the same limits. Exceeding a limit raises `UsageLimitExceeded` and records a
failed turn event.

After a run, inspect `agent.usage` for `turns`, `tool_calls`, `input_tokens`, and
`output_tokens`.

## Conversations and Storage

Pass a `conversation_id` to continue a known conversation. Agent IDs are
internal implementation details and are never supplied by application code.

```python
from hyperforge.harness_sdk import AgentHarness, InMemoryHarnessStorage

storage = InMemoryHarnessStorage()

first = AgentHarness(
    model="your-model",
    model_client=model_client,
    conversation_id="support-thread-42",
    storage=storage,
)
async for _ in first.run("My order is late"):
    pass

resumed = AgentHarness(
    model="your-model",
    model_client=model_client,
    conversation_id="support-thread-42",
    storage=storage,
)
await resumed.load(create=False)
async for _ in resumed.run("What should I do next?"):
    pass
```

`InMemoryHarnessStorage` is process-local and intended for tests and ephemeral
agents. Production applications should implement `HarnessStorageProtocol` with
durable conversation, event, and memory storage. `create_agent()` is a convenience
function that constructs and loads an agent in one call.

Conversation metadata can be supplied under `conversation_metadata` in the
execution context:

```python
agent = AgentHarness(
    model="your-model",
    model_client=model_client,
    execution_context={
        "user_id": "user-42",
        "conversation_metadata": {"tenant": "acme"},
    },
)
```

The complete `execution_context` is also passed to the model client.

## Streaming Events

Iterate `run()` to receive text and reasoning deltas,
model and tool lifecycle events, inbox activity, feedback activity, and the final
turn event:

```python
import asyncio

from hyperforge.harness_sdk import HarnessEventType


async def run_with_events(agent: AgentHarness, prompt: str) -> str:
    result = "interrupted"
    async with agent:
        async for event in agent.run(prompt):
            if event.type == HarnessEventType.TEXT_DELTA:
                send_to_client(event.payload["text"])
            elif event.type == HarnessEventType.TURN_COMPLETED:
                result = event.payload["text"]
    return result
```

`TEXT_DELTA` and `REASONING_DELTA` events are live-only. Other events are
persisted. Use `history()` to iterate over persisted conversation events.

`run()` owns one turn-scoped event queue and supports one consumer. Event
publication never waits for the consumer. If the consumer falls behind by
`event_queue_size` events, additional `TEXT_DELTA` and `REASONING_DELTA` events
are dropped; persisted lifecycle events remain lossless. Events from child agents
may interleave, but `turn_id`, `agent_id`, and `parent_agent_id` identify their
origin. Use `async with agent` when a caller may stop consuming early so the
active turn and descendants are cleaned up.

At most four agents run concurrently in a conversation by default, including
the root agent. Configure `max_concurrent_agents` to change this limit. When the
limit is reached, `spawn_agent` returns `concurrency_limit_reached` and identifies
children the caller can pass to `wait_agent`; it does not start another agent.

## Steering and Interrupting

`steer()` adds a message while the run is active. The message is delivered before
the next model request:

```python
await agent.steer("Prefer a concise answer")
```

`interrupt()` stops active model work, tool calls, and descendant agents:

```python
async for event in agent.run("Perform a long task"):
    agent.interrupt()
    if event.type == HarnessEventType.TURN_INTERRUPTED:
        break
```

External task cancellation remains regular `asyncio` cancellation and is
re-raised rather than converted to an interrupted result.

## Feedback

Enable the built-in `feedback` tool when the model should be able to ask the
application for input:

```python
agent = AgentHarness(
    model="your-model",
    model_client=model_client,
    feedback_enabled=True,
)

async for event in agent.run("Prepare the report"):
    if event.type == HarnessEventType.FEEDBACK_REQUESTED:
        await agent.respond_feedback(event.payload["request_id"], "Use markdown")
```

Feedback requests time out after five minutes when invoked through the built-in
tool. `request_feedback()` is also available for application-defined tools that
need a custom schema or timeout.

## Memory and Compaction

The following core tools are available to the model by default:

- `remember`, `recall`, and `forget` manage records through the configured storage.
- `compact` replaces current history with the system prompt and a model-authored summary.
- `spawn_agent`, `send_message`, and `wait_agent` coordinate sub-agents.

Disable core tools that should not be available to an agent:

```python
agent = AgentHarness(
    model="your-model",
    model_client=model_client,
    disabled_core_tools={"remember", "forget", "spawn_agent"},
)
```

Memory scope is an application-defined string. A production storage
implementation is responsible for enforcing tenant and user isolation.

## Sub-Agents

The model can call `spawn_agent` with a self-contained prompt, then call
`wait_agent` with the returned internal ID. It can use `send_message` to steer a
running child. Children inherit the model client, model, external tools,
execution context, storage, and usage limits. They start with isolated history
unless `include_history=true` is requested. Included history contains the current
system and user messages but never an incomplete spawn tool exchange.

Delegation is limited to depth one by default. Set `max_spawn_depth` on the
root harness to change the limit; an agent at the limit does not advertise the
`spawn_agent` tool. Children are turn-scoped and are stopped when the parent turn
completes, fails, times out, or is interrupted.

Internal child IDs are only handles for these core tools. They are not part of
the application-facing agent or storage API.

## Adapting Existing Hyperforge Agents

`AgentHarness.to_tools()` converts an existing Hyperforge agent's
`__published_functions__` into harness tools. Each published function becomes one
tool with its configured name, description, and complete parameter schema. Use a
namespace when combining agents that publish the same function name.

```python
from hyperforge.harness_sdk import AgentHarness
from hyperforge_perplexity import PerplexityAgent


perplexity = PerplexityAgent(
    config=perplexity_config,
)

agent = AgentHarness(
    model="your-model",
    model_client=model_client,
    tools=AgentHarness.to_tools(perplexity, namespace="perplexity") + [application_tool],
    system_prompt=(
        "Use `perplexity__internet_search` when the answer requires current information."
    ),
    execution_context={
        "drivers": {
            perplexity_config.source: perplexity_driver,
        }
    },
)

answer = ""
async for event in agent.run("What changed in Python this week?"):
    if event.type == HarnessEventType.TURN_COMPLETED:
        answer = event.payload["text"]
```

The conversion uses each `FunctionDefinition` in `__published_functions__` and
the corresponding instance method. Model arguments are validated from the
published JSON schema and method annotations. Framework-only `memory` and
`manager` arguments are omitted from the model-facing tool schema. Schema
constraints such as enums, nested objects, defaults, bounds, and
`additionalProperties` are preserved and references are flattened.

On every tool call, the compatibility layer:

- Creates a fresh no-op `QuestionMemory` for the harness conversation.
- Creates a compatibility `Manager` that shares the NUA client when the harness
  uses `NucliaModelClient`.
- Adds prebuilt drivers from `execution_context["drivers"]`.
- Calls the published method and returns its result as structured tool context.

For a custom `ModelClient`, pass an existing `Manager` to `to_tools()` or provide
an asynchronous NUA client as `execution_context["nua_client"]`. The harness model
loop still uses chat completions; the manager exists only for unchanged legacy
published functions that call the established manager API.

Agents that discover published functions asynchronously can use
`preload_published_agent_to_tools()` with an existing manager.

Agents whose published functions only use `manager.nua` require no extra manager
configuration. Agents such as Perplexity that use `manager.drivers` need their
initialized drivers passed through `execution_context` as shown above.

`to_tools()` raises `TypeError` when the agent has no published functions or a
published function does not map to a callable method. `HarnessAgentToTool`
remains available for exceptional agents that need custom memory construction or
result adaptation.

## Testing Applications

Use a small `ModelClient` test double that yields deterministic `ModelDelta`
objects. This exercises the real message, tool, event, and persistence loop
without making a model request:

```python
from collections.abc import AsyncIterator

from hyperforge.harness_sdk import AgentHarness, ModelDelta


class TestModel:
    async def stream(self, **kwargs) -> AsyncIterator[ModelDelta]:
        yield ModelDelta(text="test answer", input_tokens=2, output_tokens=2)


async def test_agent_answer() -> None:
    agent = AgentHarness(model="test", model_client=TestModel())
    events = [event async for event in agent.run("question")]
    assert events[-1].payload["text"] == "test answer"
```

For tool-loop tests, return a `HarnessToolCall` from the first model request and a
final text delta after the harness appends the tool result. See
`hyperforge/tests/harness_sdk/` for complete happy-path examples.
