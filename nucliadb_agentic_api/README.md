# NucliaDB Agentic API

The NucliaDB Agentic API package exposes NucliaDB-oriented agentic capabilities
for Hyperforge, including ASK/search flows and MCP integrations.

## Install

From the workspace root:

```bash
uv sync
```

## Run

Start the service:

```bash
uv run nucliadb-agentic-api
```

Useful endpoints:

- `/health/ready`
- `/health/alive`
- `/metrics`

## Configuration

Runtime configuration is provided through environment variables consumed by
Pydantic settings. Common settings include:

- `HTTP_HOST` and `HTTP_PORT`
- `MEMORY_READER_NUCLIADB`, `MEMORY_WRITER_NUCLIADB`, and
  `MEMORY_SEARCH_NUCLIADB`
- `MEMORY_APIKEY_NUCLIADB`
- `VALKEY_URL`
- `IDP_REGIONAL_GRPC`
- `LOAD_MODULES`

## Development

Run the package tests from the workspace root:

```bash
make test
```


# Objective
Enhance the RAG experience by offering RAO features directly in a KB.

# How?
The same way we store search configs in the KB, we could store agentic configs, and when calling the `/ask` endpoint we can refer to a given agentic config.

# Scope

The corresponding RAO workflow will be:
- a Rephrase (optionally)
- a SmartAgent
- a Summarize

Possible sources for the SmartAgent:
- The current KB (possibly several times with different filters)
- Sync service (using connections defined in the current KB)
- Google
- Perplexity
- MCP

Out of scope
- Other KBs
- SQL
- Snowflake (?)
- Sitefinity (?)

# Config spec

```json
{
  "rephrase": { // optional
    "ask_to": <filter_expression>, //optional
    "prompt": <string>, // optional
    "model": <model>, // optional
  },
  "smart_agent": {
    "mode": <reactive | plan_execute>,
    "extra_prompt": <string>, // optional
    "models": { // optional
      "context_validation": <model>, // optional
      "planner": <model>, // optional
      "executor": <model>, // optional
    },
    "sources": [<source list>]
  },
  "summarize": { // optional
    "user_prompt": <string>, // optional
    "system_prompt": <string>, // optional
    "conversational": <boolean>,
    "model": <model>, // optional
    // and citations must be forced to chunk-level
  }
}
```

And sources can be:

```json
{
  "type": "nucliadb",
  "description": <string>,
  "filter_expression": <filter_expression>, //optional
}

{
  "type": "sync",
  "description": <string>,
  "connection": <connection_id>
}

{
  "type": "google",
  "description": <string>
}

{
  "type": "perplexity",
  "description": <string>
}

{
  "type": "mcp",
  "description": <string>,
  <...the MCP driver params>
}
````
