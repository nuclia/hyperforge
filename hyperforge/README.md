# Hyperforge

Hyperforge is an **Agentic Framework for Orchestrated Runtime, Governance, and Execution** — the core runtime powering AI agent workflows at Nuclia. It provides the infrastructure for defining, deploying, and running agentic pipelines backed by NucliaDB for memory and retrieval, Redis/Valkey for pub/sub and session state, PostgreSQL for persistence, and a FastAPI HTTP/WebSocket API surface.

## Architecture Overview

```
hyperforge/
├── api/          # FastAPI application, REST + WebSocket endpoints, OAuth
├── broker/       # Pub/sub backends (Redis, local)
├── context/      # Execution context passed through agent runs
├── db/           # SQLAlchemy models and Alembic migrations
├── memory/       # Session memory backed by NucliaDB
├── retrieval/    # Retrieval agent and config
├── server/       # Long-running server process
├── standalone/   # Single-process standalone mode
├── agent.py      # Base Agent and AgentConfig abstractions
├── engine.py     # Initialization and state wiring
├── manager.py    # Agent lifecycle manager
├── workflows.py  # Workflow data models
└── settings.py   # Pydantic settings (env-driven)
```

Key components:

- **Agent / AgentConfig** — abstract base classes for building typed, configurable agents
- **Engine** — initializes agent state, loads external modules, wires LLM connections and memory
- **Manager** — handles agent lifecycle within a session
- **Memory** — session-scoped memory stored in NucliaDB (`SessionMemory`, `QuestionMemory`)
- **Broker** — pub/sub layer (Redis or in-process) for streaming workflow events
- **API** — FastAPI app exposing REST and WebSocket endpoints for agents, workflows, sessions, MCP, and OAuth
- **Workflows** — structured workflow definitions with parameters, rules, and required fields

## Install

From the workspace root:

```bash
uv sync
```

## Running

### API server

```bash
uv run hyperforge-api
```

### Other entry points

| Command | Description |
|---|---|
| `hyperforge-api` | Main HTTP API server |
| `hyperforge-server` | Background server process |
| `hyperforge-sandbox` | Sandbox execution environment |
| `hyperforge-standalone` | All-in-one single-process mode |
| `hyperforge-downloads-cronjob` | Downloads cleanup cron job |
| `hyperforge-workflows-cleanup-cronjob` | Stale workflow cleanup cron job |
| `hyperforge-extract-openapi` | Extract OpenAPI schema to file |

### Health and observability endpoints

- `GET /health/ready` — readiness probe
- `GET /health/alive` — liveness probe
- `GET /metrics` — Prometheus metrics

## Configuration

All settings are provided through environment variables consumed by Pydantic `BaseSettings`.

| Variable | Description |
|---|---|
| `HTTP_HOST` | Bind host (default `0.0.0.0`) |
| `HTTP_PORT` | Bind port (default `8080`) |
| `MEMORY_READER_NUCLIADB` | NucliaDB read endpoint |
| `MEMORY_WRITER_NUCLIADB` | NucliaDB write endpoint |
| `MEMORY_SEARCH_NUCLIADB` | NucliaDB search endpoint |
| `MEMORY_APIKEY_NUCLIADB` | NucliaDB API key |
| `VALKEY_URL` | Redis/Valkey connection URL |
| `LOAD_MODULES` | Comma-separated list of agent package modules to load at startup |
| `NUCLIA_ZONE` | Nuclia zone (default `arag`) |
| `NUCLIA_PUBLIC_URL` | Public Nuclia URL template (default `https://{zone}.nuclia.com`) |

## Loading Agent Packages

Hyperforge supports dynamically loading external agent packages at startup. Set the `LOAD_MODULES` environment variable to a comma-separated list of Python module names. Each module is scanned for agent and configuration definitions and registered in the global registry.

```bash
LOAD_MODULES=hyperforge_rephrase,my_custom_agent uv run hyperforge-api
```

## Development

Run tests from the workspace root:

```bash
uv run pytest hyperforge
```

Run a specific test file:

```bash
uv run pytest hyperforge/tests/test_engine.py
```

Format and lint:

```bash
make fmt
make lint
```

## License

Apache 2.0 — see [LICENSE](../LICENSE).
