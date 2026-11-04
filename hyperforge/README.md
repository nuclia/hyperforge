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
| `hyperforge-standalone` | All-in-one HTTP, MCP, and optional A2A gRPC mode |
| `hyperforge-a2a-grpc` | Database-backed SaaS A2A gRPC server |
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
| `MCP_MAX_SERVERS` | Maximum cached MCP managers (default `100`) |
| `MCP_MAX_SESSIONS_PER_SERVER` | Maximum concurrent connections per MCP manager (default `100`) |
| `MCP_SESSION_IDLE_TTL_SECONDS` | Idle lifetime of each MCP connection and cached manager (default `1800`) |

### MCP connections

API and standalone applications expose their configured MCP limits through the
typed `mcp_settings` property, without falling back to session-server settings.

Multiple clients can use the same `/api/v1/agent/{agent_id}/session/{session}/mcp`
URL, including `session/ephemeral`. Each initialization receives its own
`Mcp-Session-Id`; subsequent requests and DELETE must include that ID. DELETE
only terminates the requested connection. Missing IDs on GET or DELETE return 400;
unknown or expired IDs return 404. Clients should initialize a new connection
after a 404, not reuse the expired ID.

Managers whose last connection has closed can be evicted immediately when the
pool is full.

API replicas share an identity-bound MCP owner registry through the existing
Redis/Valkey broker. SDK transports remain in their owning process; requests
arriving on another replica are relayed through bounded Redis Streams, including
GET/SSE, elicitation replies, cancellation and DELETE. No sticky sessions or Istio
DestinationRule is required. Standalone uses this routing when
`BROKER_REDIS_DSN` is configured; without Redis it remains process-local.

One supervisor owns each process's 30-second lease, heartbeat, inbox listeners
and relayed requests. The same monotonic lease deadline gates admission and is
watched independently of Redis: a blocked renewal cannot keep the owner alive.
Successful renewals include their Redis round-trip time in that deadline; a late
acknowledgement never revives an expired owner. Registration and revocation also
have a lease-sized timeout, independent of the broker client's retry settings.
A lost lease, stopped worker or unrecoverable relay failure
immediately disables admission and retires that owner: workers and HTTP requests
are cancelled, the lease is revoked and SDK sessions are invalidated. Recovery
waits for cleanup before registering a new owner identity and restarting workers;
it never renews the failed owner's lease. SDK invalidation and lease revocation
run independently, so Redis delays cannot postpone cancelling local tools.
If revocation cannot reach Redis, the old lease expires naturally, so callers do
not depend on receiving a final error frame. Clients must initialize again after
a 404. This deliberately resets all MCP connections owned by the affected
process, not those owned by other replicas.
Ordinary request errors, exhausted capacity and disconnected callers do not
retire a healthy owner.

Live Python tasks are not migrated and tool calls are never automatically replayed:
after a connection failure, their completion may be unknown. Redis failures
return 503 (or terminate an already-started response), with no local fallback.
Use a private, trusted Redis deployment: relayed requests carry the current
request headers and body in short-lived keys, removed when accepted/completed.
Registry keys contain only hashed session identities and owner IDs.

Each owner has separate bounded queues and capacity for calls, GET/SSE streams
and control messages (notifications, elicitation replies and DELETE), with at
most 100 queued and 100 active requests per category. Calls and long-lived SSE
connections cannot exhaust the capacity needed to cancel or complete calls.
Request payloads are claimed atomically before capacity checks, so duplicate
queue entries cannot reject the original request or execute it again.
If a relay consumer's lease expires, the request returns 503 or its response
stream is terminated if headers were already sent. Each reply stream
buffers at most 16 frames of up to 64 KiB, applying backpressure instead of
dropping unread events. Complete response messages are serialized so concurrent
SSE heartbeats cannot overtake or interleave with a fragmented event.
Relay keys expire after at most 60 seconds without
activity; session records use `MCP_SESSION_IDLE_TTL_SECONDS`. All coordination
commands are single-key to support the existing Redis Cluster client.

### Standalone A2A

Standalone can expose its configured static agent over A2A from the same
`hyperforge-standalone` process. It requires a Redis/Valkey broker because A2A
task state and cross-instance feedback are durable Redis stores; the default
in-memory broker remains suitable only for HTTP/MCP local development.

```bash
BROKER_REDIS_DSN=redis://valkey:6379 \
A2A_ENABLED=true \
A2A_ACCOUNT=local \
A2A_AGENT_ID=venue \
A2A_GRPC_HOST=0.0.0.0 \
A2A_GRPC_PORT=8034 \
A2A_PUBLIC_URL=venue.example.test:8034 \
uv run hyperforge-standalone
```

Use distinct `BROKER_REDIS_ACTIVATE_SUBJECT` and `A2A_TASK_STORE_PREFIX` values
for independent Standalone installations sharing one Valkey cluster. TLS and
mTLS use the `A2A_TLS_*` variables supported by the SaaS A2A server. If the
owner process restarts while a task awaits feedback, the client must resend the
request; Redis keeps protocol task visibility but does not serialize live
workflow continuations.

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
