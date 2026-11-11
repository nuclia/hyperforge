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

API replicas share logical session metadata through the existing Redis/Valkey
broker. Each HTTP request runs in the receiving process, reconstructing the
SDK's initialization context from the stored client capabilities. No HTTP request,
authorization header, tool result or SSE frame is relayed between replicas.
This uses the existing MCP 2025 protocol and SDK 1.x, not MRTR. Standalone uses
the same adapter only when `BROKER_REDIS_DSN` is configured. Without Redis it uses
the SDK's native stateful transport, restricted to a single process/instance;
there is no in-memory implementation of the shared session store.

With Redis, tool calls use their original POST's SSE connection for elicitation,
progress and the final result. Separate GET streams and SSE reconnection using
Last-Event-ID are not supported (GET returns 405); clients must accept both JSON
and SSE. Initialization, tools/prompts listings and prompt retrieval retain JSON
responses. Standalone without Redis follows native SDK transport behavior:
request responses use SSE and GET streams are supported, without an event store
for replay.

Elicitation replies may arrive on any replica. Unique wire request IDs prevent
collisions even when concurrent calls in one MCP session execute in different
processes. Only pending replies and cancellation markers cross processes through
Redis. Replies are retained until delivered to the waiting SDK request; identical
duplicates cannot execute the tool again. Each session is bound to the complete
authenticated account/user/account-type/agent/session identity, not to an access
token or a pod. Deleting a session cancels its requests on all replicas without
affecting other clients.

Declined or cancelled URL/form elicitations fail the tool instead of resuming
the workflow with an empty response.

Sync Service Ask Agent OAuth is supported over MCP for ShareFile
(`sharefile_oauth`), Google Drive (`google_oauth`) and SharePoint/OneDrive
(`azure_oauth`). The MCP client must support URL elicitation. Authentication
uses the existing Sync service authorization flow and Hyperforge OAuth callback;
the client receives the authorization URL, not the credentials.
Credentials are encrypted and scoped by account, user, agent, provider and Sync
configuration. API deployments persist them in PostgreSQL; standalone keeps them
in process memory, so a restart requires authentication again. SharePoint
certificate connections (`azure_certificate_credentials`) are not supported by
this interactive OAuth flow.

Requests are registered before waiting for a local handler, so cancellation
received during handler creation prevents the tool from starting. Failed or
interrupted admission releases that request's cancellation marker.
Session and call TTLs are renewed during admission until transport polling
takes over; a slow handler startup does not expire an active request.

Requests renew their session and reply/cancellation mailboxes while active,
including while waiting for user input. Mailboxes use ordinary Redis keys with
TTLs, not execution leases or a distributed execution registry. Small
WATCH/MULTI transactions protect session admission and mailbox updates; there
are no custom Lua scripts. Session and registry TTLs are renewed atomically so
valid sessions remain included in admission accounting.
Store operations have bounded timeouts;
failed coordination stops the local execution instead of falling back to
uncoordinated state. Redis failures return 503 before a response starts, or
terminate an already-started stream. Idle sessions expire independently using
`MCP_SESSION_IDLE_TTL_SECONDS`. A pod shutdown does not remove shared sessions;
other replicas can handle subsequent requests.

Live Python tasks are not migrated and tools are never automatically replayed.
A pod failure or interrupted SSE connection can leave the outcome of an in-flight
call unknown; clients must not blindly retry tools with side effects. Existing
connections from the previous owner/relay implementation require reinitialization
when deploying this version. Drain the previous version before admitting traffic
to the new replicas; do not mix both session implementations behind one service.

With the Redis adapter, admission is bounded to 100 active HTTP requests per
process, including requests waiting for handler creation, and 100 pending server
requests per HTTP request. Slots are reserved before Redis call registration and
released on admission failure or response cleanup. There is no distributed
concurrent-call quota.
Control messages and replies
bypass that admission limit so full capacity cannot prevent completing or
cancelling a tool. `MCP_MAX_SESSIONS_PER_SERVER` limits shared sessions per
authenticated server identity, while `MCP_MAX_SERVERS` bounds each process's
cached handlers. Native SDK streams provide bounded in-process buffering.
`MCP_MAX_REQUEST_BYTES` limits incoming messages in both modes.
`MCP_MAX_RESPONSE_BYTES` limits each outgoing protocol message in the Redis
adapter; the native standalone transport uses SDK defaults for outgoing messages.

Use a private, trusted Redis deployment: pending elicitation replies may contain
sensitive user input. They are deleted after delivery or request cleanup; all
session and mailbox records have TTLs so a dead process cannot retain them
indefinitely. Coordination keys use hashed identities and a common
Redis Cluster hash tag so atomic operations stay within one slot.

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
