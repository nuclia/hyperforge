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
uv run pytest nucliadb_agentic_api
```
