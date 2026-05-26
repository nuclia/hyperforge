# Hyperforge

Hyperforge is the core agentic workflow framework in this repository. It
provides the runtime, HTTP API, workflow orchestration, broker integration,
persistence layer, and support for loading Hyperforge agent packages.

## Install

From the workspace root:

```bash
uv sync
```

## Run

Start the API service:

```bash
uv run hyperforge-api
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
- `LOAD_MODULES`

## Development

Run the package tests from the workspace root:

```bash
uv run pytest hyperforge
```

Format and lint with the root `Makefile`:

```bash
make fmt
make lint
```
