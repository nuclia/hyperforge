# Hyperforge

Hyperforge is a Python monorepo for building and running agentic workflows on top
of NucliaDB. It contains the Hyperforge orchestration/runtime service, a
NucliaDB Agentic API service, and a set of reusable Hyperforge agent packages.

## Repository Layout

- `hyperforge/`: core framework, REST API, workflow runtime, persistence, broker
  integration, and the optional web UI source.
- `nucliadb_agentic_api/`: NucliaDB-facing agentic API with ASK/search endpoints
  and MCP integrations.
- `agents/`: first-party Hyperforge agent packages that add workflow capabilities
  such as NucliaDB access, HTTP calls, MCP tools, static context, rephrasing,
  summarization, and routing.
- `hyperforge/frontend/`: Vue UI source bundled by the Hyperforge package when
  built.
- `.github/workflows/`: package build and release workflows for the core package
  and agent packages.

## Packages

The workspace is managed with `uv` and Python 3.12 or newer.

Core packages:

- `hyperforge`: agentic framework runtime and API.
- `nucliadb_agentic_api`: NucliaDB Agentic API service.

Framework guides:

- [Harness SDK](docs/harness-sdk.md): build asynchronous tool-loop agents with
  streaming events, persistence, limits, feedback, memory, and sub-agents.

Agent packages:

- `hyperforge_conditional`
- `hyperforge_external`
- `hyperforge_generate`
- `hyperforge_historical`
- `hyperforge_http`
- `hyperforge_mcp`
- `hyperforge_nucliadb`
- `hyperforge_passthrough`
- `hyperforge_related`
- `hyperforge_remi`
- `hyperforge_rephrase`
- `hyperforge_restart`
- `hyperforge_restricted`
- `hyperforge_smart`
- `hyperforge_static`
- `hyperforge_static_string`
- `hyperforge_summarize`

## Development

Install dependencies:

```bash
uv sync
```

Install development dependencies:

```bash
uv sync --group dev
```

Run formatting:

```bash
make fmt
```

Run linting and type checks:

```bash
make lint
```

Run tests:

```bash
uv run pytest
```

## Running Locally

Start the Hyperforge API:

```bash
uv run hyperforge-api
```

Start the NucliaDB Agentic API:

```bash
uv run nucliadb-agentic-api
```

Both services read configuration from environment variables through Pydantic
settings. Important settings include NucliaDB reader/writer/search URLs,
optional NucliaDB API keys, Valkey/Redis URL, service ports, and module loading
configuration.

## Open Source Readiness

This repository is licensed under the Apache License 2.0. See `LICENSE`.

Before publishing a release, review:

- Package metadata in each `pyproject.toml`, especially project URLs and package
  descriptions.
- Dockerfiles, which may need deployment-specific updates for public builds.
- Generated test cassettes and fixtures to ensure they do not contain secrets.
- CI/CD workflow secrets and publishing credentials.

## Contributing

Contributions are welcome. Please format the code with `make fmt`, run
`make lint`, and run the relevant tests before opening a pull request.


## License

[Apache-2.0](LICENSE)

Disclaimer: _This is not an officially supported Progress Software Corporation product._
