# AGENTS.md

## Project Overview

Hyperforge is a Python monorepo containing the core framework and first-party
agent packages. Python 3.12 and `uv` are used for local development and package
management.

## Development

- Run development commands from the relevant subproject directory, such as
  `hyperforge/` or `agents/<name>/`, rather than from the repository root.
- Install dependencies with `uv sync --group dev`.
- Format changes with `make fmt`.
- Run lint and type checks with `make lint`.
- Run the subproject's tests with `make test`.
- Keep changes focused and add or update tests for behavioral changes.

### Tests And VCR Cassettes

Many tests use VCR cassettes to replay previously recorded HTTP interactions.

- `make test` runs tests with `--record-mode=none`. It replays existing
  cassettes and does not record missing or changed interactions. Use this for
  normal development and validation. CI also uses `--record-mode=none`, so
  tests must not require an online connection. Tests that make external calls
  must use VCR and commit the required cassettes.
- `make record` runs tests with `--record-mode=rewrite`. It calls live external
  services and rewrites cassettes, so use it only when adding or intentionally
  updating recorded interactions. Supply the required credentials locally and
  review generated cassette changes carefully to ensure they contain no
  secrets or unstable data.
- Both targets accept pytest arguments through `ARGS`, for example
  `make test ARGS="tests/test_example.py -k test_name"`.

## Creating A New Agent

1. Use an existing package under `agents/` as the template for conventions and
   create `agents/<name>/`.
2. Add package metadata in `agents/<name>/pyproject.toml`, using the package name
   `hyperforge_<name>` and a dependency on `hyperforge`.
3. Implement the package under
   `agents/<name>/src/hyperforge_<name>/`. Define an `AgentConfig`, register the
   agent with `@agent`, and export the agent from `__init__.py`.
4. Add `README.md`, `CHANGELOG.md`, `VERSION`, `Makefile`, and tests consistent
   with neighboring agent packages.
5. Register the package in the root `pyproject.toml`: add its dependency,
   workspace member, and `[tool.uv.sources]` entry.
6. Run `uv lock` so `uv.lock` includes the new workspace package.
7. **Add `agents/<name>` to the `packages` list in `.github/packages.yaml`.**
   This is required for CI change detection, testing, and publishing; creating
   the package without this entry leaves it outside the package workflow.
8. From `agents/<name>/`, run formatting, linting, and tests before submitting
   the change.

Do not add agent-specific secrets globally. If CI tests require credentials,
scope them only to that package in `.github/workflows/package-tests.yaml`.
