# Contributing

Thank you for contributing to Hyperforge.

## Development Setup

Install the workspace with development dependencies:

```bash
uv sync --group dev
```

## Checks

Before opening a pull request, run:

```bash
make fmt
make lint
uv run pytest
```

If your change only affects one package, running the relevant package tests is
acceptable.

## Pull Requests

- Keep changes focused and minimal.
- Add or update tests for behavioral changes.
- Update README files when changing user-visible behavior or configuration.
- Do not commit secrets, local environment files, generated caches, or build
  artifacts.
