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

## Signing the CLA

hyperforge is an opensource project licensed a Apache 2.0.

Contributors are required to sign a Contributor License Agreement.
The process is simple and fast. Upon your first pull request, you will be prompted to
[sign our CLA by visiting this link](https://cla-assistant.io/nuclia/nucliadb).
