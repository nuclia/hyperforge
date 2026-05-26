# Open Source Publishing Checklist

Use this checklist before making the repository public or publishing packages.

## Repository

- Confirm `LICENSE`, `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and
  `SECURITY.md` are accurate for the organization.
- Review `CHANGELOG.md` and release version numbers.
- Confirm public repository URLs in every `pyproject.toml`.
- Review `.github/workflows/` for private registry URLs, internal credentials,
  and publishing targets.
- Confirm Dockerfiles build from public paths and public dependency groups.

## Secrets And Data

- Search for committed credentials, private keys, API tokens, and internal-only
  URLs.
- Review recorded cassettes under tests before publishing.
- Verify example configuration uses placeholders only.
- Ensure `.gitignore` covers local environment files and generated artifacts.

## Packages

- Build every package from a clean checkout.
- Install every package into a fresh virtual environment.
- Verify all console scripts import and start correctly.
- Confirm package READMEs render correctly on PyPI.

## Validation

```bash
uv sync --group dev
make lint
uv run pytest
```
