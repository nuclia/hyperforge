# Nuclia RAG Agent Framework

## Running Standalone

The `arag-standalone` command runs a self-contained HTTP server for one or more agents defined in a local JSON config file.

### Configuration

Copy one of the example config files and customize it:

- `standalone_config.example.json` — Google Search agent
- `standalone_config.mks.example.json` — Google Search agent (MKS variant)
- `standalone_config-static.example.json` — Static passthrough agent (no external services)

### Environment Variables

| Variable                   | Default      | Description                                                                                                      |
| -------------------------- | ------------ | ---------------------------------------------------------------------------------------------------------------- |
| `AGENTS_CONFIG`            | **required** | Path to the JSON agents config file                                                                              |
| `EXTERNAL_NUA_API_KEY`     | `None`       | Nuclia NUA API key (from https://nuclia.cloud/user/keys). Required unless using `LOCAL_OPENAI` or `INTERNAL_NUA` |
| `LOCAL_OPENAI`             | `None`       | Base URL of a local OpenAI-compatible server (e.g. `http://localhost:11434/v1`)                                  |
| `HOST`                     | `0.0.0.0`    | Listen host                                                                                                      |
| `PORT`                     | `8080`       | Listen port                                                                                                      |
| `LOG_LEVEL`                | `INFO`       | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)                                                                  |
| `DEBUG`                    | `False`      | `True` to prevent structured log output                                                                          |
| `QUESTION_TIMEOUT_SECONDS` | `300`        | Max seconds per question before timeout                                                                          |

### Running

```bash
AGENTS_CONFIG=standalone_config.example.json \
EXTERNAL_NUA_API_KEY=your-nua-api-key \
uv run arag-standalone
```

### Verify the server is running

```bash
curl http://localhost:8080/health/ready
# {"status": "ok"}
```

### Ask a question

Replace `my-agent` with the agent key from your config file and `session-1` with any session identifier:

```bash
curl -X POST http://localhost:8080/api/v1/agent/my-agent/session/session-1 \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the capital of France?"}'
```

## Installation

- For development, cd into the `arag` folder and execute `make install-test`.
- Make sure you have `uv` installed in your system.
- If you have problems downloading the stashify-protos package, make sure that the artifact APIKEY is configured for `uv`:
  `uv tool install keyring --with keyrings.google-artifactregistry-auth`

## PROTOS

If the protos are rebuilt, or if they need to be rebuilt, you can use the 'protos' target in this subproject
to rebuild the protos and reinstall them in this subproject.

```bash
make protos
```

## Snowflake Recorded Test (manual)

There is a remote recorded test for Snowflake at `tests/arag/test_snowflake_recorded.py`.

It is intentionally opt-in and does not run in CI by default because it needs real credentials.

Required environment variables:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_PASSWORD`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_DATABASE`
- `SNOWFLAKE_SCHEMA`

Run it manually:

```bash
uv run pytest tests/arag/test_snowflake_recorded.py --record-mode=none -sxv
```

Re-record cassette:

```bash
uv run pytest tests/arag/test_snowflake_recorded.py --record-mode=rewrite -sxv
```
