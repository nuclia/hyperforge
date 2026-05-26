"""Lightweight pytest fixtures for hyperforge agents.

This module only depends on pytest and the stdlib — no docker, database, or
nucliadb deps. Safe to use as a pytest plugin in any agent repo:

    # tests/conftest.py
    pytest_plugins = ["hyperforge.minimal_fixtures"]
"""

import base64
import json
import logging

import pytest


def cassette_nua_key(iss: str) -> str:
    """Return a minimal parseable JWT stub for cassette-replay runs.

    validate_nua() decodes the middle part of the JWT to extract the ``iss``
    field before making any HTTP call.  When cassettes are present VCR
    intercepts that HTTP call, so the key doesn't need to be real — it just
    needs to parse.
    """
    payload = base64.b64encode(json.dumps({"iss": iss}).encode()).decode().rstrip("=")
    return f"cassette.{payload}.stub"


class _VCRTaskExceptionFilter(logging.Filter):
    """Suppress 'Task exception was never retrieved' asyncio errors from vcrpy.

    vcrpy's httpx stub creates a background task (_record_responses) that can
    fail with an AssertionError due to a vcrpy/httpx version incompatibility.
    The exception is noisy but harmless in test runs.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not (
            record.levelno == logging.ERROR
            and "Task exception was never retrieved" in record.getMessage()
            and "_record_responses" in record.getMessage()
        )


@pytest.fixture(scope="module")
def vcr_config():
    return {
        # Replaces the actual token with 'DUMMY' in the recorded YAML
        "filter_headers": [
            ("Authorization", "DUMMY"),
            ("x-nuclia-nuakey", "DUMMY"),
            ("x-stf-nuakey", "DUMMY"),
            ("x-goog-api-key", "DUMMY"),
        ],
        # Redacts specific query parameters like API keys
        "filter_query_parameters": ["api_key", "access_token", "key"],
        # Redacts fields in POST request bodies (e.g., login forms)
        "filter_post_data_parameters": ["password", "client_secret"],
        # Decodes compressed responses so they are human-readable in the cassette
        "decode_compressed_response": True,
    }


@pytest.fixture(autouse=True, scope="session")
def suppress_test_noise() -> None:
    """Suppress known-noisy log lines that add no diagnostic value in tests."""
    logging.getLogger("hyperforge.memory").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)
    logging.getLogger("hyperforge.server").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore.connection").setLevel(logging.ERROR)
    logging.getLogger("httpcore.http11").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.INFO)

    logging.getLogger("asyncio").addFilter(_VCRTaskExceptionFilter())
