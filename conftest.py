import logging

import pytest


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
        ],
        # Redacts specific query parameters like API keys
        "filter_query_parameters": ["api_key", "access_token"],
        # Redacts fields in POST request bodies (e.g., login forms)
        "filter_post_data_parameters": ["password", "client_secret"],
        # Decodes compressed responses so they are human-readable in the cassette
        "decode_compressed_response": True,
    }


@pytest.fixture(autouse=True, scope="session")
def suppress_test_noise() -> None:
    """Suppress known-noisy log lines that add no diagnostic value in tests."""
    logging.getLogger("nucliadb_utils.utilities").setLevel(logging.ERROR)

    logging.getLogger("arag.memory").setLevel(logging.WARNING)
    logging.getLogger("mcp.server.streamable_http").setLevel(logging.WARNING)
    logging.getLogger("hyperforge.server").setLevel(logging.WARNING)

    asyncio_logger = logging.getLogger("asyncio")
    asyncio_logger.addFilter(_VCRTaskExceptionFilter())


pytest_plugins = [
    "pytest_docker_fixtures",
    "pytest_mock",
    # fixtures from dependencies
    "nuclia.tests.fixtures",
    "learning_storages.tests.fixtures",
    "nucliadb_sdk.tests.fixtures",
    "nucliadb_utils.tests.fixtures",
    "nucliadb_utils.tests.nats",
    "nucliadb_utils.tests.gcs",
    "nucliadb_utils.tests.s3",
    "nucliadb_utils.tests.azure",
    "nucliadb_utils.tests.local",
    "nucliadb_telemetry.tests.telemetry",
    # our own fixtures
    "hyperfrage.fixtures",
    "tests.ask.fixtures.standalone",
    "tests.ask.fixtures.arag_ask",
    "tests.ask.fixtures.resources",
    "tests.ask.fixtures.audit",
    "tests.ask.fixtures.predict",
]
