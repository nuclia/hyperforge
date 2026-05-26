pytest_plugins = [
    "pytest_docker_fixtures",
    "pytest_mock",
    # fixtures from dependencies
    "nuclia.tests.fixtures",
    # "learning_storages.tests.fixtures",
    "nucliadb_sdk.tests.fixtures",
    # "nucliadb_utils.tests.fixtures",
    # "nucliadb_utils.tests.nats",
    # "nucliadb_utils.tests.gcs",
    # "nucliadb_utils.tests.s3",
    # "nucliadb_utils.tests.azure",
    # "nucliadb_utils.tests.local",
    # "nucliadb_telemetry.tests.telemetry",
    # our own fixtures
    "hyperforge.fixtures",
    # "tests.ask.fixtures.standalone",
    # "tests.ask.fixtures.arag_ask",
    # "tests.ask.fixtures.resources",
    # "tests.ask.fixtures.audit",
    # "tests.ask.fixtures.predict",
]
