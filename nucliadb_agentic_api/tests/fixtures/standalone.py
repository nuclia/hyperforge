import base64
import os
from typing import Any, AsyncIterator, Iterator
from unittest.mock import patch

import docker  # type: ignore[import-untyped]
import pytest
from grpc import aio
from httpx import AsyncClient
from nucliadb_models.resource import NucliaDBRoles
from nucliadb_protos.writer_pb2_grpc import WriterStub
from nucliadb_sdk.tests.fixtures import NucliaFixture
from pytest_docker_fixtures import images  # type: ignore[import-untyped]

# docker nucliadb environment
#
# We use it to configure nucliadb and share settings with us
environ = images.settings["nucliadb"]["env"]


# Main fixtures


@pytest.fixture(scope="session")
async def standalone_nucliadb(
    # set up docker environment
    analytics_disabled,
    dummy_learning,
    endecryptor_settings,
    shared_storage,
    # standalone nucliadb on docker
    nucliadb: NucliaFixture,
) -> AsyncIterator[NucliaFixture]:
    """Session NucliaDB running on docker with shared storage (with us).

    This is a session fixture as we want to avoid starting a nucliadb for each
    test. As settings are also for the whole session, all fixtures related are
    session too.

    NOTE this is meant to work only for docker and not leveraging other options
    the `nucliadb` fixture has (like start a local nucliadb or point to a
    localhost one). There's no technical limitation but we configure the docker
    environment during test session setup

    """
    yield nucliadb


@pytest.fixture(scope="function")
async def nucliadb_reader(
    standalone_nucliadb: NucliaFixture,
) -> AsyncIterator[AsyncClient]:
    nucliadb_address = f"{standalone_nucliadb.host}:{standalone_nucliadb.port}"
    async with AsyncClient(
        base_url=f"http://{nucliadb_address}/api/v1",
        headers={
            "X-NUCLIADB-ROLES": "READER",
            "X-NUCLIADB-USER": "ndbtests",
        },
        timeout=None,
    ) as client:
        yield client


@pytest.fixture(scope="function")
async def nucliadb_writer(
    standalone_nucliadb: NucliaFixture,
) -> AsyncIterator[AsyncClient]:
    nucliadb_address = f"{standalone_nucliadb.host}:{standalone_nucliadb.port}"
    async with AsyncClient(
        base_url=f"http://{nucliadb_address}/api/v1",
        headers={
            "X-NUCLIADB-ROLES": "WRITER",
            "X-NUCLIADB-USER": "ndbtests",
        },
        timeout=None,
    ) as client:
        yield client


@pytest.fixture(scope="function")
async def nucliadb_ingest_grpc(
    standalone_nucliadb: NucliaFixture,
) -> AsyncIterator[WriterStub]:
    nucliadb_grpc_address = f"{standalone_nucliadb.host}:{standalone_nucliadb.grpc}"
    channel = aio.insecure_channel(nucliadb_grpc_address)
    stub = WriterStub(channel)
    yield stub  # type: ignore
    await channel.close(grace=None)


# Derived


@pytest.fixture(scope="function")
async def nucliadb_reader_manager(
    nucliadb_reader: AsyncClient,
) -> AsyncIterator[AsyncClient]:
    roles = [NucliaDBRoles.MANAGER, NucliaDBRoles.READER]
    nucliadb_reader.headers["X-NUCLIADB-ROLES"] = ";".join(
        [role.value for role in roles]
    )
    yield nucliadb_reader


@pytest.fixture(scope="function")
async def nucliadb_writer_manager(
    nucliadb_writer: AsyncClient,
) -> AsyncIterator[AsyncClient]:
    roles = [NucliaDBRoles.MANAGER, NucliaDBRoles.WRITER]
    extra_roles = ["OWNER"]
    nucliadb_writer.headers["X-NUCLIADB-ROLES"] = ";".join(
        [role.value for role in roles] + extra_roles
    )
    yield nucliadb_writer


# NucliaDB environment configuration


@pytest.fixture(scope="session")
def analytics_disabled() -> Iterator[None]:
    with patch.dict(environ, {"NUCLIADB_DISABLE_ANALYTICS": "true"}, clear=False):
        yield


@pytest.fixture(scope="session")
def dummy_learning() -> Iterator[None]:
    with patch.dict(
        environ,
        {
            "DUMMY_LEARNING_SERVICES": "true",
            "DUMMY_PREDICT": "true",
            "DUMMY_PROCESSING": "true",
        },
        clear=False,
    ):
        yield


@pytest.fixture(scope="session")
def endecryptor_settings() -> Iterator[None]:
    secret_key = os.urandom(32)
    encoded_secret_key = base64.b64encode(secret_key).decode("utf-8")

    with patch.dict(
        environ, {"ENCRYPTION_SECRET_KEY": encoded_secret_key}, clear=False
    ):
        yield


@pytest.fixture(scope="session")
def shared_storage(
    overwrite_gcs_command,
    session_storage_settings: tuple[dict[str, Any], dict[str, Any]],
):
    from nucliadb_utils.settings import StorageSettings
    from nucliadb_utils.storages.settings import Settings

    raw_settings, raw_extended_settings = session_storage_settings

    settings = StorageSettings(**raw_settings)
    extended_settings = Settings(**raw_extended_settings)

    env = settings.model_dump(
        mode="json", exclude_defaults=True, exclude_unset=True
    ) | extended_settings.model_dump(
        mode="json", exclude_defaults=True, exclude_unset=True
    )

    with patch.dict(
        environ,
        env,
        clear=False,
    ):
        yield


@pytest.fixture(scope="session")
def overwrite_gcs_command():
    """Hacky fixture needed to fix the command used to start fake-gcs-server so
    it works with our setup.

    Both external-url and public-host must point to the network gateway so URLs
    returned from the fake-gcs-server have an accessible IP.

    REVIEW: the current image doesn't quite work for resumable uploads. I
    suspect the mixture of JSON and XML APIs could be the culprit. However, as
    we don't need them, we don't really care

    """
    from nucliadb_utils.tests.gcs import GCS

    network_gateway = (
        docker.from_env()
        .networks.get(GCS.default_network)
        .attrs["IPAM"]["Config"][0]["Gateway"]
    )
    with (
        patch.dict(
            images.settings["gcs"],
            {
                "image": "europe-west4-docker.pkg.dev/nuclia-internal/public/fake-gcs-server"
            },
            clear=False,
        ),
        patch.dict(
            images.settings["gcs"]["options"],
            {
                "command": f"-scheme http -external-url http://{network_gateway}:{{port}} -port {{port}} -public-host {network_gateway}:{{port}}"
            },
            clear=False,
        ),
    ):
        yield
