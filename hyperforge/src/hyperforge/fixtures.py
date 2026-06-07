import asyncio
import logging
import os
import pathlib
import socket
from unittest.mock import patch
from uuid import uuid4

import alembic.command
import alembic.config
import databases
import nucliadb_sdk
import pytest
import requests
import uvicorn
from cryptography.fernet import Fernet
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from nuclia.config import NuaKey, Selection
from nuclia.data import get_auth
from nuclia.sdk import NucliaPredict
from nucliadb_models.resource import KnowledgeBoxObj
from nucliadb_sdk import NucliaDB, NucliaDBAsync
from nucliadb_sdk.tests.fixtures import NucliaFixture
from pytest_docker_fixtures import images  # type: ignore  # type: ignore
from pytest_docker_fixtures.containers.pg import pg_image  # type: ignore
from pytest_docker_fixtures.containers.valkey import valkey_image  # type: ignore
from redis.asyncio import Redis
from sqlalchemy import create_engine
from sqlalchemy_utils import (  # type: ignore
    create_database,
    database_exists,
    drop_database,
)

from hyperforge.api.app import HTTPApplication
from hyperforge.api.settings import Settings
from hyperforge.broker.redis import RedisBroker
from hyperforge.db.agents import AgentManager
from hyperforge.db.settings import DataManagerSettings
from hyperforge.models import MemoryConfig, NucliaDBMemoryConfig, Rules
from hyperforge.server.cache import ValkeyCache
from hyperforge.server.session import SessionManager
from hyperforge.server.settings import Settings as ServerSettings
from hyperforge.utils.http import SafeTransport
import hyperforge  # noqa: F401

_package_path = pathlib.Path(hyperforge.__file__).parent.absolute()

NUA = os.environ.get("NUA_KEY", "DUMMY")

images.settings["nucliadb"]["env"]["NUA_API_KEY"] = NUA
images.settings["nucliadb"]["env"]["DUMMY_PREDICT"] = "False"


NUCLIA_Make_dataset = (
    "https://storage.googleapis.com/ncl-testbed-gcp-stage-1/test_nucliadb/make.export"
)

NUCLIA_Make_article = "https://storage.googleapis.com/ncl-testbed-gcp-stage-1/test_nucliadb/articles.export"


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


async def init_fixture(
    nucliadb: NucliaFixture,
    dataset_slug: str,
    dataset_location: str,
    semantic_model: str = "en-2024-04-24",
    generative_model: str = "chatgpt-azure-4o",
    kbid: str | None = None,
):
    async with AsyncClient() as client:
        resp = await client.get(
            f"http://{nucliadb.host}:{nucliadb.port}/api/v1/config-check",
            headers={"X-NUCLIADB-ROLES": "READER"},
        )
        assert resp.status_code == 200, "NUA KEY not configured"
        assert resp.json()["nua_api_key"]["valid"], "NUA KEY not valid"
    sdk = nucliadb_sdk.NucliaDB(region="on-prem", url=nucliadb.url)
    slug = dataset_slug
    learning_configuration = {
        "semantic_model": semantic_model,
        "semantic_models": [semantic_model],
        "semantic_vector_similarity": "dot",
        "semantic_vector_size": 768,
        "semantic_threshold": 0.47,
        "semantic_matryoshka_dims": [],
        "semantic_model_configs": {
            semantic_model: {
                "similarity": 0,
                "size": 768,
                "threshold": 0.47,
                "max_tokens": 2048,
                "matryoshka_dims": [],
            }
        },
        "generative_model": generative_model,
    }

    if kbid is not None:
        auth = get_auth()
        auth._config.nuas_token = [
            NuaKey(
                client_id="nucliadb",
                region="europe-1",
                account="nuclia",
                token=NUA,
                account_type="service",
            )
        ]
        auth._config.default = Selection(nua="nucliadb")
        np = NucliaPredict()
        np.del_config(
            kbid,
        )

    kb_obj = sdk.create_knowledge_box(
        uuid=kbid, slug=slug, learning_configuration=learning_configuration
    )
    kbid = kb_obj.uuid

    import_resp = requests.get(dataset_location)  # noqa: ASYNC210
    assert import_resp.status_code == 200, (
        f"Error pulling dataset {dataset_location}:{import_resp.status_code}"
    )
    import_data = import_resp.content

    import_id = sdk.start_import(kbid=kbid, content=import_data).import_id
    assert sdk.import_status(kbid=kbid, import_id=import_id).status.value == "finished"

    return kbid


@pytest.fixture(scope="session")
def make_dataset(nucliadb: NucliaFixture):

    kbid = asyncio.run(
        init_fixture(
            nucliadb,
            "conv",
            NUCLIA_Make_dataset,
            kbid="00000000-0000-0000-0000-000000000002",
        )
    )
    yield kbid


@pytest.fixture(scope="session")
def article_dataset(nucliadb: NucliaFixture):
    kbid = asyncio.run(
        init_fixture(
            nucliadb,
            "conv",
            NUCLIA_Make_article,
            "multilingual-2024-05-06",
            "chatgpt-4.1",
            kbid="00000000-0000-0000-0000-000000000001",
        )
    )
    yield kbid


@pytest.fixture
async def arag_settings(sdk_async: NucliaDBAsync, valkey_url: str):
    yield Settings(
        running_environment="test",
        valkey_url=valkey_url,
        valkey_cluster_mode=False,
        memory_reader_nucliadb=sdk_async.base_url,
        memory_writer_nucliadb=sdk_async.base_url,
        memory_search_nucliadb=sdk_async.base_url,
        dummy_idp=True,
    )


images.settings["postgresql"].update(
    {
        "version": "16.1",
        "env": {
            "POSTGRES_PASSWORD": "postgres",
            "POSTGRES_DB": "postgres",
            "POSTGRES_USER": "postgres",
        },
    }
)


@pytest.fixture(scope="session")
def pg():
    host, port = pg_image.run()
    yield host, port
    pg_image.stop()


@pytest.fixture(scope="session")
def pg_dsn(pg):
    host, port = pg
    dsn = f"postgresql://postgres:postgres@{host}:{port}/test_db"

    if database_exists(dsn):
        drop_database(dsn)
    create_database(dsn)
    config = alembic.config.Config(str(_package_path) + "/alembic.ini")
    config.set_main_option("sqlalchemy.url", dsn)
    alembic.command.upgrade(config, "head")
    yield dsn


@pytest.fixture(scope="function")
def test_db(pg_dsn):
    engine = create_engine(pg_dsn)
    with engine.connect() as conn:
        yield conn


@pytest.fixture
async def data_manager_settings(pg_dsn):
    yield DataManagerSettings(postgresql_dsn=pg_dsn)


@pytest.fixture
async def arag_api_app(
    arag_settings: Settings,
    data_manager_settings: DataManagerSettings,
):
    application = HTTPApplication(
        settings=arag_settings,
        data_manager_settings=data_manager_settings,
    )

    await application.startup()

    yield application

    await application.shutdown()


@pytest.fixture
async def arag_api(arag_api_app: HTTPApplication, load_agents):
    yield AsyncClient(transport=ASGITransport(app=arag_api_app), base_url="http://test")


@pytest.fixture
async def arag_api_http(
    arag_api_app: HTTPApplication,
):
    """Serve the already-started arag_api_app over real HTTP/WebSocket.

    Reuses the same HTTPApplication instance as arag_api_app so that only one
    PredictEngine (and one aiohttp session) is created per test. Uvicorn is
    started with lifespan="off" to avoid calling startup/shutdown a second time.
    """
    http_port = free_port()
    config = uvicorn.Config(
        arag_api_app,
        host="127.0.0.1",
        port=http_port,
        log_level="warning",
        lifespan="off",
    )
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())
    while not server.started and not server.should_exit:
        await asyncio.sleep(0.01)
    if not server.started:
        await server_task
        raise RuntimeError("arag_api_http failed to start")

    yield f"127.0.0.1:{http_port}"

    server.should_exit = True
    await server_task


@pytest.fixture
async def arag_api_http_client(
    arag_api_http: str,
):
    """
    Fixture to provide an HTTP client for the Arag API.
    """
    async with AsyncClient(base_url=f"http://{arag_api_http}") as client:
        yield client


@pytest.fixture
async def arag_api_http_session(
    arag_api_http_client: AsyncClient,
    arag_kb: KnowledgeBoxObj,
):
    resp = await arag_api_http_client.post(
        f"/api/v1/agent/{arag_kb.uuid}/sessions",
        json={
            "slug": "slug1",
            "name": "My Title",
            "summary": "This is a nice user",
            "data": '{"age": "46"}',
            "format": "JSON",
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200
    session_id = resp.json()["uuid"]
    yield session_id
    await arag_api_http_client.delete(
        f"/api/v1/agent/{arag_kb.uuid}/session/{session_id}",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )


@pytest.fixture
async def arag(
    request,
    arag_kb: KnowledgeBoxObj,
    arag_kb_legacy: KnowledgeBoxObj,
    arag_no_memory: str,
):
    """Indirect fixture to support parametrization of different arag KB types."""
    fixture_map = {
        "arag_kb": arag_kb.uuid,
        "arag_kb_legacy": arag_kb_legacy.uuid,
        "arag_no_memory": arag_no_memory,
    }
    return fixture_map[request.param]


@pytest.fixture
async def arag_kb(sdk: NucliaDB, arag_api_app: HTTPApplication):
    kb = await create_arag_kb(
        arag_api_app.agent_manager, sdk, "test_basic_worker", "nuclia"
    )
    yield kb
    await delete_arag_kb(arag_api_app.agent_manager, sdk, kb.uuid, "nuclia")


@pytest.fixture
async def arag_no_memory(arag_api_app: HTTPApplication):
    arag_id = await create_arag_no_memory(arag_api_app.agent_manager, "nuclia")
    yield arag_id
    await delete_arag_no_memory(arag_api_app.agent_manager, arag_id, "nuclia")


@pytest.fixture
async def arag_kb_legacy(sdk: NucliaDB, arag_api_app: HTTPApplication):
    """
    We have some rows that have memory set to null but actually have a memory KB.
    (Legacy from before we added the no KB option)
    """
    kb = await create_arag_kb(
        arag_api_app.agent_manager, sdk, "test_basic_worker_legacy", "nuclia"
    )
    # Go to the DB and set memory to null
    db: databases.Database = arag_api_app.agent_manager.database
    from hyperforge.db.agents import retrieval_agent_config

    statement = (
        retrieval_agent_config.update()
        .where(retrieval_agent_config.c.agent_id == kb.uuid)
        .values(memory=None)
    )
    await db.execute(statement)
    yield kb
    await delete_arag_kb(arag_api_app.agent_manager, sdk, kb.uuid, "nuclia")


@pytest.fixture
async def agent_db_server(data_manager_settings: DataManagerSettings):
    agent_manager = await AgentManager.from_settings(settings=data_manager_settings)
    await agent_manager.initialize()
    return agent_manager


@pytest.fixture
async def arag_server(
    sdk: NucliaDB,
    agent_db_server: AgentManager,
    valkey,
):
    valkey_host, valkey_port = valkey
    valkey_url = f"redis://{valkey_host}:{valkey_port}"
    settings = ServerSettings(
        valkey_url=valkey_url,
        valkey_cluster_mode=False,
        internal_nucliadb=True,
        internal_nucliadb_url=sdk.base_url,
        internal_nua=False,
        local_openai=None,
        external_nua_api_key=NUA,
    )
    broker = RedisBroker.from_url(
        url=valkey_url,
        activate_subject=settings.activate_subject,
        keepalive_ms=int(settings.pubsub_keepalive_seconds * 1000),
        cluster_mode=settings.valkey_cluster_mode,
    )
    session = SessionManager(
        settings=settings,
        broker=broker,
        agent_manager=agent_db_server,
        cache=ValkeyCache(
            Redis(host=valkey_host, port=valkey_port, decode_responses=True)
        ),
    )
    await session.initialize()
    yield session
    await session.finalize()


@pytest.fixture
async def arag_server_kb(sdk_async: NucliaDBAsync, arag_server: SessionManager):
    kb: KnowledgeBoxObj = await sdk_async.create_knowledge_box(slug="test_basic_worker")

    assert isinstance(arag_server.agent_manager, AgentManager)
    await arag_server.agent_manager.add_agent(
        account="nuclia",
        agent_id=kb.uuid,
        rules=Rules(rules=[]),
        memory=MemoryConfig(
            nucliadb=NucliaDBMemoryConfig(url=sdk_async.base_url, kbid=kb.uuid)
        ),
    )

    yield kb

    await sdk_async.delete_knowledge_box(kbid=kb.uuid)


@pytest.fixture
async def arag_api_session(
    arag_api: AsyncClient,
    arag_kb: KnowledgeBoxObj,
):
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/sessions",
        json={
            "slug": "slug1",
            "name": "My Title",
            "summary": "This is a nice user",
            "data": '{"age": "46"}',
            "format": "JSON",
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200
    session_id = resp.json()["uuid"]
    yield session_id
    # Best effort cleanup - don't fail if deletion fails
    try:
        await arag_api.delete(
            f"/api/v1/agent/{arag_kb.uuid}/session/{session_id}",
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )
    except Exception:
        # Ignore cleanup errors
        pass


@pytest.fixture(scope="session")
async def valkey():
    host, port = valkey_image.run()
    yield host, port
    valkey_image.stop()


@pytest.fixture
async def valkey_cache(valkey):
    yield ValkeyCache(Redis(host=valkey[0], port=valkey[1], decode_responses=True))


@pytest.fixture
async def valkey_url(valkey):
    yield f"redis://{valkey[0]}:{valkey[1]}"


@pytest.fixture(autouse=True, scope="session")
def setup_encryption_key():
    os.environ["ENCRYPTION_SECRET_KEY"] = Fernet.generate_key().decode()
    yield


@pytest.fixture
async def disable_safe_transport():
    with patch.object(SafeTransport, "is_private_address", return_value=False):
        yield


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


# Import vcr_config and suppress_test_noise from minimal_fixtures (single source of truth)
from hyperforge.minimal_fixtures import (  # noqa: E402, F401
    suppress_test_noise,
    vcr_config,
)


async def delete_arag_kb(
    agent_db: AgentManager, sdk: NucliaDB, kbid: str, account: str
) -> None:
    sdk.delete_knowledge_box(kbid=kbid)
    await agent_db.delete_agent(
        account=account,
        agent_id=kbid,
    )


async def create_arag_kb(
    agent_db: AgentManager, sdk: NucliaDB, slug: str, account: str
) -> KnowledgeBoxObj:
    kb: KnowledgeBoxObj = sdk.create_knowledge_box(slug=slug)
    await agent_db.add_agent(
        account=account,
        agent_id=kb.uuid,
        rules=Rules(rules=[]),
        memory=MemoryConfig(
            nucliadb=NucliaDBMemoryConfig(url=sdk.base_url, kbid=kb.uuid, internal=True)
        ),
    )

    return kb


async def create_arag_no_memory(agent_db: AgentManager, account: str) -> str:
    uuid = str(uuid4())
    await agent_db.add_agent(
        account=account,
        agent_id=uuid,
        rules=Rules(rules=[]),
        memory=MemoryConfig(),
    )

    return uuid


async def delete_arag_no_memory(
    agent_db: AgentManager, agent_id: str, account: str
) -> None:
    await agent_db.delete_agent(
        account=account,
        agent_id=agent_id,
    )


@pytest.fixture
def load_agents():
    from hyperforge.configure import load_all_configurations, scan

    for module in [
        "hyperforge_external",
        "hyperforge_conditional",
        "hyperforge_nucliadb",
        "hyperforge_smart",
        "hyperforge_remi",
        "hyperforge_restricted",
        "hyperforge_summarize",
        "hyperforge_static",
        "hyperforge_rephrase",
        "hyperforge_mcp",
        "hyperforge_remi",
    ]:
        scan(module)
        load_all_configurations(module)
