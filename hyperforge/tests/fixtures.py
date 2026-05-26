import asyncio
import os
import pathlib
import socket
import uuid
from typing import AsyncIterator
from unittest.mock import patch

import alembic.command
import alembic.config
import asyncpg  # type: ignore
import databases
import nucliadb_sdk
import pytest
import requests
import uvicorn
from cryptography.fernet import Fernet
from grpc.aio import insecure_channel
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from hyperforge.api.app import HTTPApplication
from hyperforge.api.settings import Settings
from hyperforge.broker.redis import RedisBroker
from hyperforge.memory import MemoryConfig, NucliaDBMemoryConfig, Rules
from hyperforge.utils.http import SafeTransport
from hyperforge_database.agents import AgentManager
from hyperforge_database.settings import DataManagerSettings
from hyperforge_server.cache import ValkeyCache
from hyperforge_server.session import SessionManager
from hyperforge_server.settings import Settings as ServerSettings
from nuclia.config import NuaKey, Selection
from nuclia.data import get_auth
from nuclia.sdk import NucliaPredict
from nucliadb_models.resource import KnowledgeBoxObj
from nucliadb_sdk import NucliaDB, NucliaDBAsync
from nucliadb_sdk.tests.fixtures import NucliaFixture
from pytest_docker_fixtures import images  # type: ignore
from pytest_docker_fixtures.containers._base import BaseImage  # type: ignore
from pytest_docker_fixtures.containers.pg import pg_image  # type: ignore
from pytest_docker_fixtures.containers.valkey import valkey_image  # type: ignore
from redis.asyncio import Redis
from sqlalchemy import create_engine
from sqlalchemy_utils import (  # type: ignore
    create_database,
    database_exists,
    drop_database,
)

_dir = pathlib.Path(__file__).parent.absolute()
_package_path = _dir.parent.absolute()

images.settings["nucliadb"]["env"]["NUCLIA_SERVICE_ACCOUNT"] = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6Im51YSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3MzcwNjE2NjUsInN1YiI6IjAzZDQ3OTk4LWY2NzItNGE5Yi1hNTdiLWJkZTNhOWMzZWU4YyIsImp0aSI6IjQ1YTQ3NjJkLTIyMzgtNGI2OC05YTI2LWQ4N2QzNjJhZThmYiIsImV4cCI6MjUzMzcwNzY0ODAwLCJrZXkiOiI0N2JjZDU4ZC04NDExLTQ1NTgtYWIzZS0wNGMyMWI4NWI2ZWEiLCJhbGxvd19rYl9tYW5hZ2VtZW50IjpmYWxzZX0.Ljgv780vMuwviospTcRQYxrFV_H7XXR0hJeeSyFIfwVjni7hyyrxB189R5rQyLLI2n85iAdNGshvc8etDQRkXr8n8IWFsy_FOWcru-LZFZwGCpsY6hKK4TdWXR9v5sxA5xyKA7lmWw1LZ8dfNbcdx11OY15BfmGuMpiq_auIs1F90C8T8_LmXbz0SbdYzPIoEP0JFBX92jHqDoJNUTlMELUrcjupK9ao2pZahI47zQHrWjGuw2KrSjghdZgzwjC0YEa7C8quEVZ9SoLOkJvJV7XV4LrlGGcsxZzng8kLBGRBS-i8p26n5vFvMqiZKqDWpq68cVzZhAsL93wkzHVZCAHpfEsHQ4DUb-Da53xUrrnVnyl1w79iXiLYwP0wxh3b34B1b1ca3rRKuifbd1e762gf11qw6LHpJ9qKYhRv6O3KZ18_amwjLhqYna5uUfrP7f59tJZ9vzTG1oTZ5KlMBeVfu_IvhAmMbGpTygqEoxXqNrH3lWOsEPLhRVBC6D5t84xy7WLe4XsGR4xWduLWHsjxPYbmTrLMysGSqBSNGPwUi8jMTrH16-xprNJRiWVHcvgz_FGQ7sT7RucaAxhmFlZY9h3BFw7u_6awOeX4ymhH6_iDzWxBc0Fx5JsDgQm9jkhlYIHqZG36N5XfsmqfCyM12gNa37j-8MPOt7eU0XQ"
)
images.settings["nucliadb"]["env"]["NUA_API_KEY"] = (
    "eyJhbGciOiJSUzI1NiIsImtpZCI6Im51YSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3MzcwNjE2NjUsInN1YiI6IjAzZDQ3OTk4LWY2NzItNGE5Yi1hNTdiLWJkZTNhOWMzZWU4YyIsImp0aSI6IjQ1YTQ3NjJkLTIyMzgtNGI2OC05YTI2LWQ4N2QzNjJhZThmYiIsImV4cCI6MjUzMzcwNzY0ODAwLCJrZXkiOiI0N2JjZDU4ZC04NDExLTQ1NTgtYWIzZS0wNGMyMWI4NWI2ZWEiLCJhbGxvd19rYl9tYW5hZ2VtZW50IjpmYWxzZX0.Ljgv780vMuwviospTcRQYxrFV_H7XXR0hJeeSyFIfwVjni7hyyrxB189R5rQyLLI2n85iAdNGshvc8etDQRkXr8n8IWFsy_FOWcru-LZFZwGCpsY6hKK4TdWXR9v5sxA5xyKA7lmWw1LZ8dfNbcdx11OY15BfmGuMpiq_auIs1F90C8T8_LmXbz0SbdYzPIoEP0JFBX92jHqDoJNUTlMELUrcjupK9ao2pZahI47zQHrWjGuw2KrSjghdZgzwjC0YEa7C8quEVZ9SoLOkJvJV7XV4LrlGGcsxZzng8kLBGRBS-i8p26n5vFvMqiZKqDWpq68cVzZhAsL93wkzHVZCAHpfEsHQ4DUb-Da53xUrrnVnyl1w79iXiLYwP0wxh3b34B1b1ca3rRKuifbd1e762gf11qw6LHpJ9qKYhRv6O3KZ18_amwjLhqYna5uUfrP7f59tJZ9vzTG1oTZ5KlMBeVfu_IvhAmMbGpTygqEoxXqNrH3lWOsEPLhRVBC6D5t84xy7WLe4XsGR4xWduLWHsjxPYbmTrLMysGSqBSNGPwUi8jMTrH16-xprNJRiWVHcvgz_FGQ7sT7RucaAxhmFlZY9h3BFw7u_6awOeX4ymhH6_iDzWxBc0Fx5JsDgQm9jkhlYIHqZG36N5XfsmqfCyM12gNa37j-8MPOt7eU0XQ"
)

images.settings["nucliadb"]["env"]["DUMMY_PREDICT"] = "False"


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


images.settings["neo4j"] = {
    "image": "neo4j",
    "version": "latest",
    "env": {},
    "options": {
        "ports": {"7474": ("0.0.0.0", 0), "7687": ("0.0.0.0", 0)},
        "publish_all_ports": False,
    },
}

NUA_NUCLIADB = "eyJhbGciOiJSUzI1NiIsImtpZCI6Im51YSIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3MzcwNjE2NjUsInN1YiI6IjAzZDQ3OTk4LWY2NzItNGE5Yi1hNTdiLWJkZTNhOWMzZWU4YyIsImp0aSI6IjQ1YTQ3NjJkLTIyMzgtNGI2OC05YTI2LWQ4N2QzNjJhZThmYiIsImV4cCI6MjUzMzcwNzY0ODAwLCJrZXkiOiI0N2JjZDU4ZC04NDExLTQ1NTgtYWIzZS0wNGMyMWI4NWI2ZWEiLCJhbGxvd19rYl9tYW5hZ2VtZW50IjpmYWxzZX0.Ljgv780vMuwviospTcRQYxrFV_H7XXR0hJeeSyFIfwVjni7hyyrxB189R5rQyLLI2n85iAdNGshvc8etDQRkXr8n8IWFsy_FOWcru-LZFZwGCpsY6hKK4TdWXR9v5sxA5xyKA7lmWw1LZ8dfNbcdx11OY15BfmGuMpiq_auIs1F90C8T8_LmXbz0SbdYzPIoEP0JFBX92jHqDoJNUTlMELUrcjupK9ao2pZahI47zQHrWjGuw2KrSjghdZgzwjC0YEa7C8quEVZ9SoLOkJvJV7XV4LrlGGcsxZzng8kLBGRBS-i8p26n5vFvMqiZKqDWpq68cVzZhAsL93wkzHVZCAHpfEsHQ4DUb-Da53xUrrnVnyl1w79iXiLYwP0wxh3b34B1b1ca3rRKuifbd1e762gf11qw6LHpJ9qKYhRv6O3KZ18_amwjLhqYna5uUfrP7f59tJZ9vzTG1oTZ5KlMBeVfu_IvhAmMbGpTygqEoxXqNrH3lWOsEPLhRVBC6D5t84xy7WLe4XsGR4xWduLWHsjxPYbmTrLMysGSqBSNGPwUi8jMTrH16-xprNJRiWVHcvgz_FGQ7sT7RucaAxhmFlZY9h3BFw7u_6awOeX4ymhH6_iDzWxBc0Fx5JsDgQm9jkhlYIHqZG36N5XfsmqfCyM12gNa37j-8MPOt7eU0XQ"

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
                token=NUA_NUCLIADB,
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


class Neo4jImage(BaseImage):
    name = "neo4j"


@pytest.fixture(scope="session")
def neo4j():
    neo4j = Neo4jImage()
    host, port = neo4j.run()
    yield host, port
    neo4j.stop()


@pytest.fixture(scope="session")
def pg():
    host, port = pg_image.run()
    yield host, port
    pg_image.stop()


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
async def pg_example(pg):
    host, port = pg
    dsn = f"postgresql://postgres:postgres@{host}:{port}/test_db_sql"

    if not database_exists(dsn):
        create_database(dsn)

    conn = await asyncpg.connect(dsn)
    await conn.execute(
        """
CREATE TABLE cars (
  brand VARCHAR(255),
  model VARCHAR(255),
  year INT
);
"""
    )
    await conn.execute(
        """
INSERT INTO cars (brand, model, year) VALUES ('peugeot', 'partner', 2020);
INSERT INTO cars (brand, model, year) VALUES ('seat', 'toledo', 2021);
"""
    )
    await conn.close()
    yield dsn
    drop_database(dsn)


@pytest.fixture
async def pg_shoping_example(pg):
    host, port = pg
    dsn = f"postgresql://postgres:postgres@{host}:{port}/test_db2_sql"

    if not database_exists(dsn):
        create_database(dsn)

    conn = await asyncpg.connect(dsn)
    await conn.execute(
        """
CREATE TABLE shopping (
  userid VARCHAR(255),
  product VARCHAR(255),
  year INT
);
"""
    )
    await conn.execute(
        """
INSERT INTO shopping (userid, product, year) VALUES ('user1', 'iphone 17', 2020);
INSERT INTO shopping (userid, product, year) VALUES ('user2', 'Samsung Galaxy S21', 2021);
"""
    )
    await conn.close()
    yield dsn
    drop_database(dsn)


@pytest.fixture
async def arag_settings(sdk_async: NucliaDBAsync, audit, valkey_url: str):
    yield Settings(
        running_environment="test",
        valkey_url=valkey_url,
        valkey_cluster_mode=False,
        memory_reader_nucliadb=sdk_async.base_url,
        memory_writer_nucliadb=sdk_async.base_url,
        memory_search_nucliadb=sdk_async.base_url,
        dummy_idp=True,
    )


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
async def arag_api(
    arag_api_app: HTTPApplication,
):
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
async def arag_kb(create_arag_kb, delete_arag_kb):
    kb = await create_arag_kb("test_basic_worker", "nuclia")
    yield kb
    await delete_arag_kb(kb.uuid, "nuclia")


@pytest.fixture
async def arag_no_memory(create_arag_no_memory, delete_arag_no_memory):
    arag_id = await create_arag_no_memory("nuclia")
    yield arag_id
    await delete_arag_no_memory(arag_id, "nuclia")


@pytest.fixture
async def arag_kb_legacy(create_arag_kb, delete_arag_kb, arag_api_app):
    """
    We have some rows that have memory set to null but actually have a memory KB.
    (Legacy from before we added the no KB option)
    """
    kb = await create_arag_kb("test_basic_worker_legacy", "nuclia")
    # Go to the DB and set memory to null
    db: databases.Database = arag_api_app.agent_manager.database
    from hyperforge_database.agents import retrieval_agent_config

    statement = (
        retrieval_agent_config.update()
        .where(retrieval_agent_config.c.agent_id == kb.uuid)
        .values(memory=None)
    )
    await db.execute(statement)
    yield kb
    await delete_arag_kb(kb.uuid, "nuclia")


@pytest.fixture
async def arag_server(
    sdk: NucliaDB,
    data_manager_settings: DataManagerSettings,
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
        external_nua_api_key=NUA_NUCLIADB,
    )
    agent_manager = await AgentManager.from_settings(settings=data_manager_settings)
    await agent_manager.initialize()
    broker = RedisBroker.from_url(
        url=valkey_url,
        activate_subject=settings.activate_subject,
        keepalive_ms=int(settings.pubsub_keepalive_seconds * 1000),
        cluster_mode=settings.valkey_cluster_mode,
    )
    session = SessionManager(
        settings=settings,
        broker=broker,
        agent_manager=agent_manager,
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
