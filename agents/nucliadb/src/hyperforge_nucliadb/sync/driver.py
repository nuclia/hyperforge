import asyncio
import datetime
import random
import time
from typing import Any, Dict, List, cast
from urllib.parse import urlencode
from uuid import UUID

from httpx import (
    AsyncClient,
    ConnectError,
    ConnectTimeout,
    ReadTimeout,
    RemoteProtocolError,
    Response,
)
from hyperforge import logger
from hyperforge.configure import driver
from hyperforge.interaction import Provider
from hyperforge.utils.http import SafeTransport
from nucliadb_models import SyncMetadata
from pydantic import BaseModel

from hyperforge_nucliadb.driver import (
    NucliaDBDriver,
    connect,
    manager_connect,
)
from hyperforge_nucliadb.driver_config import (
    NucliaDBConnection,
)
from hyperforge_nucliadb.sync.config_driver import (
    SYNC_HTTP_TIMEOUT,
    SyncConnection,
    SyncDriverConfig,
)

SYNC_VALIDATE_ATTEMPTS = 2
SYNC_VALIDATE_RETRY_BASE_SECONDS = 0.5
SYNC_VALIDATE_RETRY_JITTER_SECONDS = 0.25
SYNC_VALIDATE_RETRY_STATUS_CODES = frozenset({502, 503, 504})
SYNC_VALIDATE_RETRY_EXCEPTIONS = (
    ConnectError,
    ConnectTimeout,
    ReadTimeout,
    RemoteProtocolError,
)


async def sync_connect(conn: SyncConnection):
    headers: Dict[str, str] = {}
    if "http://localhost" in conn.kb_url:
        headers = {
            "X-NUCLIADB-ROLES": "READER",
        }
    else:
        headers = {"AUTHORIZATION": "Bearer " + conn.key}  # type: ignore
    return AsyncClient(
        headers=headers,
        base_url=conn.kb_url,
        timeout=SYNC_HTTP_TIMEOUT,
        transport=SafeTransport(),
    )


class ExternalConnectionOutput(BaseModel):
    """External connection output without sensitive credential data."""

    id: UUID
    kb_id: UUID
    created_by: UUID
    created_at: datetime.datetime
    updated_at: datetime.datetime
    provider: Provider


@driver(
    id="sync",
    title="Sync Source",
    description="Source for interacting with the Sync API.",
    config_schema=SyncDriverConfig,
)
class SyncDriver(NucliaDBDriver):
    async_driver: AsyncClient
    config: SyncConnection
    information: Dict[str, ExternalConnectionOutput]
    sync_configs: Dict[str, list[str]]

    @classmethod
    async def init(cls, driver: Any) -> "SyncDriver":
        sync_driver = cast(SyncDriverConfig, driver)
        client = await sync_connect(sync_driver.config)
        information = {}
        sync_configs: Dict[str, list[str]] = {}
        for sync_config_id in sync_driver.config.connection_ids:
            response = await client.get(f"/sync_config/{sync_config_id}")
            response.raise_for_status()
            data = response.json()
            connection_id = data["external_connection"]["id"]
            response = await client.get(f"/external_connection/{connection_id}")
            response.raise_for_status()
            sync_configs.setdefault(sync_config_id, []).append(connection_id)
            data = response.json()
            information[connection_id] = ExternalConnectionOutput(**data)
        return cls(
            provider=sync_driver.provider,
            name=sync_driver.name,
            async_driver=client,
            information=information,
            sync_configs=sync_configs,
            config=sync_driver.config,
            driver=await connect(cast(NucliaDBConnection, sync_driver.config)),
            manager=await manager_connect(cast(NucliaDBConnection, sync_driver.config)),
            _synonyms=None,
        )

    async def get_oauth_url(
        self,
        sync_config: str,
        rao_redirect_url: str,
        oauth_uuid: str,
        connection_id: str,
        question_id: str,
    ) -> str:
        sep = "&" if "?" in rao_redirect_url else "?"
        rao_redirect_url = (
            f"{rao_redirect_url}{sep}{urlencode({'question_id': question_id})}"
        )

        response = await self.async_driver.post(
            f"{self.config.kb_url}/sync_config/{sync_config}/authorize",
            json={
                "connection_id": connection_id,
                "rao_redirect_url": rao_redirect_url,
                "oauth_uuid": oauth_uuid,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["authorize_url"]

    async def validate_resources(
        self,
        resource_ids: List[str],
        credentials: str,
        connection_id: str,
        sync_config_id: str,
        sync_metadata_by_resource: Dict[str, SyncMetadata],
    ) -> List[str]:
        path = f"/sync_config/{sync_config_id}/validate_resources"
        payload = {
            "resources": [x.model_dump() for x in sync_metadata_by_resource.values()],
            "credentials": credentials,
        }

        response: Response | None = None
        attempt = 0
        overall_started_at = time.monotonic()
        for attempt in range(1, SYNC_VALIDATE_ATTEMPTS + 1):
            attempt_started_at = time.monotonic()
            try:
                response = await self.async_driver.post(path, json=payload)
                if (
                    response.status_code not in SYNC_VALIDATE_RETRY_STATUS_CODES
                    or attempt == SYNC_VALIDATE_ATTEMPTS
                ):
                    if response.status_code in SYNC_VALIDATE_RETRY_STATUS_CODES:
                        logger.warning(
                            "Sync resource validation got retriable status but reached max attempts",
                            extra={
                                "sync_config_id": sync_config_id,
                                "connection_id": connection_id,
                                "resource_count": len(resource_ids),
                                "attempt": attempt,
                                "duration_seconds": time.monotonic()
                                - attempt_started_at,
                                "status_code": response.status_code,
                            },
                        )
                    else:
                        logger.info(
                            "Sync resource validation received non-retriable status",
                            extra={
                                "sync_config_id": sync_config_id,
                                "connection_id": connection_id,
                                "resource_count": len(resource_ids),
                                "attempt": attempt,
                                "duration_seconds": time.monotonic()
                                - attempt_started_at,
                                "status_code": response.status_code,
                            },
                        )
                    break
                error_type = f"HTTP {response.status_code}"
            except SYNC_VALIDATE_RETRY_EXCEPTIONS as exc:
                if attempt == SYNC_VALIDATE_ATTEMPTS:
                    logger.warning(
                        "Sync resource validation failed after retries",
                        extra={
                            "sync_config_id": sync_config_id,
                            "connection_id": connection_id,
                            "resource_count": len(resource_ids),
                            "attempt": attempt,
                            "duration_seconds": time.monotonic() - attempt_started_at,
                            "exception_type": type(exc).__name__,
                        },
                    )
                    raise
                error_type = type(exc).__name__

            logger.warning(
                "Transient Sync resource validation failure, retrying",
                extra={
                    "sync_config_id": sync_config_id,
                    "connection_id": connection_id,
                    "resource_count": len(resource_ids),
                    "attempt": attempt,
                    "duration_seconds": time.monotonic() - attempt_started_at,
                    "exception_type": error_type,
                },
            )
            delay = SYNC_VALIDATE_RETRY_BASE_SECONDS + random.uniform(
                0, SYNC_VALIDATE_RETRY_JITTER_SECONDS
            )
            await asyncio.sleep(delay)

        if response is None:
            raise RuntimeError("Sync resource validation made no request attempts")

        response.raise_for_status()
        logger.info(
            "Sync resource validation completed",
            extra={
                "sync_config_id": sync_config_id,
                "connection_id": connection_id,
                "resource_count": len(resource_ids),
                "attempt": attempt,
                "duration_seconds": time.monotonic() - overall_started_at,
            },
        )
        data = response.json()
        # The API validates by file_id (origin storage identifier) and returns those back,
        # but resource_filters in AskRequest requires NucliaDB resource UUIDs, which are
        # the keys of sync_metadata_by_resource.
        file_id_to_resource_uuid = {
            metadata.file_id: resource_uuid
            for resource_uuid, metadata in sync_metadata_by_resource.items()
        }
        return [
            file_id_to_resource_uuid[resource["file_id"]]
            for resource in data.get("valid_resources", [])
            if resource["file_id"] in file_id_to_resource_uuid
        ]
