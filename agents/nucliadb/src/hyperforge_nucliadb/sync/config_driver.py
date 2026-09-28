from typing import Literal

from httpx import Timeout
from hyperforge.driver import DriverConfig
from pydantic import Field, field_validator
from pydantic.config import ConfigDict

from hyperforge_nucliadb.driver_config import (
    NucliaDBConnection,
)

SYNC_HTTP_TIMEOUT = Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


class SyncConnection(NucliaDBConnection):
    connection_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Sync configuration IDs to restrict searches to. When omitted or empty, "
            "the entire knowledge box is searched and synced resources are authorized "
            "individually."
        ),
    )

    @field_validator("connection_ids")
    @classmethod
    def validate_connection_ids(cls, connection_ids: list[str]) -> list[str]:
        if any(not connection_id.strip() for connection_id in connection_ids):
            raise ValueError("connection_ids cannot contain blank values")
        return connection_ids

    @property
    def kb_url(self) -> str:
        return f"{self.url}/v1/kb/{self.kbid}"


class SyncDriverConfig(DriverConfig[SyncConnection]):
    model_config = ConfigDict(title="Knowledge Box Sync Service connection")
    provider: Literal["sync"]
    config: SyncConnection
