from typing import Literal

from httpx import Timeout
from hyperforge.driver import DriverConfig
from pydantic.config import ConfigDict

from hyperforge_nucliadb.driver_config import (
    NucliaDBConnection,
)

SYNC_HTTP_TIMEOUT = Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


class SyncConnection(NucliaDBConnection):
    connection_ids: list[str]

    @property
    def kb_url(self) -> str:
        return f"{self.url}/v1/kb/{self.kbid}"


class SyncDriverConfig(DriverConfig[SyncConnection]):
    model_config = ConfigDict(title="Knowledge Box Sync Service connection")
    provider: Literal["sync"]
    config: SyncConnection
