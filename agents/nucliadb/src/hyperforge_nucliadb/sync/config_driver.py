from typing import Literal

from hyperforge.driver import DriverConfig
from pydantic.config import ConfigDict

from hyperforge_nucliadb.driver_config import (
    NucliaDBConnection,
)


class SyncConnection(NucliaDBConnection):
    connection_ids: list[str]

    @property
    def kb_url(self) -> str:
        return f"{self.url}/v1/kb/{self.kbid}"


class SyncDriverConfig(DriverConfig[SyncConnection]):
    model_config = ConfigDict(title="Knowledge Box Sync Service connection")
    provider: Literal["sync"]
    config: SyncConnection
