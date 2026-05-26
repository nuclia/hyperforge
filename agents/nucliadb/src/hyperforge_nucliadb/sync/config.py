from typing import Literal

from pydantic import ConfigDict

from hyperforge_nucliadb.basic_ask_config import (
    BasicAskAgentConfig,
)


class SyncAskAgentConfig(BasicAskAgentConfig):
    model_config = ConfigDict(title="Knowledge Box Sync Service")
    module: Literal["sync"] = "sync"  # type: ignore[assignment]
