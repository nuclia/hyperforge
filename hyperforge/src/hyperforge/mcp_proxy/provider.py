from pathlib import Path
from typing import Protocol, runtime_checkable

from hyperforge.mcp_proxy.config import ProxyConfig, load_config_file


@runtime_checkable
class ConfigProvider(Protocol):
    """Source of proxy configuration.

    Implement this to back the proxy with a database or remote service
    (e.g. one config per tenant in a SaaS deployment).
    """

    async def get_config(self) -> ProxyConfig: ...


class StaticConfigProvider:
    def __init__(self, config: ProxyConfig):
        self.config = config

    async def get_config(self) -> ProxyConfig:
        return self.config


class FileConfigProvider:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    async def get_config(self) -> ProxyConfig:
        return load_config_file(self.path)
