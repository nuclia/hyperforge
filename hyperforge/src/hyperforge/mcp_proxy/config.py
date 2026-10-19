import json
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class UpstreamServerConfig(BaseModel):
    """Declarative definition of an upstream MCP server to proxy."""

    name: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = ""
    transport: Literal["stdio", "http"] = "stdio"
    enabled: bool = True

    # stdio transport
    command: Optional[str] = None
    args: list[str] = []
    env: dict[str, str] = {}
    cwd: Optional[str] = None

    # http (streamable) transport
    url: Optional[str] = None
    headers: dict[str, str] = {}

    timeout_seconds: float = Field(default=30, gt=0)
    include_tools: Optional[list[str]] = None
    exclude_tools: list[str] = []
    tags: list[str] = []

    @model_validator(mode="after")
    def _check_transport(self) -> "UpstreamServerConfig":
        if self.transport == "stdio" and not self.command:
            raise ValueError(f"server {self.name!r}: stdio transport needs 'command'")
        if self.transport == "http" and not self.url:
            raise ValueError(f"server {self.name!r}: http transport needs 'url'")
        return self

    def exposes(self, tool_name: str) -> bool:
        if tool_name in self.exclude_tools:
            return False
        return self.include_tools is None or tool_name in self.include_tools


class ProxyConfig(BaseModel):
    """Complete configuration of one proxy instance.

    Independent of where it comes from (file, database, API payload).
    """

    name: str = "hyperforge-mcp-proxy"
    instructions: Optional[str] = None
    servers: list[UpstreamServerConfig] = []
    default_search_limit: int = Field(default=10, ge=1)
    max_search_limit: int = Field(default=50, ge=1)
    tool_name_separator: str = "."

    @model_validator(mode="after")
    def _unique_names(self) -> "ProxyConfig":
        names = [s.name for s in self.servers]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate server names: {sorted(dupes)}")
        return self

    @property
    def enabled_servers(self) -> list[UpstreamServerConfig]:
        return [s for s in self.servers if s.enabled]


def parse_config(data: dict[str, Any]) -> ProxyConfig:
    # Accept the common `mcpServers: {name: {...}}` mapping as well as a list.
    servers = data.get("servers", data.get("mcpServers"))
    if isinstance(servers, dict):
        data = {
            **{k: v for k, v in data.items() if k != "mcpServers"},
            "servers": [{"name": name, **cfg} for name, cfg in servers.items()],
        }
    return ProxyConfig.model_validate(data)


def load_config_file(path: str | Path) -> ProxyConfig:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}
    return parse_config(data)
