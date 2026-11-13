"""Progressive discovery MCP proxy.

Exposes `search` and `execute` tools over a set of upstream MCP servers
instead of flattening all upstream tools into one list.
"""

from hyperforge.mcp_proxy.config import (
    ProxyConfig,
    UpstreamServerConfig,
    load_config_file,
    parse_config,
)
from hyperforge.mcp_proxy.provider import (
    ConfigProvider,
    FileConfigProvider,
    StaticConfigProvider,
)
from hyperforge.mcp_proxy.proxy import MCPProxy, SearchResult, ToolNotFound
from hyperforge.mcp_proxy.server import build_mcp_server

__all__ = [
    "ConfigProvider",
    "FileConfigProvider",
    "MCPProxy",
    "ProxyConfig",
    "SearchResult",
    "StaticConfigProvider",
    "ToolNotFound",
    "UpstreamServerConfig",
    "build_mcp_server",
    "load_config_file",
    "parse_config",
]
