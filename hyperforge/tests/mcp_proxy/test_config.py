import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hyperforge.mcp_proxy.config import load_config_file, parse_config


def test_load_yaml_list_form(tmp_path: Path):
    path = tmp_path / "proxy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "name": "my-proxy",
                "default_search_limit": 5,
                "servers": [
                    {"name": "weather", "command": "python", "args": ["w.py"]},
                    {"name": "remote", "transport": "http", "url": "http://x/mcp"},
                    {"name": "off", "command": "x", "enabled": False},
                ],
            }
        )
    )
    cfg = load_config_file(path)
    assert cfg.name == "my-proxy"
    assert cfg.default_search_limit == 5
    assert [s.name for s in cfg.enabled_servers] == ["weather", "remote"]
    assert cfg.servers[1].transport == "http"


def test_load_json_mcp_servers_mapping(tmp_path: Path):
    path = tmp_path / "proxy.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "files": {
                        "command": "npx",
                        "args": ["files"],
                        "exclude_tools": ["read_file"],
                    },
                    "math": {"transport": "http", "url": "http://m/mcp"},
                }
            }
        )
    )
    cfg = load_config_file(path)
    assert [s.name for s in cfg.servers] == ["files", "math"]
    assert not cfg.servers[0].exposes("read_file")
    assert cfg.servers[0].exposes("list_directory")


def test_empty_yaml_is_valid(tmp_path: Path):
    path = tmp_path / "empty.yml"
    path.write_text("")
    assert load_config_file(path).servers == []


@pytest.mark.parametrize(
    "data",
    [
        {"servers": [{"name": "a", "command": "x"}, {"name": "a", "command": "y"}]},
        {"servers": [{"name": "a"}]},
        {"servers": [{"name": "a", "transport": "http"}]},
        {"servers": [{"name": "bad name", "command": "x"}]},
    ],
)
def test_invalid_configs_rejected(data):
    with pytest.raises(ValidationError):
        parse_config(data)
