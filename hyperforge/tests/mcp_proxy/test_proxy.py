import json

import pytest

from hyperforge.mcp_proxy.config import parse_config
from hyperforge.mcp_proxy.provider import StaticConfigProvider
from hyperforge.mcp_proxy.proxy import MCPProxy, ToolNotFound

from .conftest import stdio_server


def make_proxy(servers: list[dict], **extra) -> MCPProxy:
    return MCPProxy(StaticConfigProvider(parse_config({"servers": servers, **extra})))


def text(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


async def test_search_ranks_and_execute(standard_servers, tmp_path):
    async with make_proxy(standard_servers) as proxy:
        assert proxy.unavailable_servers == {}
        assert proxy.search("").total_tools == 7

        expectations = {
            "weather forecast for a city": "weather.get_forecast",
            "storm warnings": "weather.get_alerts",
            "read contents of a file": "files.read_file",
            "list folder": "files.list_directory",
            "find files matching glob": "files.search_files",
            "multiply numbers": "math.multiply",
            "sum": "math.add",
        }
        for query, expected in expectations.items():
            res = proxy.search(query, limit=3)
            assert res.results[0].name == expected, (query, res.results)

        top = proxy.search("forecast").results[0]
        assert top.input_schema["properties"]["city"]["type"] == "string"

        result = await proxy.execute("math.multiply", {"a": 6, "b": 7})
        assert not result.isError
        assert float(text(result)) == 42

        (tmp_path / "notes.txt").write_text("hello proxy")
        result = await proxy.execute(
            "files.read_file", {"path": str(tmp_path / "notes.txt")}
        )
        assert text(result) == "hello proxy"


async def test_search_limit_is_capped(standard_servers):
    async with make_proxy(standard_servers, max_search_limit=2) as proxy:
        assert len(proxy.search("", limit=100).results) == 2


async def test_unknown_tool_suggests_alternatives(standard_servers):
    async with make_proxy(standard_servers) as proxy:
        with pytest.raises(ToolNotFound) as exc:
            await proxy.execute("weather.forecast", {"city": "x"})
        assert "weather.get_forecast" in exc.value.suggestions


async def test_include_exclude_filtering():
    servers = [
        stdio_server(
            "files",
            include_tools=["read_file", "search_files"],
            exclude_tools=["search_files"],
        ),
        stdio_server("weather", exclude_tools=["get_alerts"]),
    ]
    async with make_proxy(servers) as proxy:
        names = {m.name for m in proxy.search("", limit=50).results}
        assert names == {"files.read_file", "weather.get_forecast"}
        with pytest.raises(ToolNotFound):
            await proxy.execute("weather.get_alerts", {"region": "x"})


async def test_broken_upstream_does_not_break_others():
    servers = [
        {
            "name": "broken",
            "command": "this-command-does-not-exist-xyz",
            "timeout_seconds": 5,
        },
        stdio_server("math"),
    ]
    async with make_proxy(servers) as proxy:
        res = proxy.search("add numbers")
        assert set(res.unavailable_servers) == {"broken"}
        assert res.results[0].name == "math.add"
        result = await proxy.execute("math.add", {"a": 1, "b": 2})
        assert float(text(result)) == 3


async def test_http_upstream(http_math_url):
    servers = [{"name": "math", "transport": "http", "url": http_math_url}]
    async with make_proxy(servers) as proxy:
        assert proxy.unavailable_servers == {}
        result = await proxy.execute("math.add", {"a": 2, "b": 3})
        assert float(text(result)) == 5


async def test_reload_picks_up_config_changes(tmp_path):
    from hyperforge.mcp_proxy.provider import FileConfigProvider

    path = tmp_path / "proxy.json"
    path.write_text(json.dumps({"servers": [stdio_server("math")]}))
    async with MCPProxy(FileConfigProvider(path)) as proxy:
        assert proxy.search("").total_tools == 2
        path.write_text(json.dumps({"servers": [stdio_server("weather")]}))
        await proxy.reload()
        assert {m.server for m in proxy.search("", limit=50).results} == {"weather"}
