import json
from pathlib import Path

import yaml
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from hyperforge.mcp_proxy.app import create_app
from hyperforge.mcp_proxy.provider import FileConfigProvider

from .conftest import ServedApp, stdio_server


def write_config(tmp_path: Path, servers: list[dict]) -> Path:
    path = tmp_path / "proxy.yaml"
    path.write_text(yaml.safe_dump({"servers": servers}))
    return path


def text(result) -> str:
    return "".join(c.text for c in result.content if c.type == "text")


async def exercise(session: ClientSession) -> None:
    await session.initialize()
    tools = await session.list_tools()
    assert {t.name for t in tools.tools} == {"search", "execute"}

    found = await session.call_tool(
        "search", {"query": "multiply two numbers", "limit": 2}
    )
    assert not found.isError
    payload = json.loads(text(found))
    assert payload["results"][0]["name"] == "math.multiply"
    assert found.structuredContent["results"][0]["name"] == "math.multiply"

    result = await session.call_tool(
        "execute", {"tool_name": "math.multiply", "params": {"a": 3, "b": 4}}
    )
    assert not result.isError
    assert float(text(result)) == 12

    missing = await session.call_tool("execute", {"tool_name": "math.mul"})
    assert missing.isError
    assert "math.multiply" in text(missing)


async def test_stdio_cli_end_to_end(tmp_path, standard_servers):
    config = write_config(tmp_path, standard_servers)
    params = StdioServerParameters(
        command="uv",
        args=["run", "--no-sync", "hyperforge-mcp-proxy", "--config", str(config)],
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await exercise(session)
        weather = await session.call_tool(
            "execute",
            {"tool_name": "weather.get_forecast", "params": {"city": "Paris"}},
        )
        assert "Paris" in text(weather)


async def test_http_app_end_to_end(tmp_path, http_math_url):
    servers = [
        {"name": "math", "transport": "http", "url": http_math_url},
        stdio_server("weather"),
    ]
    app = create_app(FileConfigProvider(write_config(tmp_path, servers)))
    async with (
        ServedApp(app) as served,
        streamablehttp_client(f"{served.base_url}/mcp") as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await exercise(session)
