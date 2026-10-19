"""Sanity check: talk to the proxy over stdio and run search + execute."""

import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[3]


async def main() -> None:
    params = StdioServerParameters(
        command="uv",
        args=[
            "run",
            "--no-sync",
            "hyperforge-mcp-proxy",
            "--config",
            "hyperforge/examples/mcp_proxy/proxy.yaml",
        ],
        cwd=str(ROOT),
    )
    async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
        await s.initialize()
        print("tools:", [t.name for t in (await s.list_tools()).tools])
        res = await s.call_tool("search", {"query": "inventory stock sku", "limit": 3})
        print(
            "search:",
            [m["name"] for m in res.structuredContent["results"]],
            res.structuredContent["unavailable_servers"],
        )
        res = await s.call_tool(
            "execute",
            {
                "tool_name": "inventory.get_inventory_stock",
                "params": {"sku": "WIDGET-42"},
            },
        )
        print("execute:", res.content[0].text)
        res = await s.call_tool(
            "execute",
            {
                "tool_name": "filesystem.list_directory",
                "params": {"path": str(Path(__file__).parent)},
            },
        )
        print("list_directory:", res.content[0].text)


asyncio.run(main())
