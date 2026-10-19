import sys

from mcp.server.fastmcp import FastMCP


def main(mcp: FastMCP) -> None:
    """Run over stdio, or streamable HTTP with `--http PORT`."""
    if len(sys.argv) >= 3 and sys.argv[1] == "--http":
        mcp.settings.host = "127.0.0.1"
        mcp.settings.port = int(sys.argv[2])
        mcp.run(transport="streamable-http")
    else:
        mcp.run()
