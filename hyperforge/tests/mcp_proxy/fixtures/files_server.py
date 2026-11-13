from pathlib import Path

from _run import main
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("files")


@mcp.tool()
def read_file(path: str) -> str:
    """Read the full text contents of a file from disk."""
    return Path(path).read_text()


@mcp.tool()
def list_directory(path: str) -> list[str]:
    """List the names of files and folders inside a directory."""
    return sorted(p.name for p in Path(path).iterdir())


@mcp.tool()
def search_files(root: str, pattern: str) -> list[str]:
    """Search recursively for files whose name matches a glob pattern under a root directory."""
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob(pattern))


if __name__ == "__main__":
    main(mcp)
