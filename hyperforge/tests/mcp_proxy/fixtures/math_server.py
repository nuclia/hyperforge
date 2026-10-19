from _run import main
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math")


@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers together and return the sum."""
    return a + b


@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the product."""
    return a * b


if __name__ == "__main__":
    main(mcp)
