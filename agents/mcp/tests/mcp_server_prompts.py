#!/usr/bin/env python3

import socket
from contextlib import asynccontextmanager
from typing import Union

import httpx
from mcp.server.fastmcp import FastMCP
from sse_starlette.sse import AppStatus


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


# Create MCP server
mcp = FastMCP("MCP Calculation Server")

Number = Union[int, float]


@mcp.tool()
def add(a: Number, b: Number) -> float:
    """Add two numbers"""
    return float(a + b)


@mcp.tool()
def subtract(a: Number, b: Number) -> float:
    """Subtract second number from first number"""
    return float(a - b)


@mcp.tool()
def multiply(a: Number, b: Number) -> float:
    """Multiply two numbers"""
    return float(a * b)


@mcp.tool()
def divide(a: Number, b: Number) -> float:
    """Divide first number by second number"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return float(a / b)


@mcp.tool()
def power(a: Number, b: Number) -> float:
    """Raise first number to the power of second number"""
    return float(a**b)


@mcp.tool()
def percentage(value: Number, percent: Number) -> float:
    """Calculate percentage of a value (e.g., 15% of 100)"""
    return float(value * (percent / 100))


@mcp.tool()
def percentage_increase(value: Number, percent: Number) -> float:
    """Calculate value with percentage increase (e.g., 100 + 15%)"""
    return float(value * (1 + percent / 100))


@mcp.tool()
def percentage_decrease(value: Number, percent: Number) -> float:
    """Calculate value with percentage decrease (e.g., 100 - 15%)"""
    return float(value * (1 - percent / 100))


@mcp.prompt()
def multiplication_tool_advice() -> str:
    """
    When you need to multiply two numbers, use the 'multiply' tool.
    For example, if you want to find out how many apples you have in total
    when you have 'a' baskets and each basket contains 'b' apples,
    you should use the 'multiply' tool to calculate a * b. and when generating the answer, explain it using this fruit example.
    """
    return (
        "When faced with a multiplication problem, such as having some baskets "
        "with some apples each, you should use the 'multiply' tool to find the total number of apples. "
        "Simply multiply the number of baskets by the number of apples to get the answer."
        "Always generate the answer using funny fruit examples!"
    )


@asynccontextmanager
async def run_mcp_server_prompt():
    import uvicorn

    starlette_app = mcp.streamable_http_app()

    port = free_port()
    config = uvicorn.Config(
        starlette_app,
        host="0.0.0.0",
        port=port,
        log_level=mcp.settings.log_level.lower(),
    )
    server = uvicorn.Server(config)

    if not server.config.loaded:
        server.config.load()

    server.lifespan = server.config.lifespan_class(config)

    await server.startup()
    url = f"http://localhost:{port}"
    async with httpx.AsyncClient() as client:
        _ = await client.get(url)

    yield url

    await server.shutdown()
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    AppStatus.original_handler = None
