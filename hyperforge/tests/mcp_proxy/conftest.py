import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

FIXTURES = Path(__file__).parent / "fixtures"


def stdio_server(name: str, **extra) -> dict:
    return {
        "name": name,
        "command": sys.executable,
        "args": [str(FIXTURES / f"{name}_server.py")],
        **extra,
    }


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_port(port: int, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"port {port} not listening")


@pytest.fixture
def standard_servers() -> list[dict]:
    return [stdio_server("weather"), stdio_server("files"), stdio_server("math")]


@pytest.fixture(scope="session")
def http_math_url():
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, str(FIXTURES / "math_server.py"), "--http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_port(port)
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(10)


class ServedApp:
    def __init__(self, app):
        self.port = free_port()
        self.server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="warning")
        )
        self.base_url = f"http://127.0.0.1:{self.port}"

    async def __aenter__(self) -> "ServedApp":
        self.task = asyncio.create_task(self.server.serve())
        async with httpx.AsyncClient() as client:
            for _ in range(200):
                if self.server.started:
                    resp = await client.get(f"{self.base_url}/health/ready")
                    if resp.status_code == 200:
                        return self
                await asyncio.sleep(0.05)
        raise TimeoutError("uvicorn did not start")

    async def __aexit__(self, *exc) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.task, 15)
