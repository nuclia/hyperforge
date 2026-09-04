import asyncio
import tempfile
from pathlib import Path

import pytest

from hyperforge.codemode.model import WorkerExecutionRequest, deserialize
from hyperforge.codemode.sandbox import (
    MAX_PACKET_BYTES,
    SandboxReader,
    SandboxRunner,
    SandboxSettings,
    run_sandbox_server,
)
from hyperforge.definition import FunctionDefinition


def _empty_request() -> WorkerExecutionRequest:
    return WorkerExecutionRequest(
        code="", local_vars={}, global_vars={}, function_names={}
    )


def _function_names(*names: str) -> dict[str, dict[str, FunctionDefinition]]:
    return {
        "harness": {
            name: FunctionDefinition(name=name, description="", parameters={})
            for name in names
        }
    }


@pytest.fixture
def socket_path():
    with tempfile.TemporaryDirectory(prefix="sbx", dir="/tmp") as directory:
        yield str(Path(directory) / "s")


def test_deserialize_preserves_unknown_model_marker() -> None:
    value = {"__model__": "customer-data", "value": 1}

    assert deserialize(value) == value


@pytest.mark.asyncio
async def test_reader_rejects_oversized_packet_before_payload() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data((MAX_PACKET_BYTES + 1).to_bytes(4, "little"))

    with pytest.raises(ValueError, match="maximum size"):
        await SandboxReader(reader)._read_packet()


async def _serve(monkeypatch, socket_path: str, **kwargs) -> asyncio.Task:
    monkeypatch.setenv("SANDBOX_SOCKET", socket_path)
    server_task = asyncio.create_task(run_sandbox_server(**kwargs))
    for _ in range(200):
        if Path(socket_path).exists():
            break
        await asyncio.sleep(0.01)
    return server_task


async def _run_client(
    socket_path: str,
    code: str,
    *,
    token,
    callback=None,
    function_names: dict[str, dict[str, FunctionDefinition]] | None = None,
) -> None:
    async def default_callback(task):
        return None

    runner = SandboxRunner.remote(
        socket_path, callback or default_callback, token=token
    )
    request = WorkerExecutionRequest(
        code=code,
        local_vars={},
        global_vars={},
        function_names=function_names or {},
    )
    await runner.run(request)


@pytest.mark.asyncio
async def test_sandbox_server_accepts_programmatic_client_token(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.setenv("SANDBOX_TOKEN", "server-secret")
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")
    server_task = await _serve(monkeypatch, socket_path)
    monkeypatch.delenv("SANDBOX_TOKEN")
    try:
        await _run_client(socket_path, "value = 1", token="server-secret")
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_sandbox_server_rejects_invalid_client_token(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.setenv("SANDBOX_TOKEN", "server-secret")
    server_task = await _serve(monkeypatch, socket_path)
    try:
        with pytest.raises(RuntimeError, match="closed unexpectedly"):
            await _run_client(socket_path, "value = 1", token="wrong")
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_sandbox_server_uses_token_verifier_hook(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")
    seen: list[str] = []

    async def verifier(token: str) -> bool:
        seen.append(token)
        return token.startswith("scoped-")

    server_task = await _serve(monkeypatch, socket_path, token_verifier=verifier)
    try:
        await _run_client(
            socket_path,
            "value = 1",
            token=lambda: "scoped-abc",
        )
        with pytest.raises(RuntimeError, match="closed unexpectedly"):
            await _run_client(socket_path, "value = 1", token="denied")
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)
    assert seen == ["scoped-abc", "denied"]


@pytest.mark.asyncio
async def test_sandbox_server_requires_token_without_verifier(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    monkeypatch.setenv("SANDBOX_SOCKET", socket_path)

    with pytest.raises(RuntimeError, match="SANDBOX_TOKEN is required"):
        await run_sandbox_server()


@pytest.mark.asyncio
async def test_remote_runner_requires_token(monkeypatch, socket_path: str) -> None:
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)

    async def callback(task):
        return None

    runner = SandboxRunner.remote(socket_path, callback)

    with pytest.raises(RuntimeError, match="SANDBOX_TOKEN is required"):
        await runner.run(_empty_request())


@pytest.mark.asyncio
async def test_callback_wait_seconds_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "2.5")

    assert SandboxSettings().sandbox_callback_wait_seconds == 2.5


@pytest.mark.asyncio
async def test_watchdog_kills_worker_that_never_calls_back(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.setenv("SANDBOX_TOKEN", "server-secret")
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "0.1")

    server_task = await _serve(monkeypatch, socket_path)
    try:
        with pytest.raises(RuntimeError, match="timeout"):
            await _run_client(
                socket_path,
                "while True:\n    pass",
                token="server-secret",
            )
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_slow_controller_callback_does_not_trip_watchdog(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.setenv("SANDBOX_TOKEN", "server-secret")
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "5")
    result: dict[str, object] = {}

    async def callback(task):
        if task.function == "slow":
            await asyncio.sleep(8)
            return 42
        if task.function == "output":
            result["value"] = task.args[0] if task.args else None
        return None

    server_task = await _serve(monkeypatch, socket_path)
    try:
        await _run_client(
            socket_path,
            "value = slow()\noutput(value)",
            token="server-secret",
            callback=callback,
            function_names=_function_names("slow", "output"),
        )
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    assert result["value"] == 42
