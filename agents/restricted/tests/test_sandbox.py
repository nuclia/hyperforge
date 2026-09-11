import asyncio
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor

import pytest
from hyperforge.codemode import sandbox
from hyperforge.codemode.model import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerExecutionRequest,
)
from hyperforge.definition import FunctionDefinition

from hyperforge_restricted.agent import PythonAgent
from hyperforge_restricted.config import PythonAgentConfig


@pytest.fixture
async def sandbox_conn(monkeypatch):
    with tempfile.TemporaryDirectory() as path:
        socket = f"{path}/sandbox.sock"
        monkeypatch.setenv("SANDBOX_SOCKET", socket)
        monkeypatch.setenv("SANDBOX_VERIFY", "false")
        monkeypatch.setenv("SANDBOX_TOKEN", "test-token")
        task = asyncio.create_task(sandbox.run_sandbox_server())

        await asyncio.sleep(0.1)
        assert oct(os.stat(socket).st_mode & 0o777) == "0o600"
        (rx, tx) = await asyncio.open_unix_connection(socket)
        yield (sandbox.SandboxReader(rx), sandbox.SandboxWriter(tx))

        tx.close()
        await tx.wait_closed()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
async def tokenless_sandbox_conn(monkeypatch):
    with tempfile.TemporaryDirectory() as path:
        socket = f"{path}/sandbox.sock"
        monkeypatch.setenv("SANDBOX_SOCKET", socket)
        monkeypatch.setenv("SANDBOX_VERIFY", "false")
        monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
        task = asyncio.create_task(sandbox.run_sandbox_server())
        await asyncio.sleep(0.1)
        rx, tx = await asyncio.open_unix_connection(socket)
        yield (sandbox.SandboxReader(rx), sandbox.SandboxWriter(tx))

        tx.close()
        await tx.wait_closed()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
async def long_timeout(monkeypatch):
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")
    yield


async def test_sandbox_no_code(sandbox_conn, long_timeout):
    request = WorkerExecutionRequest(
        code="", question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request, token="test-token"))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Done)


async def test_sandbox_token_is_optional(tokenless_sandbox_conn, long_timeout):
    request = WorkerExecutionRequest(
        code="", question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = tokenless_sandbox_conn
    await tx.write_message(SandboxMessage.Run(request))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Done)


async def test_sandbox_bad_code(sandbox_conn, long_timeout):
    code = 'return "patata"'  # This does not compile (return not supported)
    request = WorkerExecutionRequest(
        code=code, question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request, token="test-token"))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Error)
    assert "'return' outside function" in message.error


async def test_sandbox_infinite_loop(sandbox_conn):
    code = "while True: pass"  # An infinite loop and should be interrupted
    request = WorkerExecutionRequest(
        code=code, question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request, token="test-token"))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Error)
    assert "timeout" in message.error


async def test_sandbox_runtime_limit(sandbox_conn, long_timeout):
    request = WorkerExecutionRequest(
        code="while True: pass",
        question="Q?",
        local_vars={},
        global_vars={},
        function_names={},
        max_runtime_seconds=0.1,
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request, token="test-token"))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Error)
    assert "timed out" in message.error


async def test_sandbox_enforces_server_runtime_ceiling(
    sandbox_conn, long_timeout, monkeypatch
):
    monkeypatch.setattr(sandbox.settings, "sandbox_max_session_runtime_seconds", 0.1)
    request = WorkerExecutionRequest(
        code="while True: pass",
        question="Q?",
        local_vars={},
        global_vars={},
        function_names={},
    )
    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request, token="test-token"))

    message = await rx.read_message()

    assert isinstance(message, SandboxMessage.Error)
    assert "timed out" in message.error


async def test_sandbox_function_call(sandbox_conn, long_timeout):
    request = WorkerExecutionRequest(
        code="show(msg=foo())",
        question="Q?",
        local_vars={},
        global_vars={},
        function_names={
            "self": {
                "foo": FunctionDefinition(
                    name="foo", description="Do foo", parameters={}
                ),
                "show": FunctionDefinition(
                    name="show",
                    description="show string message",
                    parameters={
                        "msg": {
                            "type": "string",
                            "description": "The message to show.",
                        },
                    },
                ),
            }
        },
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request, token="test-token"))

    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Request)
    assert message.task == RestrictedPythonTask(
        function="foo", agent="self", args=(), keyword_args={}
    )
    await tx.write_message(SandboxMessage.Response(result="Hello from agent"))

    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Request)
    assert message.task == RestrictedPythonTask(
        function="show", agent="self", args=(), keyword_args={"msg": "Hello from agent"}
    )
    await tx.write_message(SandboxMessage.Response(result=None))

    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Done)


def test_restricted_agent_config_cannot_enable_debug_mode():
    config = PythonAgentConfig.model_validate({"code": "save()", "debug": True})

    assert "debug" not in config.model_dump()


async def test_restricted_agent_rejects_undeclared_callback():
    agent = PythonAgent.__new__(PythonAgent)
    agent.function_names = {"child": {}}
    task = RestrictedPythonTask(
        function="private_function",
        agent="child",
        args=(),
        keyword_args={},
    )

    with pytest.raises(ValueError, match="not authorized"):
        await agent.handle_queue_item(None, None, task)


async def test_sandbox_rejects_wrong_token(sandbox_conn):
    request = WorkerExecutionRequest(
        code="", question="Q?", local_vars={}, global_vars={}, function_names={}
    )
    (rx, tx) = sandbox_conn

    await tx.write_message(SandboxMessage.Run(request, token="wrong-token"))

    with pytest.raises(asyncio.IncompleteReadError):
        await rx.read_message()


async def test_remote_runner_allows_missing_token(monkeypatch):
    with tempfile.TemporaryDirectory() as path:
        socket = f"{path}/sandbox.sock"
        monkeypatch.setenv("SANDBOX_SOCKET", socket)
        monkeypatch.setenv("SANDBOX_VERIFY", "false")
        monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")
        monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
        task = asyncio.create_task(sandbox.run_sandbox_server())
        for _ in range(100):
            if os.path.exists(socket):
                break
            await asyncio.sleep(0.01)

        async def callback(_task):
            return None

        runner = sandbox.SandboxRunner.remote(socket, callback)
        request = WorkerExecutionRequest(
            code="", question="Q?", local_vars={}, global_vars={}, function_names={}
        )
        try:
            await runner.run(request)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_local_runner_preserves_legacy_python_values():
    received = []

    async def callback(task):
        received.append(task.keyword_args["value"])
        return task.keyword_args["value"]

    with ProcessPoolExecutor(max_workers=1) as pool:
        runner = sandbox.SandboxRunner.with_pool(pool, callback)
        await runner.run(
            WorkerExecutionRequest(
                code="passthrough(value=value)",
                question="Q?",
                local_vars={"value": b"legacy-pickle-value"},
                global_vars={},
                function_names={
                    "self": {
                        "passthrough": FunctionDefinition(
                            name="passthrough", description="", parameters={}
                        )
                    }
                },
            )
        )

    assert received == [b"legacy-pickle-value"]


async def test_sandbox_rejects_concurrent_session(
    sandbox_conn, long_timeout, monkeypatch
):
    request = WorkerExecutionRequest(
        code="foo()",
        question="Q?",
        local_vars={},
        global_vars={},
        function_names={
            "self": {
                "foo": FunctionDefinition(name="foo", description="", parameters={})
            }
        },
    )
    monkeypatch.setattr(sandbox.settings, "sandbox_max_concurrent_sessions", 1)
    first_rx, first_tx = sandbox_conn
    await first_tx.write_message(SandboxMessage.Run(request, token="test-token"))
    assert isinstance(await first_rx.read_message(), SandboxMessage.Request)

    second_rx_raw, second_tx_raw = await asyncio.open_unix_connection(
        os.environ["SANDBOX_SOCKET"]
    )
    second_rx = sandbox.SandboxReader(second_rx_raw)
    second_tx = sandbox.SandboxWriter(second_tx_raw)
    try:
        await second_tx.write_message(SandboxMessage.Run(request, token="test-token"))
        response = await second_rx.read_message()
        assert isinstance(response, SandboxMessage.Error)
        assert "concurrency limit" in response.error
    finally:
        second_tx.close()
        await second_tx.wait_closed()

    await first_tx.write_message(SandboxMessage.Response(result=None))
    assert isinstance(await first_rx.read_message(), SandboxMessage.Done)


def test_sandbox_settings_preserve_legacy_metrics_defaults() -> None:
    configured = sandbox.SandboxSettings()

    assert configured.sandbox_metrics_enabled is True
    assert configured.sandbox_metrics_host == "0.0.0.0"
    assert configured.sandbox_socket_mode == "0600"
    assert configured.sandbox_max_concurrent_sessions is None
    assert configured.sandbox_max_session_runtime_seconds is None
    assert configured.sandbox_max_session_memory_bytes is None


def test_sandbox_settings_accept_shared_group_socket_mode() -> None:
    configured = sandbox.SandboxSettings(
        sandbox_socket_mode="0660", sandbox_socket_group="sandbox"
    )

    assert configured.sandbox_socket_mode == "0660"
    assert configured.sandbox_socket_group == "sandbox"


def test_sandbox_settings_reject_invalid_socket_mode() -> None:
    with pytest.raises(ValueError, match="octal mode"):
        sandbox.SandboxSettings(sandbox_socket_mode="invalid")
