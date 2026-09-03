import asyncio
import os
import tempfile
from unittest.mock import patch

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
async def sandbox_conn():
    with tempfile.TemporaryDirectory() as path:
        socket = f"{path}/sandbox.sock"
        with (
            patch(
                "hyperforge.codemode.sandbox.settings.sandbox_socket",
                socket,
            ),
            patch(
                "hyperforge.codemode.sandbox.settings.sandbox_verify",
                False,
            ),
            patch(
                "hyperforge.codemode.sandbox.settings.sandbox_token",
                "test-token",
            ),
        ):
            task = asyncio.create_task(sandbox.run_sandbox_server())

            await asyncio.sleep(0.1)  # Wait for the server to start
            assert oct(os.stat(socket).st_mode & 0o777) == "0o600"
            (rx, tx) = await asyncio.open_unix_connection(socket)
            yield (sandbox.SandboxReader(rx), sandbox.SandboxWriter(tx))

            tx.close()
            await tx.wait_closed()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
async def tokenless_sandbox_conn():
    with tempfile.TemporaryDirectory() as path:
        socket = f"{path}/sandbox.sock"
        with (
            patch("hyperforge.codemode.sandbox.settings.sandbox_socket", socket),
            patch("hyperforge.codemode.sandbox.settings.sandbox_verify", False),
            patch("hyperforge.codemode.sandbox.settings.sandbox_token", None),
        ):
            task = asyncio.create_task(sandbox.run_sandbox_server())
            await asyncio.sleep(0.1)
            rx, tx = await asyncio.open_unix_connection(socket)
            yield (sandbox.SandboxReader(rx), sandbox.SandboxWriter(tx))

            tx.close()
            await tx.wait_closed()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.fixture
async def long_timeout():
    """
    On CI, process startup plus RestrictedPython compilation can exceed the default
    timeout under coverage. Use a larger limit here to avoid flaky failures.
    """
    with patch(
        "hyperforge.codemode.sandbox.WORKER_CPU_LIMIT",
        15,
    ):
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
