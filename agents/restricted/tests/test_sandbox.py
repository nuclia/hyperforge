import asyncio
import tempfile
from unittest.mock import patch

import pytest
from hyperforge.definition import FunctionDefinition
from hyperforge_restricted import sandbox
from hyperforge_restricted.model import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerExecutionRequest,
)


@pytest.fixture
async def sandbox_conn():
    with tempfile.TemporaryDirectory() as path:
        socket = f"{path}/sandbox.sock"
        with patch(
            "hyperforge_restricted.sandbox.settings.sandbox_socket",
            socket,
        ):
            task = asyncio.create_task(sandbox.run_sandbox_server())

            await asyncio.sleep(0.1)  # Wait for the server to start
            (rx, tx) = await asyncio.open_unix_connection(socket)
            yield (sandbox.SandboxReader(rx), sandbox.SandboxWriter(tx))

            task.cancel()


@pytest.fixture
async def long_timeout():
    """
    On CI, process startup plus RestrictedPython compilation can exceed the default
    timeout under coverage. Use a larger limit here to avoid flaky failures.
    """
    with patch(
        "hyperforge_restricted.sandbox.WORKER_CPU_LIMIT",
        15,
    ):
        yield


async def test_sandbox_no_code(sandbox_conn, long_timeout):
    request = WorkerExecutionRequest(
        code="", question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Done)


async def test_sandbox_bad_code(sandbox_conn, long_timeout):
    code = 'return "patata"'  # This does not compile (return not supported)
    request = WorkerExecutionRequest(
        code=code, question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Error)
    assert "'return' outside function" in message.error


async def test_sandbox_infinite_loop(sandbox_conn):
    code = "while True: pass"  # An infinite loop and should be interrupted
    request = WorkerExecutionRequest(
        code=code, question="Q?", local_vars={}, global_vars={}, function_names={}
    )

    (rx, tx) = sandbox_conn
    await tx.write_message(SandboxMessage.Run(request))
    message = await rx.read_message()
    assert isinstance(message, SandboxMessage.Error)
    assert "timeout" in message.error


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
    await tx.write_message(SandboxMessage.Run(request))

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
