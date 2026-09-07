import asyncio
import tempfile
import threading
from pathlib import Path

import pytest

from hyperforge.codemode import sandbox
from hyperforge.codemode.model import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerExecutionRequest,
    deserialize,
)
from hyperforge.codemode.sandbox import (
    MAX_PACKET_BYTES,
    SandboxReader,
    SandboxRunner,
    SandboxSettings,
    SandboxWriter,
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


@pytest.mark.asyncio
async def test_reader_rejects_deep_packet_as_value_error() -> None:
    payload = b'{"_":"response","result":' + b"[" * 1000 + b"0" + b"]" * 1000 + b"}"
    reader = asyncio.StreamReader()
    reader.feed_data(len(payload).to_bytes(4, "little") + payload)

    with pytest.raises(ValueError, match="not valid JSON"):
        await SandboxReader(reader).read_message()


@pytest.mark.asyncio
async def test_writer_rejects_oversized_packet() -> None:
    class Writer:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            pass

    with pytest.raises(ValueError, match="maximum size"):
        await SandboxWriter(Writer()).write_message(  # type: ignore[arg-type]
            SandboxMessage.Response(result="x" * MAX_PACKET_BYTES)
        )


@pytest.mark.asyncio
async def test_writer_rejects_deep_packet_as_value_error() -> None:
    class Writer:
        def write(self, data: bytes) -> None:
            del data

        async def drain(self) -> None:
            pass

    value: object = 0
    for _ in range(1000):
        value = [value]

    with pytest.raises(ValueError, match="not serializable"):
        await SandboxWriter(Writer()).write_message(  # type: ignore[arg-type]
            SandboxMessage.Response(result=value)  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_reader_normalizes_malformed_model_marker() -> None:
    payload = (
        b'{"_":"response","result":{"__model__":"RestrictedPythonTask",'
        b'"function":"test","agent":"agent","args":[],"keyword_args":[]}}'
    )
    reader = asyncio.StreamReader()
    reader.feed_data(len(payload).to_bytes(4, "little") + payload)

    with pytest.raises(ValueError, match="Invalid sandbox message"):
        await SandboxReader(reader).read_message()


@pytest.mark.asyncio
async def test_local_protocol_rejects_oversized_callback_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("hyperforge.codemode.sandbox.MAX_PACKET_BYTES", 512)

    async def callback(_task: RestrictedPythonTask):
        return "x" * 512

    runner = SandboxRunner.isolated_process(callback)

    with pytest.raises(RuntimeError, match="maximum size"):
        await runner.run(
            WorkerExecutionRequest(
                code="capability()",
                local_vars={},
                global_vars={},
                function_names={
                    "test": {
                        "capability": FunctionDefinition(
                            name="capability", description="", parameters={}
                        )
                    }
                },
                max_runtime_seconds=1,
            )
        )


@pytest.mark.asyncio
async def test_local_runner_preserves_python_request_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    runner = SandboxRunner.with_pool(object(), callback)  # type: ignore[arg-type]
    captured = None

    async def run_in_pool(request: WorkerExecutionRequest) -> None:
        nonlocal captured
        captured = request.local_vars["value"]

    monkeypatch.setattr(runner, "_run_in_pool", run_in_pool)

    await runner.run(
        WorkerExecutionRequest(
            code="",
            local_vars={"value": b"not-json"},
            global_vars={},
            function_names={},
        )
    )

    assert captured == b"not-json"
    assert isinstance(captured, bytes)


@pytest.mark.asyncio
async def test_remote_callback_cancellation_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    callback_done = asyncio.Event()

    async def callback(_task: RestrictedPythonTask):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()
        finally:
            callback_done.set()

    class Reader:
        async def read_message(self):
            await asyncio.Future()

    class Writer:
        async def write_message(self, _message):
            pytest.fail("cancelled callback must not send a response")

    monkeypatch.setattr("hyperforge.codemode.sandbox.CALLBACK_CANCEL_TIMEOUT", 0.01)
    runner = SandboxRunner.isolated_process(callback)
    execution = asyncio.create_task(
        runner._run_remote_callback(
            Reader(),  # type: ignore[arg-type]
            Writer(),  # type: ignore[arg-type]
            RestrictedPythonTask(
                function="capability", agent="test", args=(), keyword_args={}
            ),
        )
    )
    await started.wait()

    execution.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(execution, timeout=0.2)

    release.set()
    await asyncio.wait_for(callback_done.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_remote_runners_share_client_session_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0

    async def run_remotely(
        _runner: SandboxRunner, _request: WorkerExecutionRequest
    ) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1

    async def callback(_task: RestrictedPythonTask):
        return None

    monkeypatch.setattr(sandbox.settings, "sandbox_max_concurrent_sessions", 2)
    monkeypatch.setattr(SandboxRunner, "_run_remotely", run_remotely)
    request = WorkerExecutionRequest(
        code="", local_vars={}, global_vars={}, function_names={}
    )
    runners = [SandboxRunner.remote("/unused.sock", callback) for _ in range(5)]

    await asyncio.gather(*(runner.run(request) for runner in runners))

    assert maximum_active == 2


@pytest.mark.asyncio
async def test_remote_session_limit_is_shared_across_event_loops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = 0
    maximum_active = 0
    counter_lock = threading.Lock()

    async def run_remotely(
        _runner: SandboxRunner, _request: WorkerExecutionRequest
    ) -> None:
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            await asyncio.sleep(0.02)
        finally:
            with counter_lock:
                active -= 1

    async def callback(_task: RestrictedPythonTask):
        return None

    request = WorkerExecutionRequest(
        code="", local_vars={}, global_vars={}, function_names={}
    )

    def run_in_new_loop() -> None:
        runner = SandboxRunner.remote("/unused.sock", callback)
        asyncio.run(runner.run(request))

    monkeypatch.setattr(sandbox.settings, "sandbox_max_concurrent_sessions", 2)
    monkeypatch.setattr(SandboxRunner, "_run_remotely", run_remotely)

    await asyncio.gather(*(asyncio.to_thread(run_in_new_loop) for _ in range(6)))

    assert maximum_active == 2


@pytest.mark.asyncio
async def test_remote_run_timeout_adds_slack_to_requested_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox.settings, "sandbox_timeout_slack_seconds", 5.0)
    completed = []

    async def slow_admission(_runner: SandboxRunner, _request: WorkerExecutionRequest):
        await asyncio.sleep(0.05)
        completed.append(True)

    async def callback(_task: RestrictedPythonTask):
        return None

    monkeypatch.setattr(SandboxRunner, "_run_with_remote_admission", slow_admission)
    runner = SandboxRunner.remote("/unused.sock", callback)
    request = WorkerExecutionRequest(
        code="",
        local_vars={},
        global_vars={},
        function_names={},
        max_runtime_seconds=0.01,
    )

    await runner.run(request)

    assert completed == [True]


@pytest.mark.asyncio
async def test_remote_run_without_requested_runtime_uses_server_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox.settings, "sandbox_timeout_slack_seconds", 0.0)
    monkeypatch.setattr(sandbox.settings, "sandbox_max_session_runtime_seconds", 0.05)

    async def hanging_admission(
        _runner: SandboxRunner, _request: WorkerExecutionRequest
    ):
        await asyncio.Future()

    async def callback(_task: RestrictedPythonTask):
        return None

    monkeypatch.setattr(SandboxRunner, "_run_with_remote_admission", hanging_admission)
    runner = SandboxRunner.remote("/unused.sock", callback)
    request = WorkerExecutionRequest(
        code="", local_vars={}, global_vars={}, function_names={}
    )

    with pytest.raises(RuntimeError, match="timed out"):
        await runner.run(request)


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
