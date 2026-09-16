import asyncio
import contextvars
import io
import tempfile
import threading
import traceback
from concurrent.futures import Future as ConcurrentFuture
from contextlib import redirect_stderr
from pathlib import Path

import pytest
from nuclia_models.predict.remi import RemiResponse

from hyperforge.codemode import sandbox
from hyperforge.codemode import worker as worker_module
from hyperforge.codemode.model import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerError,
    WorkerExecutionRequest,
    decode_protocol_value,
    deserialize,
    encode_protocol_value,
)
from hyperforge.codemode.sandbox import (
    MAX_PACKET_BYTES,
    SandboxReader,
    SandboxRunner,
    SandboxSession,
    SandboxSettings,
    SandboxWriter,
    run_sandbox_server,
)
from hyperforge.codemode.worker import PythonAgentWorker
from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Context
from hyperforge.models import Chunk


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


def test_redacted_worker_transport_failure_does_not_expose_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "worker-stderr-secret"

    class ClosedPipe:
        closed = False

        def send_bytes(self, _payload: bytes) -> None:
            raise BrokenPipeError("pipe closed")

        def close(self) -> None:
            self.closed = True

    pipe = ClosedPipe()
    worker = PythonAgentWorker(pipe)  # type: ignore[arg-type]
    monkeypatch.setattr(worker_module, "_harden_process", lambda _limit: None)
    stderr = io.StringIO()

    with redirect_stderr(stderr):
        try:
            worker._process_question_context_sync(
                f"raise RuntimeError('{secret}')",
                "",
                {},
                {},
                {},
                redact_errors=True,
            )
        except BaseException:
            traceback.print_exc()

    assert pipe.closed
    assert secret not in stderr.getvalue()


def test_generic_worker_transport_failure_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ClosedPipe:
        def send_bytes(self, _payload: bytes) -> None:
            raise BrokenPipeError("pipe closed")

    worker = PythonAgentWorker(ClosedPipe())  # type: ignore[arg-type]
    monkeypatch.setattr(worker_module, "_harden_process", lambda _limit: None)

    with pytest.raises(BrokenPipeError, match="pipe closed"):
        worker._process_question_context_sync("1 + 1", "", {}, {}, {})


@pytest.mark.parametrize(
    "first_output",
    ["output({1, 2})", "output('x' * 512)"],
)
def test_redacted_worker_latches_caught_output_serialization_failure(
    monkeypatch: pytest.MonkeyPatch, first_output: str
) -> None:
    class Pipe:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def send_bytes(self, payload: bytes) -> None:
            self.sent.append(payload)

        def recv_bytes(self, _max_bytes: int) -> bytes:
            return encode_protocol_value(None, "Worker response", max_bytes=256)

    pipe = Pipe()
    worker = PythonAgentWorker(pipe)  # type: ignore[arg-type]
    monkeypatch.setattr(worker_module, "_harden_process", lambda _limit: None)
    monkeypatch.setattr(worker_module, "MAX_PROTOCOL_BYTES", 256)
    code = f"""
try:
    {first_output}
except Exception:
    pass
output('valid')
"""

    worker._process_question_context_sync(
        code,
        "",
        {},
        {},
        _function_names("output"),
        redact_errors=True,
    )

    tasks = [
        decode_protocol_value(payload, "Worker task", max_bytes=256)
        for payload in pipe.sent
    ]
    assert [
        task.function for task in tasks if isinstance(task, RestrictedPythonTask)
    ] == [
        "output",
        "_error",
        "_close",
    ]
    error = tasks[1]
    assert isinstance(error, RestrictedPythonTask)
    assert error.args == ("Generated code execution failed",)


def test_generic_worker_can_recover_from_output_serialization_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Pipe:
        def __init__(self) -> None:
            self.sent: list[bytes] = []

        def send_bytes(self, payload: bytes) -> None:
            self.sent.append(payload)

        def recv_bytes(self, _max_bytes: int) -> bytes:
            return encode_protocol_value(None, "Worker response", max_bytes=256)

    pipe = Pipe()
    worker = PythonAgentWorker(pipe)  # type: ignore[arg-type]
    monkeypatch.setattr(worker_module, "_harden_process", lambda _limit: None)
    monkeypatch.setattr(worker_module, "MAX_PROTOCOL_BYTES", 256)

    worker._process_question_context_sync(
        """
try:
    output({1, 2})
except Exception:
    pass
output('valid')
""",
        "",
        {},
        {},
        _function_names("output"),
    )

    tasks = [
        decode_protocol_value(payload, "Worker task", max_bytes=256)
        for payload in pipe.sent
    ]
    assert [
        task.function for task in tasks if isinstance(task, RestrictedPythonTask)
    ] == [
        "output",
        "_close",
    ]


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
                max_runtime_seconds=30,
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
    runner._begin_callback_run()
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
    await runner._cancel_callbacks()
    runner._finish_callback_run()


@pytest.mark.asyncio
async def test_remote_callback_keeps_frame_arriving_during_response_write() -> None:
    write_started = asyncio.Event()
    frame_read = asyncio.Event()

    class Reader:
        async def read_message(self):
            await write_started.wait()
            frame_read.set()
            return SandboxMessage.Done()

    class Writer:
        async def write_message(self, _message):
            write_started.set()
            await frame_read.wait()

    async def callback(_task: RestrictedPythonTask):
        return None

    runner = SandboxRunner.isolated_process(callback)
    runner._begin_callback_run()
    incoming = await runner._run_remote_callback(
        Reader(),  # type: ignore[arg-type]
        Writer(),  # type: ignore[arg-type]
        RestrictedPythonTask(
            function="capability", agent="test", args=(), keyword_args={}
        ),
    )

    message = await asyncio.wait_for(incoming, timeout=0.2)

    assert isinstance(message, SandboxMessage.Done)
    await runner._cancel_callbacks()
    runner._finish_callback_run()


@pytest.mark.asyncio
async def test_callback_scheduled_after_cleanup_cannot_start() -> None:
    called = False
    intent_registered = threading.Event()
    allow_schedule = threading.Event()
    registration_result = False

    async def callback(_task: RestrictedPythonTask):
        nonlocal called
        called = True
        return None

    runner = SandboxRunner.isolated_process(callback)
    runner._begin_callback_run()
    future: ConcurrentFuture = ConcurrentFuture()
    loop = asyncio.get_running_loop()

    def schedule_from_executor_thread() -> None:
        nonlocal registration_result
        registration_result = runner._register_callback_intent(future)
        intent_registered.set()
        allow_schedule.wait()
        loop.call_soon_threadsafe(
            runner._schedule_callback,
            future,
            RestrictedPythonTask(
                function="capability", agent="test", args=(), keyword_args={}
            ),
            loop,
            contextvars.copy_context(),
        )

    executor_thread = threading.Thread(target=schedule_from_executor_thread)
    executor_thread.start()
    await asyncio.to_thread(intent_registered.wait)
    assert registration_result

    await runner._cancel_callbacks()
    runner._finish_callback_run()
    allow_schedule.set()
    await asyncio.to_thread(executor_thread.join)
    await asyncio.sleep(0)

    assert not called
    with pytest.raises(RuntimeError, match="cancelled"):
        future.result()
    assert not runner.run_when_callbacks_complete(lambda: None)


@pytest.mark.asyncio
async def test_runner_rejects_reuse_until_orphaned_callback_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def callback(_task: RestrictedPythonTask):
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            await release.wait()

    monkeypatch.setattr(sandbox, "CALLBACK_CANCEL_TIMEOUT", 0.01)
    runner = SandboxRunner.isolated_process(callback)
    runner._begin_callback_run()
    future: ConcurrentFuture = ConcurrentFuture()
    assert runner._register_callback_intent(future)
    runner._schedule_callback(
        future,
        RestrictedPythonTask(
            function="capability", agent="test", args=(), keyword_args={}
        ),
        asyncio.get_running_loop(),
        contextvars.copy_context(),
    )
    callback_task = next(iter(runner._callback_tasks))
    await started.wait()

    await runner._cancel_callbacks()
    runner._finish_callback_run()

    with pytest.raises(RuntimeError, match="callbacks have completed"):
        await runner.run(_empty_request())

    release.set()
    await callback_task
    with pytest.raises(RuntimeError, match="cancelled"):
        future.result()

    async def complete(_request: WorkerExecutionRequest) -> None:
        return None

    monkeypatch.setattr(runner, "_run_isolated", complete)
    await runner.run(_empty_request())


@pytest.mark.asyncio
async def test_isolated_runner_rejects_non_json_request_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    runner = SandboxRunner.isolated_process(callback)

    def unexpected_run(_request: WorkerExecutionRequest):
        pytest.fail("invalid request reached the isolated worker")

    monkeypatch.setattr(runner, "run_in_process", unexpected_run)

    with pytest.raises(ValueError, match="not serializable"):
        await runner.run(
            WorkerExecutionRequest(
                code="",
                local_vars={"value": b"not-json"},
                global_vars={},
                function_names={},
            )
        )


@pytest.mark.asyncio
async def test_isolated_runner_rejects_oversized_request_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    runner = SandboxRunner.isolated_process(callback)

    def unexpected_run(_request: WorkerExecutionRequest):
        pytest.fail("oversized request reached the isolated worker")

    monkeypatch.setattr(runner, "run_in_process", unexpected_run)
    monkeypatch.setattr(sandbox, "MAX_PACKET_BYTES", 256)

    with pytest.raises(ValueError, match="maximum size"):
        await runner.run(
            WorkerExecutionRequest(
                code="x" * 512,
                local_vars={},
                global_vars={},
                function_names={},
            )
        )


@pytest.mark.asyncio
async def test_isolated_runner_executes_canonical_decoded_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    captured: WorkerExecutionRequest | None = None
    runner = SandboxRunner.isolated_process(callback)

    async def capture(request: WorkerExecutionRequest) -> None:
        nonlocal captured
        captured = request

    monkeypatch.setattr(runner, "_run_isolated", capture)

    await runner.run(
        WorkerExecutionRequest(
            code="",
            local_vars={"value": (1, 2)},
            global_vars={},
            function_names={},
        )
    )

    assert captured is not None
    assert captured.local_vars["value"] == [1, 2]


@pytest.mark.asyncio
async def test_isolated_runner_preserves_supported_marked_request_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    context = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        chunks=[Chunk(chunk_id="chunk", text="content")],
        source="source",
        agent="agent",
    )
    remi = RemiResponse(time=0.25)
    error = WorkerError(error="failed")
    task = RestrictedPythonTask(
        function="lookup", agent="agent", args=(), keyword_args={}
    )
    captured: WorkerExecutionRequest | None = None
    runner = SandboxRunner.isolated_process(callback)

    async def capture(request: WorkerExecutionRequest) -> None:
        nonlocal captured
        captured = request

    monkeypatch.setattr(runner, "_run_isolated", capture)

    await runner.run(
        WorkerExecutionRequest(
            code="",
            local_vars={"context": context, "remi": remi, "task": task},
            global_vars={"error": error},
            function_names={},
        )
    )

    assert captured is not None
    assert captured.local_vars == {"context": context, "remi": remi, "task": task}
    assert captured.global_vars == {"error": error}


@pytest.mark.asyncio
async def test_generic_isolated_runner_preserves_generated_runtime_error() -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    runner = SandboxRunner.isolated_process(callback)
    request = WorkerExecutionRequest(
        code="raise RuntimeError('legacy runtime detail')",
        local_vars={},
        global_vars={},
        function_names={},
    )

    with pytest.raises(RuntimeError, match="legacy runtime detail"):
        await runner.run(request)


@pytest.mark.asyncio
async def test_sandbox_session_reports_process_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Writer:
        def __init__(self) -> None:
            self.messages: list[SandboxMessage.AnyMessage] = []

        async def write_message(self, message: SandboxMessage.AnyMessage) -> None:
            self.messages.append(message)

    writer = Writer()
    session = SandboxSession(
        object(),  # type: ignore[arg-type]
        writer,  # type: ignore[arg-type]
    )

    def fail_to_start(_request: WorkerExecutionRequest):
        raise OSError("process start failed")

    monkeypatch.setattr(session.runner, "run_in_process", fail_to_start)

    await session.run(_empty_request())

    assert len(writer.messages) == 1
    response = writer.messages[0]
    assert isinstance(response, SandboxMessage.Error)
    assert response.error == "process start failed"
    assert session.task is None
    assert session.process is None
    assert session.runner._callbacks_closed
    assert not session.runner._run_active


@pytest.mark.asyncio
async def test_remote_run_consumes_frame_arriving_during_response_write(
    monkeypatch: pytest.MonkeyPatch, socket_path: str
) -> None:
    done_frame_read = asyncio.Event()
    original_read = SandboxReader.read_message
    original_write = SandboxWriter.write_message

    async def tracked_read(self):
        message = await original_read(self)
        if isinstance(message, SandboxMessage.Done):
            done_frame_read.set()
        return message

    async def stalled_write(self, message):
        await original_write(self, message)
        if isinstance(message, SandboxMessage.Response):
            await done_frame_read.wait()

    monkeypatch.setattr(SandboxReader, "read_message", tracked_read)
    monkeypatch.setattr(SandboxWriter, "write_message", stalled_write)

    async def handler(rx: asyncio.StreamReader, tx: asyncio.StreamWriter) -> None:
        reader, writer = SandboxReader(rx), SandboxWriter(tx)
        run_message = await reader.read_message()
        assert isinstance(run_message, SandboxMessage.Run)
        await writer.write_message(
            SandboxMessage.Request(
                task=RestrictedPythonTask(
                    function="capability", agent="test", args=(), keyword_args={}
                )
            )
        )
        response = await reader.read_message()
        assert isinstance(response, SandboxMessage.Response)
        await writer.write_message(SandboxMessage.Done())
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_unix_server(handler, socket_path)

    async def callback(_task: RestrictedPythonTask):
        return None

    runner = SandboxRunner.remote(socket_path, callback, token="server-secret")
    request = WorkerExecutionRequest(
        code="", local_vars={}, global_vars={}, function_names={}
    )
    try:
        await asyncio.wait_for(runner.run(request), timeout=2)
    finally:
        server.close()
        await server.wait_closed()


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
async def test_remote_admission_release_uses_acquisition_ticket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def callback(_task: RestrictedPythonTask):
        return None

    async def change_limit(
        _runner: SandboxRunner, _request: WorkerExecutionRequest
    ) -> None:
        monkeypatch.setattr(sandbox.settings, "sandbox_max_concurrent_sessions", None)

    monkeypatch.setattr(sandbox.settings, "sandbox_max_concurrent_sessions", 1)
    monkeypatch.setattr(SandboxRunner, "_run_remotely", change_limit)
    runner = SandboxRunner.remote("/unused.sock", callback)

    await runner.run(_empty_request())

    assert sandbox._remote_admission_counter.active_sessions == 0


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
async def test_sandbox_server_uses_one_settings_snapshot(
    monkeypatch: pytest.MonkeyPatch, socket_path: str
) -> None:
    snapshot = SandboxSettings(
        sandbox_socket=socket_path,
        sandbox_verify=False,
        sandbox_token=None,
        sandbox_callback_wait_seconds=15,
        sandbox_max_concurrent_sessions=1,
        sandbox_max_session_runtime_seconds=30,
        sandbox_max_session_memory_bytes=256 * 1024 * 1024,
    )
    calls = 0

    def settings_factory() -> SandboxSettings:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise AssertionError("sandbox server reloaded settings")
        return snapshot

    monkeypatch.setattr(sandbox, "SandboxSettings", settings_factory)
    server_task = asyncio.create_task(run_sandbox_server())
    for _ in range(200):
        if Path(socket_path).exists():
            break
        await asyncio.sleep(0.01)
    try:
        await _run_client(socket_path, "value = 1", token="client-token")
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)

    assert calls == 1


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
async def test_sandbox_server_accepts_missing_token_without_verifier(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")
    server_task = await _serve(monkeypatch, socket_path)
    try:
        await _run_client(socket_path, "value = 1", token=None)
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


@pytest.mark.asyncio
async def test_remote_runner_allows_missing_token(
    monkeypatch, socket_path: str
) -> None:
    monkeypatch.setenv("SANDBOX_VERIFY", "false")
    monkeypatch.delenv("SANDBOX_TOKEN", raising=False)
    monkeypatch.setenv("SANDBOX_CALLBACK_WAIT_SECONDS", "15")

    async def callback(task):
        return None

    server_task = await _serve(monkeypatch, socket_path)
    runner = SandboxRunner.remote(socket_path, callback)
    try:
        await runner.run(_empty_request())
    finally:
        server_task.cancel()
        await asyncio.gather(server_task, return_exceptions=True)


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
