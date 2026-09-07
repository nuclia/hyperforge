import asyncio
import contextvars
import hmac
import logging
import os
import shutil
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from concurrent.futures import Future as ConcurrentFuture
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Awaitable, Callable, Coroutine

import httpx
import nucliadb_telemetry.metrics
import prometheus_client
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from .model import (
    MAX_PROTOCOL_BYTES,
    RestrictedPythonTask,
    SandboxMessage,
    WorkerError,
    WorkerExecutionRequest,
    WorkerTypes,
    decode_json_value,
    decode_protocol_value,
    encode_protocol_value,
    encode_sandbox_message,
)
from .worker import PythonAgentWorker

logger = logging.getLogger("hyperforge_codemode_sandbox")
WORKER_EXIT_TIMEOUT = 1
CALLBACK_CANCEL_TIMEOUT = 1
MAX_PACKET_BYTES = MAX_PROTOCOL_BYTES


class SandboxSettings(BaseSettings):
    sandbox_verify: bool = True
    sandbox_socket: str | None = None
    sandbox_socket_mode: str = "0600"
    sandbox_socket_group: str | None = None
    sandbox_metrics_port: int = 8091
    sandbox_metrics_host: str = "127.0.0.1"
    sandbox_metrics_enabled: bool = False
    sandbox_token: str | None = None
    sandbox_max_concurrent_sessions: int = Field(default=4, gt=0)
    sandbox_max_session_runtime_seconds: float = Field(
        default=60, gt=0, allow_inf_nan=False
    )
    sandbox_max_session_memory_bytes: int = Field(default=512 * 1024 * 1024, gt=0)
    sandbox_timeout_slack_seconds: float = Field(
        default=10.0, ge=0, allow_inf_nan=False
    )
    sandbox_callback_wait_seconds: float = 1.0

    @field_validator("sandbox_socket_mode")
    @classmethod
    def validate_socket_mode(cls, value: str) -> str:
        try:
            mode = int(value, 8)
        except ValueError as exc:
            raise ValueError("SANDBOX_SOCKET_MODE must be an octal mode") from exc
        if mode < 0 or mode > 0o777:
            raise ValueError("SANDBOX_SOCKET_MODE must be between 0000 and 0777")
        return value


settings = SandboxSettings()
_remote_admission_lock = threading.Lock()
_remote_active_sessions = 0


class InsecureSandbox(Exception):
    pass


class SandboxRunner:
    @classmethod
    def with_pool(
        cls,
        pool: Executor,
        callback: Callable[[RestrictedPythonTask], Coroutine[Any, Any, WorkerTypes]],
        debug: bool = False,
    ) -> "SandboxRunner":
        return cls(pool=pool, socket=None, callback=callback, debug=debug)

    @classmethod
    def remote(
        cls,
        socket: str,
        callback: Callable[[RestrictedPythonTask], Coroutine[Any, Any, WorkerTypes]],
        debug: bool = False,
        *,
        token: str | Callable[[], str] | None = None,
    ) -> "SandboxRunner":
        return cls(
            pool=None,
            socket=socket,
            callback=callback,
            debug=debug,
            token=token,
        )

    @classmethod
    def isolated_process(
        cls,
        callback: Callable[[RestrictedPythonTask], Coroutine[Any, Any, WorkerTypes]],
    ) -> "SandboxRunner":
        runner = cls(pool=None, socket=None, callback=callback, debug=False)
        runner.isolated = True
        return runner

    def __init__(
        self,
        pool: Executor | None,
        socket: str | None,
        callback: Callable[[RestrictedPythonTask], Coroutine[Any, Any, WorkerTypes]],
        debug: bool,
        token: str | Callable[[], str] | None = None,
    ):
        self.pool = pool
        self.socket = socket
        self.callback = callback
        self.debug = debug
        self.isolated = False
        self.token = token
        self._callback_tasks: set[asyncio.Task[WorkerTypes]] = set()
        self._callback_futures: dict[
            asyncio.Task[WorkerTypes], ConcurrentFuture[WorkerTypes]
        ] = {}
        self._orphaned_callback_tasks: set[asyncio.Task[WorkerTypes]] = set()

    async def run(self, request: WorkerExecutionRequest):
        if self.pool is not None:
            return await self._run_in_pool(request)
        if self.socket is not None:
            encoded_request = encode_sandbox_message(
                SandboxMessage.Run(run=request),
                "Sandbox run request",
                max_bytes=MAX_PACKET_BYTES,
            )
            encoded_value = decode_json_value(
                encoded_request, "Sandbox run request", max_bytes=MAX_PACKET_BYTES
            )
            if not isinstance(encoded_value, dict):
                raise ValueError("Invalid sandbox run request")
            message = SandboxMessage.parse(encoded_value)
            if not isinstance(message, SandboxMessage.Run):
                raise ValueError("Invalid sandbox run request")
            try:
                runtime = message.run.max_runtime_seconds
                if runtime is None:
                    runtime = settings.sandbox_max_session_runtime_seconds
                async with asyncio.timeout(
                    runtime + settings.sandbox_timeout_slack_seconds
                ):
                    return await self._run_with_remote_admission(message.run)
            except TimeoutError as exc:
                raise RuntimeError("Codemode execution timed out") from exc
        if self.isolated:
            return await self._run_isolated(request)
        raise ValueError("SandboxRunner must be initialized with a pool or a socket")

    async def _run_with_remote_admission(self, request: WorkerExecutionRequest):
        await _acquire_remote_session()
        try:
            return await self._run_remotely(request)
        finally:
            _release_remote_session()

    async def _run_in_pool(self, request: WorkerExecutionRequest):
        loop = asyncio.get_running_loop()
        pipe_worker, pipe_restricted = Pipe()
        worker = PythonAgentWorker(pipe_worker, debug=self.debug)
        controller_task = asyncio.create_task(self.background_task(pipe_restricted))
        try:
            await loop.run_in_executor(
                self.pool,
                worker._process_question_context_sync,
                request.code,
                request.question,
                request.local_vars,
                request.global_vars,
                request.function_names,
                request.max_memory_bytes,
            )
            await controller_task
        finally:
            await self._cancel_callbacks()
            if not controller_task.done():
                controller_task.cancel()
            await asyncio.gather(controller_task, return_exceptions=True)

    async def _run_isolated(self, request: WorkerExecutionRequest):
        controller_task, process = self.run_in_process(request)
        try:
            if request.max_runtime_seconds is None:
                clean_exit = await controller_task
            else:
                async with asyncio.timeout(request.max_runtime_seconds):
                    clean_exit = await controller_task
            if not clean_exit:
                raise RuntimeError("Codemode worker exited unexpectedly")
            await asyncio.to_thread(process.join, WORKER_EXIT_TIMEOUT)
            if process.exitcode != 0:
                raise RuntimeError(
                    "Codemode worker exceeded its memory limit or exited unexpectedly"
                )
        except TimeoutError as exc:
            raise RuntimeError("Codemode execution timed out") from exc
        finally:
            if process.is_alive():
                process.kill()
            await asyncio.to_thread(process.join)
            await self._cancel_callbacks()
            if not controller_task.done():
                controller_task.cancel()
            await asyncio.gather(controller_task, return_exceptions=True)

    async def _run_remotely(self, request: WorkerExecutionRequest):
        token_source = self.token
        if token_source is None:
            token = SandboxSettings().sandbox_token
        elif isinstance(token_source, str):
            token = token_source
        else:
            token = token_source()
        if not token:
            raise RuntimeError("SANDBOX_TOKEN is required for remote codemode")
        rx, tx = await asyncio.open_unix_connection(self.socket)
        reader, writer = SandboxReader(rx), SandboxWriter(tx)
        try:
            await writer.write_message(SandboxMessage.Run(run=request, token=token))
            while True:
                try:
                    msg = await reader.read_message()
                except asyncio.IncompleteReadError as exc:
                    raise RuntimeError(
                        "Sandbox connection closed unexpectedly"
                    ) from exc
                if isinstance(msg, SandboxMessage.Done):
                    break
                if isinstance(msg, SandboxMessage.Error):
                    raise RuntimeError(f"Python agent error: {msg.error}")
                if not isinstance(msg, SandboxMessage.Request):
                    raise RuntimeError("Unexpected sandbox protocol message")
                await self._run_remote_callback(reader, writer, msg.task)
        finally:
            await self._cancel_callbacks()
            writer.close()
            await writer.wait_closed()

    async def _run_remote_callback(
        self,
        reader: "SandboxReader",
        writer: "SandboxWriter",
        task: RestrictedPythonTask,
    ) -> None:
        callback_task = asyncio.create_task(self.callback(task))
        self._callback_tasks.add(callback_task)
        incoming = asyncio.create_task(reader.read_message())
        try:
            done, _ = await asyncio.wait(
                {callback_task, incoming}, return_when=asyncio.FIRST_COMPLETED
            )
            if incoming in done:
                message = incoming.result()
                if isinstance(message, SandboxMessage.Error):
                    raise RuntimeError(f"Python agent error: {message.error}")
                raise RuntimeError(
                    "Unexpected sandbox protocol message during callback"
                )
            try:
                response = callback_task.result()
            except Exception as exc:
                response = WorkerError(error=str(exc))
            await writer.write_message(SandboxMessage.Response(result=response))
        finally:
            for candidate in (callback_task, incoming):
                if not candidate.done():
                    candidate.cancel()
            done, still_pending = await asyncio.wait(
                {callback_task, incoming}, timeout=CALLBACK_CANCEL_TIMEOUT
            )
            for completed in done:
                await asyncio.gather(completed, return_exceptions=True)
            if still_pending:
                logger.warning("Sandbox callback ignored cancellation")
            self._callback_tasks.discard(callback_task)
            for remaining in still_pending:
                if remaining is callback_task:
                    self._track_orphaned_callback(callback_task)
                else:
                    remaining.add_done_callback(_consume_task_result)

    def run_in_process(self, request: WorkerExecutionRequest):
        pipe_worker, pipe_restricted = Pipe()
        worker = PythonAgentWorker(pipe_worker, debug=self.debug)
        controller_task = asyncio.create_task(self.background_task(pipe_restricted))
        process = Process(
            target=worker._process_question_context_sync,
            args=(
                request.code,
                request.question,
                request.local_vars,
                request.global_vars,
                request.function_names,
                request.max_memory_bytes,
            ),
        )
        process.start()
        pipe_worker.close()
        return controller_task, process

    async def background_task(self, pipe: Connection) -> bool:
        loop = asyncio.get_running_loop()
        context = contextvars.copy_context()
        try:
            return await loop.run_in_executor(
                ThreadPoolExecutor(max_workers=1),
                self.read_write_queue,
                pipe,
                loop,
                context,
            )
        finally:
            pipe.close()

    async def _cancel_callbacks(self) -> None:
        tasks = list(self._callback_tasks)
        for task in tasks:
            task.cancel()
            future = self._callback_futures.pop(task, None)
            if future is not None and not future.done():
                future.set_exception(RuntimeError("Sandbox callback cancelled"))
        if tasks:
            done, pending = await asyncio.wait(tasks, timeout=CALLBACK_CANCEL_TIMEOUT)
            for completed in done:
                await asyncio.gather(completed, return_exceptions=True)
            if pending:
                logger.warning("Sandbox callbacks ignored cancellation")
            self._callback_tasks.difference_update(tasks)
            for task in pending:
                self._track_orphaned_callback(task)

    def _track_orphaned_callback(self, task: asyncio.Task[WorkerTypes]) -> None:
        self._orphaned_callback_tasks.add(task)

        def completed(done: asyncio.Task[WorkerTypes]) -> None:
            self._orphaned_callback_tasks.discard(done)
            _consume_task_result(done)

        task.add_done_callback(completed)

    def run_when_callbacks_complete(self, callback: Callable[[], None]) -> bool:
        pending = {task for task in self._orphaned_callback_tasks if not task.done()}
        if not pending:
            return False

        def completed(done: asyncio.Task[WorkerTypes]) -> None:
            pending.discard(done)
            if not pending:
                callback()

        for task in pending:
            task.add_done_callback(completed)
        return True

    def read_write_queue(
        self,
        pipe: Connection,
        loop: asyncio.AbstractEventLoop,
        context: contextvars.Context,
    ) -> bool:
        try:
            while True:
                item = decode_protocol_value(
                    pipe.recv_bytes(MAX_PACKET_BYTES),
                    "Local sandbox request",
                    max_bytes=MAX_PACKET_BYTES,
                )
                if not isinstance(item, RestrictedPythonTask):
                    raise ValueError("Invalid local sandbox request")
                if item.function == "_error":
                    raise RuntimeError(item.args[0])
                if item.function == "_close":
                    return True
                future: ConcurrentFuture[WorkerTypes] = ConcurrentFuture()

                def schedule_callback() -> None:
                    task = loop.create_task(self.callback(item), context=context.copy())
                    self._callback_tasks.add(task)
                    self._callback_futures[task] = future

                    def copy_result(completed: asyncio.Task[WorkerTypes]) -> None:
                        self._callback_tasks.discard(completed)
                        controller_future = self._callback_futures.pop(completed, None)
                        if controller_future is None or controller_future.done():
                            return
                        try:
                            controller_future.set_result(completed.result())
                        except BaseException as exc:
                            controller_future.set_exception(exc)

                    task.add_done_callback(copy_result)

                loop.call_soon_threadsafe(schedule_callback)
                try:
                    result = future.result()
                except Exception as exc:
                    result = WorkerError(error=str(exc))
                try:
                    encoded = encode_protocol_value(
                        result,
                        "Local sandbox response",
                        max_bytes=MAX_PACKET_BYTES,
                    )
                except ValueError as exc:
                    encoded = encode_protocol_value(
                        WorkerError(error=str(exc)),
                        "Local sandbox response error",
                        max_bytes=MAX_PACKET_BYTES,
                    )
                pipe.send_bytes(encoded)
        except EOFError:
            return False


def _url_reachable(url: str) -> bool:
    with httpx.Client() as client:
        try:
            client.get(url)
            return True
        except httpx.ConnectError:
            return False


def _verify_connectivity():
    kubernetes_host = os.environ.get("KUBERNETES_SERVICE_HOST")
    if kubernetes_host and _url_reachable(f"https://{kubernetes_host}"):
        raise InsecureSandbox()
    if _url_reachable("https://nuclia.com"):
        raise InsecureSandbox()


class SandboxWriter:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer

    async def _write_packet(self, msg: SandboxMessage.AnyMessage):
        msg_bytes = encode_sandbox_message(
            msg, "Sandbox message", max_bytes=MAX_PACKET_BYTES
        )
        self.writer.write(len(msg_bytes).to_bytes(4, "little"))
        self.writer.write(msg_bytes)
        await self.writer.drain()

    async def write_message(self, message: SandboxMessage.AnyMessage):
        await self._write_packet(message)

    def close(self):
        self.writer.close()

    async def wait_closed(self):
        await self.writer.wait_closed()


class SandboxReader:
    def __init__(self, reader: asyncio.StreamReader):
        self.reader = reader

    async def _read_packet(self) -> dict[str, Any]:
        len_bytes = await self.reader.readexactly(4)
        msg_len = int.from_bytes(len_bytes, "little")
        if msg_len > MAX_PACKET_BYTES:
            raise ValueError("Sandbox message exceeds maximum size")
        msg_bytes = await self.reader.readexactly(msg_len)
        value = decode_json_value(
            msg_bytes, "Sandbox message", max_bytes=MAX_PACKET_BYTES
        )
        if not isinstance(value, dict):
            raise ValueError("Sandbox message must be a JSON object")
        return value

    async def read_message(self) -> SandboxMessage.AnyMessage:
        value = await self._read_packet()
        try:
            return SandboxMessage.parse(value)
        except (
            AttributeError,
            KeyError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError("Invalid sandbox message") from exc


class SandboxSession:
    def __init__(
        self,
        reader: SandboxReader,
        writer: SandboxWriter,
        callback_wait_seconds: float = 1.0,
    ):
        self.reader = reader
        self.writer = writer
        self.callback_wait_seconds = callback_wait_seconds
        self.runner = SandboxRunner(
            pool=None, socket=None, callback=self._callback, debug=False
        )
        self.waiting_for_agent: asyncio.Task[None] | None = None
        self.timed_out = False

    async def run(self, request: WorkerExecutionRequest):
        memory = settings.sandbox_max_session_memory_bytes
        if request.max_memory_bytes is not None:
            memory = min(memory, request.max_memory_bytes)
        request = request.model_copy(update={"max_memory_bytes": memory})
        self.task, self.process = self.runner.run_in_process(request)
        try:
            await self._start_timeout(self.callback_wait_seconds)
            runtime = settings.sandbox_max_session_runtime_seconds
            if request.max_runtime_seconds is not None:
                runtime = min(runtime, request.max_runtime_seconds)
            async with asyncio.timeout(runtime):
                clean_exit = await self.task
            await self._stop_timeout()
            if self.timed_out:
                raise RuntimeError("Python agent timeout")
            if not clean_exit:
                raise RuntimeError("Codemode worker exited unexpectedly")
            await asyncio.to_thread(self.process.join, WORKER_EXIT_TIMEOUT)
            if self.process.exitcode != 0:
                raise RuntimeError(
                    "Codemode worker exceeded its memory limit or exited unexpectedly"
                )
            await self.writer.write_message(SandboxMessage.Done())
        except TimeoutError:
            await self._fail(RuntimeError("Codemode execution timed out"))
        except Exception as exc:
            await self._fail(exc)
        finally:
            await self._stop_timeout()
            if self.process.is_alive():
                self.process.kill()
            await asyncio.to_thread(self.process.join)
            await self.runner._cancel_callbacks()
            if not self.task.done():
                self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def _fail(self, exc: Exception):
        logger.error("Error in sandbox session: %s", exc)
        if self.process.is_alive():
            self.process.kill()
        await asyncio.to_thread(self.process.join)
        await self.runner._cancel_callbacks()
        if not self.task.done():
            self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)
        try:
            await self.writer.write_message(SandboxMessage.Error(error=str(exc)))
        except Exception:
            pass

    async def _callback(self, item: RestrictedPythonTask) -> WorkerTypes:
        await self._stop_timeout()
        await self.writer.write_message(SandboxMessage.Request(task=item))
        try:
            response = await self.reader.read_message()
        except asyncio.IncompleteReadError:
            logger.warning("Got disconnected from agent")
            self.process.kill()
            self.process.join()
            self.task.cancel()
            return None
        if not isinstance(response, SandboxMessage.Response):
            raise RuntimeError("Unexpected sandbox protocol response")
        await self._start_timeout(self.callback_wait_seconds)
        return response.result

    async def _start_timeout(self, timeout: float):
        self.waiting_for_agent = asyncio.create_task(self._timeout(timeout))

    async def _stop_timeout(self):
        if self.waiting_for_agent:
            self.waiting_for_agent.cancel()
            self.waiting_for_agent = None

    async def _timeout(self, timeout: float):
        await asyncio.sleep(timeout)
        self.timed_out = True
        if self.process.is_alive():
            self.process.kill()


agents_running = prometheus_client.Gauge(
    "arag_sandbox_agents_running", "Number of Python agents currently running"
)
execution_observer = nucliadb_telemetry.metrics.Observer("arag_sandbox_execution")


async def run_sandbox_server(
    *, token_verifier: Callable[[str], Awaitable[bool]] | None = None
):
    server_settings = SandboxSettings()
    if server_settings.sandbox_socket is None:
        raise RuntimeError("SANDBOX_SOCKET is required for the sandbox server")
    sandbox_token = server_settings.sandbox_token
    if token_verifier is None and not sandbox_token:
        raise RuntimeError("SANDBOX_TOKEN is required for the sandbox server")
    if server_settings.sandbox_verify:
        _verify_connectivity()

    active_sessions = 0
    admission_lock = asyncio.Lock()
    handshake_limit = asyncio.Semaphore(settings.sandbox_max_concurrent_sessions)

    async def handler(rx: asyncio.StreamReader, tx: asyncio.StreamWriter):
        nonlocal active_sessions
        reader = SandboxReader(rx)
        writer = SandboxWriter(tx)
        if handshake_limit.locked():
            writer.close()
            await writer.wait_closed()
            return
        await handshake_limit.acquire()
        try:
            async with asyncio.timeout(1):
                msg = await reader.read_message()
                if not isinstance(msg, SandboxMessage.Run):
                    raise ValueError("Expected a sandbox run request")
                token = msg.token or ""
                valid_token = (
                    await token_verifier(token)
                    if token_verifier is not None
                    else hmac.compare_digest(token, sandbox_token or "")
                )
                if not valid_token:
                    raise PermissionError("Invalid sandbox token")
        except Exception as exc:
            logger.warning("Rejected invalid sandbox connection: %r", exc)
            writer.close()
            await writer.wait_closed()
            return
        finally:
            handshake_limit.release()

        async with admission_lock:
            if active_sessions >= settings.sandbox_max_concurrent_sessions:
                await writer.write_message(
                    SandboxMessage.Error(error="Sandbox concurrency limit reached")
                )
                writer.close()
                await writer.wait_closed()
                return
            active_sessions += 1

        agents_running.inc()
        observation = execution_observer()
        observation.start()
        try:
            session = SandboxSession(
                reader, writer, SandboxSettings().sandbox_callback_wait_seconds
            )
            await session.run(msg.run)
        except Exception as exc:
            observation.set_status("error")
            await writer.write_message(SandboxMessage.Error(error=str(exc)))
        finally:
            async with admission_lock:
                active_sessions -= 1
            agents_running.dec()
            observation.end()
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_unix_server(handler, server_settings.sandbox_socket)
    if server_settings.sandbox_socket_group is not None:
        group: str | int = server_settings.sandbox_socket_group
        if server_settings.sandbox_socket_group.isdecimal():
            group = int(server_settings.sandbox_socket_group)
        shutil.chown(server_settings.sandbox_socket, group=group)
    os.chmod(server_settings.sandbox_socket, int(server_settings.sandbox_socket_mode, 8))
    async with server:
        await server.serve_forever()


def _consume_task_result(task: asyncio.Future[Any]) -> None:
    try:
        task.exception()
    except asyncio.CancelledError:
        pass


async def _acquire_remote_session() -> None:
    global _remote_active_sessions
    while True:
        with _remote_admission_lock:
            if _remote_active_sessions < settings.sandbox_max_concurrent_sessions:
                _remote_active_sessions += 1
                return
        await asyncio.sleep(0.01)


def _release_remote_session() -> None:
    global _remote_active_sessions
    with _remote_admission_lock:
        if _remote_active_sessions <= 0:
            raise RuntimeError("Remote sandbox admission counter underflow")
        _remote_active_sessions -= 1
