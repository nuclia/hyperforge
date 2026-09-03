import asyncio
import contextvars
import hmac
import json
import logging
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from concurrent.futures import Future as ConcurrentFuture
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Coroutine

import httpx
import nucliadb_telemetry.metrics
import prometheus_client
from pydantic_settings import BaseSettings

from .model import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerError,
    WorkerExecutionRequest,
    WorkerTypes,
)
from .worker import PythonAgentWorker

logger = logging.getLogger("hyperforge_codemode_sandbox")
WORKER_CPU_LIMIT = 1
WORKER_EXIT_TIMEOUT = 1
MAX_PACKET_BYTES = 16 * 1024 * 1024


class SandboxSettings(BaseSettings):
    sandbox_verify: bool = True
    sandbox_socket: str | None = None
    sandbox_metrics_port: int = 8091
    sandbox_token: str | None = None


settings = SandboxSettings()


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
    ) -> "SandboxRunner":
        return cls(pool=None, socket=socket, callback=callback, debug=debug)

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
    ):
        self.pool = pool
        self.socket = socket
        self.callback = callback
        self.debug = debug
        self.isolated = False

    async def run(self, request: WorkerExecutionRequest):
        if self.pool is not None:
            return await self._run_in_pool(request)
        if self.socket is not None:
            return await self._run_remotely(request)
        if self.isolated:
            return await self._run_isolated(request)
        raise ValueError("SandboxRunner must be initialized with a pool or a socket")

    async def _run_in_pool(self, request: WorkerExecutionRequest):
        loop = asyncio.get_running_loop()
        pipe_worker, pipe_restricted = Pipe()
        worker = PythonAgentWorker(pipe_worker, debug=self.debug)
        controller_task = asyncio.create_task(self.background_task(pipe_restricted))
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
            if not controller_task.done():
                controller_task.cancel()
            await asyncio.gather(controller_task, return_exceptions=True)

    async def _run_remotely(self, request: WorkerExecutionRequest):
        rx, tx = await asyncio.open_unix_connection(self.socket)
        reader, writer = SandboxReader(rx), SandboxWriter(tx)
        try:
            await writer.write_message(
                SandboxMessage.Run(run=request, token=settings.sandbox_token)
            )
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
                assert isinstance(msg, SandboxMessage.Request)
                try:
                    response = await self.callback(msg.task)
                except Exception as exc:
                    response = WorkerError(error=str(exc))
                await writer.write_message(SandboxMessage.Response(result=response))
        finally:
            writer.close()
            await writer.wait_closed()

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

    def read_write_queue(
        self,
        pipe: Connection,
        loop: asyncio.AbstractEventLoop,
        context: contextvars.Context,
    ) -> bool:
        try:
            while True:
                item = pipe.recv()
                if item.function == "_error":
                    raise RuntimeError(item.args[0])
                if item.function == "_close":
                    return True
                future: ConcurrentFuture[WorkerTypes] = ConcurrentFuture()

                def schedule_callback() -> None:
                    task = loop.create_task(self.callback(item), context=context.copy())

                    def copy_result(completed: asyncio.Task[WorkerTypes]) -> None:
                        try:
                            future.set_result(completed.result())
                        except Exception as exc:
                            future.set_exception(exc)

                    task.add_done_callback(copy_result)

                loop.call_soon_threadsafe(schedule_callback)
                try:
                    result = future.result()
                except Exception as exc:
                    result = WorkerError(error=str(exc))
                pipe.send(result)
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

    async def _write_packet(self, msg: dict[str, Any]):
        msg_bytes = json.dumps(msg).encode("utf-8")
        if len(msg_bytes) > MAX_PACKET_BYTES:
            raise ValueError("Sandbox message exceeds maximum size")
        self.writer.write(len(msg_bytes).to_bytes(4, "little"))
        self.writer.write(msg_bytes)
        await self.writer.drain()

    async def write_message(self, message: SandboxMessage.AnyMessage):
        await self._write_packet(SandboxMessage.serialize(message))

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
        return json.loads(msg_bytes.decode("utf-8"))

    async def read_message(self) -> SandboxMessage.AnyMessage:
        return SandboxMessage.parse(await self._read_packet())


class SandboxSession:
    def __init__(self, reader: SandboxReader, writer: SandboxWriter):
        self.reader = reader
        self.writer = writer
        self.runner = SandboxRunner(
            pool=None, socket=None, callback=self._callback, debug=False
        )
        self.waiting_for_agent: asyncio.Task[None] | None = None
        self.timed_out = False

    async def run(self, request: WorkerExecutionRequest):
        self.task, self.process = self.runner.run_in_process(request)
        try:
            await self._start_timeout(WORKER_CPU_LIMIT)
            if request.max_runtime_seconds is None:
                clean_exit = await self.task
            else:
                async with asyncio.timeout(request.max_runtime_seconds):
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
            if not self.task.done():
                self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)

    async def _fail(self, exc: Exception):
        logger.error("Error in sandbox session: %s", exc)
        if self.process.is_alive():
            self.process.kill()
        await asyncio.to_thread(self.process.join)
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
        assert isinstance(response, SandboxMessage.Response)
        await self._start_timeout(WORKER_CPU_LIMIT)
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


async def run_sandbox_server():
    assert settings.sandbox_socket is not None
    sandbox_token = settings.sandbox_token
    if settings.sandbox_verify:
        _verify_connectivity()

    async def handler(rx: asyncio.StreamReader, tx: asyncio.StreamWriter):
        reader = SandboxReader(rx)
        writer = SandboxWriter(tx)
        try:
            async with asyncio.timeout(1):
                msg = await reader.read_message()
                assert isinstance(msg, SandboxMessage.Run)
                if sandbox_token is not None and not hmac.compare_digest(
                    msg.token or "", sandbox_token
                ):
                    raise PermissionError("Invalid sandbox token")
        except (AssertionError, PermissionError, TimeoutError, ValueError):
            logger.warning("Rejected invalid sandbox connection")
            writer.close()
            await writer.wait_closed()
            return

        agents_running.inc()
        observation = execution_observer()
        observation.start()
        try:
            session = SandboxSession(reader, writer)
            await session.run(msg.run)
        except Exception as exc:
            observation.set_status("error")
            await writer.write_message(SandboxMessage.Error(error=str(exc)))
        finally:
            agents_running.dec()
            observation.end()
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_unix_server(handler, settings.sandbox_socket)
    os.chmod(settings.sandbox_socket, 0o600)
    async with server:
        await server.serve_forever()
