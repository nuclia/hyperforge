import asyncio
import json
import logging
import os
from concurrent.futures import Executor, ThreadPoolExecutor
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, Coroutine

import httpx
import nucliadb_telemetry.metrics
import prometheus_client
from pydantic_settings import BaseSettings

from agents.restricted.src.hyperforge_restricted.model import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerExecutionRequest,
    WorkerTypes,
)
from agents.restricted.src.hyperforge_restricted.worker import PythonAgentWorker

logger = logging.getLogger("arag_sandbox")

# Maximum time in seconds that the worker can execute CPU bound work. This is reset every time the worker
# calls any RAO function (another agent, LLM questions, etc.). It should be kept really low to avoid
# denial of service. It can be as low as a few milliseconds, it probably should not exceed 1 second.
WORKER_CPU_LIMIT = 1


class SandboxSettings(BaseSettings):
    # Verify that the sandbox is secure on startup or raise an exception
    sandbox_verify: bool = False

    # Socket for the sandbox server to listen to
    sandbox_socket: str | None = None

    # Port for prometheus metrics
    sandbox_metrics_port: int = 8091


settings = SandboxSettings()


class InsecureSandbox(Exception):
    pass


class SandboxRunner:
    pool: Executor | None
    socket: str | None
    callback: Callable[[RestrictedPythonTask], Coroutine[Any, Any, WorkerTypes]]
    debug: bool

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

    async def run(self, request: WorkerExecutionRequest):
        if self.pool is not None:
            return await self._run_in_pool(request)
        elif self.socket is not None:
            return await self._run_remotely(request)
        else:
            raise ValueError(
                "SandboxRunner must be initialized with a pool or a socket"
            )

    async def _run_in_pool(self, request: WorkerExecutionRequest):
        """
        Runs the two tasks needed to run a sandboxed Python agent:
          - _process_question_context_sync runs the untrusted code in a process pool executor
            under RestrictedPython and has limited I/O.
          - agent_runner_task calls read_write_queue and starts a "server" listening for messages
            from the sandbox. It then executes the desired function in behalf of the user provided
            code and sends the results back over a Pipe.
        """
        loop = asyncio.get_running_loop()
        (pipe_worker, pipe_restricted) = Pipe()
        worker = PythonAgentWorker(pipe_worker, debug=self.debug)
        agent_runner_task = asyncio.create_task(self.background_task(pipe_restricted))
        await loop.run_in_executor(
            self.pool,
            worker._process_question_context_sync,
            request.code,
            request.question,
            request.local_vars,
            request.global_vars,
            request.function_names,
        )
        # _process_question_context_sync always sends a message to the pipe on exit (finally block)
        # this causes the agent_runner_task to finish cleanly. Otherwise (e.g: if we call cancel on the task)
        # we would be leaking threads from the executor used for agent_runner_tasks
        await agent_runner_task

    async def _run_remotely(self, request: WorkerExecutionRequest):
        """
        Runs the code in a remote sandbox server (another container in the same machine)
        Communicates via Unix socket with the sandbox server and runs agent tasks for it
        until the code finishes
        """
        rx, tx = await asyncio.open_unix_connection(self.socket)
        reader, writer = SandboxReader(rx), SandboxWriter(tx)
        await writer.write_message(SandboxMessage.Run(run=request))
        while True:
            try:
                msg = await reader.read_message()
            except asyncio.IncompleteReadError:
                raise RuntimeError("Sandbox connection closed unexpectedly")

            if isinstance(msg, SandboxMessage.Done):
                break
            elif isinstance(msg, SandboxMessage.Error):
                raise RuntimeError(f"Python agent error: {msg.error}")

            assert isinstance(msg, SandboxMessage.Request)

            try:
                response = await self.callback(msg.task)
            except Exception as e:
                # On error, close the connection so the server can terminate the process
                writer.close()
                raise e

            await writer.write_message(SandboxMessage.Response(result=response))

        writer.close()

    def run_in_process(self, request: WorkerExecutionRequest):
        """
        Runs the two tasks needed to run a sandboxed Python agent:
        - _process_question_context_sync runs the untrusted code in a process under RestrictedPython
        - agent_runner_task calls read_write_queue and starts a "server" listening for messages
            from the sandbox. It then executes the desired function in behalf of the user provided
            code and sends the results back over a Pipe.
        """
        (pipe_worker, pipe_restricted) = Pipe()
        worker = PythonAgentWorker(pipe_worker, debug=self.debug)
        agent_runner_task = asyncio.create_task(self.background_task(pipe_restricted))
        process = Process(
            target=worker._process_question_context_sync,
            args=(
                request.code,
                request.question,
                request.local_vars,
                request.global_vars,
                request.function_names,
            ),
        )
        process.start()

        return agent_runner_task, process

    async def background_task(self, pipe: Connection):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            ThreadPoolExecutor(max_workers=1),
            self.read_write_queue,
            pipe,
            loop,
        )

    def read_write_queue(
        self,
        pipe: Connection,
        loop: asyncio.AbstractEventLoop,
    ):
        """
        This is a sync function that runs on a thread to read from the pipe that
        communicates with the RestrictedPython process. It then calls an async callback
        which can be the actual function in the agent, or a function in SandboxSession
        that sends a socket message to the actual agent.
        """
        try:
            while True:
                item = pipe.recv()
                if item.function == "_error":
                    raise Exception(item.args[0])
                elif item.function == "_close":
                    break
                future = asyncio.run_coroutine_threadsafe(self.callback(item), loop)
                result = future.result()
                pipe.send(result)
        except EOFError:
            pass


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
        self.writer.write(len(msg_bytes).to_bytes(4, "little"))
        self.writer.write(msg_bytes)
        await self.writer.drain()

    async def write_message(self, message: SandboxMessage.AnyMessage):
        await self._write_packet(SandboxMessage.serialize(message))

    def close(self):
        self.writer.close()


class SandboxReader:
    def __init__(self, reader: asyncio.StreamReader):
        self.reader = reader

    async def _read_packet(self) -> dict[str, Any]:
        len_bytes = await self.reader.readexactly(4)
        msg_len = int.from_bytes(len_bytes, "little")
        msg_bytes = await self.reader.readexactly(msg_len)
        msg = json.loads(msg_bytes.decode("utf-8"))

        return msg

    async def read_message(self) -> SandboxMessage.AnyMessage:
        return SandboxMessage.parse(await self._read_packet())


class SandboxSession:
    reader: SandboxReader
    writer: SandboxWriter
    runner: SandboxRunner
    task: asyncio.Task[None]
    process: Process
    waiting_for_agent: asyncio.Task[None] | None

    def __init__(
        self,
        reader: SandboxReader,
        writer: SandboxWriter,
    ):
        self.reader = reader
        self.writer = writer
        self.runner = SandboxRunner(
            pool=None, socket=None, callback=self._callback, debug=False
        )

    async def run(self, request: WorkerExecutionRequest):
        self.task, self.process = self.runner.run_in_process(request)
        try:
            await self._start_timeout()
            await self.task
            await self._stop_timeout()
            await self.writer.write_message(SandboxMessage.Done())
            self.process.join(timeout=1)
            if self.process.exitcode is None:
                logger.warning("Sandbox process did not exit cleanly, killing it")
                self.process.kill()
        except Exception as e:
            logger.error(f"Error in sandbox session: {e}")
            self.process.kill()
            self.process.join()
            self.task.cancel()
            try:
                await self.writer.write_message(SandboxMessage.Error(error=str(e)))
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
            # Irrelevant since the process has just been killed
            return None

        assert isinstance(response, SandboxMessage.Response)
        await self._start_timeout()
        return response.result

    async def _start_timeout(self):
        self.waiting_for_agent = asyncio.create_task(self._timeout())

    async def _stop_timeout(self):
        if self.waiting_for_agent:
            self.waiting_for_agent.cancel()
            self.waiting_for_agent = None

    async def _timeout(self):
        await asyncio.sleep(WORKER_CPU_LIMIT)
        await self.writer.write_message(
            SandboxMessage.Error(error="Python agent timeout")
        )
        self.process.kill()


agents_running = prometheus_client.Gauge(
    "arag_sandbox_agents_running", "Number of Python agents currently running"
)
execution_observer = nucliadb_telemetry.metrics.Observer("arag_sandbox_execution")


async def run_sandbox_server():
    assert settings.sandbox_socket is not None
    if settings.sandbox_verify:
        _verify_connectivity()

    async def handler(rx: asyncio.StreamReader, tx: asyncio.StreamWriter):
        reader = SandboxReader(rx)
        writer = SandboxWriter(tx)

        # Get initial request
        try:
            async with asyncio.timeout(1):
                msg = await reader.read_message()
                assert isinstance(msg, SandboxMessage.Run)
        except TimeoutError:
            logger.warning("Timed out waiting for Run() message. Closing.")
            return

        agents_running.inc()
        observation = execution_observer()
        observation.start()
        try:
            session = SandboxSession(reader, writer)
            await session.run(msg.run)
        except Exception as e:
            observation.set_status("error")
            await writer.write_message(SandboxMessage.Error(error=str(e)))
        finally:
            agents_running.dec()
            observation.end()
            writer.close()

    server = await asyncio.start_unix_server(handler, settings.sandbox_socket)
    async with server:
        await server.serve_forever()
