import os
import sys
from asyncio import Task, create_task
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Literal, Optional, Self, TextIO

import anyio
import anyio.lowlevel
import mcp.types as types
from anyio.abc import Process
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from anyio.streams.text import TextReceiveStream
from hyperforge.configure import driver
from hyperforge.driver import Driver
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field

from hyperforge_mcp.config_driver import MCPStdioDriverConfig, MCPStdioInnerConfig

# Environment variables to inherit by default
DEFAULT_INHERITED_ENV_VARS = ["HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER"]


class ValidConfig(BaseModel):
    command: str
    args: List[str]
    env: List[str] = Field(default_factory=list)
    envs: Dict[str, str] = Field(default_factory=dict)
    encoding: str = "utf-8"
    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"
    cwd: str | Path | None = None


VALID_MCP = {
    "github": ValidConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env=["GITHUB_PERSONAL_ACCESS_TOKEN"],
    ),
    "puppeteer": ValidConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-puppeteer"],
        envs={
            "PUPPETEER_LAUNCH_OPTIONS": '{ "headless": true, "executablePath": "/usr/bin/google-chrome-stable", "args": [] }',
            "ALLOW_DANGEROUS": "false",
        },
    ),
}


def get_default_environment() -> dict[str, str]:
    """
    Returns a default environment object including only environment variables deemed
    safe to inherit.
    """
    env: dict[str, str] = {}

    for key in DEFAULT_INHERITED_ENV_VARS:
        value = os.environ.get(key)
        if value is None:
            continue

        if value.startswith("()"):
            # Skip functions, which are a security risk
            continue

        env[key] = value

    return env


def _get_executable_command(command: str) -> str:
    """
    Get the correct executable command normalized for the current platform.

    Args:
        command: Base command (e.g., 'uvx', 'npx')

    Returns:
        str: Platform-appropriate command
    """
    return command


async def _create_platform_compatible_process(
    command: str,
    args: list[str],
    env: dict[str, str] | None = None,
    errlog: TextIO = sys.stderr,
    cwd: Path | str | None = None,
):
    """
    Creates a subprocess in a platform-compatible way.
    Returns a process handle.
    """
    process = await anyio.open_process(
        [command, *args], env=env, stderr=errlog, cwd=cwd
    )

    return process


class MCPStdioFullDriverConfig(MCPStdioInnerConfig):
    command: str
    """The executable to run to start the server."""

    args: list[str] = Field(default_factory=list)
    """Command line arguments to pass to the executable."""

    cwd: str | Path | None = None
    """The working directory to use when spawning the process."""

    encoding: str = "utf-8"
    """
    The text encoding used when sending/receiving messages to the server

    defaults to utf-8
    """

    encoding_error_handler: Literal["strict", "ignore", "replace"] = "strict"
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values
    """


@driver(
    id="mcpstdio",
    title="MCP Stdio Source",
    description="Source for interacting with the MCP Stdio API.",
    config_schema=MCPStdioDriverConfig,
)
class MCPStdioDriver(Driver):
    config: MCPStdioFullDriverConfig

    read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
    read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]

    write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
    write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

    reader_task: Optional[Task] = None
    writer_task: Optional[Task] = None

    errlog: TextIO

    @classmethod
    async def init(cls, driver: MCPStdioDriverConfig) -> Self:
        read_stream: MemoryObjectReceiveStream[types.JSONRPCMessage | Exception]
        read_stream_writer: MemoryObjectSendStream[types.JSONRPCMessage | Exception]
        write_stream: MemoryObjectSendStream[types.JSONRPCMessage]
        write_stream_reader: MemoryObjectReceiveStream[types.JSONRPCMessage]

        read_stream_writer, read_stream = anyio.create_memory_object_stream(0)
        write_stream, write_stream_reader = anyio.create_memory_object_stream(0)

        validated_config = VALID_MCP[driver.config.server]

        for env in validated_config.env:
            if driver.config.env is None or env not in driver.config.env:
                raise KeyError(
                    f"No environmental variable on MCP source {driver.config.server}: {env}"
                )

        combined_envs = {}
        if driver.config.env is not None:
            combined_envs = deepcopy(driver.config.env)
        combined_envs.update(validated_config.envs)

        full_config = MCPStdioFullDriverConfig(
            args=validated_config.args,
            command=validated_config.command,
            encoding=validated_config.encoding,
            encoding_error_handler=validated_config.encoding_error_handler,
            cwd=validated_config.cwd,
            server=driver.config.server,
            env=combined_envs,
        )

        obj = cls(
            read_stream=read_stream,
            read_stream_writer=read_stream_writer,
            write_stream=write_stream,
            write_stream_reader=write_stream_reader,
            config=full_config,
            name=driver.name,
            provider=driver.provider,
            errlog=sys.stderr,
        )
        await obj.initialize()
        return obj

    async def stdout_reader_loop(
        self,
        process: Process,
    ):
        assert process.stdout, "Opened process is missing stdout"
        try:
            buffer = ""
            async for chunk in TextReceiveStream(
                process.stdout,
                encoding=self.config.encoding,
                errors=self.config.encoding_error_handler,
            ):
                lines = (buffer + chunk).split("\n")
                buffer = lines.pop()

                for line in lines:
                    try:
                        message = types.JSONRPCMessage.model_validate_json(line)
                    except Exception as exc:
                        await self.read_stream_writer.send(exc)
                        continue

                    await self.read_stream_writer.send(message)

        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()

    async def stdin_writer_loop(self, process: Process):
        assert process.stdin, "Opened process is missing stdin"
        try:
            async with self.write_stream_reader:
                async for message in self.write_stream_reader:
                    json = message.model_dump_json(by_alias=True, exclude_none=True)
                    await process.stdin.send(
                        (json + "\n").encode(
                            encoding=self.config.encoding,
                            errors=self.config.encoding_error_handler,
                        )
                    )
        except anyio.ClosedResourceError:
            await anyio.lowlevel.checkpoint()
        finally:
            await self.write_stream.aclose()
            process.terminate()

    async def initialize(self):
        process = await _create_platform_compatible_process(
            command=_get_executable_command(self.config.command),
            args=self.config.args,
            env=(
                {**get_default_environment(), **self.config.env}
                if self.config.env is not None
                else get_default_environment()
            ),
            errlog=self.errlog,
            cwd=self.config.cwd,
        )
        self.reader_task = await create_task(self.stdout_reader_loop(process))

        self.writer_task = await create_task(self.stdin_writer_loop(process))

    async def finalize(self):
        await self.read_stream_writer.aclose()
        await self.write_stream.aclose()

    def client(self):
        """
        Returns the MCP client context manager.
        """
        # Create a StdioServerParameters object with the current configuration
        # This is used to create the MCP client
        # Cast to StdioServerParameters to ensure type compatibility

        server_params = StdioServerParameters(
            command=self.config.command,
            args=self.config.args,
            env=self.config.env,
            encoding=self.config.encoding,
            encoding_error_handler=self.config.encoding_error_handler,
            cwd=self.config.cwd,
        )
        return stdio_client(server_params)
