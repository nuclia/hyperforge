import asyncio
import inspect
import json
import keyword
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel

from hyperforge.codemode import (
    RestrictedPythonTask,
    SandboxRunner,
    WorkerExecutionRequest,
    encode_protocol_value,
)
from hyperforge.codemode.sandbox import settings
from hyperforge.definition import FunctionDefinition

from ..context import format_context
from ..execution import (
    current_tool_call_id,
    reset_current_tool_call_id,
    set_current_tool_call_id,
)
from ..models import HarnessEventType
from . import HarnessTool, ToolInheritancePolicy, tool

CODEMODE_TOOL_NAME = "codemode"
OUTPUT_FUNCTION_NAME = "output"
DEFAULT_MAX_SOURCE_BYTES = 64 * 1024
DEFAULT_MAX_RESULT_BYTES = 1024 * 1024
DEFAULT_MAX_CUMULATIVE_RESULT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
DEFAULT_MAX_NESTED_CALLS = 20
DEFAULT_MAX_CONCURRENT_EXECUTIONS = 4
RESERVED_CAPABILITY_NAMES = frozenset(
    {
        "Any",
        "Chunk",
        "Context",
        "Dict",
        "List",
        "__builtins__",
        "_getitem_",
        "_getiter_",
        "_inplacevar_",
        "_iter_unpack_sequence_",
        "_unpack_sequence_",
        "agent_id",
        "codemode",
        "dataclass",
        "output",
        "pdb",
        "print",
        "printed",
        "question",
        "save",
    }
)
_SENSITIVE_FIELD_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "passwd",
    "password",
    "private_key",
    "privatekey",
    "secret",
    "token",
)
type CodeModeResultAdapter = Callable[
    [HarnessTool[Any, Any], BaseModel], Any | Awaitable[Any]
]
type CodeModeDispatch = Callable[[RestrictedPythonTask], Awaitable[Any]]


class CodemodeInput(BaseModel):
    code: str
    question: str = ""


class CodemodeOutput(BaseModel):
    value: Any = None


def context_codemode_result_adapter(
    capability: HarnessTool[Any, Any], output: BaseModel
) -> Any:
    """Project a result to the same formatted context shown to the model."""
    return format_context(capability.context(output))


def raw_codemode_result_adapter(
    _capability: HarnessTool[Any, Any], output: BaseModel
) -> Any:
    """Explicitly expose the capability's complete JSON-mode output."""
    return output.model_dump(mode="json")


class CodeModeRunner(Protocol):
    """Executes a Code Mode request, routing worker callbacks through dispatch.

    Injections are intended for deterministic tests; production tools use the
    default remote sandbox (or the isolated local process when
    ``remote_required=False``). An injected runner owns its callback lifecycle:
    pending callbacks it leaves behind are not tracked for slot retention.
    """

    async def run(
        self, request: WorkerExecutionRequest, dispatch: CodeModeDispatch
    ) -> Any: ...


@dataclass(frozen=True)
class CodeModeCapability:
    tool: HarnessTool[Any, Any]
    result_adapter: CodeModeResultAdapter = context_codemode_result_adapter


@dataclass(frozen=True)
class CodeModeLimits:
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES
    max_result_bytes: int = DEFAULT_MAX_RESULT_BYTES
    max_cumulative_result_bytes: int = DEFAULT_MAX_CUMULATIVE_RESULT_BYTES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES
    max_nested_calls: int = DEFAULT_MAX_NESTED_CALLS

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_cumulative_result_bytes < self.max_result_bytes:
            raise ValueError("max_cumulative_result_bytes must be >= max_result_bytes")


class CodeModeExecutionLimiter:
    """Fail-fast admission control shared by one or more Code Mode tools."""

    def __init__(
        self, max_concurrent_executions: int = DEFAULT_MAX_CONCURRENT_EXECUTIONS
    ) -> None:
        if (
            not isinstance(max_concurrent_executions, int)
            or isinstance(max_concurrent_executions, bool)
            or max_concurrent_executions <= 0
        ):
            raise ValueError("max_concurrent_executions must be a positive integer")
        self._max_concurrent_executions = max_concurrent_executions
        self._active = 0
        self._lock = threading.Lock()

    @property
    def max_concurrent_executions(self) -> int:
        return self._max_concurrent_executions

    def acquire(self) -> None:
        with self._lock:
            if self._active >= self.max_concurrent_executions:
                raise RuntimeError(
                    "Code Mode concurrency limit reached: "
                    f"{self._active} >= {self.max_concurrent_executions}"
                )
            self._active += 1

    def release(self) -> None:
        with self._lock:
            self._active -= 1


_process_execution_limiter = CodeModeExecutionLimiter()


@tool(
    name=CODEMODE_TOOL_NAME,
    description=(
        "Execute restricted Python code. Registered tools are available as functions; "
        "call output(value) to return a result."
    ),
)
async def codemode(harness: Any, input_value: CodemodeInput) -> CodemodeOutput:
    tools = {
        tool.name: tool
        for tool in harness.iter_tools()
        if tool.name != CODEMODE_TOOL_NAME
    }
    result = CodemodeOutput()

    async def dispatch(task: RestrictedPythonTask) -> Any:
        if task.function == OUTPUT_FUNCTION_NAME:
            if task.args and task.keyword_args:
                raise ValueError("output accepts either a positional or keyword value")
            if len(task.args) > 1:
                raise ValueError("output accepts one value")
            result.value = task.args[0] if task.args else task.keyword_args.get("value")
            return None

        tool = tools.get(task.function)
        if tool is None:
            raise ValueError(f"Unknown codemode function: {task.function}")
        harness.usage.tool_calls += 1
        harness._check_limit("max_tool_calls", harness.usage.tool_calls)
        arguments = _tool_arguments(tool, task.args, task.keyword_args)
        output = await tool.execute(harness, arguments)
        return output.model_dump(mode="json")

    runner = (
        SandboxRunner.remote(settings.sandbox_socket, dispatch)
        if settings.sandbox_socket is not None
        else SandboxRunner.isolated_process(dispatch)
    )
    definitions = {
        name: FunctionDefinition(
            name=name,
            description=tool.description,
            parameters=tool.parameters.get("properties", {}),
        )
        for name, tool in tools.items()
    }
    definitions[OUTPUT_FUNCTION_NAME] = FunctionDefinition(
        name=OUTPUT_FUNCTION_NAME,
        description="Set the value returned by codemode.",
        parameters={"value": {}},
    )
    await runner.run(
        WorkerExecutionRequest(
            code=input_value.code,
            question=input_value.question,
            local_vars={},
            global_vars={},
            function_names={"harness": definitions},
            max_runtime_seconds=harness.usage_limits.max_codemode_runtime_seconds,
            max_memory_bytes=harness.usage_limits.max_codemode_memory_bytes,
        )
    )
    return result


def create_codemode_tool(
    *,
    capabilities: tuple[CodeModeCapability, ...],
    limits: CodeModeLimits = CodeModeLimits(),
    execution_limiter: CodeModeExecutionLimiter = _process_execution_limiter,
    remote_required: bool = True,
    runner: CodeModeRunner | None = None,
    inheritance: ToolInheritancePolicy = ToolInheritancePolicy.DO_NOT_INHERIT,
    name: str = CODEMODE_TOOL_NAME,
    description: str | None = None,
) -> HarnessTool[CodemodeInput, CodemodeOutput]:
    """Create a Code Mode tool with only the explicitly supplied capabilities.

    ``runner`` defaults to the remote sandbox (or the isolated local process
    when ``remote_required=False``); inject a ``CodeModeRunner`` for
    deterministic tests without any sandbox environment.
    """
    if not isinstance(capabilities, tuple):
        raise TypeError("capabilities must be an immutable tuple")
    capability_map: dict[str, CodeModeCapability] = {}
    for capability in capabilities:
        if not isinstance(capability, CodeModeCapability):
            raise TypeError("capabilities must contain CodeModeCapability values")
        capability_name = capability.tool.name
        if (
            not capability_name.isidentifier()
            or keyword.iskeyword(capability_name)
            or capability_name.startswith("_")
        ):
            raise ValueError(
                f"Code Mode capability name {capability_name!r} is not a valid "
                "Python identifier"
            )
        if capability_name in RESERVED_CAPABILITY_NAMES:
            raise ValueError(
                f"Code Mode capability name {capability_name!r} is reserved"
            )
        if capability_name in capability_map:
            raise ValueError(
                f"Code Mode capability names must be unique: {capability_name}"
            )
        if (
            "agent_id" in capability.tool.input_model.model_fields
            or "agent_id" in capability.tool.parameters.get("properties", {})
        ):
            raise ValueError(
                f"Code Mode capability {capability_name!r} uses the reserved "
                "'agent_id' input field"
            )
        capability_map[capability_name] = capability
    scoped_capabilities = tuple(capabilities)

    async def execute(harness: Any, input_value: CodemodeInput) -> CodemodeOutput:
        source_size = len(input_value.code.encode("utf-8"))
        if source_size > limits.max_source_bytes:
            raise ValueError(
                "Code Mode source exceeds maximum size: "
                f"{source_size} > {limits.max_source_bytes} bytes"
            )
        socket = settings.sandbox_socket
        if runner is None and remote_required:
            if socket is None:
                raise RuntimeError(
                    "Remote Code Mode execution is required but SANDBOX_SOCKET "
                    "is absent"
                )
            if not settings.sandbox_token:
                raise RuntimeError(
                    "Remote Code Mode execution is required but SANDBOX_TOKEN is absent"
                )
        execution_limiter.acquire()
        return await _execute_scoped_codemode(
            harness,
            input_value,
            scoped_capabilities,
            limits,
            socket,
            execution_limiter,
            runner,
        )

    execute.__name__ = name
    tool_description = _scoped_description(scoped_capabilities, description)
    return HarnessTool(
        name=name,
        handler=execute,
        description=tool_description,
        inheritance=inheritance,
    )


async def _execute_scoped_codemode(
    harness: Any,
    input_value: CodemodeInput,
    capabilities: tuple[CodeModeCapability, ...],
    limits: CodeModeLimits,
    socket: str | None,
    execution_limiter: CodeModeExecutionLimiter,
    runner: CodeModeRunner | None = None,
) -> CodemodeOutput:
    capability_map = {capability.tool.name: capability for capability in capabilities}
    result = CodemodeOutput()
    output_state = _OutputState()
    nested_calls = 0
    cumulative_result_bytes = 0
    parent_call_id = current_tool_call_id()

    async def dispatch(task: RestrictedPythonTask) -> Any:
        nonlocal nested_calls, cumulative_result_bytes
        if task.function == OUTPUT_FUNCTION_NAME:
            result.value = output_state.record(task, limits.max_output_bytes)
            return None

        capability = capability_map.get(task.function)
        if capability is None:
            raise ValueError(f"Unknown Code Mode capability: {task.function}")
        nested_calls += 1
        call_id = uuid.uuid4().hex
        event_context = _nested_event_context(harness)
        marker = {
            "codemode": True,
            "nested": True,
            "parent_call_id": parent_call_id,
            **event_context,
        }
        await harness.emit(
            HarnessEventType.TOOL_REQUESTED,
            {
                "call": {
                    "id": call_id,
                    "name": capability.tool.name,
                    "arguments": _sanitize_task_arguments(capability.tool, task),
                },
                **marker,
            },
        )
        token = set_current_tool_call_id(call_id)
        started = time.perf_counter()
        try:
            harness.usage.tool_calls += 1
            harness._check_limit("max_tool_calls", harness.usage.tool_calls)
            if nested_calls > limits.max_nested_calls:
                raise RuntimeError(
                    "Code Mode nested call limit exceeded: "
                    f"{nested_calls} > {limits.max_nested_calls}"
                )
            arguments = _tool_arguments(capability.tool, task.args, task.keyword_args)
            output = await capability.tool.execute(harness, arguments)
            projected = capability.result_adapter(capability.tool, output)
            if inspect.isawaitable(projected):
                projected = await projected
            normalized, result_bytes = _normalize_worker_value(
                projected,
                limits.max_result_bytes,
                f"Code Mode result from {capability.tool.name}",
            )
            cumulative_result_bytes += result_bytes
            if cumulative_result_bytes > limits.max_cumulative_result_bytes:
                raise RuntimeError(
                    "Code Mode cumulative result limit exceeded: "
                    f"{cumulative_result_bytes} > "
                    f"{limits.max_cumulative_result_bytes} bytes"
                )
            sanitized_result = _sanitize_event_value(normalized)
        except BaseException as exc:
            reset_current_tool_call_id(token)
            await harness.emit(
                HarnessEventType.TOOL_FAILED,
                {
                    "call_id": call_id,
                    "tool": capability.tool.name,
                    "result": {"error": type(exc).__name__},
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    **marker,
                },
            )
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise RuntimeError(
                f"Code Mode capability {capability.tool.name!r} failed "
                f"({type(exc).__name__})"
            ) from None
        else:
            reset_current_tool_call_id(token)
            await harness.emit(
                HarnessEventType.TOOL_COMPLETED,
                {
                    "call_id": call_id,
                    "tool": capability.tool.name,
                    "result": sanitized_result,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 3),
                    "result_bytes": result_bytes,
                    **marker,
                },
            )
            return normalized

    sandbox_runner: SandboxRunner | None = None
    if runner is None:
        sandbox_runner = (
            SandboxRunner.remote(socket, dispatch)
            if socket is not None
            else SandboxRunner.isolated_process(dispatch)
        )
    try:
        definitions = {
            capability.tool.name: FunctionDefinition(
                name=capability.tool.name,
                description=capability.tool.description,
                parameters=capability.tool.parameters.get("properties", {}),
            )
            for capability in capabilities
        }
        definitions[OUTPUT_FUNCTION_NAME] = FunctionDefinition(
            name=OUTPUT_FUNCTION_NAME,
            description="Set the value returned by Code Mode.",
            parameters={"value": {}},
        )
        worker_request = WorkerExecutionRequest(
            code=input_value.code,
            question=input_value.question,
            local_vars={},
            global_vars={},
            function_names={"harness": definitions},
            max_runtime_seconds=harness.usage_limits.max_codemode_runtime_seconds,
            max_memory_bytes=harness.usage_limits.max_codemode_memory_bytes,
        )
        if runner is not None:
            await runner.run(worker_request, dispatch)
        elif sandbox_runner is not None:
            await sandbox_runner.run(worker_request)
    finally:
        hold = (
            sandbox_runner.run_when_callbacks_complete
            if sandbox_runner is not None
            else None
        )
        if hold is None or not hold(execution_limiter.release):
            execution_limiter.release()
    if not output_state.called:
        raise ValueError("Code Mode must call output(value) exactly once")
    return result


class _OutputState:
    """Validate output(value) calls: exactly one call, exactly one value."""

    def __init__(self) -> None:
        self.called = False

    def record(self, task: RestrictedPythonTask, max_output_bytes: int) -> Any:
        if self.called:
            raise ValueError("output(value) may only be called once")
        value = _output_value(task)
        normalized, _ = _normalize_worker_value(
            value, max_output_bytes, "Code Mode output"
        )
        self.called = True
        return normalized


def _output_value(task: RestrictedPythonTask) -> Any:
    if task.args and task.keyword_args:
        raise ValueError("output accepts either a positional or keyword value")
    if len(task.args) > 1:
        raise ValueError("output accepts one value")
    if task.keyword_args.keys() - {"value"}:
        raise ValueError("output accepts only the 'value' keyword")
    if not task.args and "value" not in task.keyword_args:
        raise ValueError("output requires a value")
    return task.args[0] if task.args else task.keyword_args["value"]


def _normalize_worker_value(value: Any, max_bytes: int, label: str) -> tuple[Any, int]:
    if isinstance(value, BaseModel):
        raise TypeError(
            f"{label} must be projected to serializable worker values, not a model"
        )
    try:
        _reject_reserved_model_markers(value, label)
        encoded = encode_protocol_value(value, label, max_bytes=max_bytes)
        return json.loads(encoded), len(encoded)
    except (RecursionError, TypeError, ValueError) as exc:
        if "exceeds maximum size" in str(exc):
            raise ValueError(str(exc)) from exc
        raise TypeError(f"{label} is not a serializable worker value") from exc


def _reject_reserved_model_markers(value: Any, label: str) -> None:
    if isinstance(value, dict):
        if "__model__" in value:
            raise ValueError(f"{label} contains the reserved '__model__' key")
        for nested in value.values():
            _reject_reserved_model_markers(nested, label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_reserved_model_markers(nested, label)


def _sanitize_event_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): (
                "[REDACTED]"
                if any(part in str(key).casefold() for part in _SENSITIVE_FIELD_PARTS)
                else _sanitize_event_value(nested)
            )
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_event_value(nested) for nested in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, RecursionError, ValueError):
            return {"type": "string", "utf8_bytes": len(value.encode("utf-8"))}
        if isinstance(parsed, (dict, list)):
            return _sanitize_event_value(parsed)
        return {"type": "string", "utf8_bytes": len(value.encode("utf-8"))}
    if isinstance(value, float) and not math.isfinite(value):
        return "<non-finite number>"
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return f"<{type(value).__name__}>"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _sanitize_task_arguments(
    capability: HarnessTool[Any, Any], task: RestrictedPythonTask
) -> Any:
    try:
        arguments = _tool_arguments(capability, task.args, task.keyword_args)
    except ValueError:
        return {"invalid": True}
    return _sanitize_event_value(arguments)


def _nested_event_context(harness: Any) -> dict[str, Any]:
    context = {}
    for key in ("actor", "actor_id", "tenant", "tenant_id", "user_id"):
        value = harness.execution_context.get(key)
        if isinstance(value, (str, int, float, bool)):
            context[key] = value
    return {"execution_context": context} if context else {}


def _scoped_description(
    capabilities: tuple[CodeModeCapability, ...], description: str | None = None
) -> str:
    introduction = description or (
        "Execute restricted Python code using only the scoped capabilities below; "
        "call output(value) exactly once to return a result."
    )
    if not capabilities:
        return introduction
    definitions = []
    for capability in capabilities:
        schema = json.dumps(
            capability.tool.parameters,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        definitions.append(
            f"{capability.tool.name}: {capability.tool.description or 'No description.'} "
            f"Arguments JSON Schema: {schema}"
        )
    return f"{introduction}\nCapability functions:\n" + "\n".join(definitions)


def _tool_arguments(
    tool: HarnessTool[Any, Any],
    args: tuple[Any, ...],
    keyword_args: dict[str, Any],
) -> dict[str, Any]:
    if not args:
        return keyword_args
    names = list(tool.input_model.model_fields)
    if len(args) > len(names):
        raise ValueError(f"Too many positional arguments for {tool.name}")
    arguments = dict(zip(names, args, strict=False))
    duplicates = arguments.keys() & keyword_args.keys()
    if duplicates:
        duplicate = next(iter(duplicates))
        raise ValueError(f"Multiple values for argument {duplicate!r}")
    arguments.update(keyword_args)
    return arguments


__all__ = [
    "CodeModeCapability",
    "CodeModeDispatch",
    "CodeModeExecutionLimiter",
    "CodeModeLimits",
    "CodeModeResultAdapter",
    "CodeModeRunner",
    "CodemodeInput",
    "CodemodeOutput",
    "RESERVED_CAPABILITY_NAMES",
    "codemode",
    "context_codemode_result_adapter",
    "create_codemode_tool",
    "raw_codemode_result_adapter",
]
