import asyncio
import inspect
import json
import keyword
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, NoReturn, Protocol, cast

from pydantic import BaseModel

from hyperforge.codemode import (
    RestrictedPythonTask,
    SandboxMessage,
    SandboxRunner,
    WorkerExecutionRequest,
    encode_protocol_value,
    encode_sandbox_message,
)
from hyperforge.codemode.sandbox import MAX_PACKET_BYTES, settings
from hyperforge.definition import FunctionDefinition

from ..context import format_context
from ..models import HarnessEventType
from ..usage import UsageLimitExceeded
from . import HarnessTool, ToolCallContext, ToolInheritancePolicy, tool

CODEMODE_TOOL_NAME = "codemode"
OUTPUT_FUNCTION_NAME = "output"
CODEMODE_USAGE_GUIDANCE = (
    "Return the final result by calling output(value) exactly once. "
    "Do not use print(); it is unavailable. "
    "Available common built-ins are abs, bool, bytes, chr, complex, divmod, float, "
    "hash, hex, id, int, isinstance, issubclass, len, oct, ord, pow, range, repr, "
    "round, slice, sorted, str, sum, tuple, and zip; helpers such as all, any, min, "
    "and max are unavailable. "
    "For example, assign the result to a variable and finish with output(result)."
)
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
_EVENT_SANITIZE_MAX_NODES = 64
_EVENT_SANITIZE_MAX_ITEMS = 16
_EVENT_SANITIZE_MAX_DEPTH = 6
_EVENT_STRING_CHUNK_CHARS = 4096
_EVENT_CONTEXT_MAX_STRING_BYTES = 1024
_EVENT_CONTEXT_MAX_INTEGER_BYTES = 1024
_EVENT_CONTEXT_MAX_TOTAL_BYTES = 4096
_SOURCE_SIZE_CHUNK_CHARS = 4096
_CAPABILITY_EVENT_FAILURE = "Code Mode capability failed"
_POLICY_CHECK_FAILURE = "Code Mode policy check failed"
type CodeModeResultAdapter[OutputT: BaseModel] = Callable[
    [HarnessTool[Any, OutputT], OutputT], Any | Awaitable[Any]
]
type CodeModeDispatch = Callable[[RestrictedPythonTask], Awaitable[Any]]


class _CapabilityEventFailure(RuntimeError):
    pass


class CodemodeInput(BaseModel):
    code: str
    question: str = ""


class CodemodeOutput(BaseModel):
    value: Any = None


def context_codemode_result_adapter[OutputT: BaseModel](
    capability: HarnessTool[Any, OutputT], output: OutputT
) -> Any:
    """Project a result to the same formatted context shown to the model."""
    return format_context(capability.context(output))


def raw_codemode_result_adapter[OutputT: BaseModel](
    _capability: HarnessTool[Any, OutputT], output: OutputT
) -> Any:
    """Explicitly expose the capability's complete JSON-mode output."""
    return output.model_dump(mode="json")


class CodeModeRunner(Protocol):
    """Execute a request and route worker callbacks through dispatch.

    Injected runners are explicitly trusted and own their callback lifecycle.
    The production default uses the remote sandbox, or an isolated local process
    only when ``remote_required=False``.
    """

    async def run(
        self, request: WorkerExecutionRequest, dispatch: CodeModeDispatch
    ) -> Any: ...


@dataclass(frozen=True)
class CodeModeCapability[OutputT: BaseModel]:
    tool: HarnessTool[Any, OutputT]
    result_adapter: CodeModeResultAdapter[OutputT] = context_codemode_result_adapter


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
        f"{CODEMODE_USAGE_GUIDANCE}"
    ),
)
async def codemode(
    context: ToolCallContext, input_value: CodemodeInput
) -> CodemodeOutput:
    harness = context.harness
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
        output = await tool.execute(
            ToolCallContext(harness=harness, name=tool.name), arguments
        )
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
    capabilities: tuple[CodeModeCapability[Any], ...],
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
    capability_map: dict[str, CodeModeCapability[Any]] = {}
    for capability in capabilities:
        if not isinstance(capability, CodeModeCapability):
            raise TypeError("capabilities must contain CodeModeCapability values")
        capability_name = capability.tool.name
        if (
            not capability_name.isascii()
            or not capability_name.isidentifier()
            or keyword.iskeyword(capability_name)
            or capability_name.startswith("_")
        ):
            raise ValueError(
                f"Code Mode capability name {capability_name!r} must be an ASCII "
                "public Python identifier"
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
    if inheritance == ToolInheritancePolicy.INHERIT:
        non_inherited = sorted(
            capability.tool.name
            for capability in capabilities
            if capability.tool.inheritance == ToolInheritancePolicy.DO_NOT_INHERIT
        )
        if non_inherited:
            raise ValueError(
                "An inherited Code Mode tool cannot include capabilities marked "
                f"DO_NOT_INHERIT: {', '.join(non_inherited)}"
            )

    async def execute(
        context: ToolCallContext, input_value: CodemodeInput
    ) -> CodemodeOutput:
        harness = context.harness
        turn_id = harness.turn_id
        if _utf8_exceeds_limit(input_value.code, limits.max_source_bytes):
            raise ValueError(
                "Code Mode source exceeds maximum size: "
                f"> {limits.max_source_bytes} bytes"
            )
        socket = settings.sandbox_socket if runner is None and remote_required else None
        sandbox_token = (
            settings.sandbox_token if runner is None and remote_required else None
        )
        if runner is None and remote_required:
            if socket is None:
                raise RuntimeError(
                    "Remote Code Mode execution is required but SANDBOX_SOCKET "
                    "is absent"
                )
            if not sandbox_token:
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
            sandbox_token,
            execution_limiter,
            runner,
            remote_required,
            parent_call_id=context.id,
            turn_id=turn_id,
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
    capabilities: tuple[CodeModeCapability[Any], ...],
    limits: CodeModeLimits,
    socket: str | None,
    sandbox_token: str | None,
    execution_limiter: CodeModeExecutionLimiter,
    runner: CodeModeRunner | None = None,
    remote_required: bool = True,
    *,
    parent_call_id: str | None = None,
    turn_id: str | None = None,
) -> CodemodeOutput:
    capability_map = {capability.tool.name: capability for capability in capabilities}
    result = CodemodeOutput()
    invocation_state = _InvocationState()
    output_state = _OutputState(invocation_state)
    nested_calls = 0
    cumulative_result_bytes = 0
    try:
        event_context = _nested_event_context(harness)
    except (TypeError, ValueError) as exc:
        invocation_state.latch(exc)
        event_context = {}

    def event_failure() -> Exception:
        error = _CapabilityEventFailure(_CAPABILITY_EVENT_FAILURE)
        invocation_state.latch(error)
        return error

    async def emit_nested(
        event_type: HarnessEventType, payload: dict[str, Any]
    ) -> None:
        try:
            await harness.emit(
                event_type,
                payload,
                parent_call_id=parent_call_id,
                turn_id=turn_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise event_failure() from None

    async def dispatch(task: RestrictedPythonTask) -> Any:
        nonlocal nested_calls, cumulative_result_bytes
        if task.function == OUTPUT_FUNCTION_NAME:
            result.value = output_state.record(task, limits.max_output_bytes)
            return None

        invocation_state.raise_if_failed()
        capability = capability_map.get(task.function)
        if capability is None:
            invocation_state.fail(
                ValueError(f"Unknown Code Mode capability: {task.function}")
            )
        if nested_calls >= limits.max_nested_calls:
            invocation_state.fail(
                RuntimeError(
                    "Code Mode nested call limit exceeded: "
                    f"{nested_calls + 1} > {limits.max_nested_calls}"
                )
            )
        nested_calls += 1
        harness.usage.tool_calls += 1
        try:
            harness._check_limit("max_tool_calls", harness.usage.tool_calls)
        except UsageLimitExceeded as exc:
            invocation_state.fail(exc)
        except Exception:
            invocation_state.fail(RuntimeError(_POLICY_CHECK_FAILURE))
        call_id = uuid.uuid4().hex
        marker = {
            "codemode": True,
            "nested": True,
            "parent_call_id": parent_call_id,
            **event_context,
        }
        await emit_nested(
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
        started = time.perf_counter()
        try:
            arguments = _scoped_tool_arguments(
                capability.tool, task.args, task.keyword_args
            )
            output = await capability.tool.execute(
                ToolCallContext(
                    harness=harness,
                    name=capability.tool.name,
                    id=call_id,
                    turn_id=turn_id,
                    _emit_failure=event_failure,
                ),
                arguments,
            )
            invocation_state.raise_if_failed()
            projected = capability.result_adapter(capability.tool, output)
            if inspect.isawaitable(projected):
                projected = await projected
            try:
                normalized, result_bytes = _normalize_worker_value(
                    projected,
                    limits.max_result_bytes,
                    f"Code Mode result from {capability.tool.name}",
                )
            except (TypeError, ValueError) as exc:
                invocation_state.latch(exc)
                raise
            try:
                transport_label = (
                    f"Code Mode result transport from {capability.tool.name}"
                )
                if remote_required:
                    encode_sandbox_message(
                        SandboxMessage.Response(result=normalized),
                        transport_label,
                        max_bytes=MAX_PACKET_BYTES,
                    )
                else:
                    encode_protocol_value(
                        normalized,
                        transport_label,
                        max_bytes=MAX_PACKET_BYTES,
                    )
            except (TypeError, ValueError) as exc:
                invocation_state.latch(exc)
                raise
            projected_cumulative_bytes = cumulative_result_bytes + result_bytes
            if projected_cumulative_bytes > limits.max_cumulative_result_bytes:
                error = RuntimeError(
                    "Code Mode cumulative result limit exceeded: "
                    f"{projected_cumulative_bytes} > "
                    f"{limits.max_cumulative_result_bytes} bytes"
                )
                invocation_state.latch(error)
                raise error
            cumulative_result_bytes = projected_cumulative_bytes
            sanitized_result = _sanitize_event_value(normalized)
        except BaseException as exc:
            await emit_nested(
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
            if isinstance(exc, _CapabilityEventFailure):
                raise RuntimeError(_CAPABILITY_EVENT_FAILURE) from None
            raise RuntimeError(
                f"Code Mode capability {capability.tool.name!r} failed "
                f"({type(exc).__name__})"
            ) from None
        else:
            await emit_nested(
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
    try:
        invocation_state.raise_if_failed()
        if runner is None:
            sandbox_runner = (
                SandboxRunner.remote(cast(str, socket), dispatch, token=sandbox_token)
                if remote_required
                else SandboxRunner.isolated_process(dispatch)
            )
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
            redact_errors=True,
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
    invocation_state.raise_if_failed()
    if not output_state.called:
        raise ValueError("Code Mode must call output(value) exactly once")
    return result


class _InvocationState:
    def __init__(self) -> None:
        self.violation: tuple[type[Exception], str] | None = None

    def latch(self, exc: Exception) -> None:
        if self.violation is None:
            self.violation = (type(exc), str(exc))

    def fail(self, exc: Exception) -> NoReturn:
        self.latch(exc)
        if self.violation is None:
            raise AssertionError("Code Mode violation was not latched")
        exception_type, message = self.violation
        raise exception_type(message) from None

    def raise_if_failed(self) -> None:
        if self.violation is not None:
            exception_type, message = self.violation
            raise exception_type(message) from None


class _OutputState:
    """Validate output(value) calls: exactly one call, exactly one value."""

    def __init__(self, invocation_state: _InvocationState) -> None:
        self.invocation_state = invocation_state
        self.attempts = 0
        self.called = False

    def record(self, task: RestrictedPythonTask, max_output_bytes: int) -> Any:
        self.attempts += 1
        self.invocation_state.raise_if_failed()
        if self.attempts > 1:
            error = ValueError("output(value) may only be called once")
            self.invocation_state.latch(error)
            raise error
        try:
            value = _output_value(task)
            normalized, _ = _normalize_worker_value(
                value, max_output_bytes, "Code Mode output"
            )
        except (TypeError, ValueError) as exc:
            self.invocation_state.latch(exc)
            raise
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
    try:
        encoded = encode_protocol_value(
            value,
            label,
            max_bytes=max_bytes,
            strict_worker_value=True,
        )
        return json.loads(encoded), len(encoded)
    except TypeError:
        raise
    except (RecursionError, ValueError) as exc:
        if "exceeds maximum size" in str(exc):
            raise ValueError(str(exc)) from exc
        if "reserved '__model__' key" in str(exc):
            raise ValueError(str(exc)) from exc
        raise TypeError(f"{label} is not a serializable worker value") from exc


def _utf8_exceeds_limit(value: str, max_bytes: int) -> bool:
    observed = 0
    for start in range(0, len(value), _SOURCE_SIZE_CHUNK_CHARS):
        observed += len(value[start : start + _SOURCE_SIZE_CHUNK_CHARS].encode("utf-8"))
        if observed > max_bytes:
            return True
    return False


class _EventSanitizer:
    def __init__(self) -> None:
        self.remaining_nodes = _EVENT_SANITIZE_MAX_NODES

    def sanitize(self, value: Any, *, depth: int = 0) -> Any:
        self.remaining_nodes -= 1
        if isinstance(value, dict):
            return self._sanitize_dict(value, depth)
        if isinstance(value, (list, tuple)):
            return self._sanitize_sequence(value, depth)
        if isinstance(value, str):
            return {"type": "string", "utf8_bytes": _utf8_size(value)}
        if isinstance(value, float) and not math.isfinite(value):
            return "<non-finite number>"
        if isinstance(value, (bool, int, float)) or value is None:
            return value
        return {"type": "other"}

    def _sanitize_dict(self, value: dict[Any, Any], depth: int) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        result: dict[str, Any] = {"type": "object", "entries": entries}
        total = len(value)
        if depth >= _EVENT_SANITIZE_MAX_DEPTH:
            if total:
                result["truncated"] = {
                    "reason": "depth_limit",
                    "omitted_entries": total,
                }
            return result

        reason = "item_limit"
        for key, nested in value.items():
            if len(entries) >= _EVENT_SANITIZE_MAX_ITEMS:
                break
            if self.remaining_nodes <= 0:
                reason = "node_budget"
                break
            key_text = _event_key_text(key)
            sensitive = any(
                part in key_text.casefold() for part in _SENSITIVE_FIELD_PARTS
            )
            if sensitive:
                self.remaining_nodes -= 1
                sanitized = "[REDACTED]"
            else:
                sanitized = self.sanitize(nested, depth=depth + 1)
            entries.append(
                {
                    "key": (
                        {"type": "string", "utf8_bytes": _utf8_size(key_text)}
                        if isinstance(key, str)
                        else {"type": "other"}
                    ),
                    "value": sanitized,
                }
            )
        if len(entries) < total:
            result["truncated"] = {
                "reason": reason,
                "omitted_entries": total - len(entries),
            }
        return result

    def _sanitize_sequence(
        self, value: list[Any] | tuple[Any, ...], depth: int
    ) -> dict[str, Any]:
        items: list[Any] = []
        result: dict[str, Any] = {"type": "array", "items": items}
        total = len(value)
        if depth >= _EVENT_SANITIZE_MAX_DEPTH:
            if total:
                result["truncated"] = {
                    "reason": "depth_limit",
                    "omitted_items": total,
                }
            return result

        reason = "item_limit"
        for nested in value:
            if len(items) >= _EVENT_SANITIZE_MAX_ITEMS:
                break
            if self.remaining_nodes <= 0:
                reason = "node_budget"
                break
            items.append(self.sanitize(nested, depth=depth + 1))
        if len(items) < total:
            result["truncated"] = {
                "reason": reason,
                "omitted_items": total - len(items),
            }
        return result


def _sanitize_event_value(value: Any) -> Any:
    return _EventSanitizer().sanitize(value)


def _event_key_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return str(value)
    return ""


def _utf8_size(value: str) -> int:
    return sum(
        len(value[start : start + _EVENT_STRING_CHUNK_CHARS].encode("utf-8"))
        for start in range(0, len(value), _EVENT_STRING_CHUNK_CHARS)
    )


def _sanitize_task_arguments(
    capability: HarnessTool[Any, Any], task: RestrictedPythonTask
) -> Any:
    try:
        arguments = _scoped_tool_arguments(capability, task.args, task.keyword_args)
    except ValueError:
        return {"invalid": True}
    return _sanitize_event_value(arguments)


def _nested_event_context(harness: Any) -> dict[str, Any]:
    context: dict[str, str | int | float | bool] = {}
    total_value_bytes = 0
    for key in ("actor", "actor_id", "tenant", "tenant_id", "user_id"):
        value = harness.execution_context.get(key)
        if isinstance(value, str):
            if _utf8_exceeds_limit(value, _EVENT_CONTEXT_MAX_STRING_BYTES):
                raise ValueError(
                    f"Code Mode execution context marker {key!r} exceeds maximum size"
                )
            value_bytes = _utf8_size(value)
            context[key] = value
        elif isinstance(value, bool):
            value_bytes = len(str(value).lower())
            context[key] = value
        elif isinstance(value, int):
            if int.bit_length(value) > _EVENT_CONTEXT_MAX_INTEGER_BYTES * 4:
                raise ValueError(
                    f"Code Mode execution context marker {key!r} exceeds maximum size"
                )
            value_bytes = len(int.__repr__(value))
            if value_bytes > _EVENT_CONTEXT_MAX_INTEGER_BYTES:
                raise ValueError(
                    f"Code Mode execution context marker {key!r} exceeds maximum size"
                )
            context[key] = value
        elif isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError(
                    "Code Mode execution context markers must contain finite numbers"
                )
            value_bytes = len(json.dumps(value, allow_nan=False))
            context[key] = value
        else:
            continue
        total_value_bytes += value_bytes
        if total_value_bytes > _EVENT_CONTEXT_MAX_TOTAL_BYTES:
            raise ValueError(
                "Code Mode execution context markers exceed maximum total size"
            )
    return {"execution_context": context} if context else {}


def _scoped_description(
    capabilities: tuple[CodeModeCapability[Any], ...], description: str | None = None
) -> str:
    introduction = description or "Execute restricted Python code using only the scoped capabilities below."
    introduction = f"{introduction} {CODEMODE_USAGE_GUIDANCE}"
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


def _scoped_tool_arguments(
    tool: HarnessTool[Any, Any],
    args: tuple[Any, ...],
    keyword_args: dict[str, Any],
) -> dict[str, Any]:
    if not args:
        return keyword_args
    names = list(tool.parameters.get("properties", {}))
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
