import inspect
import json
import keyword
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from pydantic import BaseModel

from hyperforge.codemode import (
    RestrictedPythonTask,
    SandboxRunner,
    WorkerExecutionRequest,
)
from hyperforge.codemode.sandbox import settings
from hyperforge.definition import FunctionDefinition

from ..context import format_context
from . import HarnessTool, ToolCallContext, ToolInheritancePolicy, tool

CODEMODE_TOOL_NAME = "codemode"
OUTPUT_FUNCTION_NAME = "output"
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
type CodeModeResultAdapter[OutputT] = Callable[
    [HarnessTool[Any, OutputT], OutputT], Any | Awaitable[Any]
]
type CodeModeDispatch = Callable[[RestrictedPythonTask], Awaitable[Any]]


class CodemodeInput(BaseModel):
    code: str
    question: str = ""


class CodemodeOutput(BaseModel):
    value: Any = None


def context_codemode_result_adapter[OutputT](
    capability: HarnessTool[Any, OutputT], output: OutputT
) -> Any:
    """Project a result to the same formatted context shown to the model."""
    return format_context(capability.context(output))


def raw_codemode_result_adapter[OutputT](
    capability: HarnessTool[Any, OutputT], output: OutputT
) -> Any:
    """Explicitly expose the capability's complete JSON-mode output."""
    return capability.dump_output(output)


class CodeModeRunner(Protocol):
    async def run(
        self, request: WorkerExecutionRequest, dispatch: CodeModeDispatch
    ) -> Any: ...


@dataclass(frozen=True)
class CodeModeCapability[OutputT]:
    tool: HarnessTool[Any, OutputT]
    result_adapter: CodeModeResultAdapter[OutputT] = context_codemode_result_adapter


@tool(
    name=CODEMODE_TOOL_NAME,
    description=(
        "Execute restricted Python code. Registered tools are available as functions; "
        "call output(value) to return a result."
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
        return tool.dump_output(output)

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
    runner: CodeModeRunner | None = None,
    inheritance: ToolInheritancePolicy = ToolInheritancePolicy.DO_NOT_INHERIT,
    name: str = CODEMODE_TOOL_NAME,
    description: str | None = None,
) -> HarnessTool[CodemodeInput, CodemodeOutput]:
    """Create a Code Mode tool from an explicit immutable capability set."""
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
        result = CodemodeOutput()
        output_calls = 0
        output_succeeded = False

        async def dispatch(task: RestrictedPythonTask) -> Any:
            nonlocal output_calls, output_succeeded
            if task.function == OUTPUT_FUNCTION_NAME:
                output_calls += 1
                if output_calls > 1:
                    raise ValueError("output(value) may only be called once")
                if task.args and task.keyword_args:
                    raise ValueError(
                        "output accepts either a positional or keyword value"
                    )
                if len(task.args) > 1:
                    raise ValueError("output accepts one value")
                if not task.args and set(task.keyword_args) != {"value"}:
                    raise ValueError("output requires exactly one 'value' argument")
                value = task.args[0] if task.args else task.keyword_args["value"]
                result.value = _normalize_json_value(
                    value,
                    label="Code Mode output",
                )
                output_succeeded = True
                return None

            capability = capability_map.get(task.function)
            if capability is None:
                raise ValueError(f"Unknown Code Mode capability: {task.function}")
            harness.usage.tool_calls += 1
            harness._check_limit("max_tool_calls", harness.usage.tool_calls)
            arguments = _scoped_tool_arguments(
                capability.tool, task.args, task.keyword_args
            )
            output = await capability.tool.execute(
                ToolCallContext(
                    harness=harness,
                    name=capability.tool.name,
                    id=uuid.uuid4().hex,
                ),
                arguments,
            )
            projected = capability.result_adapter(capability.tool, output)
            if inspect.isawaitable(projected):
                projected = await projected
            return _normalize_json_value(
                projected,
                label="Code Mode result adapter",
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
        request = WorkerExecutionRequest(
            code=input_value.code,
            question=input_value.question,
            local_vars={},
            global_vars={},
            function_names={"harness": definitions},
            max_runtime_seconds=harness.usage_limits.max_codemode_runtime_seconds,
            max_memory_bytes=harness.usage_limits.max_codemode_memory_bytes,
        )
        if runner is not None:
            await runner.run(request, dispatch)
        else:
            sandbox_runner = (
                SandboxRunner.remote(settings.sandbox_socket, dispatch)
                if settings.sandbox_socket is not None
                else SandboxRunner.isolated_process(dispatch)
            )
            await sandbox_runner.run(request)
        if output_calls != 1 or not output_succeeded:
            raise ValueError("Code Mode must call output(value) exactly once")
        return result

    execute.__name__ = name
    return HarnessTool(
        name=name,
        handler=execute,
        description=_scoped_description(capabilities, description),
        inheritance=inheritance,
    )


def _normalize_json_value(value: Any, *, label: str) -> Any:
    _reject_reserved_json_values(value, label=label)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except TypeError as exc:
        raise TypeError(f"{label} must be JSON-serializable") from exc
    except ValueError as exc:
        raise ValueError(f"{label} must be a strict JSON value: {exc}") from exc


def _reject_reserved_json_values(value: Any, *, label: str) -> None:
    if isinstance(value, BaseModel):
        raise TypeError(f"{label} must not contain Pydantic model values")
    if isinstance(value, dict):
        if "__model__" in value:
            raise ValueError(f"{label} cannot contain the reserved '__model__' key")
        for key, nested in value.items():
            _reject_reserved_json_values(key, label=label)
            _reject_reserved_json_values(nested, label=label)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _reject_reserved_json_values(nested, label=label)


def _scoped_description(
    capabilities: tuple[CodeModeCapability[Any], ...], description: str | None
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
            f"{capability.tool.name}: "
            f"{capability.tool.description or 'No description.'} "
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
