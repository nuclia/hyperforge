from typing import Any

from pydantic import BaseModel

from hyperforge.codemode import (
    RestrictedPythonTask,
    SandboxRunner,
    WorkerExecutionRequest,
)
from hyperforge.codemode.sandbox import settings
from hyperforge.definition import FunctionDefinition

from . import HarnessTool, tool

CODEMODE_TOOL_NAME = "codemode"
OUTPUT_FUNCTION_NAME = "output"


class CodemodeInput(BaseModel):
    code: str


class CodemodeOutput(BaseModel):
    value: Any = None


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
            local_vars={},
            global_vars={},
            function_names={"harness": definitions},
            max_runtime_seconds=harness.usage_limits.max_codemode_runtime_seconds,
            max_memory_bytes=harness.usage_limits.max_codemode_memory_bytes,
        )
    )
    return result


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


__all__ = ["CodemodeInput", "CodemodeOutput", "codemode"]
