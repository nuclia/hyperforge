from __future__ import annotations

import inspect
from dataclasses import dataclass
from types import GenericAlias, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

from pydantic import BaseModel, Field, create_model

from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory
from hyperforge.models import Context, MemoryConfig, Rules

from .models import HarnessContextReference, HarnessContextType
from .tools import AgentContext, HarnessTool

if TYPE_CHECKING:
    from hyperforge.agent import Agent as HyperforgeAgent
    from hyperforge.manager import Manager
    from hyperforge.memory.memory import QuestionMemory

    from .harness import AgentHarness


class AgentToolInput(BaseModel):
    prompt: str


class AgentToolOutput(BaseModel):
    value: Any


class PublishedFunctionOutput(BaseModel):
    value: Any


type MemoryFactory = Callable[
    ["AgentHarness", str], "Awaitable[QuestionMemory] | QuestionMemory"
]
type ResultAdapter = Callable[[Any, "QuestionMemory"], "Awaitable[Any] | Any"]


@dataclass(frozen=True)
class HarnessAgentToTool:
    """Adapts an unchanged legacy Hyperforge agent to a harness tool."""

    name: str
    agent: HyperforgeAgent
    manager: Manager
    memory_factory: MemoryFactory
    result_adapter: ResultAdapter
    description: str = ""

    def as_tool(self) -> HarnessTool[AgentToolInput, AgentToolOutput]:
        async def execute(
            context: AgentContext, input_value: AgentToolInput
        ) -> AgentToolOutput:
            harness = context.harness
            memory = self.memory_factory(harness, input_value.prompt)
            if inspect.isawaitable(memory):
                memory = await memory
            memory = cast("QuestionMemory", memory)
            result = await self.agent(memory=memory, manager=self.manager)
            adapted = self.result_adapter(result, memory)
            if inspect.isawaitable(adapted):
                adapted = await adapted
            return AgentToolOutput(value=adapted)

        return HarnessTool(
            name=self.name,
            description=self.description or f"Run the {self.name} Hyperforge agent",
            handler=execute,
            context_type=HarnessContextType.STRUCTURED,
        )


def published_agent_to_tools(
    agent: Any,
    *,
    namespace: str | None = None,
    manager: Manager | None = None,
) -> list[HarnessTool[Any, PublishedFunctionOutput]]:
    published: dict[str, FunctionDefinition] = getattr(
        agent, "__published_functions__", {}
    )
    if not published:
        raise TypeError(
            f"{type(agent).__name__} does not define any published functions"
        )

    return [
        _published_function_tool(
            agent, function_id, definition, namespace=namespace, manager=manager
        )
        for function_id, definition in published.items()
    ]


async def preload_published_agent_to_tools(
    agent: Any,
    *,
    namespace: str | None = None,
    manager: Manager,
) -> list[HarnessTool[Any, PublishedFunctionOutput]]:
    preload = getattr(agent, "preload", None)
    if callable(preload):
        session = BaseSessionMemory.from_config(
            MemoryConfig(), agent_id="harness", workflow_id="harness", rules=Rules()
        )
        session.init("harness")
        memory = session.start_question("preload", headers={})
        result = preload(manager, memory)
        if inspect.isawaitable(result):
            await result
    return published_agent_to_tools(agent, namespace=namespace, manager=manager)


def _published_function_tool(
    agent: Any,
    function_id: str,
    definition: FunctionDefinition,
    *,
    namespace: str | None,
    manager: Manager | None,
) -> HarnessTool[Any, PublishedFunctionOutput]:
    method_name = definition.method or function_id
    method = getattr(agent, method_name, None)
    if method is None or not callable(method):
        raise TypeError(
            f"Published function {function_id!r} is not callable on {type(agent).__name__}"
        )

    input_model = _published_input_model(method, function_id, definition)

    async def execute(
        context: AgentContext, input_value: BaseModel
    ) -> PublishedFunctionOutput:
        harness = context.harness
        compat_manager = manager or _compat_manager(harness)
        memory = harness.execution_context.get("memory")
        if memory is None:
            session = BaseSessionMemory.from_config(
                MemoryConfig(),
                agent_id="harness",
                workflow_id="harness",
                rules=Rules(),
            )
            session.init(harness.conversation_id)
            memory = session.start_question(
                str(input_value.model_dump().get("question", function_id)),
                headers={"conversation-id": harness.conversation_id},
            )
        arguments = input_value.model_dump(exclude_unset=True)
        signature = inspect.signature(method)
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if "memory" in signature.parameters or accepts_kwargs:
            arguments["memory"] = memory
        if "manager" in signature.parameters or accepts_kwargs:
            arguments["manager"] = compat_manager
        result = method(**arguments)
        if inspect.isawaitable(result):
            result = await result
        return PublishedFunctionOutput(value=result)

    execute.__annotations__["input_value"] = input_model
    return HarnessTool(
        name=f"{namespace}__{definition.name}" if namespace else definition.name,
        description=definition.description,
        handler=execute,
        context_type=HarnessContextType.STRUCTURED,
        parameters_schema=_published_schema(method, definition),
        context_factory=_published_context,
        lazy_load=definition.lazy_load,
    )


def _compat_manager(harness: AgentHarness) -> Manager:
    manager = Manager()
    nua = getattr(harness.model_client, "nua", None)
    if nua is None:
        nua = harness.execution_context.get("nua_client")
    if nua is not None:
        manager.nua = nua
    drivers = harness.execution_context.get("drivers")
    if drivers is not None:
        if not isinstance(drivers, dict):
            raise TypeError("execution_context['drivers'] must be a dictionary")
        manager.drivers.update(drivers)
    return manager


def _published_schema(
    method: Callable[..., Any], definition: FunctionDefinition
) -> dict[str, Any]:
    if definition.input_schema is not None:
        return definition.input_schema
    signature = inspect.signature(method)
    properties = {name: dict(schema) for name, schema in definition.parameters.items()}
    required = []
    for name, schema in properties.items():
        parameter = signature.parameters.get(name)
        if parameter is not None and parameter.kind != inspect.Parameter.VAR_KEYWORD:
            if parameter.default is inspect.Parameter.empty:
                required.append(name)
            else:
                schema.setdefault("default", parameter.default)
        elif "default" not in schema:
            schema.setdefault("default", None)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _published_context(output: PublishedFunctionOutput) -> HarnessContextReference:
    value = output.value
    if isinstance(value, Context):
        content = value.model_dump(mode="json")
        context_type = HarnessContextType.RETRIEVAL
    elif isinstance(value, list) and all(isinstance(item, Context) for item in value):
        content = {"items": [item.model_dump(mode="json") for item in value]}
        context_type = HarnessContextType.RETRIEVAL
    elif isinstance(value, BaseModel):
        content = {"value": value.model_dump(mode="json")}
        context_type = HarnessContextType.STRUCTURED
    else:
        content = {"value": value}
        context_type = HarnessContextType.TOOL_RESULT
    return HarnessContextReference(type=context_type, content=content)


def _published_input_model(
    method: Callable[..., Any],
    function_id: str,
    definition: FunctionDefinition,
) -> type[BaseModel]:
    signature = inspect.signature(method)
    schema = _published_schema(method, definition)
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    try:
        hints = get_type_hints(method)
    except (NameError, TypeError):
        hints = {}
    fields: dict[str, Any] = {}
    for name, property_schema in properties.items():
        parameter = signature.parameters.get(name)
        annotation = hints.get(name)
        if annotation is None:
            annotation = _schema_type(property_schema)
        default: Any = ...
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            default = parameter.default
        elif name not in required:
            default = property_schema.get("default", None)
        fields[name] = (
            annotation,
            Field(default=default, description=property_schema.get("description")),
        )
    model_name = f"{function_id.title().replace('_', '')}Input"
    return create_model(model_name, **fields)


def _schema_type(schema: dict[str, Any]) -> Any:
    schema_type = schema.get("type")
    if schema_type == "string":
        return str
    if schema_type == "integer":
        return int
    if schema_type == "number":
        return float
    if schema_type == "boolean":
        return bool
    if schema_type == "array":
        return GenericAlias(list, _schema_type(schema.get("items", {})))
    if schema_type == "object":
        return dict[str, Any]
    return Any


def _allows_none(annotation: Any) -> bool:
    origin = get_origin(annotation)
    return origin in {Union, UnionType} and type(None) in get_args(annotation)
