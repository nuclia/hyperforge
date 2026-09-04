from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast, get_type_hints

from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate as validate_json_schema
from pydantic import BaseModel, ValidationError

from ..context import make_context
from ..models import HarnessContextReference, HarnessContextType
from ..schema import flatten_json_schema

if TYPE_CHECKING:
    from ..harness import AgentHarness

type ToolHandler[InputT: BaseModel, OutputT: BaseModel] = Callable[
    ["AgentHarness", InputT], Awaitable[OutputT]
]
type ContextFactory[OutputT: BaseModel] = Callable[[OutputT], HarnessContextReference]


class ToolInheritancePolicy(StrEnum):
    INHERIT = "inherit"
    DO_NOT_INHERIT = "do_not_inherit"


@dataclass(frozen=True)
class HarnessTool[InputT: BaseModel, OutputT: BaseModel]:
    name: str
    handler: ToolHandler[InputT, OutputT]
    description: str = ""
    context_type: HarnessContextType = HarnessContextType.TOOL_RESULT
    parameters_schema: dict[str, Any] | None = None
    context_factory: ContextFactory[OutputT] | None = None
    lazy_load: bool = False
    inheritance: ToolInheritancePolicy = ToolInheritancePolicy.INHERIT
    input_model: type[InputT] = field(init=False)
    output_model: type[OutputT] = field(init=False)
    _parameters: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parameters = list(inspect.signature(self.handler).parameters.values())
        if len(parameters) != 2:
            raise TypeError(
                f"Tool handler {self.name!r} must accept harness and input parameters"
            )
        globalns = dict(getattr(self.handler, "__globals__", {}))
        globalns.setdefault("AgentHarness", object)
        hints = get_type_hints(self.handler, globalns=globalns)
        input_model = hints.get(parameters[1].name)
        output_model = hints.get("return")
        if not _is_model_type(input_model):
            raise TypeError(
                f"Tool handler {self.name!r} input must be annotated with a Pydantic model"
            )
        if not _is_model_type(output_model):
            raise TypeError(
                f"Tool handler {self.name!r} return must be annotated with a Pydantic model"
            )
        input_model = cast(type[InputT], input_model)
        output_model = cast(type[OutputT], output_model)
        object.__setattr__(self, "input_model", input_model)
        object.__setattr__(self, "output_model", output_model)
        schema = self.parameters_schema or input_model.model_json_schema()
        object.__setattr__(self, "_parameters", flatten_json_schema(schema))

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    def __call__(
        self, harness: AgentHarness, input_value: InputT
    ) -> Awaitable[OutputT]:
        return self.handler(harness, input_value)

    async def execute(
        self, harness: AgentHarness, arguments: dict[str, Any]
    ) -> OutputT:
        if error := arguments.get("_tool_error"):
            raise ValueError(str(error))
        if self.parameters_schema is not None:
            try:
                validate_json_schema(arguments, self.parameters_schema)
            except JsonSchemaValidationError as exc:
                raise ValueError(
                    f"Invalid {self.name} arguments: {exc.message}"
                ) from exc
        try:
            value = self.input_model.model_validate(arguments)
        except ValidationError as exc:
            raise ValueError(f"Invalid {self.name} arguments: {exc}") from exc
        result = await self.handler(harness, value)
        return self.output_model.model_validate(result)

    def context(self, output: OutputT) -> HarnessContextReference:
        if self.context_factory is not None:
            return self.context_factory(output)
        return make_context(self.context_type, output)


def tool(
    name: str | None = None,
    *,
    description: str = "",
    context_type: HarnessContextType = HarnessContextType.TOOL_RESULT,
    parameters_schema: dict[str, Any] | None = None,
    context_factory: ContextFactory[Any] | None = None,
    lazy_load: bool = False,
    inheritance: ToolInheritancePolicy = ToolInheritancePolicy.INHERIT,
) -> Callable[[ToolHandler[Any, Any]], HarnessTool[Any, Any]]:
    """Create a harness tool from an annotated async handler."""

    def decorate(handler: ToolHandler[Any, Any]) -> HarnessTool[Any, Any]:
        return HarnessTool(
            name=name or cast(Any, handler).__name__,
            handler=handler,
            description=description,
            context_type=context_type,
            parameters_schema=parameters_schema,
            context_factory=context_factory,
            lazy_load=lazy_load,
            inheritance=inheritance,
        )

    return decorate


def _is_model_type(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, BaseModel)


__all__ = [
    "ContextFactory",
    "HarnessTool",
    "ToolHandler",
    "ToolInheritancePolicy",
    "tool",
]
