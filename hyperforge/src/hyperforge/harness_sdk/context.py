from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ConfigDict

from .models import HarnessContextReference, HarnessContextType

ContextModelT = TypeVar("ContextModelT", bound=BaseModel)
type ContextFormatter = Callable[[Any], str]


@dataclass(frozen=True)
class ContextDefinition:
    schema: type[BaseModel]
    formatter: ContextFormatter | None = None


_DEFINITIONS: dict[HarnessContextType, ContextDefinition] = {}


class RetrievalContext(BaseModel):
    model_config = ConfigDict(extra="allow")


def register_context(
    context_type: HarnessContextType,
    schema: type[ContextModelT],
) -> Callable[[Callable[[ContextModelT], str]], Callable[[ContextModelT], str]]:
    def decorator(
        formatter: Callable[[ContextModelT], str],
    ) -> Callable[[ContextModelT], str]:
        _DEFINITIONS[context_type] = ContextDefinition(
            schema=schema, formatter=formatter
        )
        return formatter

    return decorator


@register_context(HarnessContextType.RETRIEVAL, RetrievalContext)
def _format_retrieval_context(context: RetrievalContext) -> str:
    content = {
        key: (
            [
                {
                    item_key: item_value
                    for item_key, item_value in item.items()
                    if item_key not in {"images"}
                    and item_value not in (None, "", [], {})
                }
                for item in value
            ]
            if key == "items" and isinstance(value, list)
            else value
        )
        for key, value in context.model_dump(mode="json").items()
        if key not in {"images"} and value not in (None, "", [], {})
    }
    return json.dumps(content, sort_keys=True, default=str, separators=(",", ":"))


def make_context(
    context_type: HarnessContextType, output: BaseModel
) -> HarnessContextReference:
    definition = _DEFINITIONS.get(context_type)
    if definition is None:
        return HarnessContextReference(
            type=context_type, content=output.model_dump(mode="json")
        )
    value = definition.schema.model_validate(output.model_dump(mode="json"))
    return HarnessContextReference(
        type=context_type, content=value.model_dump(mode="json")
    )


def format_context(reference: HarnessContextReference) -> str:
    definition = _DEFINITIONS.get(reference.type)
    content = reference.content
    if definition is not None:
        value = definition.schema.model_validate(content)
        if definition.formatter is not None:
            return definition.formatter(value)
        content = value.model_dump(mode="json")
    return json.dumps(
        content,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def context_schema(context_type: HarnessContextType) -> dict[str, Any]:
    return _DEFINITIONS[context_type].schema.model_json_schema()
