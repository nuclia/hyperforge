from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel

from .models import HarnessContextReference, HarnessContextType

type ContextFormatter = Callable[[BaseModel], str]


@dataclass(frozen=True)
class ContextDefinition:
    schema: type[BaseModel]
    formatter: ContextFormatter | None = None


_DEFINITIONS: dict[HarnessContextType, ContextDefinition] = {}


def register_context(
    context_type: HarnessContextType,
    schema: type[BaseModel],
    formatter: ContextFormatter | None = None,
) -> None:
    _DEFINITIONS[context_type] = ContextDefinition(schema=schema, formatter=formatter)


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
    if reference.type == HarnessContextType.RETRIEVAL:
        values = reference.content.get("items", [reference.content])
        visible = [
            {
                key: value
                for key, value in item.items()
                if key not in {"images"} and value not in (None, "", [], {})
            }
            for item in values
        ]
        return json.dumps(visible, sort_keys=True, default=str, separators=(",", ":"))
    definition = _DEFINITIONS.get(reference.type)
    if definition is None:
        return json.dumps(
            reference.content, sort_keys=True, default=str, separators=(",", ":")
        )
    value = definition.schema.model_validate(reference.content)
    if definition.formatter is not None:
        return definition.formatter(value)
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def context_schema(context_type: HarnessContextType) -> dict[str, Any]:
    return _DEFINITIONS[context_type].schema.model_json_schema()
