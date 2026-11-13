from __future__ import annotations

import json
from copy import deepcopy
from itertools import islice
from typing import Any

from pydantic import BaseModel
from pydantic.json_schema import GenerateJsonSchema, JsonSchemaValue
from pydantic_core import core_schema

_JSON_SCALAR_TYPES = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "null": "None",
}


class _ToolOutputSchema(GenerateJsonSchema):
    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        # model_dump() without a by_alias override honors each nested model's
        # serialize_by_alias setting, rather than forcing one flag globally.
        previous = self.by_alias
        self.by_alias = bool(
            schema["cls"].model_config.get("serialize_by_alias", False)
        )
        try:
            return super().model_schema(schema)
        finally:
            self.by_alias = previous


def compact_model_output_type(model: type[BaseModel]) -> str:
    return compact_json_type(
        model.model_json_schema(
            mode="serialization", schema_generator=_ToolOutputSchema
        )
    )


def flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline local schema references and remove definition containers."""
    root = deepcopy(schema)

    def resolve(ref: str, stack: tuple[str, ...]) -> dict[str, Any]:
        if not ref.startswith("#/"):
            raise ValueError(
                f"External JSON Schema references are not supported: {ref}"
            )
        if ref in stack:
            raise ValueError(
                f"Recursive JSON Schema reference cannot be flattened: {ref}"
            )
        value: Any = root
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(value, dict) or part not in value:
                raise ValueError(f"JSON Schema reference not found: {ref}")
            value = value[part]
        if not isinstance(value, dict):
            raise ValueError(f"JSON Schema reference must resolve to an object: {ref}")
        return walk(deepcopy(value), (*stack, ref))

    def walk(value: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, list):
            return [walk(item, stack) for item in value]
        if not isinstance(value, dict):
            return value
        current = dict(value)
        ref = current.pop("$ref", None)
        if ref is not None:
            if not isinstance(ref, str):
                raise ValueError("JSON Schema $ref must be a string")
            current = {**resolve(ref, stack), **current}
        current.pop("$defs", None)
        current.pop("definitions", None)
        return {key: walk(item, stack) for key, item in current.items()}

    return walk(root)


def compact_json_type(schema: dict[str, Any]) -> str:
    """Describe a JSON value without schema metadata or unbounded expansion."""
    remaining_nodes = 64

    def render(
        value: dict[str, Any], depth: int = 0, refs: frozenset[str] = frozenset()
    ) -> str:
        nonlocal remaining_nodes
        remaining_nodes -= 1
        if remaining_nodes < 0:
            return "Any"
        if ref := value.get("$ref"):
            if ref in refs or not ref.startswith("#/"):
                return "Any"
            resolved: Any = schema
            for part in ref[2:].split("/"):
                if not isinstance(resolved, dict):
                    return "Any"
                resolved = resolved.get(part.replace("~1", "/").replace("~0", "~"))
            if not isinstance(resolved, dict):
                return "Any"
            return render(resolved, depth, refs | {ref})
        alternatives = value.get("anyOf") or value.get("oneOf")
        if alternatives:
            return " | ".join(
                dict.fromkeys(render(item, depth, refs) for item in alternatives)
            )
        value_type = value.get("type")
        if isinstance(value_type, list):
            return " | ".join(
                dict.fromkeys(
                    render({**value, "type": item}, depth, refs) for item in value_type
                )
            )
        if value_type in _JSON_SCALAR_TYPES:
            return _JSON_SCALAR_TYPES[value_type]
        if value_type == "array":
            if depth >= 3:
                return "list[Any]"
            items = value.get("items", {})
            item_type = (
                render(items, depth + 1, refs) if isinstance(items, dict) else "Any"
            )
            if prefix := value.get("prefixItems"):
                item_type = " | ".join(
                    dict.fromkeys(render(item, depth + 1, refs) for item in prefix)
                )
            return f"list[{item_type}]"
        if value_type == "object" or "properties" in value:
            if depth >= 3:
                return "dict[str, Any]"
            properties = value.get("properties")
            if properties is not None:
                fields = [
                    f"{json.dumps(name)}: {render(field, depth + 1, refs)}"
                    for name, field in islice(properties.items(), 12)
                ]
                if len(properties) > 12 or value.get("additionalProperties"):
                    fields.append("...")
                return "TypedDict[{" + ", ".join(fields) + "}]"
            additional = value.get("additionalProperties")
            item_type = (
                render(additional, depth + 1, refs)
                if isinstance(additional, dict)
                else "Any"
            )
            return f"dict[str, {item_type}]"
        return "Any"

    return render(schema)
