from __future__ import annotations

from copy import deepcopy
from typing import Any


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
