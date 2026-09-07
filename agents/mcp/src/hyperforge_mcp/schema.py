from copy import deepcopy
from typing import Any


class IncompatibleToolSchema(ValueError):
    def __init__(self, path: tuple[str, ...], reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{self.json_path}: {reason}")

    @property
    def json_path(self) -> str:
        if not self.path:
            return "/"
        return "/" + "/".join(
            part.replace("~", "~0").replace("/", "~1") for part in self.path
        )


def normalize_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return an LLM-compatible copy of an MCP tool input schema."""
    if not isinstance(schema, dict):
        raise IncompatibleToolSchema((), "schema must be an object")

    root = deepcopy(schema)
    normalized = _normalize_schema(root, root, (), ())
    normalized.pop("$defs", None)
    normalized.pop("definitions", None)
    return normalized


def _normalize_schema(
    schema: Any,
    root: dict[str, Any],
    path: tuple[str, ...],
    resolving: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise IncompatibleToolSchema(path, "schema must be an object")

    current = deepcopy(schema)
    reference = current.pop("$ref", None)
    if reference is not None:
        if not isinstance(reference, str) or not reference.startswith("#/"):
            raise IncompatibleToolSchema(path + ("$ref",), "only local references are supported")
        if reference in resolving:
            raise IncompatibleToolSchema(path + ("$ref",), "cyclic local reference")
        target = _resolve_reference(root, reference, path + ("$ref",))
        resolved = _normalize_schema(
            target, root, path, resolving + (reference,)
        )
        resolved.update(current)
        current = resolved

    properties = current.get("properties")
    if properties is not None:
        if not isinstance(properties, dict):
            raise IncompatibleToolSchema(path + ("properties",), "must be an object")
        current["properties"] = {
            name: _normalize_schema(
                child, root, path + ("properties", name), resolving
            )
            for name, child in properties.items()
        }

    if "items" in current:
        current["items"] = _normalize_schema(
            current["items"], root, path + ("items",), resolving
        )

    for keyword in ("allOf", "anyOf", "oneOf"):
        alternatives = current.get(keyword)
        if alternatives is None:
            continue
        if not isinstance(alternatives, list) or not alternatives:
            raise IncompatibleToolSchema(path + (keyword,), "must be a non-empty array")
        current[keyword] = [
            _normalize_schema(
                alternative, root, path + (keyword, str(index)), resolving
            )
            for index, alternative in enumerate(alternatives)
        ]

    if "type" not in current and not any(
        keyword in current for keyword in ("allOf", "anyOf", "oneOf")
    ):
        inferred_type = _infer_type(current)
        if inferred_type is None:
            raise IncompatibleToolSchema(
                path, "schema must declare or imply a supported type"
            )
        current["type"] = inferred_type

    return current


def _resolve_reference(
    root: dict[str, Any], reference: str, path: tuple[str, ...]
) -> dict[str, Any]:
    target: Any = root
    for encoded_part in reference[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or part not in target:
            raise IncompatibleToolSchema(path, f"unresolved local reference {reference!r}")
        target = target[part]
    if not isinstance(target, dict):
        raise IncompatibleToolSchema(path, "local reference must resolve to an object")
    return target


def _infer_type(schema: dict[str, Any]) -> str | None:
    if "properties" in schema or "additionalProperties" in schema:
        return "object"
    if "items" in schema:
        return "array"

    values = schema.get("enum")
    if not isinstance(values, list) or not values:
        values = [schema["const"]] if "const" in schema else []
    inferred = {_json_type(value) for value in values}
    return inferred.pop() if len(inferred) == 1 else None


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"