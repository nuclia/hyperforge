from copy import deepcopy
from typing import Any

JSON_SCHEMA_TYPES = {
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
}


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
    if not schema:
        return {"type": "object", "properties": {}}

    root = deepcopy(schema)
    normalized = _normalize_schema(root, root, (), ())
    dangling_reference = _find_reference(normalized)
    if dangling_reference is not None:
        raise IncompatibleToolSchema(
            dangling_reference, "reference in unsupported schema container"
        )
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
            raise IncompatibleToolSchema(
                path + ("$ref",), "only local references are supported"
            )
        if reference in resolving:
            raise IncompatibleToolSchema(path + ("$ref",), "cyclic local reference")
        target = _resolve_reference(root, reference, path + ("$ref",))
        resolved = _normalize_schema(target, root, path, resolving + (reference,))
        if current:
            siblings = _normalize_schema(current, root, path, resolving + (reference,))
            current = {"allOf": [resolved, siblings]}
        else:
            current = resolved

    for keyword in ("properties", "patternProperties", "dependentSchemas"):
        _normalize_schema_map(current, keyword, root, path, resolving)

    for keyword in (
        "items",
        "contains",
        "not",
        "propertyNames",
        "if",
        "then",
        "else",
        "unevaluatedItems",
    ):
        _normalize_schema_value(current, keyword, root, path, resolving)

    for keyword in ("additionalProperties", "unevaluatedProperties"):
        if keyword in current and not isinstance(current[keyword], bool):
            _normalize_schema_value(current, keyword, root, path, resolving)

    _normalize_schema_list(current, "prefixItems", root, path, resolving)

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

    if "type" in current:
        _validate_type(current["type"], path + ("type",))

    if "type" not in current and not any(
        keyword in current for keyword in ("allOf", "anyOf", "oneOf")
    ):
        inferred_type = _infer_type(current)
        if inferred_type is None:
            raise IncompatibleToolSchema(
                path, "schema must declare or imply a supported type"
            )
        current["type"] = inferred_type

    current.pop("$defs", None)
    current.pop("definitions", None)
    return current


def _validate_type(value: Any, path: tuple[str, ...]) -> None:
    if isinstance(value, str):
        valid = value in JSON_SCHEMA_TYPES
    elif isinstance(value, list):
        valid = (
            bool(value)
            and all(
                isinstance(item, str) and item in JSON_SCHEMA_TYPES for item in value
            )
            and len(value) == len(set(value))
        )
    else:
        valid = False

    if not valid:
        raise IncompatibleToolSchema(path, "invalid JSON Schema type")


def _normalize_schema_value(
    schema: dict[str, Any],
    keyword: str,
    root: dict[str, Any],
    path: tuple[str, ...],
    resolving: tuple[str, ...],
) -> None:
    if keyword in schema:
        schema[keyword] = _normalize_schema(
            schema[keyword], root, path + (keyword,), resolving
        )


def _normalize_schema_map(
    schema: dict[str, Any],
    keyword: str,
    root: dict[str, Any],
    path: tuple[str, ...],
    resolving: tuple[str, ...],
) -> None:
    values = schema.get(keyword)
    if values is None:
        return
    if not isinstance(values, dict):
        raise IncompatibleToolSchema(path + (keyword,), "must be an object")
    schema[keyword] = {
        name: _normalize_schema(value, root, path + (keyword, name), resolving)
        for name, value in values.items()
    }


def _normalize_schema_list(
    schema: dict[str, Any],
    keyword: str,
    root: dict[str, Any],
    path: tuple[str, ...],
    resolving: tuple[str, ...],
) -> None:
    values = schema.get(keyword)
    if values is None:
        return
    if not isinstance(values, list):
        raise IncompatibleToolSchema(path + (keyword,), "must be an array")
    schema[keyword] = [
        _normalize_schema(value, root, path + (keyword, str(index)), resolving)
        for index, value in enumerate(values)
    ]


def _find_reference(value: Any, path: tuple[str, ...] = ()) -> tuple[str, ...] | None:
    if isinstance(value, dict):
        if "$ref" in value:
            return path + ("$ref",)
        for key, child in value.items():
            found = _find_reference(child, path + (key,))
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _find_reference(child, path + (str(index),))
            if found is not None:
                return found
    return None


def _resolve_reference(
    root: dict[str, Any], reference: str, path: tuple[str, ...]
) -> dict[str, Any]:
    target: Any = root
    for encoded_part in reference[2:].split("/"):
        part = encoded_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(target, dict) or part not in target:
            raise IncompatibleToolSchema(
                path, f"unresolved local reference {reference!r}"
            )
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
