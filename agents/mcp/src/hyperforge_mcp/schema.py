from copy import deepcopy
from typing import Any

JSON_SCHEMA_TYPES = frozenset(
    {"array", "boolean", "integer", "number", "object", "string"}
)
ANNOTATION_KEYWORDS = {
    "default",
    "description",
    "title",
    "example",
}
IGNORED_ANNOTATION_KEYWORDS = {
    "$comment",
    "deprecated",
    "examples",
    "readOnly",
    "writeOnly",
}
SCHEMA_MAP_KEYWORDS = {"properties"}
SCHEMA_VALUE_KEYWORDS = {"additionalProperties", "items"}
SCHEMA_LIST_KEYWORDS = {"anyOf"}
SUPPORTED_KEYWORDS = (
    ANNOTATION_KEYWORDS
    | IGNORED_ANNOTATION_KEYWORDS
    | SCHEMA_MAP_KEYWORDS
    | SCHEMA_VALUE_KEYWORDS
    | SCHEMA_LIST_KEYWORDS
    | {
        "$defs",
        "$id",
        "$ref",
        "$schema",
        "definitions",
        "enum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "format",
        "maxItems",
        "maxLength",
        "maxProperties",
        "maximum",
        "minItems",
        "minLength",
        "minProperties",
        "minimum",
        "pattern",
        "required",
        "type",
    }
)


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
    if normalized.get("type") != "object":
        raise IncompatibleToolSchema(
            ("type",), "tool input schema must have type 'object'"
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
    for keyword in IGNORED_ANNOTATION_KEYWORDS:
        current.pop(keyword, None)
    unsupported = next(
        (keyword for keyword in current if keyword not in SUPPORTED_KEYWORDS), None
    )
    if unsupported is not None:
        raise IncompatibleToolSchema(
            path + (unsupported,), "keyword is not supported by all NUA providers"
        )

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
        annotations = {
            keyword: current.pop(keyword)
            for keyword in tuple(current)
            if keyword in ANNOTATION_KEYWORDS
        }
        current.pop("$defs", None)
        current.pop("definitions", None)
        if current:
            raise IncompatibleToolSchema(
                path,
                "$ref siblings with validation constraints are not supported by all NUA providers",
            )
        current = resolved
        current.update(annotations)

    for keyword in SCHEMA_MAP_KEYWORDS:
        _normalize_schema_map(current, keyword, root, path, resolving)

    _normalize_schema_value(current, "items", root, path, resolving)
    if "additionalProperties" in current and not isinstance(
        current["additionalProperties"], bool
    ):
        _normalize_schema_value(current, "additionalProperties", root, path, resolving)

    for keyword in SCHEMA_LIST_KEYWORDS:
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

    _validate_keyword_values(current, path)

    if "type" in current:
        _validate_type(current["type"], path + ("type",))

    if "type" not in current and "anyOf" not in current:
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
            len(value) == 2
            and "null" in value
            and sum(
                item in JSON_SCHEMA_TYPES for item in value if isinstance(item, str)
            )
            == 1
        )
    else:
        valid = False

    if not valid:
        raise IncompatibleToolSchema(path, "type is not supported by all NUA providers")


def _validate_keyword_values(schema: dict[str, Any], path: tuple[str, ...]) -> None:
    required = schema.get("required")
    if required is not None and (
        not isinstance(required, list)
        or not all(isinstance(name, str) for name in required)
        or len(required) != len(set(required))
    ):
        raise IncompatibleToolSchema(
            path + ("required",), "must be an array of unique strings"
        )

    enum = schema.get("enum")
    if enum is not None and (
        not isinstance(enum, list)
        or not enum
        or not all(isinstance(value, str) for value in enum)
    ):
        raise IncompatibleToolSchema(
            path + ("enum",), "must be a non-empty array of strings"
        )

    for keyword in ("title", "description", "format", "pattern", "$id", "$schema"):
        if keyword in schema and not isinstance(schema[keyword], str):
            raise IncompatibleToolSchema(path + (keyword,), "must be a string")

    for keyword in (
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minProperties",
        "maxProperties",
    ):
        if keyword in schema and (
            not isinstance(schema[keyword], int)
            or isinstance(schema[keyword], bool)
            or schema[keyword] < 0
        ):
            raise IncompatibleToolSchema(
                path + (keyword,), "must be a non-negative integer"
            )

    for keyword in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if keyword in schema and (
            not isinstance(schema[keyword], (int, float))
            or isinstance(schema[keyword], bool)
        ):
            raise IncompatibleToolSchema(path + (keyword,), "must be a number")


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
    if "enum" in schema:
        return "string"
    return None
