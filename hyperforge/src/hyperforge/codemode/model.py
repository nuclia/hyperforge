import json
import math
from dataclasses import dataclass, is_dataclass
from typing import Any, Iterator, Sequence, assert_never, cast

from nuclia_models.predict.remi import RemiResponse
from pydantic import BaseModel, Field

from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Context

MAX_PROTOCOL_BYTES = 16 * 1024 * 1024
MAX_PROTOCOL_DEPTH = 256
_JSON_STRING_CHUNK_CHARS = 4096


class _ProtocolValueTooLarge(ValueError):
    pass


@dataclass(frozen=True)
class _PlainModel:
    value: BaseModel


class WorkerExecutionRequest(BaseModel):
    code: str
    question: str = ""
    local_vars: dict[str, Any]
    global_vars: dict[str, Any]
    function_names: dict[str, dict[str, FunctionDefinition]]
    max_runtime_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_memory_bytes: int | None = Field(default=None, gt=0)


class RestrictedPythonTask(BaseModel):
    function: str
    agent: str
    args: tuple[Any, ...]
    keyword_args: dict[str, Any]


class WorkerError(BaseModel):
    error: str


WorkerTypes = (
    str
    | bool
    | int
    | float
    | Context
    | Sequence["WorkerTypes"]
    | dict[str, "WorkerTypes"]
    | None
    | RemiResponse
    | RestrictedPythonTask
    | WorkerError
)
WorkerModels: dict[str, type[BaseModel]] = {
    "Context": Context,
    "RemiResponse": RemiResponse,
    "WorkerError": WorkerError,
}


def deserialize(msg: Any, *, _depth: int = 0) -> WorkerTypes:
    if _depth > MAX_PROTOCOL_DEPTH:
        raise ValueError("Protocol value is too deeply nested")
    if isinstance(msg, dict) and "__model__" in msg:
        model_name = msg["__model__"]
        if not isinstance(model_name, str):
            raise ValueError("Invalid protocol model marker")
        if model_name == "RestrictedPythonTask":
            args = msg.get("args")
            keyword_args = msg.get("keyword_args")
            if not isinstance(args, (list, tuple)) or not isinstance(
                keyword_args, dict
            ):
                raise ValueError("Invalid RestrictedPythonTask payload")
            return RestrictedPythonTask(
                function=msg["function"],
                agent=msg["agent"],
                args=tuple(deserialize(a, _depth=_depth + 1) for a in args),
                keyword_args={
                    k: deserialize(v, _depth=_depth + 1)
                    for k, v in keyword_args.items()
                },
            )
        model = WorkerModels.get(model_name)
        if model is not None:
            return cast(WorkerTypes, model.model_validate(msg))
    if isinstance(msg, list):
        return [deserialize(m, _depth=_depth + 1) for m in msg]
    if isinstance(msg, dict):
        return {k: deserialize(v, _depth=_depth + 1) for k, v in msg.items()}
    return msg


def serialize(message: Any) -> Any:
    if isinstance(message, RestrictedPythonTask):
        return {
            "__model__": "RestrictedPythonTask",
            "function": message.function,
            "agent": message.agent,
            "args": [serialize(a) for a in message.args],
            "keyword_args": {k: serialize(v) for k, v in message.keyword_args.items()},
        }
    if isinstance(message, BaseModel):
        converted = message.model_dump()
        converted["__model__"] = message.__class__.__name__
        return converted
    if isinstance(message, (list, tuple)):
        return [serialize(m) for m in message]
    if isinstance(message, dict):
        return {k: serialize(v) for k, v in message.items()}
    return message


def validate_protocol_value(
    value: Any, label: str, *, max_bytes: int = MAX_PROTOCOL_BYTES
) -> None:
    encode_protocol_value(value, label, max_bytes=max_bytes)


def encode_protocol_value(
    value: Any, label: str, *, max_bytes: int = MAX_PROTOCOL_BYTES
) -> bytes:
    try:
        encoded = bytearray()
        for chunk in _iter_json_bytes(
            value,
            depth=0,
            markers=set(),
            max_bytes=max_bytes,
            include_model_markers=True,
        ):
            if len(encoded) + len(chunk) > max_bytes:
                raise _ProtocolValueTooLarge
            encoded.extend(chunk)
    except _ProtocolValueTooLarge as exc:
        raise ValueError(f"{label} exceeds maximum size") from exc
    except (RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError(f"{label} is not serializable") from exc
    return bytes(encoded)


def _iter_json_bytes(
    value: Any,
    *,
    depth: int,
    markers: set[int],
    max_bytes: int,
    include_model_markers: bool,
) -> Iterator[bytes]:
    if depth > MAX_PROTOCOL_DEPTH:
        raise ValueError("Protocol value is too deeply nested")
    if is_dataclass(value) and not isinstance(value, _PlainModel):
        raise TypeError("Dataclass values are not supported by the sandbox protocol")
    if isinstance(value, _PlainModel):
        model = value.value
        yield from _iter_json_object_bytes(
            _model_items(model, include_marker=False),
            marker=id(model),
            depth=depth,
            markers=markers,
            max_bytes=max_bytes,
            include_model_markers=False,
        )
        return
    if isinstance(value, RestrictedPythonTask):
        yield from _iter_json_object_bytes(
            (
                ("__model__", "RestrictedPythonTask"),
                ("function", value.function),
                ("agent", value.agent),
                ("args", value.args),
                ("keyword_args", value.keyword_args),
            ),
            marker=id(value),
            depth=depth,
            markers=markers,
            max_bytes=max_bytes,
            include_model_markers=True,
        )
        return
    elif isinstance(value, BaseModel):
        yield from _iter_json_object_bytes(
            _model_items(value, include_marker=include_model_markers),
            marker=id(value),
            depth=depth,
            markers=markers,
            max_bytes=max_bytes,
            include_model_markers=False,
        )
        return
    if value is None:
        yield b"null"
        return
    if value is True:
        yield b"true"
        return
    if value is False:
        yield b"false"
        return
    if isinstance(value, str):
        yield from _iter_json_string_bytes(value)
        return
    if isinstance(value, int):
        if int.bit_length(value) > max_bytes * 4:
            raise _ProtocolValueTooLarge
        yield int.__repr__(value).encode("ascii")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Out of range float values are not JSON compliant")
        yield json.dumps(value, allow_nan=False).encode("ascii")
        return
    if isinstance(value, (list, tuple)):
        marker = id(value)
        if marker in markers:
            raise ValueError("Circular reference detected")
        markers.add(marker)
        try:
            yield b"["
            for index, item in enumerate(value):
                if index:
                    yield b","
                yield from _iter_json_bytes(
                    item,
                    depth=depth + 1,
                    markers=markers,
                    max_bytes=max_bytes,
                    include_model_markers=include_model_markers,
                )
            yield b"]"
        finally:
            markers.remove(marker)
        return
    if isinstance(value, dict):
        marker = id(value)
        if marker in markers:
            raise ValueError("Circular reference detected")
        markers.add(marker)
        try:
            yield b"{"
            for index, (key, item) in enumerate(value.items()):
                if index:
                    yield b","
                yield from _iter_json_string_bytes(_json_key(key, max_bytes))
                yield b":"
                yield from _iter_json_bytes(
                    item,
                    depth=depth + 1,
                    markers=markers,
                    max_bytes=max_bytes,
                    include_model_markers=include_model_markers,
                )
            yield b"}"
        finally:
            markers.remove(marker)
        return
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _iter_json_object_bytes(
    items: Iterator[tuple[str, Any]] | Sequence[tuple[str, Any]],
    *,
    marker: int,
    depth: int,
    markers: set[int],
    max_bytes: int,
    include_model_markers: bool,
) -> Iterator[bytes]:
    if marker in markers:
        raise ValueError("Circular reference detected")
    markers.add(marker)
    try:
        yield b"{"
        for index, (key, item) in enumerate(items):
            if index:
                yield b","
            yield from _iter_json_string_bytes(key)
            yield b":"
            yield from _iter_json_bytes(
                item,
                depth=depth + 1,
                markers=markers,
                max_bytes=max_bytes,
                include_model_markers=include_model_markers,
            )
        yield b"}"
    finally:
        markers.remove(marker)


def _model_items(
    value: BaseModel, *, include_marker: bool
) -> Iterator[tuple[str, Any]]:
    decorators = value.__pydantic_decorators__
    if (
        decorators.field_serializers
        or decorators.model_serializers
        or _schema_has_serialization(value.__class__.__pydantic_core_schema__)
    ):
        raise TypeError(
            "Pydantic models with custom serializers are not supported by the "
            "sandbox protocol"
        )
    field_names: set[str] = set()
    for name, field in value.__class__.model_fields.items():
        if field.exclude is True:
            continue
        item = getattr(value, name)
        exclude_if = getattr(field, "exclude_if", None)
        if exclude_if is not None and exclude_if(item):
            continue
        field_names.add(name)
        yield name, item
    for name, item in (value.__pydantic_extra__ or {}).items():
        if name not in field_names:
            yield name, item
    for name in value.__class__.model_computed_fields:
        if name not in field_names:
            yield name, getattr(value, name)
    if include_marker:
        yield "__model__", value.__class__.__name__


def _schema_has_serialization(schema: Any) -> bool:
    pending = [iter((schema,))]
    visited: set[int] = set()
    while pending:
        values = pending[-1]
        try:
            value = next(values)
        except StopIteration:
            pending.pop()
            continue
        if isinstance(value, dict):
            if "serialization" in value:
                return True
            marker = id(value)
            if marker not in visited:
                visited.add(marker)
                pending.append(iter(value.values()))
        elif isinstance(value, (list, tuple)):
            marker = id(value)
            if marker not in visited:
                visited.add(marker)
                pending.append(iter(value))
    return False


def _iter_json_string_bytes(value: str) -> Iterator[bytes]:
    yield b'"'
    for start in range(0, len(value), _JSON_STRING_CHUNK_CHARS):
        encoded = json.dumps(
            value[start : start + _JSON_STRING_CHUNK_CHARS], ensure_ascii=False
        )
        yield encoded[1:-1].encode("utf-8")
    yield b'"'


def _json_key(value: Any, max_bytes: int) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        if int.bit_length(value) > max_bytes * 4:
            raise _ProtocolValueTooLarge
        return int.__repr__(value)
    if isinstance(value, float) and math.isfinite(value):
        return json.dumps(value, allow_nan=False)
    raise TypeError(
        f"keys must be str, int, float, bool or None, not {type(value).__name__}"
    )


def decode_protocol_value(
    encoded: bytes, label: str, *, max_bytes: int = MAX_PROTOCOL_BYTES
) -> WorkerTypes:
    value = decode_json_value(encoded, label, max_bytes=max_bytes)
    try:
        return deserialize(value)
    except (AttributeError, KeyError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} is invalid") from exc


def decode_json_value(
    encoded: bytes, label: str, *, max_bytes: int = MAX_PROTOCOL_BYTES
) -> Any:
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} exceeds maximum size")
    try:
        value = json.loads(
            encoded.decode("utf-8"), parse_constant=_reject_json_constant
        )
        _validate_json_depth(value)
        return value
    except (
        RecursionError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _validate_json_depth(value: Any) -> None:
    pending = [(iter((value,)), 0)]
    while pending:
        values, depth = pending[-1]
        try:
            current = next(values)
        except StopIteration:
            pending.pop()
            continue
        if depth > MAX_PROTOCOL_DEPTH:
            raise ValueError("Protocol value is too deeply nested")
        if isinstance(current, dict):
            pending.append((iter(current.values()), depth + 1))
        elif isinstance(current, list):
            pending.append((iter(current), depth + 1))
        elif isinstance(current, float) and not math.isfinite(current):
            raise ValueError("Protocol value contains a non-finite number")


class SandboxMessage:
    @dataclass
    class Run:
        run: WorkerExecutionRequest
        token: str | None = None

    @dataclass
    class Request:
        task: RestrictedPythonTask

    @dataclass
    class Response:
        result: WorkerTypes

    @dataclass
    class Done:
        pass

    @dataclass
    class Error:
        error: str

    AnyMessage = Run | Request | Response | Done | Error

    @classmethod
    def parse(cls, data: dict[str, Any]) -> AnyMessage:
        match data["_"]:
            case "run":
                return cls.Run(
                    run=WorkerExecutionRequest.model_validate(data["run"]),
                    token=data.get("token"),
                )
            case "request":
                task = deserialize(data["task"])
                if not isinstance(task, RestrictedPythonTask):
                    raise ValueError("Invalid sandbox request task")
                return cls.Request(task=task)
            case "response":
                return cls.Response(result=deserialize(data["result"]))
            case "done":
                return cls.Done()
            case "error":
                return cls.Error(error=data["error"])
            case _:
                raise ValueError("Invalid message type")

    @classmethod
    def serialize(cls, message: AnyMessage) -> dict[str, Any]:
        match message:
            case cls.Run():
                return {
                    "_": "run",
                    "run": message.run.model_dump(),
                    "token": message.token,
                }
            case cls.Request():
                return {"_": "request", "task": serialize(message.task)}
            case cls.Response():
                return {"_": "response", "result": serialize(message.result)}
            case cls.Done():
                return {"_": "done"}
            case cls.Error():
                return {"_": "error", "error": message.error}
            case _:
                assert_never(message)


def encode_sandbox_message(
    message: SandboxMessage.AnyMessage,
    label: str,
    *,
    max_bytes: int = MAX_PROTOCOL_BYTES,
) -> bytes:
    match message:
        case SandboxMessage.Run():
            value = {
                "_": "run",
                "run": _PlainModel(message.run),
                "token": message.token,
            }
        case SandboxMessage.Request():
            value = {"_": "request", "task": message.task}
        case SandboxMessage.Response():
            value = {"_": "response", "result": message.result}
        case SandboxMessage.Done():
            value = {"_": "done"}
        case SandboxMessage.Error():
            value = {"_": "error", "error": message.error}
        case _:
            assert_never(message)
    return encode_protocol_value(value, label, max_bytes=max_bytes)
