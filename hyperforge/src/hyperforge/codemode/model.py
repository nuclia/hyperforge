import json
import math
from dataclasses import dataclass, is_dataclass
from typing import Any, Iterator, Sequence, assert_never, cast

from nuclia_models.common.consumption import Consumption, TokensDetail
from nuclia_models.predict.remi import AnswerRelevance, RemiResponse
from pydantic import BaseModel, Field

from hyperforge.definition import FunctionDefinition
from hyperforge.memory import Context
from hyperforge.models import Chunk, Image, JSONObject, Prompt

MAX_PROTOCOL_BYTES = 16 * 1024 * 1024
MAX_PROTOCOL_DEPTH = 256
_JSON_STRING_CHUNK_CHARS = 4096
_PLAIN_DICT_MODEL = "__hyperforge_plain_dict__"


class _ProtocolValueTooLarge(ValueError):
    pass


class _StrictWorkerModelValue(TypeError):
    pass


class _ReservedWorkerKey(ValueError):
    pass


@dataclass(frozen=True)
class _PlainModel:
    value: BaseModel


@dataclass(frozen=True)
class _MarkedProtocolValue:
    value: Any


@dataclass(frozen=True)
class _RawPlainDict:
    value: dict[Any, Any]


class WorkerExecutionRequest(BaseModel):
    code: str
    question: str = ""
    local_vars: dict[str, Any]
    global_vars: dict[str, Any]
    function_names: dict[str, dict[str, FunctionDefinition]]
    max_runtime_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_memory_bytes: int | None = Field(default=None, gt=0)
    redact_errors: bool = False


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
_MARKED_PROTOCOL_MODELS = frozenset(WorkerModels.values())
_UNMARKED_PROTOCOL_MODELS = frozenset(
    {
        AnswerRelevance,
        Chunk,
        Consumption,
        FunctionDefinition,
        Image,
        JSONObject,
        Prompt,
        TokensDetail,
    }
)


def deserialize(msg: Any, *, _depth: int = 0) -> WorkerTypes:
    if _depth > MAX_PROTOCOL_DEPTH:
        raise ValueError("Protocol value is too deeply nested")
    if isinstance(msg, dict) and "__model__" in msg:
        model_name = msg["__model__"]
        if not isinstance(model_name, str):
            raise ValueError("Invalid protocol model marker")
        if model_name == _PLAIN_DICT_MODEL:
            value = msg.get("value")
            if set(msg) != {"__model__", "value"} or not isinstance(value, dict):
                raise ValueError("Invalid escaped protocol dictionary")
            return {
                key: deserialize(item, _depth=_depth + 1) for key, item in value.items()
            }
        if model_name == "RestrictedPythonTask":
            args = deserialize(msg.get("args"), _depth=_depth + 1)
            keyword_args = deserialize(msg.get("keyword_args"), _depth=_depth + 1)
            if not isinstance(args, (list, tuple)) or not isinstance(
                keyword_args, dict
            ):
                raise ValueError("Invalid RestrictedPythonTask payload")
            return RestrictedPythonTask(
                function=msg["function"],
                agent=msg["agent"],
                args=tuple(args),
                keyword_args=cast(dict[str, Any], keyword_args),
            )
        model = WorkerModels.get(model_name)
        if model is not None:
            payload = {
                key: deserialize(value, _depth=_depth + 1)
                for key, value in msg.items()
                if key != "__model__"
            }
            return cast(WorkerTypes, model.model_validate(payload))
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
            "args": serialize(message.args),
            "keyword_args": serialize(message.keyword_args),
        }
    if isinstance(message, BaseModel):
        if type(message) not in _MARKED_PROTOCOL_MODELS:
            raise TypeError(
                f"Pydantic model {type(message).__name__} is not supported by the "
                "sandbox protocol"
            )
        converted = {
            key: serialize(value) for key, value in message.model_dump().items()
        }
        converted["__model__"] = message.__class__.__name__
        return converted
    if isinstance(message, (list, tuple)):
        return [serialize(m) for m in message]
    if isinstance(message, dict):
        converted = {k: serialize(v) for k, v in message.items()}
        if "__model__" in message:
            return {"__model__": _PLAIN_DICT_MODEL, "value": converted}
        return converted
    return message


def validate_protocol_value(
    value: Any, label: str, *, max_bytes: int = MAX_PROTOCOL_BYTES
) -> None:
    encode_protocol_value(value, label, max_bytes=max_bytes)


def encode_protocol_value(
    value: Any,
    label: str,
    *,
    max_bytes: int = MAX_PROTOCOL_BYTES,
    strict_worker_value: bool = False,
) -> bytes:
    try:
        encoded = bytearray()
        for chunk in _iter_json_bytes(
            value,
            depth=0,
            markers=set(),
            max_bytes=max_bytes,
            include_model_markers=True,
            strict_worker_value=strict_worker_value,
        ):
            if len(encoded) + len(chunk) > max_bytes:
                raise _ProtocolValueTooLarge
            encoded.extend(chunk)
    except _ProtocolValueTooLarge as exc:
        raise ValueError(f"{label} exceeds maximum size") from exc
    except _StrictWorkerModelValue as exc:
        raise TypeError(f"{label} must not contain Pydantic model values") from exc
    except _ReservedWorkerKey as exc:
        raise ValueError(
            f"{label} cannot contain the reserved '__model__' key"
        ) from exc
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
    strict_worker_value: bool,
) -> Iterator[bytes]:
    if depth > MAX_PROTOCOL_DEPTH:
        raise ValueError("Protocol value is too deeply nested")
    if strict_worker_value and isinstance(value, BaseModel):
        raise _StrictWorkerModelValue
    if is_dataclass(value) and not isinstance(
        value, (_PlainModel, _MarkedProtocolValue, _RawPlainDict)
    ):
        raise TypeError("Dataclass values are not supported by the sandbox protocol")
    if isinstance(value, _PlainModel):
        model = value.value
        if type(model) is not WorkerExecutionRequest:
            raise TypeError(
                f"Pydantic model {type(model).__name__} is not supported by the "
                "sandbox protocol"
            )
        yield from _iter_json_object_bytes(
            _worker_request_items(cast(WorkerExecutionRequest, model)),
            marker=id(model),
            depth=depth,
            markers=markers,
            max_bytes=max_bytes,
            include_model_markers=False,
            strict_worker_value=strict_worker_value,
        )
        return
    if isinstance(value, _MarkedProtocolValue):
        yield from _iter_json_bytes(
            value.value,
            depth=depth,
            markers=markers,
            max_bytes=max_bytes,
            include_model_markers=True,
            strict_worker_value=strict_worker_value,
        )
        return
    if isinstance(value, _RawPlainDict):
        yield b"{"
        for index, (key, item) in enumerate(value.value.items()):
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
                strict_worker_value=strict_worker_value,
            )
        yield b"}"
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
            strict_worker_value=strict_worker_value,
        )
        return
    elif isinstance(value, BaseModel):
        supported_models = (
            _MARKED_PROTOCOL_MODELS
            if include_model_markers
            else _UNMARKED_PROTOCOL_MODELS
        )
        if type(value) not in supported_models:
            raise TypeError(
                f"Pydantic model {type(value).__name__} is not supported by the "
                "sandbox protocol"
            )
        yield from _iter_json_object_bytes(
            _model_items(value, include_marker=include_model_markers),
            marker=id(value),
            depth=depth,
            markers=markers,
            max_bytes=max_bytes,
            include_model_markers=False,
            strict_worker_value=strict_worker_value,
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
                    strict_worker_value=strict_worker_value,
                )
            yield b"]"
        finally:
            markers.remove(marker)
        return
    if isinstance(value, dict):
        if strict_worker_value and "__model__" in value:
            raise _ReservedWorkerKey
        if "__model__" in value:
            yield from _iter_json_object_bytes(
                (
                    ("__model__", _PLAIN_DICT_MODEL),
                    ("value", _RawPlainDict(value)),
                ),
                marker=id(value),
                depth=depth,
                markers=markers,
                max_bytes=max_bytes,
                include_model_markers=include_model_markers,
                strict_worker_value=strict_worker_value,
            )
            return
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
                    strict_worker_value=strict_worker_value,
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
    strict_worker_value: bool,
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
                strict_worker_value=strict_worker_value,
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


def _worker_request_items(
    value: WorkerExecutionRequest,
) -> Iterator[tuple[str, Any]]:
    for name, item in _model_items(value, include_marker=False):
        if name in {"local_vars", "global_vars"}:
            item = _MarkedProtocolValue(item)
        yield name, item


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
                run = dict(data["run"])
                for name in ("local_vars", "global_vars"):
                    values = deserialize(run.get(name))
                    if not isinstance(values, dict):
                        raise ValueError(f"Invalid run request {name}")
                    run[name] = values
                return cls.Run(
                    run=WorkerExecutionRequest.model_validate(run),
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
                run = message.run.model_dump()
                run["local_vars"] = serialize(message.run.local_vars)
                run["global_vars"] = serialize(message.run.global_vars)
                return {
                    "_": "run",
                    "run": run,
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
