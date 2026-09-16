import json
import tracemalloc
from dataclasses import dataclass
from typing import Annotated

import pytest
from nuclia_models.predict.remi import RemiResponse
from pydantic import (
    BaseModel,
    Field,
    PlainSerializer,
    RootModel,
    ValidationError,
    field_serializer,
)

from hyperforge.codemode import (
    RestrictedPythonTask,
    SandboxMessage,
    WorkerError,
    WorkerExecutionRequest,
    decode_protocol_value,
    deserialize,
    encode_protocol_value,
    encode_sandbox_message,
    serialize,
)
from hyperforge.codemode import model as model_module
from hyperforge.memory import Context
from hyperforge.models import Chunk


def test_nested_json_values_round_trip() -> None:
    value = {"score": 0.75, "matches": [{"id": 1}, {"id": 2}]}

    assert deserialize(serialize(value)) == value


def test_sandbox_run_token_round_trip() -> None:
    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars={},
            global_vars={},
            function_names={},
        ),
        token="secret",
    )

    parsed = SandboxMessage.parse(SandboxMessage.serialize(message))

    assert isinstance(parsed, SandboxMessage.Run)
    assert parsed.token == "secret"


def test_known_protocol_models_keep_exact_wire_encoding() -> None:
    error = WorkerError(error="failed")
    task = RestrictedPythonTask(
        function="lookup", agent="agent", args=(1,), keyword_args={"value": 2}
    )

    assert encode_protocol_value(error, "Worker error") == (
        b'{"error":"failed","__model__":"WorkerError"}'
    )
    assert encode_protocol_value(task, "Worker task") == (
        b'{"__model__":"RestrictedPythonTask","function":"lookup",'
        b'"agent":"agent","args":[1],"keyword_args":{"value":2}}'
    )
    assert (
        decode_protocol_value(
            encode_protocol_value(error, "Worker error"), "Worker error"
        )
        == error
    )
    assert (
        decode_protocol_value(encode_protocol_value(task, "Worker task"), "Worker task")
        == task
    )


def test_context_protocol_model_keeps_existing_wire_encoding() -> None:
    context = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        chunks=[Chunk(chunk_id="chunk", text="content")],
        source="source",
        agent="agent",
    )
    expected = json.dumps(
        serialize(context), ensure_ascii=False, separators=(",", ":")
    ).encode()

    encoded = encode_protocol_value(context, "Context")

    assert encoded == expected
    assert decode_protocol_value(encoded, "Context") == context


def test_protocol_encoder_rejects_arbitrary_aliased_pydantic_model() -> None:
    class AliasedModel(BaseModel):
        value: str = Field(alias="wireValue")

    with pytest.raises(ValueError, match="not serializable"):
        encode_protocol_value(AliasedModel(wireValue="value"), "Aliased model")


def test_protocol_encoder_rejects_arbitrary_root_model() -> None:
    class Values(RootModel[list[int]]):
        pass

    with pytest.raises(ValueError, match="not serializable"):
        encode_protocol_value(Values([1, 2]), "Root model")


def test_strict_worker_encoder_rejects_nested_models_and_reserved_keys() -> None:
    class Value(BaseModel):
        item: int

    with pytest.raises(TypeError, match="must not contain Pydantic model values"):
        encode_protocol_value(
            {"nested": [Value(item=1)]},
            "Worker value",
            strict_worker_value=True,
        )
    with pytest.raises(ValueError, match="reserved '__model__' key"):
        encode_protocol_value(
            {"nested": {"__model__": "Context"}},
            "Worker value",
            strict_worker_value=True,
        )


def test_worker_request_preserves_detailed_errors_by_default() -> None:
    request = WorkerExecutionRequest(
        code="",
        local_vars={},
        global_vars={},
        function_names={},
    )

    assert request.redact_errors is False


def test_bounded_sandbox_encoding_matches_existing_wire_format() -> None:
    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="output(value)",
            local_vars={"value": (1, 2)},
            global_vars={},
            function_names={},
        ),
        token="secret",
    )

    encoded = encode_sandbox_message(message, "Sandbox message")

    expected = json.loads(json.dumps(SandboxMessage.serialize(message)))
    assert json.loads(encoded) == expected


def test_sandbox_run_round_trips_supported_marked_variable_models() -> None:
    context = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        chunks=[Chunk(chunk_id="chunk", text="content")],
        source="source",
        agent="agent",
    )
    remi = RemiResponse(time=0.25)
    error = WorkerError(error="failed")
    task = RestrictedPythonTask(
        function="lookup", agent="agent", args=(), keyword_args={}
    )
    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars={"context": context, "remi": remi, "task": task},
            global_vars={"error": error},
            function_names={},
        )
    )

    encoded = encode_sandbox_message(message, "Sandbox message")
    wire = json.loads(encoded)
    parsed = SandboxMessage.parse(wire)

    assert "__model__" not in wire["run"]
    assert wire["run"]["local_vars"]["context"]["__model__"] == "Context"
    assert wire["run"]["local_vars"]["remi"]["__model__"] == "RemiResponse"
    assert wire["run"]["local_vars"]["task"]["__model__"] == "RestrictedPythonTask"
    assert wire["run"]["global_vars"]["error"]["__model__"] == "WorkerError"
    assert isinstance(parsed, SandboxMessage.Run)
    assert parsed.run.local_vars == {"context": context, "remi": remi, "task": task}
    assert parsed.run.global_vars == {"error": error}


def test_sandbox_run_round_trips_marker_bearing_variable_containers() -> None:
    context = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        chunks=[Chunk(chunk_id="chunk", text="content")],
        source="source",
        agent="agent",
    )
    error = WorkerError(error="actual error")
    local_vars = {
        "__model__": "WorkerError",
        "error": "plain container",
        "actual_error": error,
    }
    global_vars = {
        "__model__": model_module._PLAIN_DICT_MODEL,
        "value": {"actual_context": context},
    }
    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars=local_vars,
            global_vars=global_vars,
            function_names={},
        )
    )

    parsed_streaming = SandboxMessage.parse(
        json.loads(encode_sandbox_message(message, "Sandbox message"))
    )
    parsed_compatibility = SandboxMessage.parse(SandboxMessage.serialize(message))

    assert isinstance(parsed_streaming, SandboxMessage.Run)
    assert isinstance(parsed_streaming.run.local_vars, dict)
    assert isinstance(parsed_streaming.run.global_vars, dict)
    assert parsed_streaming.run.local_vars == local_vars
    assert parsed_streaming.run.global_vars == global_vars
    assert isinstance(parsed_compatibility, SandboxMessage.Run)
    assert parsed_compatibility.run.local_vars == local_vars
    assert parsed_compatibility.run.global_vars == global_vars


@pytest.mark.parametrize("marker_kind", ["model", "malformed", "escape"])
def test_restricted_task_round_trips_marker_bearing_keyword_args(
    marker_kind: str,
) -> None:
    context = Context(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="question",
        chunks=[Chunk(chunk_id="chunk", text="content")],
        source="source",
        agent="agent",
    )
    error = WorkerError(error="actual error")
    if marker_kind == "model":
        keyword_args = {
            "__model__": "WorkerError",
            "error": "plain container",
            "actual": error,
        }
    elif marker_kind == "malformed":
        keyword_args = {"__model__": 42, "actual": error}
    else:
        keyword_args = {
            "__model__": model_module._PLAIN_DICT_MODEL,
            "value": {"actual": error},
        }
    marker_argument = {"__model__": "Context", "value": "plain argument"}
    task = RestrictedPythonTask(
        function="lookup",
        agent="agent",
        args=(marker_argument, context),
        keyword_args=keyword_args,
    )

    parsed_streaming = decode_protocol_value(
        encode_protocol_value(task, "Worker task"), "Worker task"
    )
    parsed_compatibility = deserialize(serialize(task))

    assert isinstance(parsed_streaming, RestrictedPythonTask)
    assert parsed_streaming.args == (marker_argument, context)
    assert parsed_streaming.keyword_args == keyword_args
    assert isinstance(parsed_compatibility, RestrictedPythonTask)
    assert parsed_compatibility.args == (marker_argument, context)
    assert parsed_compatibility.keyword_args == keyword_args


@pytest.mark.parametrize(
    "value",
    [
        {"__model__": "WorkerError", "error": "plain dictionary"},
        {"__model__": 42, "value": "malformed marker"},
        {
            "__model__": model_module._PLAIN_DICT_MODEL,
            "value": {"plain": True},
        },
    ],
)
def test_plain_marker_dictionaries_round_trip_without_retyping(
    value: dict[str, object],
) -> None:
    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars={"value": value},
            global_vars={"nested": [value]},
            function_names={},
        )
    )

    encoded_value = decode_protocol_value(
        encode_protocol_value(value, "Plain marker dictionary"),
        "Plain marker dictionary",
    )
    parsed_streaming = SandboxMessage.parse(
        json.loads(encode_sandbox_message(message, "Sandbox message"))
    )
    parsed_compatibility = SandboxMessage.parse(SandboxMessage.serialize(message))

    assert encoded_value == value
    assert isinstance(encoded_value, dict)
    assert deserialize(serialize(value)) == value
    assert isinstance(parsed_streaming, SandboxMessage.Run)
    assert parsed_streaming.run.local_vars["value"] == value
    assert parsed_streaming.run.global_vars["nested"] == [value]
    assert isinstance(parsed_compatibility, SandboxMessage.Run)
    assert parsed_compatibility.run.local_vars["value"] == value
    assert parsed_compatibility.run.global_vars["nested"] == [value]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_runtime_seconds", float("inf")),
        ("max_runtime_seconds", 0),
        ("max_memory_bytes", 0),
    ],
)
def test_worker_request_rejects_invalid_resource_limits(
    field: str, value: float | int
) -> None:
    with pytest.raises(ValidationError):
        WorkerExecutionRequest(
            code="",
            local_vars={},
            global_vars={},
            function_names={},
            **{field: value},
        )


def test_protocol_encoder_streams_large_strings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_dumps = model_module.json.dumps
    encoded_string_lengths = []

    def recording_dumps(value, *args, **kwargs):
        if isinstance(value, str):
            encoded_string_lengths.append(len(value))
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(model_module.json, "dumps", recording_dumps)

    encoded = encode_protocol_value("x" * 10_000, "Test value", max_bytes=10_002)

    assert len(encoded) == 10_002
    assert max(encoded_string_lengths) <= model_module._JSON_STRING_CHUNK_CHARS


def test_protocol_encoder_stops_at_byte_limit() -> None:
    with pytest.raises(ValueError, match="maximum size"):
        encode_protocol_value("x" * 1_000_000, "Test value", max_bytes=512)


def test_protocol_encoder_uses_base_integer_representation() -> None:
    class MisleadingInt(int):
        def __str__(self) -> str:
            return "not-json"

        def __repr__(self) -> str:
            return "also-not-json"

    value = MisleadingInt(7)

    assert encode_protocol_value(value, "Test value") == b"7"
    assert encode_protocol_value({value: 1}, "Test value") == b'{"7":1}'


def test_protocol_decoder_rejects_deep_json_as_value_error() -> None:
    encoded = b"[" * 1000 + b"0" + b"]" * 1000

    with pytest.raises(ValueError, match="not valid JSON"):
        decode_protocol_value(encoded, "Test value")


def test_protocol_decoder_rejects_malformed_model_marker() -> None:
    encoded = (
        b'{"__model__":"RestrictedPythonTask","function":"test","agent":"agent",'
        b'"args":[],"keyword_args":[]}'
    )

    with pytest.raises(ValueError, match="Test value is invalid"):
        decode_protocol_value(encoded, "Test value")


@pytest.mark.parametrize("encoded", [b"1e9999", b'{"value":1e9999}'])
def test_protocol_decoder_rejects_non_finite_numbers(encoded: bytes) -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        decode_protocol_value(encoded, "Test value")


def test_sandbox_encoder_rejects_custom_pydantic_serializers() -> None:
    class SecretModel(BaseModel):
        secret: str

        @field_serializer("secret")
        def redact_secret(self, _value: str) -> str:
            return "[REDACTED]"

    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars={"value": SecretModel(secret="must-not-leak")},
            global_vars={},
            function_names={},
        )
    )

    with pytest.raises(ValueError, match="not serializable"):
        encode_sandbox_message(message, "Sandbox message")


def test_sandbox_encoder_rejects_schema_level_serializers() -> None:
    class SecretModel(BaseModel):
        secret: Annotated[
            str,
            PlainSerializer(lambda _value: "[REDACTED]", return_type=str),
        ]

    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars={"value": SecretModel(secret="must-not-leak")},
            global_vars={},
            function_names={},
        )
    )

    with pytest.raises(ValueError, match="not serializable"):
        encode_sandbox_message(message, "Sandbox message")


def test_sandbox_encoder_rejects_dataclass_values() -> None:
    @dataclass
    class Value:
        item: int

    message = SandboxMessage.Run(
        run=WorkerExecutionRequest(
            code="",
            local_vars={"value": Value(item=1)},
            global_vars={},
            function_names={},
        )
    )

    with pytest.raises(ValueError, match="not serializable"):
        encode_sandbox_message(message, "Sandbox message")


def test_protocol_depth_validation_uses_bounded_auxiliary_memory() -> None:
    wide_value = [0] * 100_000
    tracemalloc.start()
    try:
        model_module._validate_json_depth(wide_value)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 1_000_000
