import json
import tracemalloc
from dataclasses import dataclass
from typing import Annotated

import pytest
from pydantic import BaseModel, PlainSerializer, ValidationError, field_serializer

from hyperforge.codemode import (
    SandboxMessage,
    WorkerExecutionRequest,
    decode_protocol_value,
    deserialize,
    encode_protocol_value,
    encode_sandbox_message,
    serialize,
)
from hyperforge.codemode import model as model_module


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
