from hyperforge.codemode import (
    SandboxMessage,
    WorkerExecutionRequest,
    deserialize,
    serialize,
)


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
