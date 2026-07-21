from hyperforge_a2a.client import (
    build_send_request,
    collect_text_from_stream_response,
    dict_to_struct,
)
from hyperforge_a2a.config import A2AAgentConfig


class _Memory:
    def __init__(self, headers):
        self.headers = headers


def _make_agent(**overrides):
    from hyperforge_a2a.agent import A2AClientAgent

    config = A2AAgentConfig(source="localhost:8034", **overrides)
    a = A2AClientAgent.__new__(A2AClientAgent)
    a.config = config
    a.agent_id = "a2a-1"
    return a


def test_build_send_request_sets_text_and_metadata():
    request = build_send_request("hello?", {"account": "acc", "agent_id": "ag"})
    assert request.message.parts[0].text == "hello?"
    assert request.metadata["account"] == "acc"
    assert request.metadata["agent_id"] == "ag"


def test_dict_to_struct_nested():
    struct = dict_to_struct({"headers": {"authorization": "Bearer x"}})
    assert struct["headers"]["authorization"] == "Bearer x"


def test_collect_text_from_artifact_update():
    from a2a.types import a2a_pb2

    response = a2a_pb2.StreamResponse(
        artifact_update=a2a_pb2.TaskArtifactUpdateEvent(
            task_id="t",
            context_id="c",
            artifact=a2a_pb2.Artifact(
                artifact_id="a",
                name="answer",
                parts=[a2a_pb2.Part(text="hello world")],
            ),
        )
    )
    assert collect_text_from_stream_response(response) == ["hello world"]


def test_build_metadata_forwards_valid_headers():
    agent = _make_agent(
        remote_account="acc",
        remote_agent_id="remote",
        remote_workflow_id="wf",
        valid_headers=["authorization"],
        extra_metadata={"custom": "1"},
    )
    memory = _Memory({"authorization": "Bearer token", "x-other": "nope"})
    metadata = agent._build_metadata(memory)
    assert metadata["account"] == "acc"
    assert metadata["agent_id"] == "remote"
    assert metadata["workflow_id"] == "wf"
    assert metadata["headers"] == {"authorization": "Bearer token"}
    assert metadata["custom"] == "1"
