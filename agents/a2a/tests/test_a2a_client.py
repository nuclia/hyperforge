import pytest

from hyperforge_a2a.client import (
    RemoteFeedbackRequest,
    build_feedback_request,
    build_send_request,
    collect_text_from_stream_response,
    dict_to_struct,
    extract_feedback_request,
    raise_for_terminal_task_error,
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


def test_tls_client_credentials_require_tls_and_a_complete_key_pair():
    with pytest.raises(ValueError, match="configured together"):
        A2AAgentConfig(
            source="a2a.example.com:443",
            use_tls=True,
            tls_client_certificate_chain_path="client.pem",
        )

    with pytest.raises(ValueError, match="require use_tls"):
        A2AAgentConfig(
            source="a2a.example.com:443",
            tls_ca_certificate_path="ca.pem",
        )


def test_extract_feedback_request_and_build_continuation():
    from a2a.types import a2a_pb2

    response = a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id="task-1",
            context_id="context-1",
            status=a2a_pb2.TaskStatus(
                state=a2a_pb2.TaskState.TASK_STATE_INPUT_REQUIRED,
                message=a2a_pb2.Message(
                    parts=[a2a_pb2.Part(text="Which region should I use?")],
                    metadata=dict_to_struct(
                        {
                            "feedback_id": "feedback-1",
                            "request_id": "request-1",
                            "response_schema": {"type": "string"},
                        }
                    ),
                ),
            ),
        )
    )

    feedback = extract_feedback_request(response)

    assert feedback == RemoteFeedbackRequest(
        task_id="task-1",
        context_id="context-1",
        feedback_id="feedback-1",
        request_id="request-1",
        question="Which region should I use?",
        response_schema={"type": "string"},
    )
    continuation = build_feedback_request("EMEA", feedback)
    assert continuation.message.task_id == "task-1"
    assert continuation.message.context_id == "context-1"
    assert continuation.message.parts[0].text == "EMEA"
    assert continuation.metadata["feedback_id"] == "feedback-1"


def test_failed_remote_task_raises_its_message():
    from a2a.types import a2a_pb2

    response = a2a_pb2.StreamResponse(
        status_update=a2a_pb2.TaskStatusUpdateEvent(
            task_id="task-1",
            context_id="context-1",
            status=a2a_pb2.TaskStatus(
                state=a2a_pb2.TaskState.TASK_STATE_FAILED,
                message=a2a_pb2.Message(parts=[a2a_pb2.Part(text="Remote failure")]),
            ),
        )
    )

    with pytest.raises(ValueError, match="Remote failure"):
        raise_for_terminal_task_error(response)
