from collections.abc import AsyncIterator

import pytest

from hyperforge.harness_sdk import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    HarnessMessage,
    HarnessToolCall,
    NucliaChatCompletionsClient,
    NucliaModelClient,
)


class FakeResponse:
    status_code = 200

    async def aiter_lines(self):
        yield 'data: {"id":"chunk","model":"model","choices":[{"index":0,"delta":{"content":"hello"}}]}'
        yield "data: [DONE]"


class ErrorResponse:
    status_code = 200

    async def aiter_lines(self):
        yield 'data: {"type":"status","code":"ERROR","details":"transient generation failure","request_id":"req-1"}'


@pytest.mark.asyncio
async def test_chat_completions_transport_parses_sse() -> None:
    class FakeNua:
        async def chat_completions_stream(self, payload, **kwargs):
            assert payload["stream"] is True
            assert payload["stream_options"] == {"include_usage": True}
            yield {
                "id": "chunk",
                "model": "model",
                "choices": [{"index": 0, "delta": {"content": "hello"}}],
            }

    client = NucliaChatCompletionsClient(FakeNua())

    chunks = [
        chunk
        async for chunk in client.stream(
            ChatCompletionRequest(messages=[{"role": "user", "content": "hi"}])
        )
    ]

    assert chunks[0].choices[0].delta.content == "hello"


@pytest.mark.asyncio
async def test_chat_completions_retries_pre_stream_generation_errors() -> None:
    class FakeNua:
        def __init__(self):
            self.attempts = 0

        async def chat_completions_stream(self, payload, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                yield {
                    "type": "status",
                    "code": "ERROR",
                    "details": "transient",
                    "request_id": "req-1",
                }
            else:
                yield {
                    "id": "chunk",
                    "model": "model",
                    "choices": [{"index": 0, "delta": {"content": "hello"}}],
                }

    nua = FakeNua()
    client = NucliaChatCompletionsClient(nua)

    chunks = [
        chunk
        async for chunk in client.stream(
            ChatCompletionRequest(
                messages=[{"role": "user", "content": "hi"}], model="model"
            )
        )
    ]

    assert nua.attempts == 3
    assert [chunk.choices[0].delta.content for chunk in chunks] == ["hello"]


@pytest.mark.asyncio
async def test_harness_client_keeps_usage_only_chunk() -> None:
    class UsageClient:
        async def stream(self, request):
            yield ChatCompletionChunk.model_validate(
                {
                    "id": "usage",
                    "model": "model",
                    "choices": [],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 5},
                }
            )

    deltas = [
        delta
        async for delta in NucliaModelClient(UsageClient()).stream(
            model="model",
            reasoning_effort=None,
            messages=[HarnessMessage(role="user", content="hi")],
            tools=[],
            execution_context={},
        )
    ]
    assert (deltas[0].input_tokens, deltas[0].output_tokens) == (11, 5)


class FakeCompletionsClient:
    async def stream(
        self, request: ChatCompletionRequest
    ) -> AsyncIterator[ChatCompletionChunk]:
        assert request.messages[0] == {"role": "system", "content": "instructions"}
        assert request.reasoning_effort == "high"
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk",
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "reasoning_content": "thinking",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"query":',
                                    },
                                }
                            ],
                        },
                    }
                ],
            }
        )
        yield ChatCompletionChunk.model_validate(
            {
                "id": "chunk",
                "model": "model",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"x"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                },
            }
        )

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_harness_client_assembles_openai_tool_deltas() -> None:
    client = NucliaModelClient(FakeCompletionsClient())
    deltas = [
        delta
        async for delta in client.stream(
            model="model",
            reasoning_effort="high",
            messages=[HarnessMessage(role="system", content="instructions")],
            tools=[],
            execution_context={"user_id": "user"},
        )
    ]

    assert deltas[0].reasoning == "thinking"
    assert deltas[-1].tool_calls == [
        HarnessToolCall(id="call-1", name="lookup", arguments={"query": "x"})
    ]
    assert deltas[-1].input_tokens == 7
    assert deltas[-1].output_tokens == 3


def test_harness_client_preserves_structured_tool_history() -> None:
    messages = NucliaModelClient._messages(
        [
            HarnessMessage(
                role="assistant",
                content="",
                tool_calls=[
                    HarnessToolCall(
                        id="call-1", name="lookup", arguments={"query": "x"}
                    )
                ],
            ),
            HarnessMessage(
                role="tool", content="result", tool_call_id="call-1", tool_name="lookup"
            ),
        ]
    )

    assert messages[0]["tool_calls"][0]["function"]["name"] == "lookup"
    assert messages[1] == {
        "role": "tool",
        "content": "result",
        "tool_call_id": "call-1",
    }
    NucliaModelClient._validate_tool_history(messages)


def test_harness_client_rejects_unmatched_tool_history() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }
            ],
        },
        {"role": "user", "content": "continue"},
    ]

    with pytest.raises(ValueError, match="missing outputs.*call-1"):
        NucliaModelClient._validate_tool_history(messages)
