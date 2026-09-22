import json

import httpx
import pytest
from nuclia.lib.nua_responses import (
    AssistantMessage,
    Author,
    ChatModel,
    MessageToolCall,
    MessageToolFunction,
    Reasoning,
    Tool,
    ToolChoiceAuto,
    ToolChoiceForced,
    ToolChoiceRequired,
    ToolMessage,
    UserPrompt,
)
from nuclia_models.predict.generative_responses import (
    ConsumptionGenerative,
    ReasoningGenerativeResponse,
    TextGenerativeResponse,
    ToolsGenerativeResponse,
)

from hyperforge.engine import get_state
from hyperforge.llm import AsyncLocalOpenAIClient
from hyperforge.manager import Manager
from hyperforge.retrieval.config import RetrievalAgentConfig


def sse_response(*chunks: dict) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + (
        "data: [DONE]\n\n"
    )


@pytest.mark.asyncio
async def test_local_openai_generate_uses_chat_completions() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer secret"
        assert payload["model"] == "configured-model"
        assert payload["stream"] is True
        assert payload["reasoning_effort"] == "high"
        assert payload["messages"] == [
            {"content": "instructions", "role": "system"},
            {"content": "question", "role": "user"},
        ]
        return httpx.Response(
            200,
            text=sse_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning_content": "think ",
                                "content": "hello ",
                            },
                        }
                    ]
                },
                {
                    "choices": [{"index": 0, "delta": {"content": "world"}}],
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    },
                },
            ),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncLocalOpenAIClient(
        "https://openai.example/v1",
        api_key="secret",
        model="configured-model",
        http_client=http_client,
    )

    response = await client.generate(
        ChatModel(
            question="",
            system="instructions",
            user_prompt=UserPrompt(prompt="question"),
            format_prompt=False,
            reasoning=Reasoning(effort="high"),
        )
    )

    assert response.answer == "hello world"
    assert response.reasoning == "think "
    assert response.consumption is not None
    assert response.consumption.normalized_tokens.input == 4
    assert response.consumption.normalized_tokens.output == 2
    await http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_choice", "expected_tool_choice"),
    [
        (ToolChoiceAuto(), "auto"),
        (ToolChoiceRequired(), "required"),
        (ToolChoiceForced(name="lookup"), "required"),
    ],
)
async def test_local_openai_generate_stream_translates_tool_deltas(
    tool_choice, expected_tool_choice
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tool_choice"] == expected_tool_choice
        assert payload["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look something up",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "additionalProperties": False,
                        "required": ["query"],
                    },
                    "strict": True,
                },
            }
        ]
        return httpx.Response(
            200,
            text=sse_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query":',
                                        },
                                    }
                                ]
                            },
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": '"x"}'},
                                    }
                                ]
                            },
                        }
                    ]
                },
            ),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncLocalOpenAIClient(
        "https://openai.example/v1", http_client=http_client
    )
    stream = client.generate_stream(
        ChatModel(
            question="use a tool",
            generative_model="request-model",
            tool_choice=tool_choice,
            tools=[
                Tool(
                    name="lookup",
                    description="Look something up",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ],
        )
    )

    chunks = [chunk async for chunk in stream]

    assert isinstance(chunks[0].chunk, ToolsGenerativeResponse)
    call = chunks[0].chunk.tools["lookup"][0]
    assert call.id == "call-1"
    assert call.function.arguments == {"query": "x"}
    await http_client.aclose()


@pytest.mark.asyncio
async def test_local_openai_converts_tool_history_to_chat_completions() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["messages"][1:] == [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query": "x"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        ]
        assert "tools" not in payload
        return httpx.Response(200, text=sse_response())

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncLocalOpenAIClient(
        "https://openai.example/v1", model="model", http_client=http_client
    )

    await client.generate(
        ChatModel(
            question="",
            chat_history=[
                AssistantMessage(
                    author=Author.NUCLIA,
                    tool_calls=[
                        MessageToolCall(
                            id="call-1",
                            function=MessageToolFunction(
                                name="lookup", arguments={"query": "x"}
                            ),
                        )
                    ],
                ),
                ToolMessage(
                    author=Author.USER,
                    tool_call_id="call-1",
                    content="result",
                ),
            ],
        )
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_local_openai_stream_preserves_standard_workflow_chunks() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=sse_response(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "reasoning_content": "thinking",
                                "content": "answer",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            ),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncLocalOpenAIClient(
        "https://openai.example/v1", model="model", http_client=http_client
    )
    chunks = [
        chunk
        async for chunk in client.generate_stream(
            ChatModel(question="question", user_id="user")
        )
    ]

    assert isinstance(chunks[0].chunk, ReasoningGenerativeResponse)
    assert isinstance(chunks[1].chunk, TextGenerativeResponse)
    assert isinstance(chunks[2].chunk, ConsumptionGenerative)
    await http_client.aclose()


@pytest.mark.asyncio
async def test_manager_execute_works_with_local_openai() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=sse_response(
                {"choices": [{"index": 0, "delta": {"content": "workflow answer"}}]}
            ),
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = AsyncLocalOpenAIClient(
        "https://openai.example/v1", http_client=http_client
    )
    manager = Manager()
    manager.nua = client  # type: ignore[assignment]

    answer, input_tokens, output_tokens, code = await manager.execute(
        prompt="question", user_id="user", model="model"
    )

    assert (answer, input_tokens, output_tokens, code) == (
        "workflow answer",
        0,
        0,
        None,
    )
    await http_client.aclose()


@pytest.mark.asyncio
async def test_get_state_selects_local_openai_for_standard_workflows() -> None:
    state = await get_state(
        agent_id="agent",
        config=RetrievalAgentConfig.model_validate(
            {
                "drivers": [],
                "rules": {"rules": []},
                "memory": {},
                "workflow": {
                    "id": "workflow",
                    "name": "Workflow",
                    "description": None,
                    "parameters": None,
                },
                "preprocess": [],
                "context": [],
                "generation": [],
                "postprocess": [],
            }
        ),
        local_openai="https://openai.example/v1",
        local_openai_model="model",
        external_nua_api_key="secret",
    )

    assert isinstance(state.manager.nua, AsyncLocalOpenAIClient)
    assert state.manager.nua.model == "model"
    assert state.manager.nua.headers["authorization"] == "Bearer secret"
    await state.manager.aclose()
