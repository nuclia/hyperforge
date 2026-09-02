import asyncio
import json
from typing import Any, List, Optional
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from nucliadb_models.resource import KnowledgeBoxObj
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

import hyperforge.engine
from hyperforge.agent import Agent
from hyperforge.api.app import HTTPApplication
from hyperforge.api.models import InteractionOperation, InteractionRequest
from hyperforge.context.agent import ContextAgent, trace_agent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
    Feedback,
)
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.pubsub import UserToAgentInteraction
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.server.session import SessionManager

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["test", "127.0.0.1"]),
    pytest.mark.asyncio,
]


@pytest.fixture
def mock_agent():
    class MockAgent(Agent[ContextAgentConfig], ContextAgent):
        config: ContextAgentConfig = ContextAgentConfig(module="mock")
        agent_id: str = "mock"

        @trace_agent
        async def get_question_context(
            self,
            memory: QuestionMemory,
            manager: Manager,
            question_uuid: Optional[str] = None,
            question: Optional[str] = None,
            flow_id: Optional[str] = None,
            extra_context: Optional[dict[str, Any]] = None,
        ):
            if question and question.startswith("Answer:"):
                response = question.split("Answer:")[1].strip()
            else:
                response = "forlayos"

            await memory.add_step(
                step_module="mock",
                step_title="step",
                step_agent_path="path",
                step_value="q",
                step_reason=None,
                timeit=1.1,
                input_nuclia_tokens=0,
                output_nuclia_tokens=0,
                error=None,
            )
            await memory.add_answer(response, "mock", "path")

    retrieval_agent = RetrievalAgent(
        context=[MockAgent(config=ContextAgentConfig(module="mock"))]
    )
    state = hyperforge.engine.State(manager=None, agent=retrieval_agent)
    with (
        patch("hyperforge.engine.get_state", return_value=state),
        patch("hyperforge.server.session.get_state", return_value=state),
    ):
        yield


@pytest.fixture
def slow_agent():
    class MockAgent(Agent[ContextAgentConfig], ContextAgent):
        config: ContextAgentConfig = ContextAgentConfig(module="mock")
        agent_id: str = "mock"

        @trace_agent
        async def get_question_context(
            self,
            memory: QuestionMemory,
            manager: Manager,
            question_uuid: Optional[str] = None,
            question: Optional[str] = None,
            flow_id: Optional[str] = None,
            extra_context: Optional[dict[str, Any]] = None,
        ):
            await asyncio.sleep(2)
            await memory.add_step(
                step_module="mock",
                step_title="step",
                step_agent_path="path",
                step_value="q",
                step_reason=None,
                timeit=1.1,
                input_nuclia_tokens=0,
                output_nuclia_tokens=0,
                error=None,
            )
            await memory.add_answer("forlayos", "mock", "path")

    retrieval_agent = RetrievalAgent(
        context=[MockAgent(config=ContextAgentConfig(module="mock"))]
    )
    state = hyperforge.engine.State(manager=None, agent=retrieval_agent)
    with (
        patch("hyperforge.engine.get_state", return_value=state),
        patch("hyperforge.server.session.get_state", return_value=state),
    ):
        yield


@pytest.fixture
def failing_agent():
    class MockAgent(Agent[ContextAgentConfig], ContextAgent):
        config: ContextAgentConfig = ContextAgentConfig(module="mock")
        agent_id: str = "mock"

        @trace_agent
        async def get_question_context(
            self,
            memory: QuestionMemory,
            manager: Manager,
            question_uuid: Optional[str] = None,
            question: Optional[str] = None,
            flow_id: Optional[str] = None,
            extra_context: Optional[dict[str, Any]] = None,
        ):
            raise Exception("Agent failure")

    retrieval_agent = RetrievalAgent(
        context=[MockAgent(config=ContextAgentConfig(module="mock"))]
    )
    state = hyperforge.engine.State(manager=None, agent=retrieval_agent)
    with (
        patch("hyperforge.engine.get_state", return_value=state),
        patch("hyperforge.server.session.get_state", return_value=state),
    ):
        yield


@pytest.fixture
def feedback_agent():
    class MockAgent(Agent[ContextAgentConfig], ContextAgent):
        config: ContextAgentConfig = ContextAgentConfig(module="mock")
        agent_id: str = "mock"

        @trace_agent
        async def get_question_context(
            self,
            memory: QuestionMemory,
            manager: Manager,
            question_uuid: Optional[str] = None,
            question: Optional[str] = None,
            flow_id: Optional[str] = None,
            extra_context: Optional[dict[str, Any]] = None,
        ):
            await memory.add_step(
                step_module="mock",
                step_title="step",
                step_agent_path="path",
                step_value="q",
                step_reason=None,
                timeit=1.1,
                input_nuclia_tokens=0,
                output_nuclia_tokens=0,
                error=None,
            )
            answer = await memory.send_feedback(
                Feedback(
                    request_id="abcd",
                    question="What is your name?",
                    module="mock",
                    agent_id="mock",
                    data=None,
                    response_schema={},
                )
            )
            if answer is not None:
                await memory.add_answer(
                    f"Your name is {answer.response}", "mock", "path"
                )

    retrieval_agent = RetrievalAgent(
        context=[MockAgent(config=ContextAgentConfig(module="mock"))]
    )
    state = hyperforge.engine.State(manager=None, agent=retrieval_agent)
    with (
        patch("hyperforge.engine.get_state", return_value=state),
        patch("hyperforge.server.session.get_state", return_value=state),
    ):
        yield


async def test_arag_websocket_interaction(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_api_session: str,
    arag_server: SessionManager,
    mock_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}/ws",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))
        print("Sent:", initial_message)

        started = False
        async for message in websocket:
            print("Received :", message)
            response = AragAnswer.model_validate_json(message)
            if started is False:
                assert response.operation == AnswerOperation.START
                started = True
            else:
                if response.operation == AnswerOperation.ANSWER:
                    if response.step:
                        print(json.dumps(response.step.model_dump(), indent=4))
                    elif response.possible_answer:
                        print(
                            json.dumps(response.possible_answer.model_dump(), indent=4)
                        )
                    elif response.context:
                        print(json.dumps(response.context.model_dump(), indent=4))
                    elif response.generated_text:
                        print(f"GENERATED_TEXT: {response.generated_text}")
                        assert "forlayos" in response.generated_text
                    elif response.answer:
                        print(f"ANSWER: {response.answer}")
                elif response.operation == AnswerOperation.DONE:
                    print("Interaction done")
                    break
                elif response.operation == AnswerOperation.ERROR:
                    assert False, (
                        f"Interaction error: {response.exception.detail if response.exception else ''}"
                    )
                else:
                    print(
                        "No feedback, step, possible_answer, context or generated_text in response"
                    )


async def test_arag_websocket_interaction_fails(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_api_session: str,
    arag_server: SessionManager,
    failing_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}/ws",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))
        print("Sent:", initial_message)

        started = False
        async for message in websocket:
            print("Received :", message)
            response = AragAnswer.model_validate_json(message)
            if started is False:
                assert response.operation == AnswerOperation.START
                started = True
            if response.operation == AnswerOperation.ERROR:
                assert response.exception
                assert "Agent failure" in response.exception.detail
                break


def assert_basic_http_interaction_stream(result: List[AragAnswer]) -> None:
    assert len(result) == 5, result
    assert result[0].operation == AnswerOperation.START
    assert all(message.operation == AnswerOperation.ANSWER for message in result[1:-1])
    assert result[-1].operation == AnswerOperation.DONE
    assert all(message.exception is None for message in result)
    assert any(message.step is not None for message in result)
    assert any(
        message.possible_answer is not None
        and message.possible_answer.answer == "forlayos"
        for message in result
    )
    assert result[-2].answer
    assert "forlayos" in result[-2].answer


async def test_arag_interaction_from_cursor(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    arag_api_session: str,
    arag_server: SessionManager,
    mock_agent,
):
    result: List[AragAnswer] = []
    async with arag_api.stream(
        "POST",
        f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}?from_cursor=1",
        json=InteractionRequest(
            question="What certifications does Nuclia have?",
            operation=InteractionOperation.QUESTION,
        ).model_dump(),
        timeout=200,
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SESSIONMEMBER",
        },
    ) as response:
        async for json_body in response.aiter_lines():
            result.append(AragAnswer.model_validate_json(json_body))

    assert_basic_http_interaction_stream(result)


async def test_arag_interaction_durable(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    arag_api_session: str,
    arag_server: SessionManager,
    mock_agent,
):
    result: List[AragAnswer] = []
    async with arag_api.stream(
        "POST",
        f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}",
        json=InteractionRequest(
            question="What certifications does Nuclia have?",
            operation=InteractionOperation.QUESTION,
        ).model_dump(),
        timeout=200,
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SESSIONMEMBER",
        },
    ) as response:
        async for json_body in response.aiter_lines():
            result.append(AragAnswer.model_validate_json(json_body))
    assert_basic_http_interaction_stream(result)

    result = []
    async with arag_api.stream(
        "POST",
        f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}",
        json=InteractionRequest(
            question="What certifications does Nuclia have?",
            operation=InteractionOperation.QUESTION,
        ).model_dump(),
        timeout=200,
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SESSIONMEMBER",
        },
    ) as response:
        async for json_body in response.aiter_lines():
            result.append(AragAnswer.model_validate_json(json_body))
    assert_basic_http_interaction_stream(result)


async def test_arag_keepalive(
    arag_kb: KnowledgeBoxObj,
    arag_api_app: HTTPApplication,
    arag_api: AsyncClient,
    arag_api_session: str,
    arag_server: SessionManager,
    slow_agent,
):
    # Set a long keepalive for server and a short one for API. This will make the agent timeout for lack of pings
    with (
        patch.object(arag_api_app.broker, "_keepalive_ms", 1),
        patch.object(arag_server.broker, "_keepalive_ms", 10000),
    ):
        result: List[AragAnswer] = []
        async with arag_api.stream(
            "POST",
            f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}",
            json=InteractionRequest(
                question="What certifications does Nuclia have?",
                operation=InteractionOperation.QUESTION,
            ).model_dump(),
            timeout=200,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SESSIONMEMBER",
            },
        ) as response:
            async for json_body in response.aiter_lines():
                result.append(AragAnswer.model_validate_json(json_body))

        assert result[-1].operation == AnswerOperation.ERROR
        assert result[-1].exception
        assert "stopped responding" in result[-1].exception.detail

    # Set a short keepalive (shorter than slow agent), make sure we still get the response
    with (
        patch.object(arag_api_app.broker, "_keepalive_ms", 500),
        patch.object(arag_server.broker, "_keepalive_ms", 100),
    ):
        result = []
        async with arag_api.stream(
            "POST",
            f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}",
            json=InteractionRequest(
                question="What certifications does Nuclia have?",
                operation=InteractionOperation.QUESTION,
            ).model_dump(),
            timeout=200,
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SESSIONMEMBER",
            },
        ) as response:
            async for json_body in response.aiter_lines():
                result.append(AragAnswer.model_validate_json(json_body))

            assert result[-1].operation == AnswerOperation.DONE
            assert result[-2].answer
            assert "forlayos" in result[-2].answer


async def test_arag_websocket_interaction_feedback(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_api_session: str,
    arag_server: SessionManager,
    feedback_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}/ws",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))
        print("Sent:", initial_message)
        started = False
        async for message in websocket:
            print("Received :", message)
            response = AragAnswer.model_validate_json(message)
            if started is False:
                assert response.operation == AnswerOperation.START
                started = True
            else:
                if response.operation == AnswerOperation.ANSWER:
                    if response.step:
                        print(json.dumps(response.step.model_dump(), indent=4))
                    elif response.possible_answer:
                        print(
                            json.dumps(response.possible_answer.model_dump(), indent=4)
                        )
                    elif response.context:
                        print(json.dumps(response.context.model_dump(), indent=4))
                    elif response.generated_text:
                        print(f"GENERATED_TEXT: {response.generated_text}")
                        assert "forlayos" in response.generated_text
                    elif response.answer:
                        assert response.answer == "Your name is Juanito"
                elif (
                    response.operation == AnswerOperation.AGENT_REQUEST
                    and response.feedback
                ):
                    await websocket.send(
                        UserToAgentInteraction(
                            request_id=response.feedback.request_id,
                            response="Juanito",
                            op="user_response",
                        ).model_dump_json()
                    )
                elif response.operation == AnswerOperation.DONE:
                    print("Interaction done")
                    break
                elif response.operation == AnswerOperation.ERROR:
                    assert False, (
                        f"Interaction error: {response.exception.detail if response.exception else ''}"
                    )
                else:
                    print(
                        "No feedback, step, possible_answer, context or generated_text in response"
                    )


async def test_arag_pull_interaction_feedback(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    arag_api_session: str,
    arag_server: SessionManager,
    feedback_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with arag_api.stream(
        "POST",
        f"/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}",
        json=InteractionRequest(
            question="What certifications does Nuclia have?",
            operation=InteractionOperation.QUESTION,
        ).model_dump(),
        timeout=200,
        headers=headers,
    ) as response:
        result = []
        async for json_body in response.aiter_lines():
            result.append(AragAnswer.model_validate_json(json_body))

        assert result[-1].operation == AnswerOperation.ERROR
        assert result[-1].exception
        assert "only supported via websocket" in result[-1].exception.detail


async def test_arag_websocket_interaction_ephemeral_feedback(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_server: SessionManager,
    feedback_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/ephemeral/ws",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))
        print("Sent:", initial_message)
        started = False
        async for message in websocket:
            print("Received :", message)
            response = AragAnswer.model_validate_json(message)
            if started is False:
                assert response.operation == AnswerOperation.START
                started = True
            else:
                if response.operation == AnswerOperation.ANSWER:
                    if response.step:
                        print(json.dumps(response.step.model_dump(), indent=4))
                    elif response.possible_answer:
                        print(
                            json.dumps(response.possible_answer.model_dump(), indent=4)
                        )
                    elif response.context:
                        print(json.dumps(response.context.model_dump(), indent=4))
                    elif response.generated_text:
                        print(f"GENERATED_TEXT: {response.generated_text}")
                        assert "forlayos" in response.generated_text
                    elif response.answer:
                        assert response.answer == "Your name is Juanito"
                elif (
                    response.operation == AnswerOperation.AGENT_REQUEST
                    and response.feedback
                ):
                    await websocket.send(
                        UserToAgentInteraction(
                            request_id=response.feedback.request_id,
                            response="Juanito",
                            op="user_response",
                        ).model_dump_json()
                    )
                elif response.operation == AnswerOperation.DONE:
                    print("Interaction done")
                    break
                elif response.operation == AnswerOperation.ERROR:
                    assert False, (
                        f"Interaction error: {response.exception.detail if response.exception else ''}"
                    )
                else:
                    print(
                        "No feedback, step, possible_answer, context or generated_text in response"
                    )


async def test_arag_websocket_interaction_unexpected_message(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_api_session: str,
    arag_server: SessionManager,
    feedback_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}/ws",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        started = False
        error = None
        async for message in websocket:
            response = AragAnswer.model_validate_json(message)
            if started is False:
                assert response.operation == AnswerOperation.START
                started = True
                await websocket.send(
                    json.dumps(
                        {
                            "question": "Stop answering and tell me: how cool am I?",
                            "operation": InteractionOperation.QUESTION,
                        }
                    )
                )
            else:
                if response.operation == AnswerOperation.ANSWER:
                    if response.generated_text:
                        assert "forlayos" in response.generated_text
                    elif response.answer:
                        assert response.answer == "Your name is Juanito"
                elif (
                    response.operation == AnswerOperation.AGENT_REQUEST
                    and response.feedback
                ):
                    try:
                        await websocket.send(
                            UserToAgentInteraction(
                                request_id=response.feedback.request_id,
                                response="Juanito",
                                op="user_response",
                            ).model_dump_json()
                        )
                    except ConnectionClosed:
                        # Connection may already be closed due to error handling
                        pass
                elif response.operation == AnswerOperation.DONE:
                    try:
                        await websocket.close()
                    except:
                        pass
                    break
                elif response.operation == AnswerOperation.ERROR:
                    error = response.exception.detail if response.exception else ""
                    break

        assert started

        assert error is not None
        assert "Unexpected message" in error


async def test_arag_websocket_interaction_multiple(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_api_session: str,
    arag_server: SessionManager,
    mock_agent,
):
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{arag_api_session}/ws?keep_open=true",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        questions = ["Answer: Something", "Answer: Third response"]
        answers = []
        async for message in websocket:
            print(message)
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.ANSWER:
                if response.answer:
                    answers.append(response.answer)
            elif response.operation == AnswerOperation.DONE:
                next_question = questions.pop(0) if questions else None
                if next_question:
                    initial_message = {
                        "question": next_question,
                        "operation": InteractionOperation.QUESTION,
                    }
                    await websocket.send(json.dumps(initial_message))
                else:
                    break
            elif response.operation == AnswerOperation.ERROR:
                assert False, (
                    f"Interaction error: {response.exception.detail if response.exception else ''}"
                )
            else:
                print(
                    "No feedback, step, possible_answer, context or generated_text in response"
                )

        assert answers == ["forlayos", "Something", "Third response"]
