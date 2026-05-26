import json
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import hyperforge.engine
import pytest
from httpx import AsyncClient
from hyperforge.agent import Agent
from hyperforge.api.models import InteractionOperation
from hyperforge.context.agent import ContextAgent, trace_agent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.interaction import (
    AnswerOperation,
    AragAnswer,
)
from hyperforge.manager import Manager
from hyperforge.memory import QuestionMemory
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.server.session import SessionManager
from nucliadb_models.resource import KnowledgeBoxObj
from websockets.asyncio.client import connect

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["test", "127.0.0.1"]),
    pytest.mark.asyncio,
]


@pytest.fixture
def mock_agent_mem():
    """Mock agent fixture for testing interactions."""

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
            )
            await memory.add_answer(response, "mock", "path")

    retrieval_agent = RetrievalAgent(
        context=[MockAgent(config=ContextAgentConfig(module="mock"))]
    )
    with (
        patch(
            "hyperforge.engine.get_state",
            return_value=hyperforge.engine.State(manager=None, agent=retrieval_agent),
        ),
        patch(
            "hyperforge.api.v1.interaction.agent_has_nucliadb_memory",
            new_callable=AsyncMock,
            return_value=True,
        ),
    ):
        yield


@pytest.fixture
def mock_agent_no_memory():
    """Mock agent fixture that doesn't patch agent_has_nucliadb_memory."""

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
            )
            await memory.add_answer(response, "mock", "path")

    retrieval_agent = RetrievalAgent(
        context=[MockAgent(config=ContextAgentConfig(module="mock"))]
    )
    with patch(
        "hyperforge.engine.get_state",
        return_value=hyperforge.engine.State(manager=None, agent=retrieval_agent),
    ):
        yield


async def test_arag_session(
    arag_kb: KnowledgeBoxObj, arag_kb_legacy: KnowledgeBoxObj, arag_api: AsyncClient
):
    for arag in [arag_kb, arag_kb_legacy]:
        resp = await arag_api.post(
            f"/api/v1/agent/{arag.uuid}/sessions",
            json={
                "slug": "slug1",
                "name": "My Title",
                "summary": "This is a nice user",
                "data": json.dumps({"age": "46"}),
                "format": "JSON",
            },
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )
        assert resp.status_code == 200

        session_id = resp.json()["uuid"]

        resp = await arag_api.get(
            f"/api/v1/agent/{arag.uuid}/sessions",
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )
        assert resp.status_code == 200

        data = resp.json()
        assert len(data["resources"]) == 1
        assert data["resources"][0]["title"] == "My Title"
        assert data["resources"][0]["summary"] == "This is a nice user"
        assert data["resources"][0]["slug"] == "slug1"

        resp = await arag_api.get(
            f"/api/v1/agent/{arag.uuid}/session/{session_id}",
            headers={
                "X-STF-USER": "user1",
                "X-STF-ACCOUNT": "nuclia",
                "X-STF-ACCOUNT-TYPE": "basic",
                "X-STF-ROLES": "SOWNER",
            },
        )
        assert resp.status_code == 200

        data = resp.json()
        assert data["title"] == "My Title"
        assert data["summary"] == "This is a nice user"
        assert data["slug"] == "slug1"
        assert data["data"]["texts"]["info"]["value"]["body"] == '{"age": "46"}'


async def test_arag_session_no_memory(arag_api: AsyncClient, arag_no_memory: str):
    # All endpoints  should return 400
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_no_memory}/sessions",
        json={
            "slug": "slug1",
            "name": "My Title",
            "summary": "This is a nice user",
            "data": json.dumps({"age": "46"}),
            "format": "JSON",
        },
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 400
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_no_memory}/sessions",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 400
    resp = await arag_api.get(
        f"/api/v1/agent/{arag_no_memory}/session/some-session-id",
        headers={
            "X-STF-USER": "user1",
            "X-STF-ACCOUNT": "nuclia",
            "X-STF-ACCOUNT-TYPE": "basic",
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 400


async def test_arag_websocket_session_auto_create(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_server: SessionManager,
    mock_agent_mem,
):
    """Test that websocket auto-creates session when create_session_if_not_exists=True (default)"""
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    non_existent_session = "auto-created-session-123"

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{non_existent_session}/ws",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        # Should get START message, not an error
        message = await websocket.recv()
        response = AragAnswer.model_validate_json(message)
        assert response.operation == AnswerOperation.START, (
            "Session should be auto-created"
        )

        # Consume remaining messages
        async for message in websocket:
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.DONE:
                break


async def test_arag_websocket_session_no_auto_create_error(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_server: SessionManager,
    mock_agent_mem,
):
    """Test that websocket returns error when create_session_if_not_exists=False and session doesn't exist"""
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    non_existent_session = "non-existent-session-456"

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{non_existent_session}/ws?create_session_if_not_exists=false",
        additional_headers=headers,
    ) as websocket:
        # Should receive error immediately
        message = await websocket.recv()
        response = AragAnswer.model_validate_json(message)
        assert response.operation == AnswerOperation.ERROR
        assert response.exception is not None
        assert "does not exist" in response.exception.detail


async def test_arag_websocket_session_no_auto_create_existing_session(
    arag_kb: KnowledgeBoxObj,
    arag_api: AsyncClient,
    arag_api_http: str,
    arag_server: SessionManager,
    mock_agent_mem,
):
    """Test that websocket works with create_session_if_not_exists=False when session already exists"""
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    # First, create the session
    resp = await arag_api.post(
        f"/api/v1/agent/{arag_kb.uuid}/sessions",
        json={
            "slug": "existing-session-789",
            "name": "Existing Session",
            "summary": "This session exists",
            "data": json.dumps({"test": "data"}),
            "format": "JSON",
        },
        headers={
            **headers,
            "X-STF-ROLES": "SOWNER",
        },
    )
    assert resp.status_code == 200
    session_id = resp.json()["uuid"]

    # Now connect to websocket with create_session_if_not_exists=False
    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/{session_id}/ws?create_session_if_not_exists=false",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        # Should get START message since session exists
        message = await websocket.recv()
        response = AragAnswer.model_validate_json(message)
        assert response.operation == AnswerOperation.START, (
            "Should work with existing session"
        )

        # Consume remaining messages
        async for message in websocket:
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.DONE:
                break


async def test_arag_websocket_no_memory_no_session_check(
    arag_api_http: str,
    arag_no_memory: str,
    arag_server: SessionManager,
    mock_agent_no_memory,
):
    """Test that websocket doesn't check session for agents without NucliaDB memory"""
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    # Use a non-existent session ID - should not fail because agent has no memory
    non_existent_session = "no-memory-session-999"

    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_no_memory}/session/{non_existent_session}/ws?create_session_if_not_exists=false",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        # Should get START message - session check should be skipped for agents without memory
        message = await websocket.recv()
        response = AragAnswer.model_validate_json(message)
        assert response.operation == AnswerOperation.START, (
            "Session check should be skipped for agents without memory"
        )

        # Consume remaining messages
        async for message in websocket:
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.DONE:
                break


async def test_arag_websocket_ephemeral_session_always_works(
    arag_kb: KnowledgeBoxObj,
    arag_api_http: str,
    arag_server: SessionManager,
    mock_agent_mem,
):
    """Test that ephemeral sessions work regardless of create_session_if_not_exists flag"""
    headers = {
        "X-STF-ROLES": "SESSIONMEMBER",
        "X-STF-USER": "user1",
        "X-STF-ACCOUNT": "nuclia",
        "X-STF-ACCOUNT-TYPE": "basic",
    }

    # Ephemeral session should work even with create_session_if_not_exists=False
    async with connect(
        f"ws://{arag_api_http}/api/v1/agent/{arag_kb.uuid}/session/ephemeral/ws?create_session_if_not_exists=false",
        additional_headers=headers,
    ) as websocket:
        initial_message = {
            "question": "What certifications does Nuclia have?",
            "operation": InteractionOperation.QUESTION,
        }
        await websocket.send(json.dumps(initial_message))

        # Should get START message
        message = await websocket.recv()
        response = AragAnswer.model_validate_json(message)
        assert response.operation == AnswerOperation.START, (
            "Ephemeral sessions should always work"
        )

        # Consume remaining messages
        async for message in websocket:
            response = AragAnswer.model_validate_json(message)
            if response.operation == AnswerOperation.DONE:
                break
