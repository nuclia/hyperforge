from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hyperforge.pubsub import StartInteraction
from hyperforge.server.session import SessionManager


@pytest.mark.asyncio
async def test_ephemeral_session_is_not_cached():
    manager = SessionManager(
        settings=SimpleNamespace(
            answers_subject="arag.{account}.{agent_id}.{workflow_id}.{session}.{question}.answer",
            internal_nucliadb_url="",
            internal_nua_api="",
            internal_nua=False,
            local_openai=None,
            external_nua_api_key=None,
            standalone=False,
            allow_private_network_endpoints=False,
        ),
        broker=None,  # type: ignore[arg-type]
        agent_manager=SimpleNamespace(get_agent_config=AsyncMock()),
        cache=None,  # type: ignore[arg-type]
    )
    manager.agent_manager.get_agent_config.return_value = SimpleNamespace(
        memory=SimpleNamespace(),
        rules=SimpleNamespace(rules=[]),
    )

    memory = SimpleNamespace(rules=None)
    memory.start_question = MagicMock(return_value=SimpleNamespace())
    task = MagicMock()
    message = StartInteraction(
        account="account",
        agent_id="agent",
        session="ephemeral",
        question_id="question-id",
        question="question",
    )

    with (
        patch(
            "hyperforge.server.session.get_memory",
            new_callable=AsyncMock,
            return_value=memory,
        ) as get_memory,
        patch(
            "hyperforge.server.session.get_state",
            new_callable=AsyncMock,
            return_value=SimpleNamespace(),
        ),
        patch.object(manager, "answer", new=MagicMock(return_value=object())),
        patch("hyperforge.server.session.asyncio.create_task", return_value=task),
    ):
        await manager.activate(message)

    get_memory.assert_awaited_once()
    assert "ephemeral" not in manager.memory
