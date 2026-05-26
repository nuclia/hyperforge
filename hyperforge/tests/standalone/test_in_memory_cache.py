from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from httpx._transports.asgi import ASGITransport
from hyperforge.api.models import InteractionRequest
from hyperforge.context.agent import ContextAgent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.engine import State
from hyperforge.models import HistoryQuestionAnswer, MemoryConfig
from hyperforge.memory.memory import (
    EphemeralSessionMemory,
)
from hyperforge.retrieval.agent import RetrievalAgent
from hyperforge.trace import trace_agent
from hyperforge.server.cache import (
    CachedSessionQA,
    InMemoryCache,
)
from hyperforge.standalone.app import StandaloneApplication
from hyperforge.standalone.config import StandaloneConfig
from hyperforge.standalone.settings import StandaloneSettings

pytestmark = pytest.mark.asyncio

# ---------------------------------------------------------------------------
# Minimal standalone config reused across tests
# ---------------------------------------------------------------------------

_AGENT_ID = "mem-test-agent"


@pytest.fixture
def local_agents_config(load_agents):
    return StandaloneConfig.validate_python(
        {
            _AGENT_ID: {
                "title": "Memory Test Agent",
                "instructions": "Test agent for cache tests.",
                "workflows": {
                    "default": {
                        "name": "Default",
                        "generation": [{"module": "summarize"}],
                    },
                },
            }
        }
    )


_STANDALONE_SETTINGS = StandaloneSettings(
    agents_config=Path("/dev/null"),
    external_nua_api_key="dummy",
    # broker_redis_dsn intentionally omitted → InMemoryCache must be used
)


# ---------------------------------------------------------------------------
# Mock context agent (same pattern as test_standalone.py)
# ---------------------------------------------------------------------------


class _AnswerContextAgent(ContextAgent):
    """Context agent whose answer text is configurable per test."""

    config: ContextAgentConfig = ContextAgentConfig(module="mock-mem")
    agent_id: str = "mock-mem"

    # Override to inject a custom answer at test time.
    answer_text: str = "default answer"

    @classmethod
    def config_class(cls):
        return ContextAgentConfig

    @trace_agent
    async def get_question_context(
        self,
        memory,
        manager,
        question_uuid=None,
        question=None,
        flow_id=None,
        extra_context=None,
    ):
        await memory.add_step(
            step_module="mock-mem",
            step_title="mock step",
            step_agent_path="path",
            step_value="context",
            step_reason=None,
            timeit=0.01,
            input_nuclia_tokens=0,
            output_nuclia_tokens=0,
            error=None,
        )
        await memory.add_answer(self.answer_text, "mock-mem", "path")


_mock_context_agent = _AnswerContextAgent()
_mock_retrieval_agent = RetrievalAgent(context=[_mock_context_agent])
_mock_state = State(manager=None, agent=_mock_retrieval_agent)  # type: ignore[arg-type]


@pytest.fixture
def standalone_app(load_agents, local_agents_config):
    app = StandaloneApplication(local_agents_config, _STANDALONE_SETTINGS)
    with patch("hyperforge.server.session.get_state", return_value=_mock_state):
        yield app


@pytest.fixture
async def standalone_client(load_agents, standalone_app: StandaloneApplication):
    async with (
        standalone_app.router.lifespan_context(standalone_app),
        AsyncClient(
            transport=ASGITransport(app=standalone_app),
            base_url="http://test",
        ) as client,
    ):
        yield client


async def test_in_memory_cache_get_missing_key():
    cache = InMemoryCache()
    assert await cache.get("no-such-key") is None
    assert await cache.get_list("no-such-key") is None
    await cache.set("k1", "hello", expire=60)
    assert await cache.get("k1") == "hello"
    await cache.set("k1", "second", expire=60)
    assert await cache.get("k1") == "second"
    await cache.append("lst", ["a", "b"], limit=10, expire=60)
    result = await cache.get_list("lst")
    assert result == ["a", "b"]


async def test_in_memory_cache_append_extends_existing_list():
    cache = InMemoryCache()
    await cache.append("lst", ["a"], limit=10, expire=60)
    await cache.append("lst", ["b", "c"], limit=10, expire=60)
    assert await cache.get_list("lst") == ["a", "b", "c"]


# ===========================================================================
# 2.  CachedSessionQA backed by InMemoryCache
# ===========================================================================


async def test_cached_session_qa_empty_on_first_access():
    cache = InMemoryCache()
    qa = CachedSessionQA(cache, agent_id="agent1", session="sess1")
    assert await qa.get() is None


async def test_cached_session_qa_append_and_retrieve():
    cache = InMemoryCache()
    qa = CachedSessionQA(cache, agent_id="agent1", session="sess1")

    entry = HistoryQuestionAnswer(question="What?", answer="This.")
    await qa.append(entry)

    result = await qa.get()
    assert result == [entry]


async def test_cached_session_qa_append_all():
    cache = InMemoryCache()
    qa = CachedSessionQA(cache, agent_id="agent1", session="sess1")

    entries = [
        HistoryQuestionAnswer(question="Q1", answer="A1"),
        HistoryQuestionAnswer(question="Q2", answer="A2"),
    ]
    await qa.append_all(entries)

    result = await qa.get()
    assert result == entries


# ===========================================================================
# 4.  EphemeralSessionMemory uses InMemoryCache for history
# ===========================================================================


async def test_ephemeral_session_memory_history_via_in_memory_cache():
    """EphemeralSessionMemory.save() stores Q&A via CachedSessionQA, and
    qa_history() retrieves it from the cache on the next call."""
    cache = InMemoryCache()
    config = MemoryConfig()
    memory = EphemeralSessionMemory(
        config=config, agent_id="agent1", workflow_id="default", cache=cache
    )
    memory.init("session-1")

    # Simulate two questions being answered and saved.
    q1 = memory.start_question("What is 2+2?", question_id="q1")
    q1.final_answer = "4"
    await memory.save(q1)

    q2 = memory.start_question("And 3+3?", question_id="q2")
    q2.final_answer = "6"
    await memory.save(q2)

    history = await memory.qa_history()
    assert len(history) == 2
    assert history[0] == HistoryQuestionAnswer(question="What is 2+2?", answer="4")
    assert history[1] == HistoryQuestionAnswer(question="And 3+3?", answer="6")


async def test_standalone_app_uses_in_memory_cache_without_redis(
    standalone_client: AsyncClient,
    standalone_app: StandaloneApplication,
):
    """When broker_redis_dsn is not configured, StandaloneApplication must
    wire an InMemoryCache into the SessionManager."""
    # Trigger startup via the ASGI lifespan (already done by the fixture).
    assert isinstance(
        standalone_app.session_manager.cache,
        InMemoryCache,
    ), (
        "Expected InMemoryCache when no broker_redis_dsn is provided, "
        f"got {type(standalone_app.session_manager.cache)}"
    )


async def test_standalone_session_cache_stores_qa_after_interaction(
    standalone_app: StandaloneApplication,
):
    """After answering a question in a named session, the InMemoryCache backing
    the SessionManager must contain a Q&A entry for that session."""

    with patch("hyperforge.server.session.get_state", return_value=_mock_state):
        async with (
            standalone_app.router.lifespan_context(standalone_app),
            AsyncClient(
                transport=ASGITransport(app=standalone_app),
                base_url="http://test",
            ) as client,
        ):
            session_id = "cache-check-session"

            async with client.stream(
                "POST",
                f"/api/v1/agent/{_AGENT_ID}/session/{session_id}",
                json=InteractionRequest(
                    question="Test question for cache"
                ).model_dump(),
                timeout=30,
            ) as resp:
                assert resp.status_code == 200
                # Drain the stream to ensure the session manager finishes saving.
                async for _ in resp.aiter_lines():
                    pass

            # The cache backing the session manager should now hold a Q&A for
            # our session.  We inspect it directly via CachedSessionQA.
            cache: InMemoryCache = standalone_app.session_manager.cache  # type: ignore[assignment]
            cached_qa = CachedSessionQA(cache, agent_id=_AGENT_ID, session=session_id)
            history = await cached_qa.get()

            assert history is not None, (
                "Expected Q&A history in InMemoryCache after answering a question"
            )
            assert len(history) >= 1
            assert history[0].question == "Test question for cache"
