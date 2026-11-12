import os
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from hyperforge.context.agent import ContextAgent
from hyperforge.context.config import ContextAgentConfig
from hyperforge.llm import NUAConnection
from hyperforge.manager import Manager
from hyperforge.memory.memory import BaseSessionMemory
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import Chunk, Context, MemoryConfig, Rules

NUA_KEY = os.environ.get("NUA_KEY") or cassette_nua_key(
    "https://europe-1.dp.progress.cloud/"
)
VALIDATION_MODEL = "gemini-2.5-flash"


class StubContextAgent(ContextAgent):
    def __init__(self, *, real_model: bool = False) -> None:
        self.config = ContextAgentConfig(
            module="test",
            id="test",
            context_validation_model=VALIDATION_MODEL if real_model else "test",
            prune_context=True,
        )
        self.agent_id = "test"


@pytest_asyncio.fixture
async def validation_manager():
    nua = await NUAConnection(key=NUA_KEY).connect()
    return await Manager.from_config(drivers=[], nua=nua)


def make_memory(question: str = "question"):
    session = BaseSessionMemory.from_config(
        MemoryConfig(), agent_id="root", workflow_id="default", rules=Rules()
    )
    session.init("session")
    return session.start_question(question)


def make_context(
    agent_id: str,
    summary: str = "",
    chunks: list[str] | None = None,
) -> Context:
    return Context(
        id=f"context-{agent_id}",
        agent_id=agent_id,
        original_question_uuid="original",
        actual_question_uuid="actual",
        question="question",
        source=agent_id,
        agent=agent_id,
        summary=summary,
        chunks=[
            Chunk(chunk_id=f"{agent_id}-{index}", text=text)
            for index, text in enumerate(chunks or [f"{agent_id} text"])
        ],
    )


@pytest.mark.asyncio
async def test_validate_contexts_maps_blocks_to_original_payloads() -> None:
    agent = StubContextAgent()
    manager = AsyncMock()
    contexts = [
        make_context("first", "First summary"),
        make_context("second"),
        make_context("omitted"),
    ]
    manager.execute_json.return_value = (
        {
            "missing_info_query": "",
            "contexts": [
                {
                    "context_id": contexts[0].id,
                    "answer": "First partial answer",
                    "citations": [],
                },
                {
                    "context_id": contexts[1].id,
                    "answer": "Second partial answer",
                    "citations": ["block-AB"],
                },
            ],
        },
        10,
        5,
    )

    memory = make_memory()
    result = await agent.validate_contexts_and_answer(
        memory, manager, contexts, "question"
    )

    assert result == ("yes", None, None)
    assert contexts[0].citations == []
    assert contexts[1].citations == ["second-0"]
    assert contexts[2].citations is None
    assert contexts[0].summary == "First summary"
    assert contexts[1].summary == "Second partial answer"
    assert contexts[2].summary == ""
    assert memory.steps[-1].title == "agent: Context validation"


@pytest.mark.asyncio
async def test_multi_context_validation_merges_duplicate_context_results() -> None:
    agent = StubContextAgent()
    manager = AsyncMock()
    context = make_context(
        "cities",
        chunks=["Haverhill answer", "Ashland answer", "York answer"],
    )
    manager.execute_json.return_value = (
        {
            "missing_info_query": "",
            "contexts": [
                {
                    "context_id": context.id,
                    "answer": "City of Haverhill",
                    "citations": ["block-AA"],
                },
                {
                    "context_id": context.id,
                    "answer": "City of Ashland",
                    "citations": ["block-AB"],
                },
            ],
        },
        10,
        5,
    )

    memory = make_memory()
    await agent.save_contexts_and_return_missing(
        memory=memory,
        manager=manager,
        question="Which cities?",
        contexts=[context],
        flow_id="flow",
    )

    assert memory.contexts == [context]
    assert context.summary == "City of Haverhill\nCity of Ashland"
    assert context.citations == ["cities-0", "cities-1"]
    assert [chunk.chunk_id for chunk in context.chunks] == ["cities-0", "cities-1"]


@pytest.mark.asyncio
async def test_complete_source_contexts_skip_second_validation() -> None:
    agent = StubContextAgent()
    manager = AsyncMock()
    context = make_context(
        "source",
        summary="The source answer.",
        chunks=["Relevant", "Unrelated"],
    )
    context.citations = ["source-0"]
    memory = make_memory()

    await agent.save_contexts_and_return_missing(
        memory=memory,
        manager=manager,
        question="question",
        contexts=[context],
        flow_id="flow",
    )

    manager.execute_json.assert_not_awaited()
    assert memory.contexts == [context]
    assert [chunk.chunk_id for chunk in context.chunks] == ["source-0"]


@pytest.mark.asyncio
async def test_multi_context_validation_keeps_unmentioned_source_answer() -> None:
    agent = StubContextAgent()
    manager = AsyncMock()
    context = make_context("haverhill", summary="City of Haverhill")
    context.citations = ["haverhill-0"]
    manager.execute_json.return_value = (
        {"missing_info_query": "", "contexts": []},
        10,
        5,
    )

    memory = make_memory()
    await agent.save_contexts_and_return_missing(
        memory=memory,
        manager=manager,
        question="Which city?",
        contexts=[context],
        flow_id="flow",
    )

    assert memory.contexts == [context]
    assert context.summary == "City of Haverhill"
    assert context.citations == ["haverhill-0"]


def test_prune_to_structured_citations_removes_chunks() -> None:
    context = make_context("mixed", chunks=["Unrelated chunk"])
    context.structured = ["Relevant structured data", "Unrelated structured data"]
    context.citations = ["structured-0"]

    context.prune_to_citations()

    assert context.chunks == []
    assert context.structured == ["Relevant structured data"]


@pytest.mark.asyncio
async def test_single_context_validation_preserves_legacy_mutations() -> None:
    agent = StubContextAgent()
    manager = AsyncMock()
    manager.execute_json.return_value = (
        {
            "answer": "Answer",
            "missing_info_query": "",
            "useful": "yes",
            "reason": "Relevant",
            "citations": ["block-AA"],
        },
        10,
        5,
    )
    context = make_context("first")

    useful, missing, error = await agent.validate_ctx_and_answer(
        make_memory(), manager, context, "question"
    )

    assert (useful, missing, error) == ("yes", None, None)
    assert context.summary == "Answer"
    assert context.citations == ["first-0"]


@pytest.mark.asyncio
async def test_multi_context_save_indexes_children_under_parent_agent() -> None:
    agent = StubContextAgent()
    agent.config.prune_context = False
    context = make_context("child", summary="Child answer")
    memory = make_memory()

    await agent.save_contexts_and_return_missing(
        memory=memory,
        manager=AsyncMock(),
        question="question",
        contexts=[context],
        flow_id="flow",
    )

    assert memory.contexts == [context]
    assert context.agent_id == "child"
    assert memory.get_agent_answer_summaries("flow", "test") == ["Child answer"]


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_multi_context_validation_keeps_relevant_contexts(
    validation_manager: Manager,
) -> None:
    question = "What kind of animal is Pixel, and what color is Pixel's collar?"
    animal = make_context("animal", chunks=["Pixel is a rescue cat."])
    collar = make_context("collar", chunks=["Pixel wears a green collar."])
    irrelevant = make_context(
        "database", chunks=["The reporting service stores data in PostgreSQL."]
    )

    missing = await StubContextAgent(real_model=True).save_contexts_and_return_missing(
        memory=make_memory(question),
        manager=validation_manager,
        question=question,
        contexts=[animal, collar, irrelevant],
        flow_id="flow",
    )

    assert missing is None
    assert animal.citations == ["animal-0"]
    assert collar.citations == ["collar-0"]
    assert irrelevant.citations is None
    assert "cat" in animal.summary.lower()
    assert "green" in collar.summary.lower()


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_multi_context_validation_keeps_summary_and_discards_chunks(
    validation_manager: Manager,
) -> None:
    question = "On what date is the Atlas launch scheduled?"
    summary_only = make_context(
        "launch",
        summary="The Atlas launch is scheduled for 14 September 2032.",
        chunks=["The cafeteria serves soup on Tuesdays."],
    )
    irrelevant = make_context(
        "weather", chunks=["The coastal forecast calls for light rain."]
    )
    memory = make_memory(question)

    missing = await StubContextAgent(real_model=True).save_contexts_and_return_missing(
        memory=memory,
        manager=validation_manager,
        question=question,
        contexts=[summary_only, irrelevant],
        flow_id="flow",
    )

    assert missing is None
    assert memory.contexts == [summary_only]
    assert summary_only.citations == []
    assert summary_only.chunks == []
    assert "14" in summary_only.summary
    assert "2032" in summary_only.summary


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_multi_context_validation_prunes_irrelevant_chunks(
    validation_manager: Manager,
) -> None:
    question = "Who won the Northbridge design award?"
    awards = make_context(
        "awards",
        chunks=[
            "Mira Chen won the Northbridge design award.",
            "The ceremony menu included mushroom risotto.",
        ],
    )
    memory = make_memory(question)

    missing = await StubContextAgent(real_model=True).save_contexts_and_return_missing(
        memory=memory,
        manager=validation_manager,
        question=question,
        contexts=[awards],
        flow_id="flow",
    )

    assert missing is None
    assert memory.contexts == [awards]
    assert awards.citations == ["awards-0"]
    assert [chunk.chunk_id for chunk in awards.chunks] == ["awards-0"]
    assert "mira chen" in awards.summary.lower()


@pytest.mark.asyncio
@pytest.mark.vcr(ignore_localhost=True)
async def test_multi_context_validation_reports_globally_missing_information(
    validation_manager: Manager,
) -> None:
    question = "What is Project Aurora's codename and approved budget?"
    project = make_context(
        "project", chunks=["The internal codename for the project is Aurora."]
    )

    missing = await StubContextAgent(real_model=True).save_contexts_and_return_missing(
        memory=make_memory(question),
        manager=validation_manager,
        question=question,
        contexts=[project],
        flow_id="flow",
    )

    assert missing is not None
    assert "budget" in missing[1].lower()
    assert project.citations == ["project-0"]
    assert "aurora" in project.summary.lower()
