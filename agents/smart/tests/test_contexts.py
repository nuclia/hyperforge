from hyperforge.models import Chunk, Context

from hyperforge_smart.agent import SmartAgent
from hyperforge_smart.config import SmartAgentConfig


def make_context(
    agent_id: str,
    *,
    summary: str = "",
    chunks: list[str] | None = None,
    structured: list[str] | None = None,
) -> Context:
    return Context(
        agent_id=agent_id,
        original_question_uuid="original",
        actual_question_uuid="actual",
        question="question",
        source=agent_id,
        agent=agent_id,
        title=agent_id,
        summary=summary,
        chunks=[
            Chunk(chunk_id=f"{agent_id}-{index}", text=text)
            for index, text in enumerate(chunks or [])
        ],
        structured=structured or [],
    )


def make_agent() -> SmartAgent:
    return SmartAgent(
        SmartAgentConfig(
            module="smart",
            id="smart",
            registered_agents=[],
        )
    )


def test_process_results_preserves_original_contexts() -> None:
    agent = make_agent()
    first = make_context("first", summary="First answer", chunks=["first chunk"])
    second = make_context("second", chunks=["second chunk"])
    collected: list[Context] = []

    result_texts = agent._process_results(
        [("ask_agent of nucliadb", [first, second])],
        collected_contexts=collected,
    )

    assert collected == [first, second]
    assert collected[0].summary == "First answer"
    assert collected[0].source == "first"
    assert collected[0].chunks[0].action == "ask_agent of nucliadb"
    assert result_texts[0] == "[ask_agent of nucliadb]:\nFirst answer"
