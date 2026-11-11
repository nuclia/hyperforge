from unittest.mock import AsyncMock, Mock

import pytest
from hyperforge.models import Context, Source

from hyperforge_nucliadb.basic_ask_agent import BasicAskAgent
from hyperforge_nucliadb.basic_ask_config import BasicAskAgentConfig


@pytest.mark.asyncio
@pytest.mark.parametrize("label_sets", [[], ["topic"]])
async def test_facets_count_handles_label_selection(mocker, label_sets):
    agent = BasicAskAgent(BasicAskAgentConfig(sources=["kb"]), agent_id="facets-agent")
    source = Source.model_construct(id="kb")
    mocker.patch(
        "hyperforge_nucliadb.basic_ask_agent.choose_source",
        new=AsyncMock(return_value=[source]),
    )
    driver = Mock(
        labels=AsyncMock(return_value={"topic": ["science"]}),
        field_labels=AsyncMock(return_value=({"topic": {"science": 2}}, 2)),
    )
    mocker.patch(
        "hyperforge_nucliadb.basic_ask_agent.get_ndb_driver", return_value=driver
    )
    memory = Mock(original_question_uuid="question-id", add_step=AsyncMock())
    manager = Mock(
        execute_json=AsyncMock(
            return_value=(
                {"label_sets": label_sets, "reasoning": "Label selection"},
                1,
                2,
            )
        ),
        execute=AsyncMock(return_value=("There are 2 science documents.", 3, 4, 0)),
    )

    contexts = await agent.facets_count(memory, manager, question="How many?")

    assert len(contexts) == 1
    context = contexts[0]
    assert isinstance(context, Context)
    assert context.agent_id == "facets-agent"
    assert context.original_question_uuid == "question-id"
    assert context.actual_question_uuid is None
    assert context.question == "How many?"
    assert context.source == "kb"
    assert context.agent == "basic_ask_facets"
    assert context.title == "Faceted search on kb Knowledge Box"
    memory.add_step.assert_awaited_once()
    step = memory.add_step.call_args.kwargs
    assert step["step_value"] == "Label selection"
    assert step["step_agent_path"] == "/context/facets-agent/facets_count"

    if label_sets:
        assert context.missing is None
        assert len(context.chunks) == 1
        assert context.chunks[0].chunk_id == "facets_count_result-0"
        assert context.chunks[0].text == "There are 2 science documents."
        driver.field_labels.assert_awaited_once_with(labelsets=label_sets)
        manager.execute.assert_awaited_once()
        assert step["step_reason"] == "Faceted search performed"
        assert step["input_nuclia_tokens"] == 4
        assert step["output_nuclia_tokens"] == 6
    else:
        assert context.missing == "How many?"
        assert context.chunks == []
        driver.field_labels.assert_not_awaited()
        manager.execute.assert_not_awaited()
        assert step["step_reason"] == "Faceted search skipped"
        assert step["input_nuclia_tokens"] == 1
        assert step["output_nuclia_tokens"] == 2
