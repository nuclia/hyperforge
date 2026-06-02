import pytest
from hyperforge.engine import main as arag_main
from hyperforge.interaction import AragAnswer

pytestmark = pytest.mark.asyncio


CONFIG = {
    "drivers": [],
    "rules": {},
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [],
    "postprocess": [],
    "context": [
        {
            "module": "static",
            "title": "Static Agent",
            "context": "My data",
            "structured": '{"source": "static"}',
            "prune_context": False,
        }
    ],
    "generation": [
        {
            "module": "passthrough",
            "title": "Passthrough",
            "rich_context": True,
        }
    ],
}


async def test_passthrough_rich_context_emits_possible_answer_from_pipeline():
    answers: list[AragAnswer] = []

    async def callback(obj: AragAnswer):
        answers.append(obj)

    await arag_main(
        agent_id="default",
        question="Return the static context",
        config=CONFIG,
        callback=callback,
        loaded_modules=["hyperforge_passthrough", "hyperforge_static"],
    )

    context_msg = next(
        (answer for answer in answers if answer.context is not None), None
    )
    assert context_msg is not None
    assert len(context_msg.context.chunks) == 1
    assert context_msg.context.chunks[0].text == "My data"
    assert context_msg.context.structured == ['{"source": "static"}']

    possible_answer_msg = next(
        (answer for answer in answers if answer.possible_answer is not None),
        None,
    )
    assert possible_answer_msg is not None
    assert possible_answer_msg.possible_answer.answer == ""
    assert possible_answer_msg.possible_answer.chunks[0].text == "My data"
    assert possible_answer_msg.possible_answer.structured == ['{"source": "static"}']
