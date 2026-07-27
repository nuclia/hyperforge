import pytest

from hyperforge.interaction import AragAnswer
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import ExternalUsage, MemoryConfig, Rules


def test_external_usage_defaults_and_roundtrip():
    usage = ExternalUsage(provider="google", model="gemini-2.5-flash")

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.requests == 1
    assert ExternalUsage.model_validate(usage.model_dump()) == usage


@pytest.mark.asyncio
async def test_add_step_forwards_external_usage():
    memory_session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(),
        agent_id="agent",
        workflow_id="workflow",
        rules=Rules(),
    )
    memory_session.init("session")
    memory = memory_session.start_question("Question")
    answers: list[AragAnswer] = []

    async def callback(answer: AragAnswer):
        answers.append(answer)

    memory.set_callback_fn(callback)
    usage = ExternalUsage(
        provider="perplexity",
        model="sonar-pro",
        input_tokens=12,
        output_tokens=7,
    )

    await memory.add_step(
        step_module="perplexity",
        step_title="Search results",
        step_agent_path="/context/perplexity",
        timeit=0.1,
        external_usage=[usage],
    )

    assert memory.steps[0].external_usage == [usage]
    assert answers[0].step is not None
    assert answers[0].step.external_usage == [usage]
