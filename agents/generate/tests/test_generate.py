from collections import OrderedDict
from unittest.mock import AsyncMock, MagicMock

import pytest
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.models import MemoryConfig, Rules
from hyperforge_generate.agent import GenerateAgent
from hyperforge_generate.config import GenerateAgentConfig

pytestmark = pytest.mark.asyncio


async def test_generate_agent():
    # 1. Create a mocked manager
    manager = MagicMock(spec=Manager)

    # Mock execute_raw response.
    # resp must be an object with an `answer` attribute.
    resp_mock = MagicMock()
    resp_mock.answer = "This is a generated response based on context."
    manager.execute_raw = AsyncMock(return_value=(resp_mock, 10, 20))

    # 2. Create the configuration for GenerateAgent
    config = GenerateAgentConfig(
        prompt="Synthesize an answer.", model="chatgpt-azure-4o-mini"
    )

    # 3. Create the GenerateAgent instance
    agent = GenerateAgent(config=config)

    # 4. Set up EphemeralSessionMemory and QuestionMemory
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(), agent_id="test", workflow_id="test", rules=Rules()
    )
    session.init("test-session")
    memory = session.start_question("What is the capital of France?")

    # Let's add some contexts and rules to memory to verify prompt rendering
    context_obj = Context(
        id="context-1",
        original_question_uuid="question-1",
        actual_question_uuid="question-1",
        question="What is the capital of France?",
        source="test-source",
        agent="static",
        title="Static Context",
        chunks=[Chunk(chunk_id="chunk-1", text="The capital of France is Paris.")],
    )
    memory.contexts.append(context_obj)

    # Add generation rules
    memory.generation_rules = OrderedDict([("Only use the context.", "")])

    # 5. Call the agent
    await agent(memory=memory, manager=manager)

    # 6. Assertions
    # Verify execute_raw was called with expected arguments
    assert manager.execute_raw.called
    call_args = manager.execute_raw.call_args[0]
    chat_model = call_args[0]

    # Verify that prompt template rendered correctly containing context, question, and rules
    assert "The capital of France is Paris." in chat_model.user_prompt.prompt
    assert "What is the capital of France?" in chat_model.user_prompt.prompt
    assert "Only use the context." in chat_model.user_prompt.prompt
    assert "Synthesize an answer." in chat_model.user_prompt.prompt

    # Verify the generated text was added to memory
    assert memory.is_answered is True
    assert len(memory.generated_texts) == 2
    for val in memory.generated_texts.values():
        assert val == "This is a generated response based on context."

    # Verify a step was added
    assert len(memory.steps) == 1
    step = memory.steps[0]
    assert step.module == "generate"
    assert step.title == "Generate: Generate"
    assert step.input_nuclia_tokens == 10
    assert step.output_nuclia_tokens == 20
