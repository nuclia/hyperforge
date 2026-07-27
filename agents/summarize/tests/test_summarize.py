import os
from copy import deepcopy

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context
from hyperforge.memory.memory import EphemeralSessionMemory
from hyperforge.minimal_fixtures import cassette_nua_key
from hyperforge.models import MemoryConfig, Rules
from nuclia.lib.nua import AsyncNuaClient

from hyperforge_summarize.agent import SummarizeAgent
from hyperforge_summarize.config import SummarizeAgentConfig

# Real key used when recording; the stub is sufficient for cassette replay.
NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

pytestmark = [pytest.mark.vcr(ignore_localhost=True), pytest.mark.asyncio]

DE48CFAA_3209_4041_BB64_8604AFF061FB = os.environ.get(
    "KB_DE48CFAA_3209_4041_BB64_8604AFF061FB"
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

DF8B4C24_2807_4888_AD6C_AE97357A638B = os.environ.get(
    "KB_DF8B4C24_2807_4888_AD6C_AE97357A638B"
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

CONFIG = {
    "drivers": [
        {
            "name": "nuclia-conversation",
            "provider": "nucliadb",
            "identifier": "nucliadb-1",
            "config": {
                "url": "https://europe-1.stashify.cloud/api",
                "manager": "https://europe-1.stashify.cloud/api",
                "kbid": "de48cfaa-3209-4041-bb64-8604aff061FB",
                "key": DE48CFAA_3209_4041_BB64_8604AFF061FB,
                "filters": [],
                "description": "Make Discourse Conversation",
            },
        },
        {
            "name": "nuclia-docs",
            "provider": "nucliadb",
            "identifier": "nucliadb-2",
            "config": {
                "identifier": "nucliadb-2",
                "url": "https://europe-1.nuclia.cloud/api",
                "manager": "https://europe-1.nuclia.cloud/api",
                "kbid": "df8b4c24-2807-4888-ad6c-ae97357a638b",
                "key": DF8B4C24_2807_4888_AD6C_AE97357A638B,
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
            },
        },
    ],
    "rules": {
        "rules": [
            {"prompt": "Be polite"},
            {
                "prompt": "The documentation of Nuclia is hosted at https://docs.nuclia.dev"
            },
        ]
    },
    "memory": {},
    "workflow": {
        "id": "default",
        "name": "Default workflow",
        "description": "Default workflow for testing",
        "parameters": {},
    },
    "preprocess": [],
    "context": [],
    "generation": [],
    "postprocess": [],
}


async def test_summarize_uses_context_summary_with_forced_chunk_citations():
    manager = await Manager.from_config(
        drivers=[],
        nua=AsyncNuaClient(region="europe-1", account="test", token=NUA_KEY),
    )
    agent = SummarizeAgent(
        SummarizeAgentConfig(
            citations=True,
            force_chunk_level_citations=True,
        )
    )
    session = EphemeralSessionMemory.from_config(
        config=MemoryConfig(),
        agent_id="test",
        workflow_id="test",
        rules=Rules(),
    )
    session.init("test-session")
    memory = session.start_question("What is the launch code for the Aurora project?")
    memory.contexts.append(
        Context(
            original_question_uuid="question-1",
            actual_question_uuid="question-1",
            question="What is the launch code for the Aurora project?",
            source="test-source",
            agent="upstream",
            summary="The Aurora project launch code is LANTERN-47.",
            chunks=[
                Chunk(
                    chunk_id="chunk-1",
                    text="The Aurora project launch code is documented in the deployment runbook.",
                )
            ],
        )
    )

    await agent(memory, manager)

    answer, citations = memory.answers[-1]
    assert "lantern-47" in answer.lower()
    assert citations is not None
    assert citations.metadata["block-AA-0"].chunk_index == 0


async def test_summarize_answers():
    config = deepcopy(CONFIG)
    config["context"] = [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        }
    ]
    config["generation"] = [
        {
            "module": "summarize",
            "conversational": False,
        }
    ]

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_magic y dime como cambiará este parametro en el futuro",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )
    assert "not enough data to answer this" in question_memory.final_answer.lower()
    config["generation"][0]["conversational"] = True  # type: ignore
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_magic y dime como cambiará este parametro en el futuro",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )
    assert "not enough data to answer this" not in question_memory.final_answer.lower()


async def test_summarize_tokens():
    config = deepcopy(CONFIG)

    config["context"] = [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        },
    ]
    config["generation"] = [
        {
            "module": "summarize",
            "conversational": False,
            "model": "claude-4-5-haiku",
        }
    ]
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )

    summarize_step = question_memory.steps[-1]
    assert summarize_step.input_nuclia_tokens >= 0  # type: ignore
    assert summarize_step.output_nuclia_tokens >= 0  # type: ignore


async def test_summarize_answers_with_citations():
    config = deepcopy(CONFIG)

    config["context"] = [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        },
    ]
    config["generation"] = [
        {"module": "summarize", "conversational": True, "citations": True},
    ]
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.magic  y dime como cambiará este parametro en el futuro",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )
    assert "not enough data to answer this" not in question_memory.final_answer.lower()
    # Check that we have citations
    assert question_memory.final_answer_citations is not None
    assert question_memory.final_answer_citations.metadata != {}

    # Check that we can refer back to the context using the citation metadata
    context_id = question_memory.final_answer_citations.metadata["block-AA"].context_id
    context = next(
        (c for c in question_memory.contexts if c.id == context_id),
        None,
    )
    assert context is not None
    assert context.chunks != []


async def test_summarize_answers_force_chunk_level_citations():
    config = deepcopy(CONFIG)
    config["context"] = [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        },
    ]
    config["generation"] = [
        {
            "module": "summarize",
            "conversational": True,
            "citations": True,
            "force_chunk_level_citations": True,
        },
    ]
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens.magic  y dime como cambiará este parametro en el futuro",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )

    assert "not enough data to answer this" not in question_memory.final_answer.lower()
    # Check that we have citations
    assert question_memory.final_answer_citations is not None
    assert question_memory.final_answer_citations.metadata != {}

    # Check that we can refer back to the cited chunks using the citation metadata
    for chunk_citation, chunk_index in [
        ("block-AA-0", 0),
        ("block-AA-1", 1),
    ]:
        assert (
            question_memory.final_answer_citations.metadata[chunk_citation].chunk_index
            == chunk_index
        )
        context_id = question_memory.final_answer_citations.metadata[
            chunk_citation
        ].context_id
        context = next(
            (c for c in question_memory.contexts if c.id == context_id),
            None,
        )
        assert context is not None
        chunk = context.chunks[chunk_index]
        assert chunk.text is not None


async def test_summarize_with_funny_system_prompt():
    """Test that system prompt can make responses fun and engaging with emojis."""
    config = deepcopy(CONFIG)
    config["context"] = [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        },
    ]

    # Test with a custom system prompt that makes responses funny and uses emojis
    funny_system_prompt = """You are a cheerful and enthusiastic technical assistant who loves making learning fun! 🎉

Your style:
- Use emojis liberally to make your explanations more engaging and visual 😊
- Make technical concepts feel approachable with friendly language
- Add humor where appropriate, but stay accurate and helpful
- Use analogies and fun comparisons to explain complex ideas
- Celebrate the user's curiosity and questions! 🌟

Guidelines:
- Sprinkle relevant emojis throughout your answers (but don't overdo it)
- Keep the friendly, upbeat tone while remaining professional
- Make technical topics feel less intimidating
- Use fun phrases and expressions to keep things lively

Remember: You're here to make learning enjoyable while providing accurate technical information! 🚀
If you can't answer, just say that they can always ask the Oracle of Nuclia for more wisdom! 🧙‍♂️"""

    config["generation"] = [
        {
            "module": "summarize",
            "conversational": True,
            "system_prompt": funny_system_prompt,
        },
    ]

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="How do I use max_tokens parameter?",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )

    assert question_memory.final_answer
    assert len(question_memory.final_answer) > 0
    # Verify that system prompt influenced the response - should contain emojis
    has_emoji = any(
        emoji in question_memory.final_answer
        for emoji in ["😊", "🎉", "🚀", "✨", "💡", "👍", "🌟", "📝", "⚡", "🔧", "💻"]
    )
    assert has_emoji, "Response should contain emojis as per the funny system prompt"

    # Test with a question about unknown/future information
    question_memory_unknown = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="What are Nuclia's plans for 2027 and beyond?",
        config=config,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )

    # Verify that it mentions the Oracle of Nuclia and still contains emojis to maintain the fun tone, even when it can't provide a concrete answer about the future
    assert question_memory_unknown.final_answer
    assert len(question_memory_unknown.final_answer) > 0
    has_emoji_unknown = any(
        emoji in question_memory_unknown.final_answer
        for emoji in [
            "😊",
            "🎉",
            "🚀",
            "✨",
            "💡",
            "👍",
            "🌟",
            "📝",
            "⚡",
            "🔧",
            "💻",
            "🧙",
            "🔮",
        ]
    )
    assert has_emoji_unknown, (
        "Response should maintain fun tone with emojis even when information is unavailable"
    )
    assert "oracle of nuclia" in question_memory_unknown.final_answer.lower()


async def test_summarize_streaming():
    """Test that streaming mode produces the same quality answer as non-streaming."""
    config = deepcopy(CONFIG)
    config["context"] = [
        {
            "module": "ask",
            "title": "",
            "sources": ["nucliadb-2"],
            "ai_parameter_search": False,
        }
    ]
    config["generation"] = [
        {
            "module": "summarize",
            "conversational": True,
        }
    ]

    # Collect streamed chunks via callback
    streamed_chunks: list[str] = []

    async def capture_callback(answer):
        from hyperforge.interaction import AnswerOperation

        if (
            answer.operation == AnswerOperation.ANSWER_CHUNK
            and answer.streaming_response_chunk
            and answer.streaming_response_chunk.text
        ):
            streamed_chunks.append(answer.streaming_response_chunk.text)

    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question="Como usar max_tokens",
        config=config,
        callback=capture_callback,
        streaming=True,
        loaded_modules=[
            "hyperforge_summarize",
            "hyperforge_nucliadb",
            "hyperforge_rephrase",
        ],
    )

    # Should have a final answer
    assert question_memory.final_answer is not None
    assert len(question_memory.final_answer) > 0

    # Streamed chunks should reassemble to the final answer
    reassembled = "".join(streamed_chunks)
    assert reassembled == question_memory.final_answer

    # Step should have token consumption recorded
    summarize_step = question_memory.steps[-1]
    assert summarize_step.input_nuclia_tokens >= 0  # type: ignore
    assert summarize_step.output_nuclia_tokens >= 0  # type: ignore
