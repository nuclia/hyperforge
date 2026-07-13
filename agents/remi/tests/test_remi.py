import os
import re

import pytest
from hyperforge.engine import main as arag_main
from hyperforge.minimal_fixtures import cassette_nua_key

from hyperforge_remi.config import ContextGranularity

NUA_KEY = os.environ.get(
    "NUA_KEY",
) or cassette_nua_key("https://europe-1.nuclia.cloud/")

KB_DF8B4C24_2807_4888_AD6C_AE97357A638B = os.environ.get(
    "KB_DF8B4C24_2807_4888_AD6C_AE97357A638B", "DUMMY_KEY"
)

pytestmark = [
    pytest.mark.vcr(ignore_localhost=True, ignore_hosts=["europe-1.nuclia.cloud"]),
    pytest.mark.asyncio,
]

ANSWERED_QUESTION = "What is Progress Agentic RAG?"
UNANSWERED_QUESTION = (
    "Como usar max_magic y dime como cambiará este parametro en el futuro"
)

CONFIG = {
    "drivers": [
        {
            "provider": "nucliadb",
            "identifier": "nuclia-docs",
            "name": "nuclia-docs",
            "config": {
                "url": "https://europe-1.nuclia.cloud/api",
                "manager": "https://europe-1.nuclia.cloud/api",
                "kbid": "df8b4c24-2807-4888-ad6c-ae97357a638b",
                "key": KB_DF8B4C24_2807_4888_AD6C_AE97357A638B,
                "filters": [],
                "description": "Documentation of the Nuclia API, recipies, reference",
            },
        },
    ],
    "rules": {
        "rules": [
            {
                "prompt": "Be polite",
            },
            {
                "prompt": "The documentation of Nuclia is hosted at https://docs.nuclia.dev",
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
    "context": [
        {
            "module": "basic_ask",
            "title": "Nuclia Docs Retrieval Agent",
            "sources": ["nuclia-docs"],
            "prune_context": False,
        },
    ],
    "generation": [
        {"module": "summarize", "title": "Summarize agent"},
    ],
    "postprocess": [{"module": "remi"}],
}


# Match on body since we send parallel requests and otherwise they get played back in a different order
@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
@pytest.mark.parametrize(
    "granularity", (ContextGranularity.PARTIAL_ANSWERS, ContextGranularity.FULL)
)
async def test_remi(granularity: ContextGranularity):
    CONFIG["postprocess"][0] = {"module": "remi", "context_granularity": granularity}  # type: ignore
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question=ANSWERED_QUESTION,
        config=CONFIG,
        loaded_modules=[
            "hyperforge_remi",
            "hyperforge_summarize",
            "hyperforge_nucliadb",
        ],
    )

    assert question_memory.final_answer is not None
    # verify the step logic is actually validating the context summary instead
    remi_contexts = [ctx for ctx in question_memory.contexts if ctx.agent == "remi"]
    assert len(remi_contexts) > 0, "Missing REMi structured context"
    remi_ctx = remi_contexts[-1]
    assert remi_ctx.title == "REMi Evaluation Breakdown"

    assert remi_ctx.summary is not None
    relevance_match = re.search(r"Answer relevance: (\d+)/5\.", remi_ctx.summary)
    assert relevance_match is not None, remi_ctx.summary
    relevance_score = int(relevance_match.group(1))
    assert relevance_score > 1, f"Relevance score {relevance_score} should be > 1"

    groundedness_match = re.search(r"Answer groundedness: (\d+)/5\.", remi_ctx.summary)
    assert groundedness_match is not None, remi_ctx.summary
    groundedness_score = int(groundedness_match.group(1))
    assert groundedness_score > 1, (
        f"Groundedness score {groundedness_score} should be > 1"
    )

    assert len(remi_ctx.chunks) > 0, "Missing evaluated chunks in REMi context"
    assert remi_ctx.chunks[0].title is not None
    assert "Groundedness" in remi_ctx.chunks[0].title, (
        f"Missing score in title: {remi_ctx.chunks[0].title}"
    )
    assert "**Groundedness:" in remi_ctx.chunks[0].text, (
        "Missing score text inside chunk text"
    )

    if granularity == ContextGranularity.FULL:
        # original text chunks check
        original_chunks = sum(
            len(ctx.chunks) for ctx in question_memory.contexts if ctx.agent != "remi"
        )
        assert len(remi_ctx.chunks) == original_chunks

        orig_ids = {
            c.chunk_id
            for ctx in question_memory.contexts
            if ctx.agent != "remi"
            for c in ctx.chunks
            if c.chunk_id
        }
        for chunk in remi_ctx.chunks:
            assert chunk.chunk_id in orig_ids or "remi_chunk_" in (chunk.chunk_id or "")
    else:
        # One chunk per context evaluated
        ctx_list = [ctx for ctx in question_memory.contexts if ctx.agent != "remi"]
        original_contexts = min(len(ctx_list), 60)
        assert len(remi_ctx.chunks) == original_contexts, (
            f"remi_ctx: {len(remi_ctx.chunks)}, orig: {original_contexts}. ctx_list: {[c.agent for c in ctx_list]}"
        )
        orig_ids = {c.id for c in ctx_list if c.id}
        for chunk in remi_ctx.chunks:
            assert chunk.chunk_id in orig_ids or (chunk.chunk_id or "").startswith(
                "remi_"
            )

    assert "Errors:" not in remi_ctx.summary


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
@pytest.mark.parametrize(
    "granularity", [ContextGranularity.PARTIAL_ANSWERS, ContextGranularity.FULL]
)
async def test_remi_not_enough_data(granularity: ContextGranularity):
    CONFIG["postprocess"][0] = {"module": "remi", "context_granularity": granularity}  # type: ignore
    question_memory = await arag_main(
        agent_id="default",
        internal_nua=False,
        external_nua_api_key=NUA_KEY,
        question=UNANSWERED_QUESTION,
        config=CONFIG,
        loaded_modules=[
            "hyperforge_remi",
            "hyperforge_summarize",
            "hyperforge_nucliadb",
        ],
    )
    assert question_memory.final_answer is not None
    assert "not enough data to answer this" in question_memory.final_answer.lower()
    assert question_memory.is_answered is False

    remi_contexts = [ctx for ctx in question_memory.contexts if ctx.agent == "remi"]
    # If there are context chunks, REMi generates a structured context
    original_chunks = sum(
        len(ctx.chunks) for ctx in question_memory.contexts if ctx.agent != "remi"
    )
    if original_chunks > 0:
        assert len(remi_contexts) > 0, "Missing REMi structured context"
        remi_ctx = remi_contexts[-1]
        assert remi_ctx.title == "REMi Evaluation Breakdown"

        assert remi_ctx.summary is not None
        relevance_match = re.search(r"Answer relevance: (\d+)/5\.", remi_ctx.summary)
        assert relevance_match is not None, remi_ctx.summary
        relevance_score = int(relevance_match.group(1))
        assert relevance_score < 1, f"Relevance score {relevance_score} should be < 1"

        groundedness_match = re.search(
            r"Answer groundedness: (\d+)/5\.", remi_ctx.summary
        )
        assert groundedness_match is not None, remi_ctx.summary
        groundedness_score = int(groundedness_match.group(1))
        assert groundedness_score == 0, (
            f"Groundedness score {groundedness_score} should be 0 when the question is not answered"
        )

        assert len(remi_ctx.chunks) > 0, "Missing evaluated chunks in REMi context"
        assert remi_ctx.chunks[0].title is not None
        assert "Groundedness" in remi_ctx.chunks[0].title, (
            f"Missing score in title: {remi_ctx.chunks[0].title}"
        )
        assert "**Groundedness:" in remi_ctx.chunks[0].text, (
            "Missing score text inside chunk text"
        )

        if granularity == ContextGranularity.FULL:
            assert len(remi_ctx.chunks) == original_chunks
            orig_ids = {
                c.chunk_id
                for ctx in question_memory.contexts
                if ctx.agent != "remi"
                for c in ctx.chunks
                if c.chunk_id
            }
            for chunk in remi_ctx.chunks:
                assert chunk.chunk_id in orig_ids or "remi_chunk_" in (
                    chunk.chunk_id or ""
                )
        else:
            # One chunk per context evaluated
            ctx_list = [ctx for ctx in question_memory.contexts if ctx.agent != "remi"]
            original_contexts = min(len(ctx_list), 60)
            assert len(remi_ctx.chunks) == original_contexts, (
                f"remi_ctx: {len(remi_ctx.chunks)}, orig: {original_contexts}. ctx_list: {[c.agent for c in ctx_list]}"
            )
            orig_ids = {c.id for c in ctx_list if c.id}
            for chunk in remi_ctx.chunks:
                assert chunk.chunk_id in orig_ids or (chunk.chunk_id or "").startswith(
                    "remi_"
                )

        assert "Errors:" not in remi_ctx.summary
