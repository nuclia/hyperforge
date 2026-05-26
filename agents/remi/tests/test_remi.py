import re

import pytest
from hyperforge.engine import main as arag_main
from tests.arag import NUA_KEY

from agents.remi.src.hyperforge_remi.config import ContextGranularity

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
                "key": "eyJhbGciOiJSUzI1NiIsImtpZCI6InNhIiwidHlwIjoiSldUIn0.eyJpc3MiOiJodHRwczovL2V1cm9wZS0xLm51Y2xpYS5jbG91ZC8iLCJpYXQiOjE3NTkyNTQ3NjQsInN1YiI6ImUwNGUwMzcyLTYwNDgtNDY5ZC04NWExLWI1ZTM1MjBmMzdlZiIsImp0aSI6IjgzMzRhY2NlLTIwMTUtNGY0MS05M2U5LTczNzk1MTg2NDdiZiIsImV4cCI6MTc5MDc5MDc2MCwia2V5IjoiMmYxMzYyNTItNjNiMy00NzA1LTg1MjQtMDhmYWJjOWUzMjUyIiwia2lkIjoiN2RmMzY2NDctOTdiOC00NzU0LWExNjUtZWZkY2ZlMDRkMzI2In0.kwHAfx9RRTI-G3S64X0iisr0iAyXRKNRhnN4C67MkLSxeu1AOAnVV8EIQuu4jpXW7O4FkSsthFXEv9ZxlRRh_CaS0z_TjPzIzDPeE6eIKskZ70Q7c-pDe949WE9DZiDyy9_dwKsdX5cnvYpKorp0ROm-GvRXrdHaTZKDSYWht3gvEtm6-0j9C1gx2BzKr2coizUAIde_qjSpLOojO4S-k8P8I9dsQFagdcrjxgGWgrAzjhAs_qkqlRmP0QP6S7ToN0nrbHmtKKb0lWmcpVvlAfH95CM20YUs7IAqU_t7-_V6mm43FstRgGeiHkoapo8nPVJtXMBSlaM7GSz0Kxf2TWQwi94mTEQLdA8CblX0skMCfIHFwbcbm1Vf-2C6LywAsSmTYAwsVPpqeQcVZdrfLMhddCjZKUFCNLSurCSb4TuN79GZicPCJDT-VEBMlNH8ayHOyRib5RyqvgXUwGN9zyM-ma7RrVk4eEwSk7923bn_9GTk-s5tYw_exbYsQ1Qa84GA6NzgJ_kNQmgJwb2zW1V5ddCpYd5k6lNEdPRk0JQKlCC2zTmSvnRcLxfDPi4SZFdLLdtG0j2hIl_QNTEC_3VtqJds4FMofy7TkmUObdbEmXjdAsOxkqj2ntGOsaBNiCI_w47BbPvG_V1LsBHDrrIo0Wo1fgAhUbtWV7Dd5J4",
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
