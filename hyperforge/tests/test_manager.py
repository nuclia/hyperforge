from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from nuclia.lib.nua import GenerateStreamResponse, QueryRequest, RephraseRequest
from nuclia.lib.nua_responses import ChatModel, RerankModel
from nuclia_models.predict.generative_responses import GenerativeChunk

from hyperforge.manager import Manager
from hyperforge.models import TrackingInfo


def manager_with_nua(*, send_rao_origin: bool = True) -> tuple[Manager, AsyncMock]:
    manager = Manager(send_rao_origin=send_rao_origin)
    nua = AsyncMock()
    manager.nua = nua
    return manager, nua


async def empty_stream() -> AsyncIterator[GenerativeChunk]:
    if False:
        yield


@pytest.mark.asyncio
async def test_manager_delegates_kb_aware_predict_calls() -> None:
    manager, nua = manager_with_nua()
    query = QueryRequest(text="question")
    rephrase = RephraseRequest(question="question", user_id="user")
    chat = ChatModel(question="question", user_id="user")
    rerank = RerankModel(question="question", user_id="user", context={"1": "text"})
    headers = {"x-test": "value"}
    tracking = TrackingInfo(rao_id="rao", session="session", message="message")
    expected_headers = {
        "x-show-consumption": "true",
        "x-origin": "RAO",
        "x-client-ident": "rao",
        "x-session": "session",
        "x-message": "message",
        **headers,
    }
    stream = empty_stream()
    nua.query_predict.return_value = object()
    nua.rephrase.return_value = object()
    nua.generate_stream.return_value = GenerateStreamResponse(
        "learning-id", "model", stream
    )
    nua.rerank.return_value = object()
    nua.tokens_predict.return_value = object()

    assert (
        await manager.query_predict(
            query, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.query_predict.return_value
    )
    assert (
        await manager.rephrase(
            rephrase, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.rephrase.return_value
    )
    assert (
        await manager.generate_stream(
            chat, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.generate_stream.return_value
    )
    assert (
        await manager.rerank(
            rerank, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.rerank.return_value
    )
    assert (
        await manager.tokens_predict(
            "text", kbid="kb", model="model", extra_headers=headers, tracking=tracking
        )
        is nua.tokens_predict.return_value
    )

    nua.query_predict.assert_awaited_once_with(
        query, kbid="kb", extra_headers=expected_headers
    )
    nua.rephrase.assert_awaited_once_with(
        rephrase, kbid="kb", extra_headers=expected_headers
    )
    nua.generate_stream.assert_awaited_once_with(
        chat, kbid="kb", extra_headers=expected_headers, return_metadata=True
    )
    nua.rerank.assert_awaited_once_with(
        rerank, kbid="kb", extra_headers=expected_headers
    )
    nua.tokens_predict.assert_awaited_once_with(
        "text", model="model", extra_headers=expected_headers, kbid="kb"
    )


@pytest.mark.asyncio
async def test_manager_aclose_closes_nua_client() -> None:
    manager, nua = manager_with_nua()

    await manager.aclose()

    nua.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_manager_can_omit_rao_origin_header() -> None:
    manager, nua = manager_with_nua(send_rao_origin=False)
    query = QueryRequest(text="question")

    await manager.query_predict(query, extra_headers={"x-test": "value"})

    nua.query_predict.assert_awaited_once_with(
        query,
        kbid=None,
        extra_headers={"x-show-consumption": "true", "x-test": "value"},
    )


@pytest.mark.asyncio
async def test_manager_uses_default_headers_for_existing_nua_calls() -> None:
    manager, nua = manager_with_nua()
    rerank = RerankModel(question="question", user_id="user", context={"1": "text"})

    await manager.rerank(rerank)
    await manager.tokens_predict("text", "model")
    await manager.remi("question", "answer", ["context"])

    headers = {"x-show-consumption": "true", "x-origin": "RAO"}
    nua.rerank.assert_awaited_once_with(rerank, extra_headers=headers, kbid=None)
    nua.tokens_predict.assert_awaited_once_with(
        "text", model="model", extra_headers=headers, kbid=None
    )
    assert nua.remi.await_args.kwargs["extra_headers"] == headers
