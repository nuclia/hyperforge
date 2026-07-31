from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from nuclia.lib.nua import PredictQueryRequest, PredictRephraseRequest
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
    query = PredictQueryRequest(text="question")
    rephrase = PredictRephraseRequest(question="question", user_id="user")
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
    nua.predict_query.return_value = object()
    nua.predict_rephrase.return_value = object()
    nua.predict_chat_stream.return_value = ("learning-id", "model", stream)
    nua.predict_rerank.return_value = object()
    nua.predict_tokens.return_value = object()

    assert (
        await manager.predict_query(
            query, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.predict_query.return_value
    )
    assert (
        await manager.predict_rephrase(
            rephrase, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.predict_rephrase.return_value
    )
    assert await manager.predict_chat_stream(
        chat, kbid="kb", extra_headers=headers, tracking=tracking
    ) == ("learning-id", "model", stream)
    assert (
        await manager.predict_rerank(
            rerank, kbid="kb", extra_headers=headers, tracking=tracking
        )
        is nua.predict_rerank.return_value
    )
    assert (
        await manager.predict_tokens(
            "text", kbid="kb", model="model", extra_headers=headers, tracking=tracking
        )
        is nua.predict_tokens.return_value
    )

    nua.predict_query.assert_awaited_once_with(
        query, kbid="kb", extra_headers=expected_headers
    )
    nua.predict_rephrase.assert_awaited_once_with(
        rephrase, kbid="kb", extra_headers=expected_headers
    )
    nua.predict_chat_stream.assert_awaited_once_with(
        chat, kbid="kb", extra_headers=expected_headers
    )
    nua.predict_rerank.assert_awaited_once_with(
        rerank, kbid="kb", extra_headers=expected_headers
    )
    nua.predict_tokens.assert_awaited_once_with(
        "text", kbid="kb", model="model", extra_headers=expected_headers
    )


@pytest.mark.asyncio
async def test_manager_aclose_closes_nua_client() -> None:
    manager, nua = manager_with_nua()

    await manager.aclose()

    nua.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_manager_can_omit_rao_origin_header() -> None:
    manager, nua = manager_with_nua(send_rao_origin=False)
    query = PredictQueryRequest(text="question")

    await manager.predict_query(query, extra_headers={"x-test": "value"})

    nua.predict_query.assert_awaited_once_with(
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
    nua.rerank.assert_awaited_once_with(rerank, extra_headers=headers)
    nua.tokens_predict.assert_awaited_once_with(
        text="text", model="model", extra_headers=headers
    )
    assert nua.remi.await_args.kwargs["extra_headers"] == headers
