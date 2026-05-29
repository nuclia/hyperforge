import base64

from httpx import AsyncClient
from nucliadb_models.search import (
    SyncAskResponse,
)
from nucliadb_protos.writer_pb2_grpc import WriterStub

from .fixtures.resources import cookie_tale_resource


async def test_ask_paragraph_image_rag_strategy(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
):
    kbid = knowledgebox
    rid = await cookie_tale_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    resp = await nucliadb_search.post(
        f"/kb/{kbid}/ask",
        headers={"x-synchronous": "true"},
        json={
            "query": "A yummy image of some cookies",
            "top_k": 1,
            "reranker": "noop",
            "debug": True,
            "rag_images_strategies": [
                {
                    "name": "paragraph_image",
                },
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = SyncAskResponse.model_validate_json(resp.content)
    assert data.predict_request is not None
    assert (
        f"{rid}/f/cookie-recipie/0-29" in data.predict_request["query_context_images"]
    )
    assert (
        base64.b64decode(
            data.predict_request["query_context_images"][
                f"{rid}/f/cookie-recipie/0-29"
            ]["b64encoded"]
        )
        == b"delicious cookies image"
    )
    assert (
        data.predict_request["query_context_images"][f"{rid}/f/cookie-recipie/0-29"][
            "content_type"
        ]
        == "image/png"
    )


async def test_ask_page_image_rag_strategy(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
):
    kbid = knowledgebox
    rid = await cookie_tale_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    resp = await nucliadb_search.post(
        f"/kb/{kbid}/ask",
        headers={"x-synchronous": "true"},
        json={
            "query": "A yummy image of some cookies",
            "reranker": "noop",
            "top_k": 1,
            "debug": True,
            "rag_images_strategies": [
                {
                    "name": "page_image",
                    "count": 2,
                },
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = SyncAskResponse.model_validate_json(resp.content)
    assert data.predict_request is not None
    assert f"{rid}/f/cookie-recipie/0" in data.predict_request["query_context_images"]
    assert (
        base64.b64decode(
            data.predict_request["query_context_images"][f"{rid}/f/cookie-recipie/0"][
                "b64encoded"
            ]
        )
        == b"A page with an image of cookies"
    )
    assert (
        data.predict_request["query_context_images"][f"{rid}/f/cookie-recipie/0"][
            "content_type"
        ]
        == "image/png"
    )


async def test_ask_table_image_rag_strategy(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
):
    kbid = knowledgebox
    rid = await cookie_tale_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    resp = await nucliadb_search.post(
        f"/kb/{kbid}/ask",
        headers={"x-synchronous": "true"},
        json={
            "query": "Ingredient: peanut butter",
            "reranker": "noop",
            "top_k": 1,
            "debug": True,
            "rag_images_strategies": [
                {
                    "name": "tables",
                },
            ],
        },
    )
    assert resp.status_code == 200, resp.text
    data = SyncAskResponse.model_validate_json(resp.content)
    assert data.predict_request is not None
    assert (
        f"{rid}/f/cookie-recipie/29-75" in data.predict_request["query_context_images"]
    )
    assert (
        base64.b64decode(
            data.predict_request["query_context_images"][
                f"{rid}/f/cookie-recipie/29-75"
            ]["b64encoded"]
        )
        == b"ingredients table"
    )
    assert (
        data.predict_request["query_context_images"][f"{rid}/f/cookie-recipie/29-75"][
            "content_type"
        ]
        == "image/png"
    )
