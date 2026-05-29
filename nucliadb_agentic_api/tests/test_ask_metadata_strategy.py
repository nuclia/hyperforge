import pytest
from httpx import AsyncClient
from nucliadb_models.search import (
    SyncAskResponse,
)
from nucliadb_protos import resources_pb2
from nucliadb_protos.writer_pb2 import BrokerMessage
from nucliadb_protos.writer_pb2_grpc import WriterStub

from .utils import inject_message
from .utils.broker_messages import BrokerMessageBuilder
from .utils.dirty_index import wait_for_sync


async def test_ask_rag_strategy_metadata_extension(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_reader: AsyncClient,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
    metadata_resource: str,
):
    kbid = knowledgebox

    resp = await nucliadb_search.post(
        f"/kb/{kbid}/ask",
        json={
            "query": "title",
            "rag_strategies": [
                {
                    "name": "metadata_extension",
                    "types": [
                        "origin",
                        "extra_metadata",
                        "classification_labels",
                        "ners",
                    ],
                }
            ],
            "debug": True,
        },
        headers={"X-Synchronous": "True"},
    )
    assert resp.status_code == 200, resp.text
    ask_response = SyncAskResponse.model_validate_json(resp.content)
    assert ask_response.prompt_context is not None

    # Make sure the text blocks of the context are extended with the metadata
    origin_found = False
    classification_labels_found = False
    ners_found = False
    extra_found = False
    for text_block in ask_response.prompt_context:
        if "DOCUMENT METADATA AT ORIGIN" in text_block:
            origin_found = True
            assert "https://example.com/" in text_block
            assert "collaborator_" in text_block
        if "DOCUMENT EXTRA METADATA" in text_block:
            extra_found = True
            assert "metadata:\n  bar: baz" in text_block
        if "DOCUMENT CLASSIFICATION LABELS" in text_block:
            classification_labels_found = True
            # resource classification
            assert "- rs-0 (ls)" in text_block
            # field classifications
            assert "- book (object)" in text_block
            assert "- computer (object)" in text_block
        if "DOCUMENT NAMED ENTITIES (NERs)" in text_block:
            # we can't easily mock this for the coupled ask
            ners_found = True
            assert "- PLACE" in text_block
            assert "  - Amsterdam" in text_block
            assert "  - Paris" in text_block

    assert origin_found, ask_response.prompt_context
    assert extra_found, ask_response.prompt_context
    assert classification_labels_found, ask_response.prompt_context
    assert ners_found, ask_response.prompt_context

    # Try now combining metadata_extension with another strategy
    for strategy in [
        {"name": "full_resource"},
        {"name": "neighbouring_paragraphs", "before": 1, "after": 1},
        {"name": "hierarchy", "count": 40},
        {"name": "field_extension", "fields": ["a/title", "a/summary"]},
    ]:
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            json={
                "query": "title",
                "rag_strategies": [
                    {"name": "metadata_extension", "types": ["origin"]},
                    strategy,
                ],
                "debug": True,
            },
            headers={"X-Synchronous": "True"},
        )
        assert resp.status_code == 200, resp.text
        ask_response = SyncAskResponse.model_validate_json(resp.content)
        assert ask_response.prompt_context is not None

        # Make sure the text blocks of the context are extended with the metadata
        origin_found = False
        for text_block in ask_response.prompt_context:
            if "DOCUMENT METADATA AT ORIGIN" in text_block:
                origin_found = True
                assert "https://example.com/" in text_block
                assert "collaborator_" in text_block
        assert origin_found, ask_response.prompt_context


@pytest.fixture
async def metadata_resource(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    knowledgebox: str,
):
    kbid = knowledgebox

    resp = await nucliadb_writer.post(
        f"/kb/{kbid}/resources",
        json={
            "slug": "metadata-resource",
            "title": "The title",
            "summary": "The summary",
            "texts": {"text_field": {"body": "The body of the text field"}},
            "origin": {
                "url": "https://example.com/",
                "collaborators": ["collaborator_0", "collaborator_1"],
                "metadata": {"foo": "bar"},
            },
            "extra": {"metadata": {"bar": "baz"}},
            "usermetadata": {"classifications": [{"labelset": "ls", "label": "rs-0"}]},
        },
    )
    assert resp.status_code == 201
    rid = resp.json()["uuid"]

    bmb = BrokerMessageBuilder(
        kbid=kbid,
        rid=rid,
        slug="metadata-resource",
        source=BrokerMessage.MessageSource.PROCESSOR,
    )

    title_field = bmb.with_title("Metadata resource title")
    bmb.with_summary("Metadata resource summary")

    title_field._extracted_metadata.metadata.metadata.classifications.extend(
        [
            resources_pb2.Classification(labelset="object", label="computer"),
            resources_pb2.Classification(labelset="object", label="book"),
        ]
    )

    title_field._extracted_metadata.metadata.metadata.entities["PLACE"].entities.extend(
        [
            resources_pb2.FieldEntity(label="PLACE", text="Amsterdam"),
            resources_pb2.FieldEntity(label="PLACE", text="Paris"),
        ]
    )

    # build the broker message

    bm = bmb.build()

    # customize fields we don't want to overwrite from the writer BM
    bm.origin.url = "https://example.com/"
    bm.origin.colaborators.extend(["collaborator_0", "collaborator_1"])
    bm.origin.metadata["foo"] = "bar"

    # ingest the processed BM
    await inject_message(nucliadb_ingest_grpc, bm)
    await wait_for_sync()

    yield rid
