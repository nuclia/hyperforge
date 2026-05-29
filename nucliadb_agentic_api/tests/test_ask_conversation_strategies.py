from contextlib import contextmanager
from unittest.mock import patch

from httpx import AsyncClient
from nucliadb_agentic_api.ask.search import ask, rpc
from nucliadb_agentic_api.ask.search.rpc import augment
from nucliadb_models.augment import AugmentedFileField, AugmentRequest, AugmentResponse
from nucliadb_models.retrieval import (
    KeywordScore,
    Metadata,
    RetrievalMatch,
    RetrievalResponse,
    RrfScore,
    Scores,
    ScoreSource,
    ScoreType,
    SemanticScore,
)
from nucliadb_models.search import (
    Image,
    SyncAskResponse,
)
from nucliadb_protos.writer_pb2_grpc import WriterStub
from pytest_mock import MockerFixture

from .fixtures.resources.lambs import lambs_resource


async def test__validate_mocked_text_block_search(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
    mocker: MockerFixture,
):
    """Simple test to validate that we are properly mocking search"""
    kbid = knowledgebox
    rid = await lambs_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    mock_paragraph_id = f"{rid}/c/lambs/10/0-35"

    data: SyncAskResponse
    with mocked_retrieve(result_paragraph_id=mock_paragraph_id):
        # make sure we are enforcing the mock
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "debug": True,
                "reranker": "noop",
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        # validate we are enforcing the mock
        assert data.retrieval_results.best_matches == [mock_paragraph_id], (
            "we are actually enforcing this with a mock"
        )
        assert data.retrieval_results.resources.keys() == {rid}
        assert data.retrieval_results.resources[rid].fields.keys() == {"/c/lambs"}
        assert data.retrieval_results.resources[rid].fields[
            "/c/lambs"
        ].paragraphs.keys() == {mock_paragraph_id}


async def test_ask_default_prompt_context_for_conversations(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
    mocker: MockerFixture,
):
    """Test how default prompt context applies to conversation matches"""

    kbid = knowledgebox
    rid = await lambs_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    data: SyncAskResponse

    mock_paragraph_id = f"{rid}/c/lambs/4/0-26"
    with mocked_retrieve(result_paragraph_id=mock_paragraph_id):
        #
        # TEST: Matching a QUESTION (lambs/4) we'll search an answer and find lambs/5
        #
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "Where are you, Dr. Lecter?",
                "top_k": 1,
                "debug": True,
                "reranker": "noop",
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        # default prompt context doesn't fill augmented context
        augmented = data.augmented_context
        assert augmented is not None
        assert len(augmented.paragraphs) == 0
        # we matched a question and we search and find the answer
        assert data.prompt_context is not None
        assert len(data.prompt_context) == 2
        assert data.predict_request is not None
        assert data.predict_request["query_context"].keys() == {
            f"{rid}/c/lambs/4/0-26",
            f"{rid}/c/lambs/5/0-31",
        }

    mock_paragraph_id = f"{rid}/c/lambs/6/0-80"
    with mocked_retrieve(result_paragraph_id=mock_paragraph_id):
        #
        # TEST: Matching a non-QUESTION (lambs/6) we'll return a bunch of
        # messages following the message
        #
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "debug": True,
                "reranker": "noop",
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        # default prompt context doesn't fill augmented context
        augmented = data.augmented_context
        assert augmented is not None
        assert len(augmented.paragraphs) == 0
        # it actually fills propmt context with a window
        assert data.prompt_context is not None
        assert len(data.prompt_context) == 7
        assert data.predict_request is not None
        assert data.predict_request["query_context"].keys() == {
            f"{rid}/c/lambs/6/0-80",
            f"{rid}/c/lambs/7/0-191",
            f"{rid}/c/lambs/8/0-12",
            f"{rid}/c/lambs/9/0-130",
            f"{rid}/c/lambs/10/0-35",
            f"{rid}/c/lambs/11/0-72",
            f"{rid}/c/lambs/12/0-28",
        }


async def test_ask_conversational_strategy__full_conversation(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
    mocker: MockerFixture,
):
    """Test conversational strategy full=True"""

    kbid = knowledgebox
    rid = await lambs_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    data: SyncAskResponse

    mock_paragraph_id = f"{rid}/c/lambs/10/0-35"
    with mocked_retrieve(result_paragraph_id=mock_paragraph_id):
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "debug": True,
                "reranker": "noop",  # /ask coupled doesn't work with reranker scores
                "rag_strategies": [
                    {
                        "name": "conversation",
                        "full": True,
                        "max_messages": 1,
                        "attachments_text": False,
                        "attachments_images": False,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        augmented = data.augmented_context
        assert augmented is not None
        assert data.prompt_context is not None
        assert len(augmented.paragraphs) == 11
        assert (
            augmented.paragraphs[f"{rid}/c/lambs/4/0-26"].text
            == "Where are you, Dr. Lecter?"
        )
        assert len(data.prompt_context) == 12
        assert [m.id for m in data.retrieval_best_matches] == [f"{rid}/c/lambs/10/0-35"]
        assert (
            data.retrieval_results.resources[rid]
            .fields["/c/lambs"]
            .paragraphs[f"{rid}/c/lambs/10/0-35"]
            .id
            == f"{rid}/c/lambs/10/0-35"
        )


async def test_ask_conversational_strategy__max_messages(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
    mocker: MockerFixture,
):
    """Test conversational strategy max_messages"""

    kbid = knowledgebox
    rid = await lambs_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    data: SyncAskResponse

    mock_paragraph_id = f"{rid}/c/lambs/10/0-35"
    with mocked_retrieve(result_paragraph_id=mock_paragraph_id):
        #
        # TEST: max_messages=1 will return the first conversation message (that
        # nobody asked for) and the matched split and
        #
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "reranker": "noop",
                "rag_strategies": [
                    {
                        "name": "conversation",
                        "full": False,
                        "max_messages": 1,
                        "attachments_text": False,
                        "attachments_images": False,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        augmented = data.augmented_context
        assert augmented is not None
        assert augmented.paragraphs.keys() == {f"{rid}/c/lambs/1/0-9"}

        #
        # TEST: max_messages=3 will return the first conversation message (that
        # nobody asked for), the matched split and a window surrounding it
        #
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "reranker": "noop",
                "rag_strategies": [
                    {
                        "name": "conversation",
                        "full": False,
                        "max_messages": 3,
                        "attachments_text": False,
                        "attachments_images": False,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        augmented = data.augmented_context
        assert augmented is not None
        assert augmented.paragraphs.keys() == {
            f"{rid}/c/lambs/1/0-9",
            f"{rid}/c/lambs/9/0-130",
            f"{rid}/c/lambs/11/0-72",
        }

        #
        # TEST: with max_messages exceeding, we'll get an window with an offset
        #
        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "reranker": "noop",
                "rag_strategies": [
                    {
                        "name": "conversation",
                        "full": False,
                        "max_messages": 7,
                        "attachments_text": False,
                        "attachments_images": False,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data = SyncAskResponse.model_validate_json(resp.content)

        augmented = data.augmented_context
        assert augmented is not None
        assert augmented.paragraphs.keys() == {
            f"{rid}/c/lambs/1/0-9",
            f"{rid}/c/lambs/6/0-80",
            f"{rid}/c/lambs/7/0-191",
            f"{rid}/c/lambs/8/0-12",
            f"{rid}/c/lambs/9/0-130",
            f"{rid}/c/lambs/11/0-72",
            f"{rid}/c/lambs/12/0-28",
        }


async def test_ask_conversational_strategy__attachments(
    nucliadb_writer: AsyncClient,
    nucliadb_ingest_grpc: WriterStub,
    nucliadb_search: AsyncClient,
    knowledgebox: str,
    mocker: MockerFixture,
):
    """Test conversational strategy text and image attachments"""

    kbid = knowledgebox
    rid = await lambs_resource(kbid, nucliadb_writer, nucliadb_ingest_grpc)

    async def mocked_augment(sdk, kbid, item: AugmentRequest):
        if item.fields and any((augment.file_thumbnail for augment in item.fields)):
            return AugmentResponse(
                resources={},
                fields={
                    f"{rid}/f/attachment:lamb": AugmentedFileField(
                        text="A beautiful picture of some happy lambs in the middle of a green field",
                        thumbnail_image="path/to/attachment:lamb",
                    ),
                    f"{rid}/f/attachment:blue-suit": AugmentedFileField(
                        text="Clarice's blue suit",
                        thumbnail_image="path/to/attachment:blue-suit",
                    ),
                },
                paragraphs={},
            )

        else:
            # fallback to original augment API call
            return await augment(rpc.get_sdk("search"), kbid, item)

    image = Image(b64encoded="my image", content_type="image/jpeg")

    with (
        patch(
            "nuclia_arag_ask.search.rpc.augment",
            side_effect=mocked_augment,
        ),
        patch("nuclia_arag_ask.search.rpc.download_image", return_value=image),
    ):
        spy = mocker.spy(ask, "get_answer_stream")

        resp = await nucliadb_search.post(
            f"/kb/{kbid}/ask",
            headers={"x-synchronous": "true"},
            json={
                "query": "You know I can't make that promise.",
                "top_k": 1,
                "reranker": "noop",
                "rag_strategies": [
                    {
                        "name": "conversation",
                        "full": True,
                        "max_messages": 1,
                        "attachments_text": True,
                        "attachments_images": True,
                    },
                ],
            },
        )
        assert resp.status_code == 200
        data: SyncAskResponse = SyncAskResponse.model_validate_json(resp.content)

        augmented = data.augmented_context
        assert augmented is not None
        assert len(augmented.paragraphs) == 13
        assert {
            f"{rid}/f/attachment:blue-suit/0-19",
            f"{rid}/f/attachment:lamb/0-70",
        }.issubset(augmented.paragraphs.keys())

        assert (
            augmented.paragraphs[f"{rid}/f/attachment:blue-suit/0-19"].text
            == "Attachment attachment:blue-suit: Clarice's blue suit\n\n"
        )

        # images are sent to the LLM but not returned, we spy the method
        # that sends to Predict API to validate we are sending the context
        assert spy.call_count == 1
        assert spy.call_args.kwargs["item"].query_context_images == {
            f"{rid}/f/attachment:lamb/0-0": image,
            f"{rid}/f/attachment:blue-suit/0-0": image,
        }
        del spy


@contextmanager
def mocked_retrieve(*, result_paragraph_id: str):
    # FIXME(sc-13624): due to a bug in conversation field indexing, search is
    # not reliable. Thus, we mock here the results of a search to the index to
    # control the search results and properly test the RAG strategy

    async def mock_retrieve(sdk, kbid: str, item):
        nonlocal result_paragraph_id

        return RetrievalResponse(
            matches=[
                RetrievalMatch(
                    id=result_paragraph_id,
                    score=Scores(
                        value=0.05693472,
                        source=ScoreSource.RANK_FUSION,
                        type=ScoreType.RRF,
                        history=[
                            SemanticScore(
                                score=1.47892374982375,
                                source=ScoreSource.INDEX,
                                type=ScoreType.SEMANTIC,
                            ),
                            KeywordScore(
                                score=3.1989123821258545,
                                source=ScoreSource.INDEX,
                                type=ScoreType.KEYWORD,
                            ),
                            RrfScore(
                                score=0.05693472,
                                source=ScoreSource.RANK_FUSION,
                                type=ScoreType.RRF,
                            ),
                        ],
                    ),
                    metadata=Metadata(
                        field_labels=[],
                        paragraph_labels=[
                            "/l/art/film",
                            "/l/genre/thriller",
                            "/l/genre/horror",
                            "/l/genre/psychological",
                        ],
                        is_an_image=False,
                        is_a_table=False,
                        source_file=None,
                        page=None,
                        in_page_with_visual=None,
                    ),
                )
            ]
        )

    with patch("nuclia_arag_ask.search.rpc.retrieve", side_effect=mock_retrieve):
        yield
