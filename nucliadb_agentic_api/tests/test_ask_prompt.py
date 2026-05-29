import base64
from unittest import mock
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient
from nucliadb_agentic_api.ask.search import (
    prompt as chat_prompt,
)
from nucliadb_agentic_api.ask.search import rpc
from nucliadb_agentic_api.ask.search.metrics import Metrics
from nucliadb_agentic_api.ask.utils.ids import ParagraphId
from nucliadb_models.augment import AugmentedParagraph, AugmentRequest, AugmentResponse
from nucliadb_models.search import (
    SCORE_TYPE,
    AugmentedContext,
    FindField,
    FindParagraph,
    FindResource,
    HierarchyResourceStrategy,
    Image,
    KnowledgeboxFindResults,
    MinScore,
    PageImageStrategy,
    ParagraphImageStrategy,
    TableImageStrategy,
)
from nucliadb_protos import resources_pb2 as rpb2


@pytest.fixture()
def messages():
    msgs = [
        rpb2.Message(ident="1", content=rpb2.MessageContent(text="Message 1")),
        rpb2.Message(ident="2", content=rpb2.MessageContent(text="Message 2")),
        rpb2.Message(
            ident="3",
            who="1",
            content=rpb2.MessageContent(text="Message 3"),
            type=rpb2.Message.MessageType.QUESTION,
        ),
        rpb2.Message(
            ident="4",
            content=rpb2.MessageContent(text="Message 4"),
            type=rpb2.Message.MessageType.ANSWER,
            to=["1"],
        ),
        rpb2.Message(ident="5", content=rpb2.MessageContent(text="Message 5")),
    ]
    yield msgs


@pytest.fixture()
def field_obj(messages):
    mock = AsyncMock()
    mock.get_metadata.return_value = rpb2.FieldConversation(pages=1, total=5)
    mock.db_get_value.return_value = rpb2.Conversation(messages=messages)

    yield mock


@pytest.fixture()
def kb(field_obj):
    mock = AsyncMock()
    mock.get.return_value.get_field.return_value = field_obj
    yield mock


def get_ordered_paragraphs(find_results):
    paragraphs = []
    for resource in find_results.resources.values():
        for field in resource.fields.values():
            paragraphs.extend(field.paragraphs.values())
    paragraphs.sort(key=lambda p: p.order)
    return paragraphs


def _create_find_result(
    paragraph: FindParagraph,
):
    pid = ParagraphId.from_string(paragraph.id)
    rid = pid.rid
    fid = f"{pid.field_id.type}/{pid.field_id.key}"
    return FindResource(
        id=rid,
        fields={
            fid: FindField(
                paragraphs={
                    pid.full(): paragraph,
                }
            )
        },
    )


async def test_default_prompt_context(kb) -> None:
    # TODO: this was a unit test for old prompt, we may want to test something
    # with the new implementation
    pass


@pytest.fixture(scope="function")
def find_results():
    return KnowledgeboxFindResults(
        facets={},
        resources={
            "resource1": _create_find_result(
                FindParagraph(
                    id="resource1/a/title/0-10",
                    score=1,
                    score_type=SCORE_TYPE.BOTH,
                    order=1,
                    text="Resource 1",
                )
            ),
            "resource2": _create_find_result(
                FindParagraph(
                    id="resource2/a/title/0-10",
                    score=2,
                    score_type=SCORE_TYPE.VECTOR,
                    order=2,
                    text="Resource 2",
                )
            ),
        },
        min_score=MinScore(semantic=-1),
    )


async def test_prompt_context_builder_prepends_user_context(
    find_results: KnowledgeboxFindResults,
) -> None:
    builder = chat_prompt.PromptContextBuilder(
        rpc.get_sdk("search"),
        rpc.get_sdk("reader"),
        kbid="kbid",
        ordered_paragraphs=get_ordered_paragraphs(find_results),
        user_context=["Carrots are orange"],
    )

    async def _mock_build_context(context, *args, **kwargs):
        context["resource1/a/title"] = "Resource 1"
        context["resource2/a/title"] = "Resource 2"

    with mock.patch.object(builder, "_build_context", new=_mock_build_context):
        context, context_order, image_context, augmented_context = await builder.build()
        assert len(context) == 3
        assert len(context_order) == 3
        assert len(image_context) == 0
        assert context["USER_CONTEXT_0"] == "Carrots are orange"
        assert context["resource1/a/title"] == "Resource 1"
        assert context["resource2/a/title"] == "Resource 2"
        assert context_order["USER_CONTEXT_0"] == 0
        assert context_order["resource1/a/title"] == 1
        assert context_order["resource2/a/title"] == 2


def test_capped_prompt_context() -> None:
    context = chat_prompt.CappedPromptContext(max_size=2)

    # Check that output is trimmed
    context["key1"] = "123"

    context.cap()
    assert context.output == {"key1": "12"}
    assert context.size == 2

    # Check that is trimmed from the last added key
    context = chat_prompt.CappedPromptContext(max_size=2)
    context["key1"] = "12"
    context["key2"] = "34"
    context.cap()
    assert context.output == {"key1": "12"}

    # Update existing value
    context["key1"] = "foobar"
    context.cap()
    assert context.output == {"key1": "fo"}
    assert context.size == 2

    # Check text block ids
    assert context.text_block_ids() == ["key1"]

    # Check without limits
    context = chat_prompt.CappedPromptContext(max_size=None)
    context["key1"] = "foo" * int(1e6)

    context.cap()
    assert context.output == {"key1": "foo" * int(1e6)}
    assert context.size == int(3e6)

    # Check that the size is updated correctly upon deletion
    del context["key1"]
    assert context.size == 0

    # Deletion of non-existing key should not raise an error
    del context["key1337"]
    assert context.size == 0


async def test_hierarchy_prompt_context(nucliadb_search: AsyncClient, kb):
    rid = uuid4().hex

    async def mocked_augment(
        sdk, kbid: str, request: AugmentRequest
    ) -> AugmentResponse:
        return AugmentResponse(
            resources={},
            fields={},
            paragraphs={
                f"{rid}/f/f1/0-10": AugmentedParagraph(text="First paragraph text"),
                f"{rid}/a/title/0-500": AugmentedParagraph(text="Title text"),
                f"{rid}/a/summary/0-1000": AugmentedParagraph(text="Summary text"),
                f"{rid}/f/f1/10-20": AugmentedParagraph(text="Second paragraph text"),
            },
        )

    with (
        mock.patch(
            "nuclia_arag_ask.search.prompt.rpc.augment", side_effect=mocked_augment
        ) as augment,
    ):
        context = chat_prompt.CappedPromptContext(max_size=int(1e6))
        find_results = KnowledgeboxFindResults(
            resources={
                f"{rid}": FindResource(
                    id=f"{rid}",
                    fields={
                        "f/f1": FindField(
                            paragraphs={
                                f"{rid}/f/f1/0-10": FindParagraph(
                                    id=f"{rid}/f/f1/0-10",
                                    score=10,
                                    score_type=SCORE_TYPE.BM25,
                                    order=0,
                                    text="First paragraph text",
                                ),
                                f"{rid}/f/f1/10-20": FindParagraph(
                                    id=f"{rid}/f/f1/10-20",
                                    score=8,
                                    score_type=SCORE_TYPE.BM25,
                                    order=1,
                                    text="Second paragraph text",
                                ),
                            }
                        )
                    },
                )
            },
        )
        ordered_paragraphs = get_ordered_paragraphs(find_results)
        augmented_context = AugmentedContext()
        await chat_prompt.hierarchy_prompt_context(
            rpc.get_sdk("search"),
            context,
            "kbid",
            ordered_paragraphs,
            HierarchyResourceStrategy(),
            Metrics("foo"),
            augmented_context=augmented_context,
        )

        assert augment.call_count == 1
        assert augment.call_args.args[1] == "kbid"
        assert set(
            (
                paragraph.id
                for paragraph in augment.call_args.args[2].paragraphs[0].given
            )
        ) == {
            f"{rid}/f/f1/0-10",
            f"{rid}/a/title/0-500",
            f"{rid}/a/summary/0-1000",
            f"{rid}/f/f1/10-20",
        }

        assert (
            context.output[f"{rid}/f/f1/0-10"]
            == "DOCUMENT: Title text \n SUMMARY: Summary text \n RESOURCE CONTENT: \n EXTRACTED BLOCK: \n First paragraph text \n\n \n EXTRACTED BLOCK: \n Second paragraph text"
        )
        # Chec that the original text of the paragraphs is preserved
        assert ordered_paragraphs[0].text == "First paragraph text"
        assert ordered_paragraphs[1].text == "Second paragraph text"

        assert augmented_context.paragraphs[f"{rid}/f/f1/0-10"].id == f"{rid}/f/f1/0-10"
        assert augmented_context.paragraphs[f"{rid}/f/f1/0-10"].text.startswith(
            "DOCUMENT: Title"
        )
        assert (
            augmented_context.paragraphs[f"{rid}/f/f1/0-10"].augmentation_type
            == "hierarchy"
        )


async def test_prompt_context_image_context_builder() -> None:
    result_text = " ".join(["text"] * 10)
    find_results = KnowledgeboxFindResults(
        facets={},
        resources={
            "bmid": _create_find_result(
                FindParagraph(
                    id="bmid/f/file/0-1",
                    score=1,
                    score_type=SCORE_TYPE.BM25,
                    order=1,
                    text=result_text,
                    is_a_table=True,
                    reference="table_image_data",
                    page_with_visual=False,
                )
            ),
            "vecid": _create_find_result(
                FindParagraph(
                    id="vecid/f/file/0-1",
                    score=0,
                    score_type=SCORE_TYPE.VECTOR,
                    order=2,
                    text=result_text,
                    is_a_table=False,
                    reference="paragraph_image_data",
                    page_with_visual=False,
                )
            ),
            "both_id": _create_find_result(
                FindParagraph(
                    id="both_id/f/file/0-1",
                    score=2,
                    score_type=SCORE_TYPE.BOTH,
                    order=0,
                    text=result_text,
                    is_a_table=False,
                    reference="page_image_data",
                    page_with_visual=True,
                )
            ),
        },
    )

    # By default, no image strategies are provided so no images should be added
    builder = chat_prompt.PromptContextBuilder(
        rpc.get_sdk("search"),
        rpc.get_sdk("reader"),
        kbid="kbid",
        ordered_paragraphs=get_ordered_paragraphs(find_results),
        user_context=["Carrots are orange"],
        image_strategies=[],
    )
    context = chat_prompt.CappedPromptContext(max_size=int(1e6))
    await builder._build_context_images(context)
    assert len(context.images) == 0

    # Test that the image strategies are applied correctly
    builder = chat_prompt.PromptContextBuilder(
        rpc.get_sdk("search"),
        rpc.get_sdk("reader"),
        kbid="kbid",
        ordered_paragraphs=get_ordered_paragraphs(find_results),
        user_context=["Carrots are orange"],
        image_strategies=[
            PageImageStrategy(count=10),
            TableImageStrategy(),
            ParagraphImageStrategy(),
        ],
    )
    with (
        mock.patch(
            "nuclia_arag_ask.search.prompt.get_paragraph_page_number", return_value=1
        ),
        mock.patch(
            "nuclia_arag_ask.search.prompt.rpc.download_image",
            return_value=Image(b64encoded="an-image", content_type="image/png"),
        ),
    ):
        context = chat_prompt.CappedPromptContext(max_size=int(1e6))
        await builder._build_context_images(context)
        assert len(context.output) == 0
        assert len(context.images) == 6
        assert set(context.images.keys()) == {
            # The paragraph images
            "bmid/f/file/0-1",
            "both_id/f/file/0-1",
            "vecid/f/file/0-1",
            # The page images
            "bmid/f/file/1",
            "both_id/f/file/1",
            "vecid/f/file/1",
        }


async def test_prompt_context_builder_with_extra_image_context() -> None:
    image_content = base64.b64encode(b"my-image")
    user_image = Image(content_type="image/png", b64encoded=image_content)

    builder = chat_prompt.PromptContextBuilder(
        rpc.get_sdk("search"),
        rpc.get_sdk("reader"),
        kbid="kbid",
        ordered_paragraphs=[],
        user_image_context=[user_image],
    )
    with patch("nuclia_arag_ask.search.prompt.default_prompt_context"):
        # context = chat_prompt.CappedPromptContext(max_size=int(1e6))
        _, _, context_images, _ = await builder.build()

    assert len(context_images) == 1
    _, context_image = context_images.popitem()
    assert context_image == user_image


async def test_prompt_context_builder_with_query_image() -> None:
    image_content = base64.b64encode(b"my-image").decode("utf-8")
    query_image = Image(content_type="image/png", b64encoded=image_content)
    user_image = Image(content_type="image/jpg", b64encoded=image_content)

    builder = chat_prompt.PromptContextBuilder(
        rpc.get_sdk("search"),
        rpc.get_sdk("reader"),
        kbid="kbid",
        ordered_paragraphs=[],
        user_image_context=[user_image],
        query_image=query_image,
    )

    with patch("nuclia_arag_ask.search.prompt.default_prompt_context"):
        # context = chat_prompt.CappedPromptContext(max_size=int(1e6))
        _, _, context_images, _ = await builder.build()

    # User image should not be included in the context images if a query image is provided
    assert len(context_images) == 1
    _, context_image = context_images.popitem()
    assert context_image == query_image


def test_get_neighbouring_indices() -> None:
    field_pids = [ParagraphId.from_string(f"r1/f/f1/0-{i}") for i in range(10)]
    index = 5

    for (before, after), expected in [
        ((0, 0), []),
        ((1, 0), [4]),
        ((0, 1), [6]),
        ((1, 1), [4, 6]),
        ((2, 0), [3, 4]),
        ((0, 2), [6, 7]),
        ((2, 2), [3, 4, 6, 7]),
        ((100, 100), [0, 1, 2, 3, 4, 6, 7, 8, 9]),
    ]:
        assert (
            chat_prompt.get_neighbouring_indices(index, before, after, field_pids)
            == expected
        )
