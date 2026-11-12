"""Unit tests for convert_arag_answer_to_content."""

import json

import pytest
from mcp.types import EmbeddedResource, ImageContent, TextContent, TextResourceContents
from nuclia.lib.nua_responses import Image

from hyperforge.api.v1.mcp_content import convert_arag_answer_to_content
from hyperforge.interaction import AnswerOperation, AragAnswer, ARAGException
from hyperforge.models import (
    Answer,
    AnswerCitations,
    Chunk,
    CitationMetadata,
    Context,
    Step,
    VegaLiteVisualization,
)

# ============================================================================
# Fixtures & Helpers
# ============================================================================


def make_answer(**kwargs) -> AragAnswer:
    """Create an AragAnswer with defaults."""
    return AragAnswer(**{"operation": AnswerOperation.ANSWER, **kwargs})


def find_by_uri_pattern(result, pattern: str) -> list:
    """Find EmbeddedResources matching a URI pattern."""
    return [
        c
        for c in result
        if isinstance(c, EmbeddedResource) and pattern in str(c.resource.uri)
    ]


def find_text_blocks(result) -> list[TextContent]:
    """Find all TextContent blocks."""
    return [c for c in result if isinstance(c, TextContent)]


def find_image_blocks(result) -> list[ImageContent]:
    """Find all ImageContent blocks."""
    return [c for c in result if isinstance(c, ImageContent)]


@pytest.fixture
def minimal_chunk():
    """A chunk with only required fields."""
    return Chunk(chunk_id="c1", text="Chunk text")


@pytest.fixture
def rich_chunk():
    """A chunk with all metadata fields."""
    return Chunk(
        chunk_id="c1",
        text="Chunk text",
        title="Doc title",
        source="my-source",
        labels=["label1"],
        origin_url="https://origin.com",
    )


@pytest.fixture
def minimal_context(minimal_chunk):
    """A context with only required fields."""
    return Context(
        id="ctx-id",
        original_question_uuid=None,
        actual_question_uuid=None,
        question="Q",
        chunks=[minimal_chunk],
        source="s",
        agent="a",
    )


@pytest.fixture
def rich_context(rich_chunk):
    """A context with all fields populated."""
    return Context(
        id="ctx-id",
        original_question_uuid="q1",
        actual_question_uuid="q1",
        question="What?",
        chunks=[rich_chunk],
        images={"img-1": Image(content_type="image/png", b64encoded="aGVsbG8=")},
        structured=['{"key": "value"}'],
        image_urls=["https://example.com/img.png"],
        source="nucliadb",
        agent="test-agent",
    )


# ---------------------------------------------------------------------------
# Basic answer text
# ---------------------------------------------------------------------------


def test_empty_answer_returns_empty_list():
    msg = make_answer()
    assert convert_arag_answer_to_content(msg) == []


def test_plain_text_answer():
    msg = make_answer(answer="Hello world")
    result = convert_arag_answer_to_content(msg)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "Hello world"


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def test_answer_with_citations():
    citations = AnswerCitations(
        metadata={
            "block-1": CitationMetadata(
                context_id="ctx1",
                origin_urls=["https://example.com"],
                chunk_index=0,
            )
        }
    )
    msg = make_answer(answer="Answer", answer_citations=citations)
    result = convert_arag_answer_to_content(msg)

    assert len(result) == 2
    assert isinstance(result[0], TextContent)
    embedded = result[1]
    assert isinstance(embedded, EmbeddedResource)
    assert isinstance(embedded.resource, TextResourceContents)
    assert embedded.resource.mimeType == "application/json"
    assert str(embedded.resource.uri) == "rao-response://citations"
    data = json.loads(embedded.resource.text)
    assert "metadata" in data
    assert "block-1" in data["metadata"]


def test_empty_citations_metadata_not_included():
    citations = AnswerCitations(metadata={})
    msg = make_answer(answer="Answer", answer_citations=citations)
    result = convert_arag_answer_to_content(msg)
    # empty metadata → no EmbeddedResource for citations
    assert all(
        not (
            isinstance(c, EmbeddedResource)
            and str(c.resource.uri) == "rao-response://citations"
        )
        for c in result
    )


# ---------------------------------------------------------------------------
# Answer URLs
# ---------------------------------------------------------------------------


def test_answer_urls():
    msg = make_answer(answer="Answer", answer_urls=["https://a.com", "https://b.com"])
    result = convert_arag_answer_to_content(msg)

    urls_block = next(
        (
            c
            for c in result
            if isinstance(c, EmbeddedResource) and "answer-urls" in str(c.resource.uri)
        ),
        None,
    )
    assert urls_block is not None
    assert urls_block.resource.mimeType == "application/json"
    urls = json.loads(urls_block.resource.text)
    assert urls == ["https://a.com", "https://b.com"]


# ---------------------------------------------------------------------------
# Context chunks
# ---------------------------------------------------------------------------


def test_context_chunks():
    chunk = Chunk(
        chunk_id="c1",
        title="Doc title",
        source="my-source",
        text="Chunk text here",
        labels=["label1"],
        origin_url="https://origin.com",
    )
    context = Context(
        id="ctx-id",
        original_question_uuid="q1",
        actual_question_uuid="q1",
        question="What?",
        chunks=[chunk],
        source="nucliadb",
        agent="test-agent",
    )
    msg = make_answer(answer="Answer", context=context)
    result = convert_arag_answer_to_content(msg)

    chunk_blocks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource) and "chunk" in str(c.resource.uri)
    ]
    assert len(chunk_blocks) == 1
    block = chunk_blocks[0]
    assert isinstance(block.resource, TextResourceContents)
    assert block.resource.mimeType == "text/plain"
    assert block.resource.text == "Chunk text here"
    assert "ctx-id" in str(block.resource.uri)
    assert "c1" in str(block.resource.uri)
    meta = block.resource.meta
    assert meta["title"] == "Doc title"
    assert meta["source"] == "my-source"
    assert meta["labels"] == ["label1"]
    assert meta["origin_url"] == "https://origin.com"


def test_context_chunk_minimal_metadata():
    """Chunks with only required fields must not include None values in meta."""
    chunk = Chunk(chunk_id="c2", text="Some text")
    context = Context(
        id="ctx2",
        original_question_uuid=None,
        actual_question_uuid=None,
        question="Q",
        chunks=[chunk],
        source="s",
        agent="a",
    )
    msg = make_answer(context=context)
    result = convert_arag_answer_to_content(msg)
    block = next(
        c
        for c in result
        if isinstance(c, EmbeddedResource) and "chunk" in str(c.resource.uri)
    )
    # meta should be None or an empty dict (no spurious keys)
    assert block.resource.meta is None or block.resource.meta == {}


# ---------------------------------------------------------------------------
# Context images
# ---------------------------------------------------------------------------


def test_context_images():
    image = Image(content_type="image/png", b64encoded="aGVsbG8=")
    context = Context(
        id="ctx3",
        original_question_uuid=None,
        actual_question_uuid=None,
        question="Q",
        images={"img-1": image},
        source="s",
        agent="a",
    )
    msg = make_answer(context=context)
    result = convert_arag_answer_to_content(msg)

    img_blocks = [c for c in result if isinstance(c, ImageContent)]
    assert len(img_blocks) == 1
    assert img_blocks[0].data == "aGVsbG8="
    assert img_blocks[0].mimeType == "image/png"


# ---------------------------------------------------------------------------
# Context image_urls
# ---------------------------------------------------------------------------


def test_context_image_urls():
    context = Context(
        id="ctx4",
        original_question_uuid=None,
        actual_question_uuid=None,
        question="Q",
        image_urls=["https://img.example.com/1.png"],
        source="s",
        agent="a",
    )
    msg = make_answer(context=context)
    result = convert_arag_answer_to_content(msg)

    url_block = next(
        (
            c
            for c in result
            if isinstance(c, EmbeddedResource) and "image-urls" in str(c.resource.uri)
        ),
        None,
    )
    assert url_block is not None
    assert json.loads(url_block.resource.text) == ["https://img.example.com/1.png"]


# ---------------------------------------------------------------------------
# Context structured items
# ---------------------------------------------------------------------------


def test_context_structured():
    context = Context(
        id="ctx5",
        original_question_uuid=None,
        actual_question_uuid=None,
        question="Q",
        structured=[
            '{"col1": "val1", "col2": "val2"}',
            "| A | B |\n|---|---|\n| 1 | 2 |",
        ],
        source="s",
        agent="a",
    )
    msg = make_answer(context=context)
    result = convert_arag_answer_to_content(msg)

    structured_blocks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource) and "structured" in str(c.resource.uri)
    ]
    assert len(structured_blocks) == 2
    assert "ctx5" in str(structured_blocks[0].resource.uri)
    assert "structured/0" in str(structured_blocks[0].resource.uri)
    assert structured_blocks[0].resource.mimeType == "text/plain"
    assert structured_blocks[0].resource.text == '{"col1": "val1", "col2": "val2"}'
    assert "structured/1" in str(structured_blocks[1].resource.uri)
    assert structured_blocks[1].resource.text == "| A | B |\n|---|---|\n| 1 | 2 |"


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def test_vega_lite_visualization():
    viz = VegaLiteVisualization(vega_lite_obj={"mark": "bar", "data": {"values": []}})
    msg = make_answer(answer="Answer", data_visualizations=[viz])
    result = convert_arag_answer_to_content(msg)

    viz_blocks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource) and "visualization" in str(c.resource.uri)
    ]
    assert len(viz_blocks) == 1
    block = viz_blocks[0]
    assert block.resource.mimeType == "application/vnd.vegalite.v5+json"
    assert "visualization/0" in str(block.resource.uri)
    # Top-level visualizations do NOT have the answer/ prefix
    assert "answer/visualization" not in str(block.resource.uri)
    data = json.loads(block.resource.text)
    assert data["mark"] == "bar"


def test_multiple_visualizations_get_indexed():
    vizs = [
        VegaLiteVisualization(vega_lite_obj={"mark": "bar"}),
        VegaLiteVisualization(vega_lite_obj={"mark": "line"}),
    ]
    msg = make_answer(data_visualizations=vizs)
    result = convert_arag_answer_to_content(msg)

    viz_blocks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource) and "visualization" in str(c.resource.uri)
    ]
    assert len(viz_blocks) == 2
    assert "visualization/0" in str(viz_blocks[0].resource.uri)
    assert "visualization/1" in str(viz_blocks[1].resource.uri)


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


def test_step_produces_assistant_only_text():
    step = Step(
        original_question_uuid="q1",
        actual_question_uuid="q1",
        module="smart",
        title="Planning",
        value="Plan result",
        reason="Because logic",
        timeit=1.23,
        input_nuclia_tokens=10,
        output_nuclia_tokens=20,
        agent_path="agent",
    )
    msg = make_answer(step=step)
    result = convert_arag_answer_to_content(msg)

    step_blocks = [c for c in result if isinstance(c, TextContent)]
    assert len(step_blocks) == 1
    block = step_blocks[0]
    assert "Planning" in block.text
    assert "Plan result" in block.text
    assert "Because logic" in block.text
    assert block.annotations is not None
    assert "assistant" in block.annotations.audience


def test_step_without_optional_fields():
    step = Step(
        original_question_uuid=None,
        actual_question_uuid=None,
        module="smart",
        title="Planning",
        timeit=0.5,
        input_nuclia_tokens=None,
        output_nuclia_tokens=None,
        agent_path="agent",
    )
    msg = make_answer(step=step)
    result = convert_arag_answer_to_content(msg)
    assert len(result) == 1
    assert "Planning" in result[0].text


def test_step_metadata_defaults_to_none():
    """Step.metadata is None when not provided (backward compat)."""
    step = Step(
        original_question_uuid=None,
        actual_question_uuid=None,
        module="smart",
        title="Planning",
        timeit=0.5,
        input_nuclia_tokens=None,
        output_nuclia_tokens=None,
        agent_path="agent",
    )
    assert step.metadata is None


def test_step_metadata_accepts_dict():
    """Step.metadata stores an arbitrary key/value dict."""
    meta = {"source": "test-run", "confidence": 0.95, "tags": ["a", "b"]}
    step = Step(
        original_question_uuid="q1",
        actual_question_uuid="q1",
        module="smart",
        title="Planning",
        timeit=1.0,
        input_nuclia_tokens=5,
        output_nuclia_tokens=10,
        agent_path="agent",
        metadata=meta,
    )
    assert step.metadata == meta


def test_step_metadata_roundtrip():
    """Step.metadata survives a model_dump / model_validate round-trip."""
    meta = {"key": "value", "nested": {"inner": 42}}
    step = Step(
        original_question_uuid="q1",
        actual_question_uuid="q1",
        module="smart",
        title="Planning",
        timeit=1.0,
        input_nuclia_tokens=5,
        output_nuclia_tokens=10,
        agent_path="agent",
        metadata=meta,
    )
    dumped = step.model_dump()
    assert dumped["metadata"] == meta

    restored = Step.model_validate(dumped)
    assert restored.metadata == meta


def test_step_metadata_none_omitted_in_dump():
    """When metadata is None the serialised dict contains the key with None value."""
    step = Step(
        original_question_uuid=None,
        actual_question_uuid=None,
        module="smart",
        title="Planning",
        timeit=0.5,
        input_nuclia_tokens=None,
        output_nuclia_tokens=None,
        agent_path="agent",
    )
    dumped = step.model_dump()
    assert "metadata" in dumped
    assert dumped["metadata"] is None


# Exception


def test_exception_produces_text_content():
    msg = make_answer(exception=ARAGException(detail="Something went wrong"))
    result = convert_arag_answer_to_content(msg)
    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert "Something went wrong" in result[0].text


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_ordering_answer_then_citations_then_urls():
    citations = AnswerCitations(
        metadata={"b1": CitationMetadata(context_id="c", origin_urls=[])}
    )
    msg = make_answer(
        answer="My answer",
        answer_citations=citations,
        answer_urls=["https://example.com"],
    )
    result = convert_arag_answer_to_content(msg)
    assert isinstance(result[0], TextContent)  # answer first
    assert isinstance(result[1], EmbeddedResource)  # citations second
    assert "citations" in str(result[1].resource.uri)
    assert isinstance(result[2], EmbeddedResource)  # urls third
    assert "answer-urls" in str(result[2].resource.uri)


# possible_answer (Answer object emitted by add_answer callbacks) - current behaviour of the passthrough agent


def _make_possible_answer(**kwargs) -> Answer:
    defaults = {
        "answer": "Answer text",
        "module": "test",
        "agent_path": "/generation/test",
        "original_question_uuid": "q1",
        "actual_question_uuid": "q1",
    }
    return Answer(**{**defaults, **kwargs})


def test_possible_answer_plain_text():
    pa = _make_possible_answer(answer="Hello from possible_answer")
    msg = make_answer(possible_answer=pa)
    result = convert_arag_answer_to_content(msg)

    assert len(result) == 1
    assert isinstance(result[0], TextContent)
    assert result[0].text == "Hello from possible_answer"


def test_possible_answer_with_citations():
    citations = AnswerCitations(
        metadata={"block-1": CitationMetadata(context_id="ctx1", origin_urls=[])}
    )
    pa = _make_possible_answer(citations=citations)
    msg = make_answer(possible_answer=pa)
    result = convert_arag_answer_to_content(msg)

    assert len(result) == 2
    assert isinstance(result[0], TextContent)
    embedded = result[1]
    assert isinstance(embedded, EmbeddedResource)
    assert str(embedded.resource.uri) == "rao-response://answer/citations"
    data = json.loads(embedded.resource.text)
    assert "block-1" in data["metadata"]


def test_possible_answer_with_chunks():
    pa = _make_possible_answer(
        chunks=[
            Chunk(chunk_id="c1", text="Chunk A"),
            Chunk(chunk_id="c2", text="Chunk B"),
        ]
    )
    msg = make_answer(possible_answer=pa)
    result = convert_arag_answer_to_content(msg)

    embedded_chunks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource) and "answer/chunk" in str(c.resource.uri)
    ]
    assert len(embedded_chunks) == 2
    assert "answer/chunk/0" in str(embedded_chunks[0].resource.uri)
    assert embedded_chunks[0].resource.text == "Chunk A"
    assert "answer/chunk/1" in str(embedded_chunks[1].resource.uri)
    assert embedded_chunks[1].resource.text == "Chunk B"


def test_possible_answer_with_structured():
    pa = _make_possible_answer(structured=["col1,col2", "val1,val2"])
    msg = make_answer(possible_answer=pa)
    result = convert_arag_answer_to_content(msg)

    structured_blocks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource)
        and "answer/structured" in str(c.resource.uri)
    ]
    assert len(structured_blocks) == 2
    assert "answer/structured/0" in str(structured_blocks[0].resource.uri)
    assert structured_blocks[0].resource.text == "col1,col2"


def test_possible_answer_with_visualization():
    viz = VegaLiteVisualization(vega_lite_obj={"mark": "bar"})
    pa = _make_possible_answer(data_visualizations=[viz])
    msg = make_answer(possible_answer=pa)
    result = convert_arag_answer_to_content(msg)

    viz_blocks = [
        c
        for c in result
        if isinstance(c, EmbeddedResource)
        and "answer/visualization" in str(c.resource.uri)
    ]
    assert len(viz_blocks) == 1
    assert "answer/visualization/0" in str(viz_blocks[0].resource.uri)
    assert viz_blocks[0].resource.mimeType == "application/vnd.vegalite.v5+json"
    data = json.loads(viz_blocks[0].resource.text)
    assert data["mark"] == "bar"


def test_possible_answer_and_plain_answer_both_emitted():
    # Option A: both possible_answer and plain answer are emitted independently.
    pa = _make_possible_answer(answer="Streaming answer")
    msg = make_answer(possible_answer=pa, answer="Inline answer")
    result = convert_arag_answer_to_content(msg)

    texts = [c.text for c in result if isinstance(c, TextContent)]
    assert len(texts) == 2
    assert texts[0] == "Streaming answer"  # possible_answer first
    assert texts[1] == "Inline answer"  # then top-level answer
