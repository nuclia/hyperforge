import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from nucliadb_models.filters import (
    And,
    CatalogFilterExpression,
    FilterExpression,
    Keyword,
    Label,
    Or,
    Resource,
)

from hyperforge_nucliadb.basic_ask_agent import (
    BasicAskAgent,
    clean_citation_footnotes_from_answer,
)


@pytest.mark.parametrize(
    "answer, citations, cleaned_answer",
    [
        (
            "This is an answer.[1] More text here.[2]\n\n[1]: block-AA\n[2]: block-BB",
            {
                "block-AA": "rid/field1/chunk1",
                "block-BB": "rid/field2/chunk2",
            },
            "This is an answer. More text here.",
        ),
        (
            "No citations here, just text.",
            {},
            "No citations here, just text.",
        ),
        (
            "Answer with one citation.[1]\n\n[1]: block-CC",
            {
                "block-CC": "rid/field3/chunk3",
            },
            "Answer with one citation.",
        ),
        (
            "Citations missing.[1][2]\n\n[1]: block-DD",
            {
                "block-DD": "rid/field4/chunk4",
            },
            "Citations missing.",
        ),
    ],
)
def test_clean_citation_footnotes_from_answer(
    answer: str, citations: dict[str, str], cleaned_answer: str
):
    assert clean_citation_footnotes_from_answer(answer, citations) == cleaned_answer


@pytest.mark.asyncio
async def test_apply_filter_all_filters():
    """Test that all possible filter types are combined into a single FilterExpression."""
    agent = BasicAskAgent.__new__(BasicAskAgent)
    agent.synonyms = {}

    # Mock nucliadb_driver
    nucliadb_driver = MagicMock()
    nucliadb_driver.synonyms_raw = AsyncMock(return_value={"climbing": ["bouldering"]})
    nucliadb_driver.config.filters = ["/classification.labels/region/europe"]
    nucliadb_driver.config.filter_expression = FilterExpression(
        field=Label(labelset="status", label="active")
    )

    source = "source1"

    keyword_filters = ["climbing"]
    and_filters = [
        "/classification.labels/topic/science",
        "/classification.labels/topic/tech",
    ]
    or_filters = ["/classification.labels/lang/en", "/classification.labels/lang/es"]
    rid1 = uuid.uuid4().hex
    rid2 = uuid.uuid4().hex
    resource_filters = [rid1, rid2]

    filter_expression = FilterExpression(field=Label(labelset="cat", label="A"))

    result = await agent.build_filter_expression(
        nucliadb_driver=nucliadb_driver,
        source=source,
        keyword_filters=keyword_filters,
        and_filters=and_filters,
        or_filters=or_filters,
        resource_filters=resource_filters,
        filter_expression=filter_expression,
    )

    assert result is not None

    # The top-level should be a FilterExpression combining everything with AND
    # There are 3 expressions to combine:
    #   1. field-level And of: keyword Or, resource Or, config.filters And, and_filters And, or_filters Or
    #   2. catalog_filters
    #   3. config.filter_expression
    # They are combined via combine_filter_expressions which nests ANDs

    # Verify the result is a FilterExpression
    assert isinstance(result, FilterExpression)
    assert result.field is not None

    assert isinstance(result.field, And)
    assert len(result.field.operands) == 3

    # Verify the first operand is the combination of keyword, resource, and config filters
    first_operand = result.field.operands[0]
    assert isinstance(first_operand, And)
    assert len(first_operand.operands) == 5
    assert isinstance(first_operand.operands[0], Or)  # keyword filters
    assert all(isinstance(op, Keyword) for op in first_operand.operands[0].operands)
    assert isinstance(first_operand.operands[1], Or)  # resource filters
    assert all(isinstance(op, Resource) for op in first_operand.operands[1].operands)
    assert isinstance(first_operand.operands[2], Label)  # config.filters
    assert first_operand.operands[2].labelset == "region"
    assert first_operand.operands[2].label == "europe"
    assert isinstance(first_operand.operands[3], And)  # and_filters
    assert all(isinstance(op, Label) for op in first_operand.operands[3].operands)
    assert isinstance(first_operand.operands[4], Or)  # or_filters
    assert all(isinstance(op, Label) for op in first_operand.operands[4].operands)
    assert first_operand.operands[4].operands[0].labelset == "lang"
    assert first_operand.operands[4].operands[0].label == "en"
    assert first_operand.operands[4].operands[1].labelset == "lang"
    assert first_operand.operands[4].operands[1].label == "es"

    # Verify the second operand is the catalog_filters
    assert isinstance(result.field.operands[1], Label)  # catalog_filters
    assert result.field.operands[1].labelset == "cat"
    assert result.field.operands[1].label == "A"

    # Verify the third operand is the config.filter_expression
    assert isinstance(result.field.operands[2], Label)  # config.filter_expression
    assert result.field.operands[2].labelset == "status"
    assert result.field.operands[2].label == "active"


@pytest.mark.asyncio
async def test_apply_filter_returns_none_when_no_filters():
    """Test that apply_filter returns None when no filters are provided."""
    agent = BasicAskAgent.__new__(BasicAskAgent)
    agent.synonyms = {}

    nucliadb_driver = MagicMock()
    nucliadb_driver.config.filters = []
    nucliadb_driver.config.filter_expression = None

    result = await agent.build_filter_expression(
        nucliadb_driver=nucliadb_driver,
        source="source1",
        keyword_filters=None,
        and_filters=None,
        or_filters=None,
        filter_expression=None,
        resource_filters=None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_apply_filter_single_expression_no_and():
    """Test that apply_filter returns a single FilterExpression (not wrapped in And) when only one filter group is provided."""
    agent = BasicAskAgent.__new__(BasicAskAgent)
    agent.synonyms = {}

    nucliadb_driver = MagicMock()
    nucliadb_driver.config.filters = []
    nucliadb_driver.config.filter_expression = None

    result = await agent.build_filter_expression(
        nucliadb_driver=nucliadb_driver,
        source="source1",
        keyword_filters=None,
        and_filters=["/classification.labels/topic/science"],
        or_filters=None,
        filter_expression=None,
        resource_filters=None,
    )

    assert result is not None
    assert isinstance(result, FilterExpression)
    # With a single and_filter, the field should be a Label directly, not wrapped in And
    assert isinstance(result.field, Label)
    assert result.field.labelset == "topic"
    assert result.field.label == "science"


@pytest.mark.asyncio
async def test_build_catalog_filter_all_filters():
    """Test that all catalog filter types are combined into a single CatalogFilterExpression."""
    agent = BasicAskAgent.__new__(BasicAskAgent)
    agent.synonyms = {}

    nucliadb_driver = MagicMock()
    nucliadb_driver.config.catalog_filter_expression = CatalogFilterExpression(
        resource=Label(labelset="status", label="active")
    )

    filter_expression = CatalogFilterExpression(
        resource=Label(labelset="cat", label="A")
    )

    result = await agent.build_catalog_filter_expression(
        nucliadb_driver=nucliadb_driver,
        filters=[
            "/classification.labels/region/europe",
            "/classification.labels/region/asia",
        ],
        classification_labels=[
            "/classification.labels/topic/science",
            "/classification.labels/topic/tech",
        ],
        classification_labels_operand="or",
        filter_expression=filter_expression,
    )

    assert result is not None
    assert isinstance(result, CatalogFilterExpression)
    assert result.resource is not None
    assert isinstance(result.resource, And)
    assert len(result.resource.operands) == 3

    # Verify the first operand is the combination of filters
    first_operand = result.resource.operands[0]
    assert isinstance(first_operand, And)
    assert len(first_operand.operands) == 2
    # Classification labels
    assert isinstance(first_operand.operands[0], Or)
    assert len(first_operand.operands[0].operands) == 2
    assert isinstance(first_operand.operands[0].operands[0], Label)
    assert first_operand.operands[0].operands[0].labelset == "topic"
    assert first_operand.operands[0].operands[0].label == "science"
    assert isinstance(first_operand.operands[0].operands[1], Label)
    assert first_operand.operands[0].operands[1].labelset == "topic"
    assert first_operand.operands[0].operands[1].label == "tech"
    # Filters
    assert isinstance(first_operand.operands[1], And)
    assert len(first_operand.operands[1].operands) == 2
    assert isinstance(first_operand.operands[1].operands[0], Label)
    assert first_operand.operands[1].operands[0].labelset == "region"
    assert first_operand.operands[1].operands[0].label == "europe"
    assert isinstance(first_operand.operands[1].operands[1], Label)
    assert first_operand.operands[1].operands[1].labelset == "region"
    assert first_operand.operands[1].operands[1].label == "asia"

    # Verify the second operand is the filter_expression
    third_operand = result.resource.operands[1]
    assert isinstance(third_operand, Label)
    assert third_operand.labelset == "cat"
    assert third_operand.label == "A"

    # Verify the third operand is the driver configured catalog filter expression
    second_operand = result.resource.operands[2]
    assert isinstance(second_operand, Label)
    assert second_operand.labelset == "status"
    assert second_operand.label == "active"


@pytest.mark.asyncio
async def test_build_catalog_filter_returns_none_when_no_filters():
    """Test that build_catalog_filter_expression returns None when no filters are provided."""
    agent = BasicAskAgent.__new__(BasicAskAgent)
    agent.synonyms = {}

    nucliadb_driver = MagicMock()
    nucliadb_driver.config.catalog_filter_expression = None

    result = await agent.build_catalog_filter_expression(
        nucliadb_driver=nucliadb_driver,
        filters=None,
        classification_labels=None,
        filter_expression=None,
    )

    assert result is None


@pytest.mark.asyncio
async def test_build_catalog_filter_single_expression_no_and():
    """Test that a single catalog filter is returned without And wrapping."""
    agent = BasicAskAgent.__new__(BasicAskAgent)
    agent.synonyms = {}

    nucliadb_driver = MagicMock()
    nucliadb_driver.config.catalog_filter_expression = None

    result = await agent.build_catalog_filter_expression(
        nucliadb_driver=nucliadb_driver,
        filters=["/classification.labels/topic/science"],
        classification_labels=None,
        filter_expression=None,
    )

    assert result is not None
    assert isinstance(result, CatalogFilterExpression)
    assert isinstance(result.resource, Label)
    assert result.resource.labelset == "topic"
    assert result.resource.label == "science"
