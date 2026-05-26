import json

from hyperforge_nucliadb.ask_utils import (
    spit_by_filter_type,
)
from hyperforge_nucliadb.basic_ask_agent import (
    BasicAskAgent,
)
from hyperforge_nucliadb.basic_ask_config import (
    BasicAskAgentConfig,
)


def test_parse_selected_filters_simple():
    config = BasicAskAgentConfig(module="basic_ask", sources=["test_kb"])
    agent = BasicAskAgent(config=config, agent_id="test_agent")
    source_id = "test_kb"

    # Valid filter
    selected = {"filters": json.dumps([{"any": ["/l/topic/tech"]}])}
    filters = agent.parse_selected_filters("question", source_id, selected)
    assert len(filters) == 1
    assert filters[0].any == ["/l/topic/tech"]

    # Fixable filter
    selected = {"filters": "[{'any': ['/l/topic/tech']}]"}
    filters = agent.parse_selected_filters("question", source_id, selected)
    assert len(filters) == 1
    assert filters[0].any == ["/l/topic/tech"]

    # Invalid filter (not JSON)
    selected = {"filters": "{invalid json}"}
    filters = agent.parse_selected_filters("question", source_id, selected)
    assert filters == []

    # Missing filters key
    selected = {}
    filters = agent.parse_selected_filters("question", source_id, selected)
    assert filters == []


def test_spit_by_filter_type():
    # Using full name
    assert spit_by_filter_type("classification.labels/foo/bar") == (
        "classification.labels",
        "/foo/bar",
    )
    # Using alias
    assert spit_by_filter_type("l/foo/bar") == ("classification.labels", "/foo/bar")
    # Alias should not strip characters from the value (regression test)
    assert spit_by_filter_type("classification.labels/lang/en") == (
        "classification.labels",
        "/lang/en",
    )
    assert spit_by_filter_type("l/lang/en") == ("classification.labels", "/lang/en")
    # Icon filter
    assert spit_by_filter_type("icon/application/pdf") == ("icon", "/application/pdf")
    assert spit_by_filter_type("n/i/application/pdf") == ("icon", "/application/pdf")
    # Language filters
    assert spit_by_filter_type("metadata.language/en") == ("metadata.language", "/en")
    assert spit_by_filter_type("s/p/en") == ("metadata.language", "/en")
    assert spit_by_filter_type("metadata.languages/en") == ("metadata.languages", "/en")
    # Origin tags
    assert spit_by_filter_type("origin.tags/mytag") == ("origin.tags", "/mytag")
    assert spit_by_filter_type("t/mytag") == ("origin.tags", "/mytag")
    # Entities
    assert spit_by_filter_type("entities/person/John") == ("entities", "/person/John")
    assert spit_by_filter_type("e/person/John") == ("entities", "/person/John")
    # No match
    assert spit_by_filter_type("unknown/foo") is None
