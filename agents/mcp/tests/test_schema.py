import logging

import pytest
from mcp import types

from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig
from hyperforge_mcp.schema import IncompatibleToolSchema, normalize_tool_schema


def test_normalize_empty_root_schema_as_no_argument_object():
    assert normalize_tool_schema({}) == {"type": "object", "properties": {}}


def test_normalize_tool_schema_rejects_empty_nested_property():
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema({"type": "object", "properties": {"value": {}}})

    assert exc_info.value.json_path == "/properties/value"


@pytest.mark.parametrize(
    "declared_type",
    [42, "unknown", [], ["string", "unknown"], ["string", "string"]],
)
def test_normalize_tool_schema_rejects_invalid_declared_type(declared_type):
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {
                "type": "object",
                "properties": {"value": {"type": declared_type}},
            }
        )

    assert exc_info.value.json_path == "/properties/value/type"
    assert exc_info.value.reason == "invalid JSON Schema type"


def test_normalize_tool_schema_preserves_type_union():
    schema = {"type": "object", "properties": {"value": {"type": ["string", "null"]}}}

    assert normalize_tool_schema(schema) == schema


def test_normalize_tool_schema_preserves_nested_constraints_and_resolves_refs():
    schema = {
        "type": "object",
        "$defs": {
            "Filter": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "minLength": 1},
                    "value": {"enum": ["open", "closed"]},
                },
                "required": ["field", "value"],
                "additionalProperties": False,
            }
        },
        "properties": {
            "filters": {
                "type": "array",
                "items": {"$ref": "#/$defs/Filter"},
                "minItems": 1,
                "maxItems": 10,
            }
        },
        "required": ["filters"],
    }

    normalized = normalize_tool_schema(schema)

    assert "$defs" not in normalized
    assert normalized["properties"]["filters"] == {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "field": {"type": "string", "minLength": 1},
                "value": {"enum": ["open", "closed"], "type": "string"},
            },
            "required": ["field", "value"],
            "additionalProperties": False,
        },
        "minItems": 1,
        "maxItems": 10,
    }


def test_normalize_ref_with_siblings_preserves_conjunction():
    normalized = normalize_tool_schema(
        {
            "type": "object",
            "$defs": {
                "Target": {
                    "type": "object",
                    "properties": {"from_target": {"type": "string"}},
                    "required": ["from_target"],
                }
            },
            "properties": {
                "value": {
                    "$ref": "#/$defs/Target",
                    "type": "object",
                    "properties": {"from_sibling": {"type": "integer"}},
                    "required": ["from_sibling"],
                }
            },
        }
    )

    assert normalized["properties"]["value"] == {
        "allOf": [
            {
                "type": "object",
                "properties": {"from_target": {"type": "string"}},
                "required": ["from_target"],
            },
            {
                "type": "object",
                "properties": {"from_sibling": {"type": "integer"}},
                "required": ["from_sibling"],
            },
        ]
    }


def test_normalize_refs_in_schema_containers():
    normalized = normalize_tool_schema(
        {
            "type": "object",
            "$defs": {"Value": {"type": "string", "minLength": 1}},
            "additionalProperties": {"$ref": "#/$defs/Value"},
            "patternProperties": {
                "^x-": {"$ref": "#/$defs/Value"},
            },
            "dependentSchemas": {
                "kind": {
                    "type": "object",
                    "properties": {"value": {"$ref": "#/$defs/Value"}},
                }
            },
        }
    )

    value_schema = {"type": "string", "minLength": 1}
    assert normalized["additionalProperties"] == value_schema
    assert normalized["patternProperties"]["^x-"] == value_schema
    assert normalized["dependentSchemas"]["kind"]["properties"]["value"] == value_schema
    assert "$defs" not in normalized


def test_rejects_reference_in_unsupported_schema_container():
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {
                "type": "object",
                "$defs": {"Value": {"type": "string"}},
                "customSchema": {"$ref": "#/$defs/Value"},
            }
        )

    assert exc_info.value.json_path == "/customSchema/$ref"
    assert exc_info.value.reason == "reference in unsupported schema container"


@pytest.mark.parametrize(
    ("reference", "reason"),
    [
        ("https://example.com/schema.json", "only local references are supported"),
        ("#/$defs/Missing", "unresolved local reference"),
    ],
)
def test_normalize_tool_schema_rejects_unsupported_references(reference, reason):
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {
                "type": "object",
                "properties": {"value": {"$ref": reference}},
            }
        )

    assert exc_info.value.json_path == "/properties/value/$ref"
    assert reason in exc_info.value.reason


def test_normalize_tool_schema_rejects_cyclic_local_reference():
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {
                "type": "object",
                "$defs": {"Node": {"$ref": "#/$defs/Node"}},
                "properties": {"node": {"$ref": "#/$defs/Node"}},
            }
        )

    assert exc_info.value.json_path == "/properties/node/$ref"
    assert exc_info.value.reason == "cyclic local reference"


def test_incompatible_tool_is_isolated_with_diagnostics(caplog):
    agent = MCPAgent(
        MCPAgentConfig.model_validate(
            {"id": "mcp-test", "module": "mcp", "source": "mcphttp-01"}
        )
    )
    valid_tool = types.Tool(
        name="search",
        description="Search documents",
        inputSchema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )
    invalid_tool = types.Tool(
        name="getFieldValuesFiltered",
        description="Get filtered field values",
        inputSchema={
            "type": "object",
            "properties": {
                "filters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {}},
                    },
                }
            },
        },
    )

    with caplog.at_level(logging.WARNING, logger="hyperforge"):
        compatible = agent._compatible_tools([invalid_tool, valid_tool])

    assert [tool.name for tool in compatible] == ["search"]
    assert "server='mcphttp-01'" in caplog.text
    assert "tool='getFieldValuesFiltered'" in caplog.text
    assert "path='/properties/filters/items/properties/value'" in caplog.text
