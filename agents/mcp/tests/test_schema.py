import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp import types

from hyperforge_mcp.agent import MCPAgent
from hyperforge_mcp.config import MCPAgentConfig
from hyperforge_mcp.schema import IncompatibleToolSchema, normalize_tool_schema


def test_normalize_empty_root_schema_as_no_argument_object():
    assert normalize_tool_schema({}) == {"type": "object", "properties": {}}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"$defs": {"Value": {"type": "string"}}, "$ref": "#/$defs/Value"},
    ],
)
def test_normalize_tool_schema_rejects_non_object_root(schema):
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(schema)

    assert exc_info.value.json_path == "/type"
    assert exc_info.value.reason == "tool input schema must have type 'object'"


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
    assert exc_info.value.reason == "type is not supported by all NUA providers"


def test_normalize_tool_schema_preserves_type_union():
    schema = {"type": "object", "properties": {"value": {"type": ["string", "null"]}}}

    assert normalize_tool_schema(schema) == schema


@pytest.mark.parametrize("enum", [[1, 2], ["auto", None], []])
def test_normalize_tool_schema_rejects_non_string_enum(enum):
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {"type": "object", "properties": {"mode": {"enum": enum}}}
        )

    assert exc_info.value.json_path == "/properties/mode/enum"


def test_literal_refs_and_ref_property_name_are_not_schema_references():
    schema = {
        "type": "object",
        "properties": {
            "$ref": {"type": "string"},
            "metadata": {
                "type": "object",
                "default": {"$ref": "literal-default"},
                "example": {"$ref": "literal-example"},
            },
        },
    }

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


def test_normalize_local_reference_through_array_and_encoded_fragment():
    normalized = normalize_tool_schema(
        {
            "type": "object",
            "$defs": {
                "Choices": {
                    "anyOf": [
                        {"type": "string", "description": "Selected value"},
                    ]
                }
            },
            "properties": {
                "choice": {"$ref": "#%2F%24defs%2FChoices%2FanyOf%2F0"},
            },
        }
    )

    assert normalized["properties"]["choice"] == {
        "type": "string",
        "description": "Selected value",
    }


@pytest.mark.parametrize("reference", ["#/$defs/a~2b", "#/$defs/a%2"])
def test_normalize_rejects_invalid_local_reference(reference):
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {
                "type": "object",
                "$defs": {"a~2b": {"type": "string"}},
                "properties": {"value": {"$ref": reference}},
            }
        )

    assert exc_info.value.json_path == "/properties/value/$ref"
    assert exc_info.value.reason == f"invalid local reference {reference!r}"


def test_normalize_ref_with_validation_siblings_is_rejected():
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema(
            {
                "type": "object",
                "$defs": {"Value": {"type": "string"}},
                "properties": {
                    "value": {
                        "$ref": "#/$defs/Value",
                        "type": "string",
                        "maxLength": 10,
                    }
                },
            }
        )

    assert exc_info.value.json_path == "/properties/value"


def test_normalize_ref_with_annotation_siblings():
    normalized = normalize_tool_schema(
        {
            "type": "object",
            "$defs": {"Value": {"type": "string", "minLength": 1}},
            "properties": {
                "value": {
                    "$ref": "#/$defs/Value",
                    "title": "Value",
                    "description": "The value to use",
                    "default": "example",
                }
            },
        }
    )

    assert normalized["properties"]["value"] == {
        "type": "string",
        "minLength": 1,
        "title": "Value",
        "description": "The value to use",
        "default": "example",
    }


@pytest.mark.parametrize(
    "keyword",
    [
        "allOf",
        "oneOf",
        "patternProperties",
        "dependentSchemas",
        "contains",
        "const",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "$id",
    ],
)
def test_unsupported_provider_keyword_is_rejected(keyword):
    with pytest.raises(IncompatibleToolSchema) as exc_info:
        normalize_tool_schema({"type": "object", keyword: {}})

    assert exc_info.value.json_path == f"/{keyword}"
    assert exc_info.value.reason == "keyword is not supported by all NUA providers"


def test_normalize_refs_in_supported_schema_containers():
    normalized = normalize_tool_schema(
        {
            "type": "object",
            "$defs": {"Value": {"type": "string", "minLength": 1}},
            "additionalProperties": {"$ref": "#/$defs/Value"},
        }
    )

    value_schema = {"type": "string", "minLength": 1}
    assert normalized["additionalProperties"] == value_schema
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

    assert exc_info.value.json_path == "/customSchema"
    assert exc_info.value.reason == "keyword is not supported by all NUA providers"


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
    assert "reason='schema must declare or imply a supported type'" in caplog.text


def test_deeply_nested_tool_is_isolated():
    agent = MCPAgent(
        MCPAgentConfig.model_validate(
            {"id": "mcp-test", "module": "mcp", "source": "mcphttp-01"}
        )
    )
    nested_schema = {"type": "string"}
    for _ in range(1200):
        nested_schema = {"type": "array", "items": nested_schema}

    compatible = agent._compatible_tools(
        [
            types.Tool(
                name="too-deep",
                inputSchema={
                    "type": "object",
                    "properties": {"value": nested_schema},
                },
            ),
            types.Tool(
                name="valid",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]
    )

    assert [tool.name for tool in compatible] == ["valid"]


async def test_choose_tool_only_sends_compatible_tools_to_nua():
    agent = MCPAgent(
        MCPAgentConfig.model_validate(
            {"id": "mcp-test", "module": "mcp", "source": "mcphttp-01"}
        )
    )
    agent.session = SimpleNamespace(
        list_tools=AsyncMock(
            return_value=SimpleNamespace(
                tools=[
                    types.Tool(
                        name="invalid",
                        inputSchema={
                            "type": "object",
                            "properties": {"value": {}},
                        },
                    ),
                    types.Tool(
                        name="valid",
                        inputSchema={
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    ),
                ],
                nextCursor=None,
            )
        )
    )
    await agent.preload_tools()
    captured_items = []
    add_step_items = []

    async def execute_raw(item, tracking=None):
        captured_items.append(item)
        return SimpleNamespace(tools=None, reasoning=""), 0.0, 0.0

    async def add_step(
        step_module: str = "",
        step_title: str = "",
        step_reason: str = "",
        step_value: str = "",
        step_agent_path: str = "",
        input_nuclia_tokens: int = 0,
        output_nuclia_tokens: int = 0,
        error: str = "",
        timeit: float = 0.0,
    ):
        add_step_items.append(
            (
                step_module,
                step_title,
                step_reason,
                step_value,
                step_agent_path,
                input_nuclia_tokens,
                output_nuclia_tokens,
                error,
                timeit,
            )
        )

    await agent.choose_tool(
        manager=SimpleNamespace(execute_raw=execute_raw),
        images=[],
        messages=[],
        memory=SimpleNamespace(add_step=add_step),
    )

    assert [tool.name for tool in captured_items[0].tools] == ["valid"]


async def test_preload_tools_isolates_incompatible_tools_across_pages():
    agent = MCPAgent(
        MCPAgentConfig.model_validate(
            {"id": "mcp-test", "module": "mcp", "source": "mcphttp-01"}
        )
    )
    agent.session = SimpleNamespace(
        list_tools=AsyncMock(
            side_effect=[
                SimpleNamespace(
                    tools=[
                        types.Tool(
                            name="invalid-first-page",
                            inputSchema={"type": "object", "properties": {"value": {}}},
                        ),
                        types.Tool(
                            name="valid-first-page",
                            inputSchema={"type": "object", "properties": {}},
                        ),
                    ],
                    nextCursor="page-2",
                ),
                SimpleNamespace(
                    tools=[
                        types.Tool(
                            name="invalid-second-page",
                            inputSchema={"type": "object", "oneOf": []},
                        ),
                        types.Tool(
                            name="valid-second-page",
                            inputSchema={"type": "object", "properties": {}},
                        ),
                    ],
                    nextCursor=None,
                ),
            ]
        )
    )

    await agent.preload_tools()

    assert [tool.name for tool in agent.tools] == [
        "valid-first-page",
        "valid-second-page",
    ]
    assert agent.session.list_tools.await_count == 2
