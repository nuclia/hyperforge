from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from inline_snapshot import snapshot
from pydantic import BaseModel, Field

from hyperforge.api.utils import to_strict_json_schema
from hyperforge.models import Context, JSONObject


class Table(str, Enum):
    orders = "orders"
    customers = "customers"
    products = "products"


class Column(str, Enum):
    id = "id"
    status = "status"
    expected_delivery_date = "expected_delivery_date"
    delivered_at = "delivered_at"
    shipped_at = "shipped_at"
    ordered_at = "ordered_at"
    canceled_at = "canceled_at"


class Operator(str, Enum):
    eq = "="
    gt = ">"
    lt = "<"
    le = "<="
    ge = ">="
    ne = "!="


class OrderBy(str, Enum):
    asc = "asc"
    desc = "desc"


class DynamicValue(BaseModel):
    column_name: str


class Condition(BaseModel):
    column: str
    operator: Operator
    value: Union[str, int, DynamicValue]


class Query(BaseModel):
    name: Optional[str] = None
    table_name: Table
    columns: List[Column]
    conditions: List[Condition]
    order_by: OrderBy


def test_most_types() -> None:
    assert to_strict_json_schema(Query) == snapshot(
        {
            "$defs": {
                "Column": {
                    "enum": [
                        "id",
                        "status",
                        "expected_delivery_date",
                        "delivered_at",
                        "shipped_at",
                        "ordered_at",
                        "canceled_at",
                    ],
                    "title": "Column",
                    "type": "string",
                },
                "Condition": {
                    "properties": {
                        "column": {"title": "Column", "type": "string"},
                        "operator": {
                            "enum": ["=", ">", "<", "<=", ">=", "!="],
                            "title": "Operator",
                            "type": "string",
                        },
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {
                                    "properties": {
                                        "column_name": {
                                            "title": "Column Name",
                                            "type": "string",
                                        }
                                    },
                                    "required": ["column_name"],
                                    "title": "DynamicValue",
                                    "type": "object",
                                    "additionalProperties": False,
                                },
                            ],
                            "title": "Value",
                        },
                    },
                    "required": ["column", "operator", "value"],
                    "title": "Condition",
                    "type": "object",
                },
                "DynamicValue": {
                    "properties": {
                        "column_name": {"title": "Column Name", "type": "string"}
                    },
                    "required": ["column_name"],
                    "title": "DynamicValue",
                    "type": "object",
                },
                "Operator": {
                    "enum": ["=", ">", "<", "<=", ">=", "!="],
                    "title": "Operator",
                    "type": "string",
                },
                "OrderBy": {
                    "enum": ["asc", "desc"],
                    "title": "OrderBy",
                    "type": "string",
                },
                "Table": {
                    "enum": ["orders", "customers", "products"],
                    "title": "Table",
                    "type": "string",
                },
            },
            "properties": {
                "name": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "title": "Name",
                },
                "table_name": {
                    "enum": ["orders", "customers", "products"],
                    "title": "Table",
                    "type": "string",
                },
                "columns": {
                    "items": {
                        "enum": [
                            "id",
                            "status",
                            "expected_delivery_date",
                            "delivered_at",
                            "shipped_at",
                            "ordered_at",
                            "canceled_at",
                        ],
                        "title": "Column",
                        "type": "string",
                    },
                    "title": "Columns",
                    "type": "array",
                },
                "conditions": {
                    "items": {
                        "properties": {
                            "column": {"title": "Column", "type": "string"},
                            "operator": {
                                "enum": ["=", ">", "<", "<=", ">=", "!="],
                                "title": "Operator",
                                "type": "string",
                            },
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "integer"},
                                    {
                                        "properties": {
                                            "column_name": {
                                                "title": "Column Name",
                                                "type": "string",
                                            }
                                        },
                                        "required": ["column_name"],
                                        "title": "DynamicValue",
                                        "type": "object",
                                        "additionalProperties": False,
                                    },
                                ],
                                "title": "Value",
                            },
                        },
                        "required": ["column", "operator", "value"],
                        "title": "Condition",
                        "type": "object",
                        "additionalProperties": False,
                    },
                    "title": "Conditions",
                    "type": "array",
                },
                "order_by": {
                    "enum": ["asc", "desc"],
                    "title": "OrderBy",
                    "type": "string",
                },
            },
            "required": ["name", "table_name", "columns", "conditions", "order_by"],
            "title": "Query",
            "type": "object",
            "additionalProperties": False,
        }
    )


class Color(Enum):
    RED = "red"
    BLUE = "blue"
    GREEN = "green"


class ColorDetection(BaseModel):
    color: Color = Field(description="The detected color")
    hex_color_code: str = Field(description="The hex color code of the detected color")


def test_enums() -> None:
    assert to_strict_json_schema(ColorDetection) == snapshot(
        {
            "$defs": {
                "Color": {
                    "enum": ["red", "blue", "green"],
                    "title": "Color",
                    "type": "string",
                }
            },
            "properties": {
                "color": {
                    "description": "The detected color",
                    "enum": ["red", "blue", "green"],
                    "title": "Color",
                    "type": "string",
                },
                "hex_color_code": {
                    "description": "The hex color code of the detected color",
                    "title": "Hex Color Code",
                    "type": "string",
                },
            },
            "required": ["color", "hex_color_code"],
            "title": "ColorDetection",
            "type": "object",
            "additionalProperties": False,
        }
    )


class Star(BaseModel):
    name: str = Field(description="The name of the star.")


class Galaxy(BaseModel):
    name: str = Field(description="The name of the galaxy.")
    largest_star: Star = Field(description="The largest star in the galaxy.")


class Universe(BaseModel):
    name: str = Field(description="The name of the universe.")
    galaxy: Galaxy = Field(description="A galaxy in the universe.")


def test_nested_inline_ref_expansion() -> None:
    assert to_strict_json_schema(Universe) == snapshot(
        {
            "title": "Universe",
            "type": "object",
            "$defs": {
                "Star": {
                    "title": "Star",
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "title": "Name",
                            "description": "The name of the star.",
                        }
                    },
                    "required": ["name"],
                },
                "Galaxy": {
                    "title": "Galaxy",
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "title": "Name",
                            "description": "The name of the galaxy.",
                        },
                        "largest_star": {
                            "title": "Star",
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "title": "Name",
                                    "description": "The name of the star.",
                                }
                            },
                            "required": ["name"],
                            "description": "The largest star in the galaxy.",
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "largest_star"],
                },
            },
            "properties": {
                "name": {
                    "type": "string",
                    "title": "Name",
                    "description": "The name of the universe.",
                },
                "galaxy": {
                    "title": "Galaxy",
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "title": "Name",
                            "description": "The name of the galaxy.",
                        },
                        "largest_star": {
                            "title": "Star",
                            "type": "object",
                            "properties": {
                                "name": {
                                    "type": "string",
                                    "title": "Name",
                                    "description": "The name of the star.",
                                }
                            },
                            "required": ["name"],
                            "description": "The largest star in the galaxy.",
                            "additionalProperties": False,
                        },
                    },
                    "required": ["name", "largest_star"],
                    "description": "A galaxy in the universe.",
                    "additionalProperties": False,
                },
            },
            "required": ["name", "galaxy"],
            "additionalProperties": False,
        }
    )


# ---------------------------------------------------------------------------
# JSONObject model
# ---------------------------------------------------------------------------


def test_json_object_minimal():
    """JSONObject can be created with only json_object (all other fields optional)."""
    obj = JSONObject(json_object={"key": "value"})
    assert obj.json_object == {"key": "value"}
    assert obj.json_schema is None
    assert obj.metadata == {}
    assert obj.id is not None  # auto-generated uuid


def test_json_object_with_schema():
    """JSONObject stores a JSON schema alongside the object."""
    schema = {"type": "object", "properties": {"key": {"type": "string"}}}
    obj = JSONObject(
        json_schema=schema,
        json_object={"key": "value"},
    )
    assert obj.json_schema == schema
    assert obj.json_object == {"key": "value"}


def test_json_object_with_metadata():
    """JSONObject stores arbitrary metadata."""
    obj = JSONObject(
        json_object={"x": 1},
        metadata={"source": "test", "confidence": 0.9},
    )
    assert obj.metadata == {"source": "test", "confidence": 0.9}


def test_json_object_explicit_id():
    """JSONObject accepts an explicit id and does not override it."""
    obj = JSONObject(json_object={}, id="my-custom-id")
    assert obj.id == "my-custom-id"


def test_json_object_unique_ids():
    """Two JSONObjects without explicit ids get different auto-generated ids."""
    a = JSONObject(json_object={})
    b = JSONObject(json_object={})
    assert a.id != b.id


def test_json_object_roundtrip():
    """JSONObject survives model_dump / model_validate round-trip."""
    obj = JSONObject(
        json_schema={"type": "object"},
        json_object={"answer": 42},
        metadata={"tag": "roundtrip"},
    )
    dumped = obj.model_dump()
    restored = JSONObject.model_validate(dumped)
    assert restored.json_object == obj.json_object
    assert restored.json_schema == obj.json_schema
    assert restored.metadata == obj.metadata
    assert restored.id == obj.id


# ---------------------------------------------------------------------------
# Context.json_objects field
# ---------------------------------------------------------------------------


def _make_context(**kwargs) -> Context:
    defaults = dict(
        original_question_uuid=None,
        actual_question_uuid=None,
        question="Q",
        source="test",
        agent="test-agent",
    )
    return Context(**{**defaults, **kwargs})


def test_context_json_objects_defaults_empty():
    """Context.json_objects is an empty list when not provided."""
    ctx = _make_context()
    assert ctx.json_objects == []


def test_context_json_objects_stores_items():
    """Context.json_objects holds JSONObject instances."""
    objs = [
        JSONObject(json_object={"a": 1}),
        JSONObject(json_object={"b": 2}),
    ]
    ctx = _make_context(json_objects=objs)
    assert len(ctx.json_objects) == 2
    assert ctx.json_objects[0].json_object == {"a": 1}
    assert ctx.json_objects[1].json_object == {"b": 2}


def test_context_json_objects_roundtrip():
    """Context.json_objects survives model_dump / model_validate round-trip."""
    objs = [JSONObject(json_object={"key": "val"}, metadata={"m": 1})]
    ctx = _make_context(json_objects=objs)
    dumped = ctx.model_dump()
    restored = Context.model_validate(dumped)
    assert len(restored.json_objects) == 1
    assert restored.json_objects[0].json_object == {"key": "val"}
    assert restored.json_objects[0].metadata == {"m": 1}
