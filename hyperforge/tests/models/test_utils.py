from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from hyperforge.api.utils import to_strict_json_schema
from inline_snapshot import snapshot
from pydantic import BaseModel, Field


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
