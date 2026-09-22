import json
from copy import deepcopy
from typing import Any, Literal

from jsonschema import ValidationError
from jsonschema import validate as validate_json_schema

from hyperforge import logger
from hyperforge.exceptions import CouldNotParse, ParseJsonSchemaException


def set_additional_properties_false(obj: dict):
    """
    Recursively set 'additionalProperties': False in the given object and in every object inside '$defs'.
    Also fixes malformed additionalProperties (empty objects, etc.)
    """
    if not isinstance(obj, dict):
        return

    # Fix malformed additionalProperties (like empty {})
    if (
        "additionalProperties" in obj
        and isinstance(obj["additionalProperties"], dict)
        and not obj["additionalProperties"]
    ):
        obj["additionalProperties"] = False

    # Only set if type is object or if 'properties' exists and additionalProperties not properly set
    if (
        obj.get("type") == "object"
        or "properties" in obj
        and "additionalProperties" not in obj
    ):
        obj["additionalProperties"] = False

    # Recurse into $defs if present
    if "$defs" in obj and isinstance(obj["$defs"], dict):
        for def_obj in obj["$defs"].values():
            set_additional_properties_false(def_obj)

    # Recurse into properties
    if "properties" in obj and isinstance(obj["properties"], dict):
        for prop_obj in obj["properties"].values():
            set_additional_properties_false(prop_obj)

    # Recurse into items (for arrays)
    if "items" in obj:
        set_additional_properties_false(obj["items"])

    # Recurse into anyOf/oneOf/allOf
    for key in ("anyOf", "oneOf", "allOf"):
        if key in obj and isinstance(obj[key], list):
            for variant in obj[key]:
                set_additional_properties_false(variant)


def remove_additional_properties(obj: Any):
    """
    Recursively remove 'additionalProperties' from a JSON schema.
    """
    if isinstance(obj, dict):
        obj.pop("additionalProperties", None)
        for value in obj.values():
            remove_additional_properties(value)
    elif isinstance(obj, list):
        for value in obj:
            remove_additional_properties(value)


def convert_nullable_type_arrays(obj: Any):
    """Convert JSON Schema nullable type unions to OpenAPI-style nullable fields."""
    if isinstance(obj, dict):
        schema_type = obj.get("type")
        if (
            isinstance(schema_type, list)
            and len(schema_type) == 2
            and "null" in schema_type
        ):
            obj["type"] = next(value for value in schema_type if value != "null")
            obj["nullable"] = True
        for value in obj.values():
            convert_nullable_type_arrays(value)
    elif isinstance(obj, list):
        for value in obj:
            convert_nullable_type_arrays(value)


def add_additional_properties_and_required(
    parameters: dict[str, Any],
) -> dict[str, Any]:
    """
    Add additionalProperties: False and required fields to the parameters schema if not present.
    This is needed for OpenAI function calling to work properly.
    """
    parameters.setdefault("properties", {})
    if "type" not in parameters or parameters["type"] != "object":
        return parameters

    for kind in ["properties", "$defs"]:
        if kind not in parameters or not isinstance(parameters[kind], dict):
            continue
        for prop_name, prop_schema in parameters[kind].items():
            if isinstance(prop_schema, dict) and prop_schema.get("type") == "object":
                parameters[kind][prop_name] = add_additional_properties_and_required(
                    prop_schema
                )
            elif (
                isinstance(prop_schema, dict)
                and prop_schema.get("type") == "array"
                and isinstance(prop_schema.get("items"), dict)
                and prop_schema["items"].get("type") == "object"
            ):
                prop_schema["items"] = add_additional_properties_and_required(
                    prop_schema["items"]
                )
            elif isinstance(prop_schema, dict):
                for combinator in ("anyOf", "oneOf", "allOf"):
                    if combinator in prop_schema and isinstance(
                        prop_schema[combinator], list
                    ):
                        for i, variant in enumerate(prop_schema[combinator]):
                            if (
                                isinstance(variant, dict)
                                and variant.get("type") == "object"
                            ):
                                prop_schema[combinator][i] = (
                                    add_additional_properties_and_required(variant)
                                )
                            elif (
                                isinstance(variant, dict)
                                and variant.get("type") == "array"
                                and isinstance(variant.get("items"), dict)
                                and variant["items"].get("type") == "object"
                            ):
                                variant["items"] = (
                                    add_additional_properties_and_required(
                                        variant["items"]
                                    )
                                )

    parameters["additionalProperties"] = False
    if "properties" in parameters:
        parameters["required"] = list(parameters["properties"].keys())

    return parameters


def convert_json_schema(
    schema: dict[str, Any],
    *,
    additional_properties: Literal[
        "default_to_false", "unsupported", "noop"
    ] = "default_to_false",
    required: Literal["set_if_no_default", "force", "noop"] = "set_if_no_default",
    output_format: Literal["flat", "function_call"] = "function_call",
    unsupported_attrs: list[str] | None = None,
    nullable: Literal["keyword", "noop"] = "noop",
) -> dict[str, Any]:
    """
    Convert from json_schema to the format expected by LLMs

    We support these two types of schemas:
    **Flat JSON Schema**
    ```
    {
        "title": "example",
        "description": "This is an example schema",
        "type": "object",
        "properties": {
            "example_property": {
                "type": "string",
                "description": "This is an example property"
            },
            "example_property_array": {
                "type": "array",
                "description": "This is an example array property",
                "items": {
                    "type": "integer"
                }
            }
        },
        "required": ["example_property", "example_property_array"]
    }
    ```
    **Function Call Schema**
    ```
    {
        "name": "example_function",
        "description": "This is an example function",
        "parameters": {
            "type": "object",
            "properties": {
                "example_property": {
                    "type": "string",
                    "description": "This is an example property"
                },
                "example_property_array": {
                    "type": "array",
                    "description": "This is an example array property",
                    "items": {
                        "type": "integer"
                    }
                }
            },
            "required": ["example_property", "example_property_array"],
            "additionalProperties": False
        }
    }
    ```
    However, the providers tend to only support one of them, so this function will convert the schema to the appropriate format.

    Args:
        schema (dict): The input JSON schema to convert, in the format "Flat JSON Schema" or "Function Call Schema"
        additional_properties (Literal["default_to_false", "noop", "unsupported"]): How to handle the AdditionalProperties field in the schema, some providers require it while others do not support it
        required (Literal["set_if_no_default", "force", "noop"]): How to handle the Required field in the schema
            - "set_if_no_default": Only set required if no default is provided
            - "force": Always set all fields as required
            - "noop": Do not modify the required fields
        output_format (Literal["flat", "function_call"]): The output format to convert the schema to
        unsupported_attrs (list[str] | None): A list of unsupported attributes for fields in the schema
        nullable (Literal["keyword", "noop"]): Convert nullable type arrays to the OpenAPI-style nullable keyword, or leave them unchanged

    Returns:
        model_json_schema (dict): A dictionary modeling the expected schema by LLM providers
    """
    converted_input_schema = deepcopy(schema)

    if "name" not in converted_input_schema and "title" not in converted_input_schema:
        raise ParseJsonSchemaException(
            "Input schema must contain a 'name' or 'title' field. Please add it to your schema definition."
        )

    def to_function_call_schema():
        # We interchangeably support "name" and "title", but finally set name
        if "name" not in converted_input_schema and "title" in converted_input_schema:
            converted_input_schema["name"] = converted_input_schema["title"]
            del converted_input_schema["title"]
        converted_input_schema["name"] = converted_input_schema["name"].replace(
            " ", "_"
        )
        # Flat JSON Schema detected
        if converted_input_schema.get("properties") is not None:
            converted_schema = {
                "name": converted_input_schema["name"],
                "description": converted_input_schema.get("description") or "",
                "parameters": {
                    k: v
                    for k, v in converted_input_schema.items()
                    if k not in ("description", "name")
                },
            }
        # Function Call Schema detected
        elif converted_input_schema.get("parameters") is not None:
            converted_schema = converted_input_schema
            converted_schema.setdefault("description", "")
        else:
            raise ParseJsonSchemaException(
                "Input schema must contain a 'properties' or 'parameters' field."
            )

        return converted_schema

    def to_flat_schema():
        # We interchangeably support "name" and "title", but finally set title
        if "title" not in converted_input_schema and "name" in converted_input_schema:
            converted_input_schema["name"] = converted_input_schema["name"].replace(
                " ", "_"
            )
            converted_input_schema["title"] = converted_input_schema["name"]
            del converted_input_schema["name"]

        # Flat JSON Schema detected
        if converted_input_schema.get("properties") is not None:
            converted_input_schema.setdefault("description", "")
            converted_schema = converted_input_schema
        # Function call schema detected
        elif converted_input_schema.get("parameters") is not None:
            if "properties" not in converted_input_schema["parameters"]:
                raise ParseJsonSchemaException(
                    "Schema must contain a 'properties' field inside 'parameters'"
                )
            converted_schema = {
                "title": converted_input_schema.get("title"),
                "description": converted_input_schema.get("description") or "",
                **converted_input_schema["parameters"],
            }
        else:
            raise ParseJsonSchemaException(
                "Input schema must contain a 'properties' or 'parameters' field."
            )
        return converted_schema

    if output_format == "function_call":
        converted_schema = to_function_call_schema()
        param_container = converted_schema["parameters"]
    else:
        converted_schema = to_flat_schema()
        param_container = converted_schema

    try:
        # If user did not specify `additionalProperties` we disable them
        if (
            additional_properties == "default_to_false"
            and "additionalProperties" not in param_container
        ):
            set_additional_properties_false(param_container)

        # If the LLM does not support `additionalProperties` we remove the field
        if (
            additional_properties == "unsupported"
            and "additionalProperties" in param_container
        ):
            remove_additional_properties(param_container)

        # If the user did not specify `required``, we set all properties without defaults as required
        if (required == "set_if_no_default") and "required" not in param_container:
            param_container["required"] = sorted(
                k
                for k, v in param_container["properties"].items()
                if "default" not in v
            )
        if required == "force":
            param_container["required"] = sorted(
                k for k, v in param_container["properties"].items()
            )

        if nullable == "keyword":
            convert_nullable_type_arrays(param_container)

        # Remove unsupported attributes
        if unsupported_attrs:
            properties = param_container.get("properties")
            if properties is not None:
                for property, content in properties.items():
                    for unsupported_attr in unsupported_attrs:
                        if unsupported_attr in content.keys():
                            param_container["properties"][property].pop(
                                unsupported_attr
                            )
    except ParseJsonSchemaException:
        raise
    except Exception:
        logger.exception("Failed to parse JSON schema for generative request")
        raise ParseJsonSchemaException(
            "The provided JSON Schema could not be parsed. Please review your schema and ensure it is valid."
        )

    return converted_schema


def validate(instance: dict | str, schema: dict) -> dict:
    """
    Wrapper around jsonschema.validate to raise our own exceptions and to hold any additional future validation logic.
    We handle here the conversion from string to dict if needed and return the converted dict for convenience.
    When instance is a string, CouldNotParse is raised with raw set to the original string so
    callers can attempt domain-specific repair.
    """
    raw = instance if isinstance(instance, str) else None
    try:
        if isinstance(instance, str):
            instance_dict = json.loads(
                instance.strip()
            )  # Strip really important here to remove any leading/trailing whitespace or linebreaks that could cause json.JSONDecodeError
        else:
            instance_dict = instance
        validate_json_schema(instance=instance_dict, schema=schema)
    except ValidationError as e:
        raise CouldNotParse(raw=raw) from e
    except json.JSONDecodeError as e:
        raise CouldNotParse("Invalid JSON format", raw=raw) from e
    return instance_dict
