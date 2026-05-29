from __future__ import annotations

import inspect
from typing import Any, List, TypeVar

import pydantic
from typing_extensions import TypeGuard

_T = TypeVar("_T")

FLOW_PROPERTIES = [
    "next_agent",
    "fallback",
    "then",
    "else_",
    "agents",
    "registered_agents",
    "agent",
]


def to_strict_json_schema(
    model: type[pydantic.BaseModel] | pydantic.TypeAdapter[Any],
    exclude_properties: List[str] = [],
    exclude_defs: List[str] = [],
) -> dict[str, Any]:
    if inspect.isclass(model) and is_basemodel_type(model):
        schema = model.model_json_schema()
    elif isinstance(model, pydantic.TypeAdapter):
        schema = model.json_schema()
    else:
        raise TypeError(
            f"Non BaseModel types are only supported with Pydantic v2 - {model}"
        )

    for exclude in exclude_properties:
        if "properties" in schema and exclude in schema["properties"]:
            del schema["properties"][exclude]
    for exclude in exclude_defs:
        if "$defs" in schema and exclude in schema["$defs"]:
            del schema["$defs"][exclude]
    return _ensure_strict_json_schema(schema, path=(), root=schema)


def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    """Mutates the given JSON schema to ensure it conforms to the `strict` standard
    that the API expects.
    """
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    # defs = json_schema.get("$defs")
    # if is_dict(defs):
    #     for def_name, def_schema in defs.items():
    #         _ensure_strict_json_schema(
    #             def_schema, path=(*path, "$defs", def_name), root=root
    #         )

    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = [prop for prop in properties.keys()]
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema, path=(*path, "properties", key), root=root
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(
            items, path=(*path, "items"), root=root
        )

    # unions
    one_of = json_schema.get("oneOf")
    if isinstance(one_of, list):
        json_schema["oneOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "oneOf", str(i)), root=root
            )
            for i, variant in enumerate(one_of)
        ]

    # unions
    any_of = json_schema.get("anyOf")
    if isinstance(any_of, list):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant, path=(*path, "anyOf", str(i)), root=root
            )
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if isinstance(all_of, list):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0], path=(*path, "allOf", "0"), root=root
                )
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry, path=(*path, "allOf", str(i)), root=root
                )
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", "") is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref is not None:
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}"
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied,
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid.
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert is_dict(value), (
            f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        )
        resolved = value

    return resolved


def is_basemodel_type(typ: type) -> TypeGuard[type[pydantic.BaseModel]]:
    if not inspect.isclass(typ):
        return False
    return issubclass(typ, pydantic.BaseModel)


def is_dataclass_like_type(typ: type) -> bool:
    """Returns True if the given type likely used `@pydantic.dataclass`"""
    return hasattr(typ, "__pydantic_config__")


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)


async def clean_up_items(items: dict[str, Any], filtered: list[str]) -> dict[str, Any]:
    """Cleans up the items section of the schema by removing filtered agents and drivers from the references."""
    if "discriminator" in items and "mapping" in items["discriminator"]:
        for name, module in list(items["discriminator"]["mapping"].items()):
            if filtered and module.split("/")[-1] in filtered:
                del items["discriminator"]["mapping"][name]
    if "oneOf" in items:
        for item in list(items["oneOf"]):
            if "$ref" in item:
                module_name = item["$ref"].split("/")[-1]
                if module_name in filtered:
                    items["oneOf"].remove(item)
    return items


async def cleanup_anyof(anyof: list[dict[str, Any]], filtered: list[str]):
    """Cleans up the anyof section of the schema by removing filtered agents and drivers from the references."""
    mapping = {}
    if "discriminator" in anyof[0]:
        anyof_mapping = anyof[0]["discriminator"].get("mapping", {})
        for name, module in anyof_mapping.items():
            module_name = module.split("/")[-1]
            if all(module_name not in fa for fa in filtered):
                mapping[name] = module
        anyof[0]["discriminator"]["mapping"] = mapping
    if "oneOf" in anyof[0]:
        for module in anyof[0]["oneOf"]:
            if "$ref" in module:
                module_name = module["$ref"].split("/")[-1]
                if module_name in filtered:
                    anyof[0]["oneOf"].remove(module)
    return anyof


async def cleanup_properties(
    properties: dict[str, Any], filtered_agents: list[str], filtered_drivers: list[str]
) -> dict[str, Any]:
    """Cleans up the properties section of the schema by removing filtered agents and drivers from the references."""
    filtered = filtered_agents + filtered_drivers
    steps = ["preprocess", "context", "generation", "postprocess", "drivers"]
    for step in steps:
        if step not in properties:
            continue
        items = properties[step].get("items")
        if items:
            properties[step]["items"] = await clean_up_items(items, filtered)
    return properties


async def cleanup_definitions(
    definitions: dict[str, Any], filtered_agents: list[str]
) -> dict[str, Any]:
    """Cleans up the definitions section of the schema by removing filtered agents from the definitions."""
    for item, item_schema in definitions.items():
        if item.endswith("AgentConfig"):
            if item in filtered_agents:
                del definitions[item]
                continue
            else:
                properties = item_schema.get("properties", {})
                for property in FLOW_PROPERTIES:
                    if property in item_schema["properties"]:
                        if "anyOf" in properties[property]:
                            anyof = properties[property]["anyOf"]
                            if "items" in anyof[0]:
                                items = anyof[0]["items"]
                                properties[property]["anyOf"][0][
                                    "items"
                                ] = await clean_up_items(items, filtered_agents)
                            else:
                                properties[property]["anyOf"] = await cleanup_anyof(
                                    anyof, filtered_agents
                                )
                        elif "items" in properties[property]:
                            items = properties[property]["items"]
                            properties[property]["items"] = await clean_up_items(
                                items, filtered_agents
                            )
                        elif "discriminator" in properties[property]:
                            properties[property] = await clean_up_items(
                                properties[property], filtered_agents
                            )
                item_schema["properties"] = properties
        definitions[item] = item_schema

    return definitions
