from typing import Literal, Optional

from hyperforge import logger
from nucliadb_models import filters as ndb_filters
from nucliadb_models.labels import LABEL_QUERY_ALIASES as LABEL_ALIASES
from nucliadb_models.search import SyncAskResponse


def get_chunk_text(ask_response: SyncAskResponse, chunk_id: str) -> str:
    ids = chunk_id.split("/")
    resource_id = ids[0]
    resource = ask_response.retrieval_results.resources[resource_id]
    field_id = f"/{ids[1]}/{ids[2]}" if len(ids) > 2 else ""
    try:
        # Try to get the text from the main retrieval results first
        return resource.fields[field_id].paragraphs[chunk_id].text
    except KeyError:
        # If not found, try to get it from the augmented context, as it may be a chunk that was augmented as part of the RAG strategies.
        if ask_response.augmented_context is not None:
            try:
                return ask_response.augmented_context.paragraphs[chunk_id].text
            except KeyError:
                # If still not found, return an empty string
                pass
        return ""


def spit_by_filter_type(text: str) -> Optional[tuple[str, str]]:
    """
    Splits the filter type from the rest of the filter.
    Examples:
        /l/foo/bar -> ("classification.labels", "foo/bar")
        /classification.labels/foo/bar -> ("classification.labels", "foo/bar")
        /n/i/application/pdf -> ("icon", "application/pdf")
    """
    for alias, replacement in LABEL_ALIASES.items():
        if text.startswith(replacement + "/"):
            return alias, text[len(replacement) :]
        if text.startswith(alias + "/"):
            return alias, text[len(alias) :]
    return None


def _convert_classification_labels(filter: str) -> ndb_filters.Label:
    labelset, label = split_head_and_tail(filter)
    return ndb_filters.Label(labelset=labelset, label=label)


def _convert_to_field_mimetype(filter: str) -> ndb_filters.FieldMimetype:
    itype, isubtype = split_head_and_tail(filter)
    return ndb_filters.FieldMimetype(
        type=itype,
        subtype=isubtype,
    )


def _convert_to_language(filter_type: str, filter: str) -> ndb_filters.Language:
    only_primary = "metadata.language" == filter_type
    return ndb_filters.Language(language=filter, only_primary=only_primary)


def _convert_to_origin(
    filter_type: str, filter: str
) -> (
    ndb_filters.OriginTag
    | ndb_filters.OriginMetadata
    | ndb_filters.OriginPath
    | ndb_filters.OriginCollaborator
    | ndb_filters.OriginSource
):
    if filter_type == "origin.tags":
        return ndb_filters.OriginTag(tag=filter)
    elif filter_type == "origin.path":
        return ndb_filters.OriginPath(prefix=filter)
    elif filter_type == "origin.metadata":
        key, value = split_head_and_tail(filter)
        return ndb_filters.OriginMetadata(field=key, value=value)
    elif filter_type == "origin.collaborators":
        return ndb_filters.OriginCollaborator(collaborator=filter)
    elif filter_type == "origin.sources":
        return ndb_filters.OriginSource(id=filter)
    else:
        raise ValueError(f"Unsupported origin filter type: {filter_type}")


def _convert_to_entities(filter: str) -> ndb_filters.Entity:
    entity_type, entity_value = split_head_and_tail(filter)
    return ndb_filters.Entity(subtype=entity_type, value=entity_value)


def _convert_to_field(filter: str) -> ndb_filters.Field:
    ftype, fname = split_head_and_tail(filter)
    return ndb_filters.Field(type=ftype, name=fname)


def to_field_filter_expression(
    filter: str,
) -> Optional[ndb_filters.FieldFilterExpression]:
    """
    Converts plain string labels to FieldFilterExpression objects
    Examples:
        /l/foo/bar -> Label(labelset="foo", label="bar")
        /n/i/application/pdf -> FieldMimetype(type="application", subtype="pdf")
    """
    parts = spit_by_filter_type(filter.lstrip("/"))
    if parts is None:
        logger.error(f"Could not parse filter: {filter}")
        return None
    filter_type, filter = parts
    if filter_type == "classification.labels":
        return _convert_classification_labels(filter)
    elif filter_type == "icon":
        return _convert_to_field_mimetype(filter)
    elif filter_type in ("metadata.language", "metadata.languages"):
        return _convert_to_language(filter_type, filter)
    elif filter_type in (
        "origin.tags",
        "origin.path",
        "origin.metadata",
        "origin.collaborators",
        "origin.sources",
    ):
        return _convert_to_origin(filter_type, filter)
    elif filter_type == "entities":
        return _convert_to_entities(filter)
    elif filter_type == "field":
        return _convert_to_field(filter)
    elif filter_type == "generated.data-augmentation":
        return ndb_filters.Generated(by="data-augmentation")
    else:
        logger.error(f"Unhandled filter format: {filter}")
        return None


def to_resource_filter_expression(
    filter: str,
) -> Optional[ndb_filters.ResourceFilterExpression]:
    """
    Converts plain string labels to ResourceFilterExpression objects
    Examples:
        /l/foo/bar -> Label(labelset="foo", label="bar")
        /n/i/application/pdf -> ResourceMimetype(type="application", subtype="pdf")
    """
    parts = spit_by_filter_type(filter.lstrip("/"))
    if parts is None:
        logger.error(f"Could not parse filter: {filter}")
        return None
    filter_type, filter = parts
    if filter_type == "classification.labels":
        return _convert_classification_labels(filter)
    elif filter_type == "icon":
        itype, isubtype = split_head_and_tail(filter)
        return ndb_filters.ResourceMimetype(
            type=itype,
            subtype=isubtype,
        )
    elif filter_type in ("metadata.language", "metadata.languages"):
        return _convert_to_language(filter_type, filter)
    elif filter_type in (
        "origin.tags",
        "origin.path",
        "origin.metadata",
        "origin.collaborators",
        "origin.sources",
    ):
        return _convert_to_origin(filter_type, filter)
    else:
        logger.error(f"Unhandled filter format: {filter}")
        return None


def split_head_and_tail(text: str) -> tuple[str, Optional[str]]:
    parts = text.lstrip("/").split("/", maxsplit=1)
    head = parts[0]
    tail = None
    if len(parts) > 1:
        tail = parts[1]
    return head, tail


def combine_filter_expressions(
    expressions: list[ndb_filters.FilterExpression],
    operator: Literal["and", "or"] = "and",
) -> ndb_filters.FilterExpression:
    """
    Merge the two filter expressions into a single expression.
    """
    if len(expressions) == 0:
        raise ValueError("At least one filter expression is required to combine")
    elif len(expressions) == 1:
        return expressions[0]
    # Make sure all expressions have the same operator
    one = expressions[0]
    for other in expressions[1:]:
        if one.operator != other.operator:
            raise ValueError(
                "Cannot combine filter expressions with different operators"
            )

    operator_klass = ndb_filters.And if operator == "and" else ndb_filters.Or
    result = ndb_filters.FilterExpression(
        operator=one.operator,
    )
    field_expressions = [expr.field for expr in expressions if expr.field is not None]
    if len(field_expressions) > 1:
        result.field = operator_klass(operands=field_expressions)
    elif len(field_expressions) == 1:
        result.field = field_expressions[0]
    paragraph_expressions = [
        expr.paragraph for expr in expressions if expr.paragraph is not None
    ]
    if len(paragraph_expressions) > 1:
        result.paragraph = operator_klass(operands=paragraph_expressions)
    elif len(paragraph_expressions) == 1:
        result.paragraph = paragraph_expressions[0]
    return result


def combine_catalog_filter_expressions(
    expressions: list[ndb_filters.CatalogFilterExpression],
    operator: Literal["and", "or"] = "and",
) -> ndb_filters.CatalogFilterExpression:
    """
    Merge the two catalog filter expressions.
    """
    if len(expressions) == 0:
        raise ValueError("At least one filter expression is required to combine")
    elif len(expressions) == 1:
        return expressions[0]
    else:
        operator_klass = ndb_filters.And if operator == "and" else ndb_filters.Or
        return ndb_filters.CatalogFilterExpression(
            resource=operator_klass(operands=[expr.resource for expr in expressions])
        )
