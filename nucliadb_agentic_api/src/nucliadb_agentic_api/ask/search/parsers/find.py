from nucliadb_models import retrieval as retrieval_models
from nucliadb_models import search as search_models
from nucliadb_models.common import FieldTypeName, Paragraph
from nucliadb_models.filters import (
    And,
    DateCreated,
    DateModified,
    Entity,
    Field,
    FieldFilterExpression,
    FieldMimetype,
    FilterExpression,
    Generated,
    Keyword,
    Kind,
    Label,
    Language,
    Not,
    Or,
    OriginCollaborator,
    OriginMetadata,
    OriginPath,
    OriginSource,
    OriginTag,
    ParagraphFilterExpression,
    Resource,
    ResourceMimetype,
    Status,
)
from nucliadb_models.labels import translate_alias_to_system_label
from nucliadb_models.metadata import ResourceProcessingStatus
from nucliadb_models.retrieval import RetrievalRequest
from nucliadb_models.search import Filter, FindRequest
from nucliadb_protos import knowledgebox_pb2
from nucliadb_sdk import NucliaDBAsync
from pydantic import ValidationError

from nucliadb_agentic_api.ask import logger
from nucliadb_agentic_api.ask.exceptions import (
    InternalParserError,
    InvalidQueryError,
)
from nucliadb_agentic_api.ask.search.parsers.fetcher import (
    Fetcher,
)
from nucliadb_agentic_api.ask.search.rerankers import (
    NoopReranker,
    PredictReranker,
    Reranker,
)

# Filters that end up as a facet
FacetFilter = (
    OriginTag
    | Label
    | ResourceMimetype
    | FieldMimetype
    | Entity
    | Language
    | OriginMetadata
    | OriginPath
    | Generated
    | Kind
    | OriginCollaborator
    | OriginSource
    | Status
)


DEFAULT_GENERIC_SEMANTIC_THRESHOLD = 0.7
CLASSIFICATION_LABEL_PREFIX = "/l/"


async def parse_find(
    kbid: str, find_request: FindRequest, reader_sdk: NucliaDBAsync
) -> tuple[Fetcher, RetrievalRequest, Reranker]:
    # This is a thin layer to convert a FindRequest into a RetrievalRequest +
    # some bw/c stuff we need while refactoring and decoupling code

    fetcher = Fetcher(
        kbid,
        query=find_request.query,
        user_vector=find_request.vector,
        vectorset=find_request.vectorset,
        rephrase=find_request.rephrase,
        rephrase_prompt=find_request.rephrase_prompt,
        generative_model=find_request.generative_model,
        query_image=find_request.query_image,
    )
    parser = FindParser(kbid, find_request, fetcher)
    retrieval_request, reranker = await parser.parse(reader_sdk=reader_sdk)
    return fetcher, retrieval_request, reranker


class FindParser:
    def __init__(self, kbid: str, item: FindRequest, fetcher: Fetcher):
        self.kbid = kbid
        self.item = item
        self.fetcher = fetcher

        # cached data while parsing
        self._query: retrieval_models.RawQuery | None = None

    async def parse(
        self, reader_sdk: NucliaDBAsync
    ) -> tuple[RetrievalRequest, Reranker]:
        self._validate_request()

        top_k = self.item.top_k

        # parse search types (features)

        self._query = retrieval_models.RawQuery()

        if search_models.FindOptions.KEYWORD in self.item.features:
            self._query.keyword = await parse_keyword_query(
                self.item, fetcher=self.fetcher
            )  # type: ignore

        if search_models.FindOptions.SEMANTIC in self.item.features:
            self._query.semantic = await parse_semantic_query(
                self.item, fetcher=self.fetcher
            )  # type: ignore

        if search_models.FindOptions.RELATIONS in self.item.features:
            # skip, we'll do something about this later on
            pass

        if search_models.FindOptions.GRAPH in self.item.features:
            self._query.graph = await self._parse_graph_query()

        filters = await self._parse_filters(reader_sdk=reader_sdk)

        # rank fusion is just forwarded to /retrieve
        rank_fusion = self.item.rank_fusion

        try:
            reranker = self._parse_reranker()
        except ValidationError as exc:
            raise InternalParserError(f"Parsing error in reranker: {exc!s}") from exc

        # As we'll call /retrieve, that has rank fusion integrated, we have to
        # make sure we ask for enough results to rerank.
        if isinstance(reranker, PredictReranker):
            top_k = max(top_k, reranker.window)

        retrieval = RetrievalRequest(
            query=self._query,
            top_k=top_k,
            filters=filters,
            rank_fusion=rank_fusion,
        )
        return retrieval, reranker

    def _validate_request(self):
        # synonyms are not compatible with vector/graph search
        if (
            self.item.with_synonyms
            and self.item.query
            and (
                search_models.FindOptions.SEMANTIC in self.item.features
                or search_models.FindOptions.RELATIONS in self.item.features
                or search_models.FindOptions.GRAPH in self.item.features
            )
        ):
            raise InvalidQueryError(
                "synonyms",
                "Search with custom synonyms is only supported on paragraph and document search",
            )

        if (
            search_models.FindOptions.SEMANTIC in self.item.features
            and should_disable_vector_search(self.item)
        ):
            self.item.features.remove(search_models.FindOptions.SEMANTIC)

        if (
            self.item.graph_query
            and search_models.FindOptions.GRAPH not in self.item.features
        ):
            raise InvalidQueryError(
                "graph_query", "Using a graph query requires enabling graph feature"
            )

    async def _parse_graph_query(self) -> retrieval_models.GraphQuery:
        if self.item.graph_query is None:
            raise InvalidQueryError(
                "graph_query", "Graph query must be provided when using graph search"
            )
        return retrieval_models.GraphQuery(query=self.item.graph_query)

    async def _parse_filters(
        self, reader_sdk: NucliaDBAsync
    ) -> retrieval_models.Filters:
        assert self._query is not None, "query must be parsed before filters"

        # this is a conversion between /find filters to /retrieve filters. As
        # /find keeps maintaining old filter style, we must convert from one to
        # another

        has_old_filters = (
            len(self.item.filters) > 0
            or len(self.item.resource_filters) > 0
            or len(self.item.fields) > 0
            or len(self.item.keyword_filters) > 0
            or self.item.range_creation_start is not None
            or self.item.range_creation_end is not None
            or self.item.range_modification_start is not None
            or self.item.range_modification_end is not None
        )
        if self.item.filter_expression is not None and has_old_filters:
            raise InvalidQueryError(
                "filter_expression", "Cannot mix old filters with filter_expression"
            )

        filter_expression = None

        if has_old_filters:
            # convert old filters into a filter expression

            operator = FilterExpression.Operator.AND
            field_expression: list[FieldFilterExpression] = []
            paragraph_expression: list[ParagraphFilterExpression] = []

            if self.item.range_creation_start or self.item.range_creation_end:
                field_expression.append(
                    DateCreated(
                        since=self.item.range_creation_start,
                        until=self.item.range_creation_end,
                    )
                )

            if self.item.range_modification_start or self.item.range_modification_end:
                field_expression.append(
                    DateModified(
                        since=self.item.range_modification_start,
                        until=self.item.range_modification_end,
                    )
                )

            if self.item.filters:
                classification_labels = await self.fetcher.get_classification_labels(
                    reader_sdk=reader_sdk
                )
                field_exprs, paragraph_expr = convert_labels_to_filter_expressions(
                    self.item.filters, classification_labels
                )
                if field_exprs:
                    field_expression.extend(field_exprs)
                if paragraph_expr:
                    paragraph_expression.append(paragraph_expr)

            if self.item.keyword_filters:
                # keyword filters
                for keyword_filter in self.item.keyword_filters:
                    if isinstance(keyword_filter, str):
                        field_expression.append(Keyword(word=keyword_filter))
                    else:
                        # model validates that one and only one of these match
                        if keyword_filter.all:
                            field_expression.append(
                                And(
                                    operands=[
                                        Keyword(word=word)
                                        for word in keyword_filter.all
                                    ]
                                )
                            )
                        elif keyword_filter.any:
                            field_expression.append(
                                Or(
                                    operands=[
                                        Keyword(word=word)
                                        for word in keyword_filter.any
                                    ]
                                )
                            )
                        elif keyword_filter.none:
                            field_expression.append(
                                Not(
                                    operand=Or(
                                        operands=[
                                            Keyword(word=word)
                                            for word in keyword_filter.none
                                        ]
                                    )
                                )
                            )
                        elif keyword_filter.not_all:
                            field_expression.append(
                                Not(
                                    operand=And(
                                        operands=[
                                            Keyword(word=word)
                                            for word in keyword_filter.not_all
                                        ]
                                    )
                                )
                            )

            if self.item.fields:
                operands: list[FieldFilterExpression] = []
                for key in self.item.fields:
                    parts = key.split("/")
                    try:
                        field_type = FieldTypeName.from_abbreviation(parts[0])
                    except KeyError:  # pragma: no cover
                        raise InvalidQueryError(
                            "fields",
                            f"field filter {key} has an invalid field type: {parts[0]}",
                        )
                    field_id = parts[1] if len(parts) > 1 else None
                    operands.append(Field(type=field_type, name=field_id))

                if len(operands) == 1:
                    field_expression.append(operands[0])
                elif len(operands) > 1:
                    field_expression.append(Or(operands=operands))

            if self.item.resource_filters:
                operands = []
                for key in self.item.resource_filters:
                    parts = key.split("/")
                    if len(parts) == 1:
                        operands.append(Resource(id=parts[0]))
                    else:
                        rid = parts[0]
                        field_type = FieldTypeName.from_abbreviation(parts[1])
                        field_id = parts[2] if len(parts) > 2 else None
                        operands.append(
                            And(
                                operands=[
                                    Resource(id=rid),
                                    Field(type=field_type, name=field_id),
                                ]
                            )
                        )

                if len(operands) == 1:
                    field_expression.append(operands[0])
                elif len(operands) > 1:
                    field_expression.append(Or(operands=operands))

            field = None
            if len(field_expression) == 1:
                field = field_expression[0]
            elif len(field_expression) > 1:
                field = And(operands=field_expression)

            paragraph = None
            if len(paragraph_expression) == 1:
                paragraph = paragraph_expression[0]
            elif len(paragraph_expression) > 1:
                paragraph = And(operands=paragraph_expression)

            if field or paragraph:
                filter_expression = FilterExpression(
                    field=field, paragraph=paragraph, operator=operator
                )

        if self.item.filter_expression is not None:
            filter_expression = self.item.filter_expression

        return retrieval_models.Filters(
            filter_expression=filter_expression,
            show_hidden=self.item.show_hidden,
            security=self.item.security,
            with_duplicates=self.item.with_duplicates,
        )

    def _parse_reranker(self) -> Reranker:
        reranker: Reranker
        top_k = self.item.top_k

        if isinstance(self.item.reranker, search_models.RerankerName):
            if self.item.reranker == search_models.RerankerName.NOOP:
                reranker = NoopReranker()

            elif self.item.reranker == search_models.RerankerName.PREDICT_RERANKER:
                # for predict rearnker, by default, we want a x2 factor with a
                # top of 200 results
                reranker = PredictReranker(window=min(top_k * 2, 200))

            else:
                raise InternalParserError(
                    f"Unknown reranker algorithm: {self.item.reranker}"
                )

        elif isinstance(self.item.reranker, search_models.PredictReranker):
            user_window = self.item.reranker.window
            reranker = PredictReranker(window=min(max(user_window or 0, top_k), 200))

        else:
            raise InternalParserError(f"Unknown reranker {self.item.reranker}")

        return reranker


async def parse_keyword_query(
    item: search_models.BaseSearchRequest,
    *,
    fetcher: Fetcher,
) -> retrieval_models.KeywordQuery:
    query = item.query

    # only when a query image is used, we use the rephrased query for keyword
    # search
    if item.query_image is not None:
        rephrased_query = await fetcher.get_rephrased_query()
        if rephrased_query is not None:
            query = rephrased_query

    min_score = parse_keyword_min_score(item.min_score)

    return retrieval_models.KeywordQuery(
        query=query,
        # Synonym checks are done at the retrieval endpoint already
        with_synonyms=item.with_synonyms,
        min_score=min_score,
    )


def should_disable_vector_search(request: search_models.BaseSearchRequest) -> bool:
    if has_user_vectors(request):
        return False

    if is_exact_match_only_query(request):
        return True

    return is_empty_query(request)


def has_user_vectors(request: search_models.BaseSearchRequest) -> bool:
    return request.vector is not None and len(request.vector) > 0


def is_exact_match_only_query(request: search_models.BaseSearchRequest) -> bool:
    """
    '"something"' -> True
    'foo "something" else' -> False
    """
    query = request.query.strip()
    return len(query) > 0 and query.startswith('"') and query.endswith('"')


def is_empty_query(request: search_models.BaseSearchRequest) -> bool:
    return len(request.query) == 0


def parse_keyword_min_score(
    min_score: float | search_models.MinScore | None,
) -> float:
    # Keep backward compatibility with the deprecated min_score payload
    # parameter being a float (specifying semantic)
    if min_score is None or isinstance(min_score, float):
        return 0.0
    else:
        return min_score.bm25


async def parse_semantic_query(
    item: search_models.SearchRequest | search_models.FindRequest,
    *,
    fetcher: Fetcher,
) -> retrieval_models.SemanticQuery:
    vectorset = await fetcher.get_vectorset()
    query = await fetcher.get_query_vector()

    min_score = await parse_semantic_min_score(item.min_score, fetcher=fetcher)

    return retrieval_models.SemanticQuery(
        query=query, vectorset=vectorset, min_score=min_score
    )


async def parse_semantic_min_score(
    min_score: float | search_models.MinScore | None,
    *,
    fetcher: Fetcher,
) -> float:
    if min_score is None:
        min_score = None
    elif isinstance(min_score, float):
        min_score = min_score
    else:
        min_score = min_score.semantic
    if min_score is None:
        # min score not defined by the user, we'll try to get the default
        # from Predict API
        min_score = await fetcher.get_semantic_min_score()
        if min_score is None:
            logger.warning(
                "Semantic threshold not found in query information, using default",
                extra={"kbid": fetcher.kbid},
            )
            min_score = DEFAULT_GENERIC_SEMANTIC_THRESHOLD

    return min_score


def convert_labels_to_filter_expressions(
    label_filters: list[str] | list[Filter],
    classification_labels: knowledgebox_pb2.Labels,
) -> tuple[list[FieldFilterExpression], ParagraphFilterExpression | None]:
    field_expressions: list[FieldFilterExpression] = []
    paragraph_expressions: list[ParagraphFilterExpression] = []

    for label_filter in label_filters:
        if isinstance(label_filter, str):
            # translate_label
            if len(label_filter) == 0:
                raise InvalidQueryError("filters", "Invalid empty label")
            if label_filter[0] != "/":
                raise InvalidQueryError(
                    "filters",
                    f"Invalid label. It must start with a `/`: {label_filter}",
                )

            label = translate_label(label_filter)
            facet_filter = filter_from_facet(label)

            if is_paragraph_label(label, classification_labels):
                paragraph_expressions.append(facet_filter)  # type: ignore[arg-type]
            else:
                field_expressions.append(facet_filter)  # type: ignore[arg-type]

        else:
            combinator: (
                type[And[FieldFilterExpression]] | type[Or[FieldFilterExpression]]
            )
            if label_filter.all:
                labels = label_filter.all
                combinator, negate = And, False
            elif label_filter.any:
                labels = label_filter.any
                combinator, negate = Or, False
            elif label_filter.none:
                labels = label_filter.none
                combinator, negate = And, True
            elif label_filter.not_all:
                labels = label_filter.not_all
                combinator, negate = Or, True
            else:
                # Empty filter, should not happen due to validation, but skip just in case
                continue

            # equivalent to split_labels
            field = []
            paragraph = []
            for label in labels:
                label = translate_label(label)
                expr = filter_from_facet(label)

                if negate:
                    expr = Not(operand=expr)  # type: ignore

                if is_paragraph_label(label, classification_labels):
                    paragraph.append(expr)
                else:
                    field.append(expr)

            if len(paragraph) > 0 and not (combinator == And and negate is False):
                raise InvalidQueryError(
                    "filters",
                    "Paragraph labels can only be used with 'all' filter",
                )

            if len(field) == 1:
                field_expressions.append(field[0])  # type: ignore
            elif len(field) > 1:
                field_expressions.append(combinator(operands=field))  # type: ignore

            if len(paragraph) == 1:
                paragraph_expressions.append(paragraph[0])  # type: ignore
            elif len(paragraph) > 1:
                paragraph_expressions.append(combinator(operands=paragraph))  # type: ignore

    if len(paragraph_expressions) == 1:
        paragraph_expression = paragraph_expressions[0]  # type: ignore
    elif len(paragraph_expressions) > 1:
        paragraph_expression = And(operands=paragraph_expressions)  # type: ignore
    else:
        paragraph_expression = None

    return field_expressions, paragraph_expression


def filter_from_facet(facet: str) -> FacetFilter:
    expr: FacetFilter

    if facet.startswith("/t/"):
        value = facet.removeprefix("/t/")
        expr = OriginTag(tag=value)

    elif facet.startswith("/l/"):
        value = facet.removeprefix("/l/")
        parts = value.split("/", maxsplit=1)
        if len(parts) == 1:
            type = parts[0]
            expr = Label(labelset=type)
        else:
            type, subtype = parts
            expr = Label(labelset=type, label=subtype)

    elif facet.startswith("/n/i/"):
        value = facet.removeprefix("/n/i/")
        parts = value.split("/", maxsplit=1)
        if len(parts) == 1:
            type = parts[0]
            expr = ResourceMimetype(type=type)
        else:
            type, subtype = parts
            expr = ResourceMimetype(type=type, subtype=subtype)

    elif facet.startswith("/mt/"):
        value = facet.removeprefix("/mt/")
        parts = value.split("/", maxsplit=1)
        if len(parts) == 1:
            type = parts[0]
            expr = FieldMimetype(type=type)
        else:
            type, subtype = parts
            expr = FieldMimetype(type=type, subtype=subtype)

    elif facet.startswith("/e/"):
        value = facet.removeprefix("/e/")
        parts = value.split("/", maxsplit=1)
        if len(parts) == 1:
            subtype = parts[0]
            expr = Entity(subtype=subtype)
        else:
            subtype, value = parts
            expr = Entity(subtype=subtype, value=value)

    elif facet.startswith("/s/p"):
        value = facet.removeprefix("/s/p/")
        expr = Language(language=value, only_primary=True)

    elif facet.startswith("/s/s"):
        value = facet.removeprefix("/s/s/")
        expr = Language(language=value, only_primary=False)

    elif facet.startswith("/m/"):
        value = facet.removeprefix("/m/")
        parts = value.split("/", maxsplit=1)
        if len(parts) == 1:
            field = parts[0]
            expr = OriginMetadata(field=field)
        else:
            field, value = parts
            expr = OriginMetadata(field=field, value=value)

    elif facet.startswith("/p/"):
        value = facet.removeprefix("/p/")
        expr = OriginPath(prefix=value)

    elif facet.startswith("/g/da"):
        value = facet.removeprefix("/g/da")
        expr = expr = Generated(by="data-augmentation")
        if value.removeprefix("/"):
            expr.da_task = value.removeprefix("/")

    elif facet.startswith("/k/"):
        value = facet.removeprefix("/k/")
        try:
            kind = Paragraph.TypeParagraph(value.upper())
        except ValueError:
            raise InvalidQueryError("filters", f"invalid paragraph kind: {value}")
        expr = Kind(kind=kind)

    elif facet.startswith("/u/o/"):
        value = facet.removeprefix("/u/o/")
        expr = OriginCollaborator(collaborator=value)

    elif facet.startswith("/u/s"):
        value = facet.removeprefix("/u/s")
        expr = OriginSource()
        if value.removeprefix("/"):
            expr.id = value.removeprefix("/")

    elif facet.startswith("/n/s/"):
        value = facet.removeprefix("/n/s/")
        try:
            status = ResourceProcessingStatus(value.upper())
        except ValueError:
            raise InvalidQueryError(
                "filters", f"invalid resource processing status: {value}"
            )
        expr = Status(status=status)

    else:
        raise InvalidQueryError("filters", f"invalid filter: {facet}")

    return expr


def translate_label(literal: str) -> str:
    if len(literal) == 0:
        raise InvalidQueryError("filters", "Invalid empty label")
    if literal[0] != "/":
        raise InvalidQueryError(
            "filters", f"Invalid label. It must start with a `/`: {literal}"
        )
    return translate_alias_to_system_label(literal)


def is_paragraph_label(
    label: str, classification_labels: knowledgebox_pb2.Labels
) -> bool:
    if len(label) == 0 or label[0] != "/":
        return False
    if not label.startswith(CLASSIFICATION_LABEL_PREFIX):
        return False
    # Classification labels should have the form /l/labelset/label
    # REVIEW: there's no technical reason why this has to be like this (/l/labelset could be valid)
    parts = label.split("/")
    if len(parts) < 4:
        return False
    labelset_id = parts[2]

    try:
        labelset: knowledgebox_pb2.LabelSet | None = classification_labels.labelset.get(
            labelset_id
        )
        if labelset is None:
            return False
        return knowledgebox_pb2.LabelSet.LabelSetKind.PARAGRAPHS in labelset.kind
    except KeyError:
        # labelset_id not found
        return False
