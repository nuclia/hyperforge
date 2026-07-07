from typing import Any, List, Literal, Optional, Union

from hyperforge.context.config import ContextAgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType
from nucliadb_models.filters import FilterExpression
from nucliadb_models.search import (
    ChatOptions,
    CitationsType,
    CustomPrompt,
    Image,
    MaxTokens,
    MinScore,
    RagImagesStrategies,
    RagStrategies,
    RankFusion,
    RankFusionName,
    RerankerName,
)
from nucliadb_models.security import RequestSecurity
from pydantic import Field
from pydantic.config import ConfigDict


class AdvancedAskAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Knowledge Box advanced Ask")
    module: Literal["advanced_ask"] = "advanced_ask"
    sources: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "show_in_node": True,
        },
    )
    generative_model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.default),
        title="Generative model",
        description="Model used to generate answers",
    )
    features: Optional[list[ChatOptions]] = Field(
        default=None,
        title="Chat features",
        description="Features enabled for the chat endpoint. Semantic search is done if `semantic` is included. If `keyword` is included, the results will include matching paragraphs from the bm25 index. If `relations` is included, a graph of entities related to the answer is returned. `paragraphs` and `vectors` are deprecated, please use `keyword` and `semantic` instead",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    top_k: Optional[int] = Field(
        default=None,
        title="Top k",
        ge=1,
        le=200,
        description="The top most relevant results to fetch at the retrieval step. The maximum number of results allowed is 200.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    filter_expression: Optional[FilterExpression] = Field(
        default=None,
        title="Filter documents by an expression",
        description=(
            "Returns only documents that match this filter expression."
            "Filtering examples can be found here: https://docs.nuclia.dev/docs/rag/advanced/search-filters"
        ),
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    rephrase: Optional[bool] = Field(
        default=None,
        description=(
            "Rephrase the query for a more efficient retrieval. This will consume LLM tokens and make the request slower."
        ),
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    reranker: Optional[RerankerName] = Field(
        default=None,
        title="Reranker",
        description="Reranker to use for the search.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    security: Optional[RequestSecurity] = Field(
        default=None,
        title="Security",
        description="Security metadata for the request. If not provided, the search request is done without the security lookup phase.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    prompt: Optional[CustomPrompt] = Field(
        default=None,
        title="Prompts",
        description="Use to customize the prompts given to the generative model. Both system and user prompts can be customized.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    max_tokens: Optional[MaxTokens] = Field(
        default=None,
        title="Maximum LLM tokens to use for the request",
        description="Use to limit the amount of tokens used in the LLM context and/or for generating the answer. If not provided, the default maximum tokens of the generative model will be used. If an integer is provided, it is interpreted as the maximum tokens for the answer.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    show_hidden: Optional[bool] = Field(
        default=None,
        title="Show hidden resources",
        description="If set to false (default), excludes hidden resources from search",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    rag_strategies: Optional[List[RagStrategies]] = Field(
        default=None,
        title="RAG context building strategies",
        description=(
            """Options for tweaking how the context for the LLM model is crafted:
- `full_resource` will add the full text of the matching resources to the context. This strategy cannot be combined with `hierarchy`, `neighbouring_paragraphs`, or `field_extension`.
- `field_extension` will add the text of the matching resource's specified fields to the context.
- `hierarchy` will add the title and summary text of the parent resource to the context for each matching paragraph.
- `neighbouring_paragraphs` will add the sorrounding paragraphs to the context for each matching paragraph.
- `metadata_extension` will add the metadata of the matching paragraphs or its resources to the context.
- `prequeries` allows to run multiple retrieval queries before the main query and add the results to the context. The results of specific queries can be boosted by the specifying weights.

If empty, the default strategy is used, which simply adds the text of the matching paragraphs to the context.
"""
        ),
        examples=[
            [{"name": "full_resource", "count": 2}],
            [
                {"name": "field_extension", "fields": ["t/amend", "a/title"]},
            ],
            [{"name": "hierarchy", "count": 2}],
            [{"name": "neighbouring_paragraphs", "before": 2, "after": 2}],
            [
                {
                    "name": "metadata_extension",
                    "types": ["origin", "classification_labels"],
                }
            ],
            [
                {
                    "name": "prequeries",
                    "queries": [
                        {
                            "request": {
                                "query": "What is the capital of France?",
                                "features": ["keyword"],
                            },
                            "weight": 0.5,
                        },
                        {
                            "request": {
                                "query": "What is the capital of Germany?",
                            },
                            "weight": 0.5,
                        },
                    ],
                }
            ],
        ],
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    citations: Optional[Union[bool, CitationsType]] = Field(
        default=None,
        description="Whether to include citations in the response. "
        "If set to None or False, no citations will be computed. "
        "If set to True or 'default', citations will be computed after answer generation and send as a separate `CitationsGenerativeResponse` chunk. "
        "If set to 'llm_footnotes', citations will be included in the LLM's response as markdown-styled footnotes. A `FootnoteCitationsGenerativeResponse` chunk will also be sent to map footnote ids to context keys in the `query_context`.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    citation_threshold: Optional[float] = Field(
        default=None,
        description="If citations is set to True or 'default', this will be the similarity threshold. Value between 0 and 1, lower values will produce more citations. If not set, it will be set to the optimized threshold found by Nuclia.",
        ge=0.0,
        le=1.0,
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    answer_json_schema: Optional[dict[str, Any]] = Field(
        default=None,
        title="Answer JSON schema",
        description="""Desired JSON schema for the LLM answer.
This schema is passed to the LLM so that it answers in a scructured format following the schema. If not provided, textual response is returned.
Note that when using this parameter, the answer in the generative response will not be returned in chunks, the whole response text will be returned instead.
Using this feature also disables the `citations` parameter. For maximal accuracy, please include a `description` for each field of the schema.
""",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    generate_answer: Optional[bool] = Field(
        default=None,
        description="Whether to generate an answer using the generative model. If set to false, the response will only contain the retrieval results.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    min_score: Optional[Union[float, MinScore]] = Field(
        default=None,
        title="Minimum score",
        description="Minimum score to filter search results. Results with a lower score will be ignored. Accepts either a float or a dictionary with the minimum scores for the bm25 and vector indexes. If a float is provided, it is interpreted as the minimum score for vector index search.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    extra_context: Optional[list[str]] = Field(
        default=None,
        title="Extra query context",
        description="""Additional context that is added to the retrieval context sent to the LLM.
        It allows extending the chat feature with content that may not be in the Knowledge Box.""",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    extra_context_images: Optional[list[Image]] = Field(
        default=None,
        title="Extra query context images",
        description="""Additional images added to the retrieval context sent to the LLM."
        It allows extending the chat feature with content that may not be in the Knowledge Box.""",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    query_image: Optional[Image] = Field(
        default=None,
        title="Query image",
        description="Image that will be used together with the query text for retrieval and then sent to the LLM as part of the context. "
        "If a query image is provided, the `extra_context_images` and `rag_images_strategies` will be disabled.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    rank_fusion: Optional[Union[RankFusionName, RankFusion]] = Field(
        default=None,
        title="Rank fusion",
        description="Rank fusion algorithm to use to merge results from multiple retrievers (keyword, semantic)",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    rag_images_strategies: Optional[list[RagImagesStrategies]] = Field(
        default=None,
        title="RAG image context building strategies",
        description=(
            "Options for tweaking how the image based context for the LLM model is crafted:\n"
            "- `page_image` will add the full page image of the matching resources to the context.\n"
            "- `tables` will send the table images for the paragraphs that contain tables and matched the retrieval query.\n"
            "- `paragraph_image` will add the images of the paragraphs that contain images (images for tables are not included).\n"
            "No image strategy is used by default. Note that this is only available for LLM models that support visual inputs. If the model does not support visual inputs, the image strategies will be ignored."
        ),
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    resource_filters: Optional[list[str]] = Field(
        default=None,
        title="Resources filter",
        description="List of resource ids to filter search results for. Only paragraphs from the specified resources will be returned.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    vectorset: Optional[str] = Field(
        default=None,
        title="Vectorset",
        description="Vectors index to perform the search in. If not provided, NucliaDB will use the default one",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    search_configuration: Optional[str] = Field(
        default=None,
        description="Load ask parameters from this configuration. Parameters in the request override parameters from the configuration.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
