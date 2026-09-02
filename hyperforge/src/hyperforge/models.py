from __future__ import annotations

import uuid
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)

from nuclia.lib.nua_responses import Image, StoredLearningConfiguration
from nucliadb_models.resource import (
    ConversationFieldData,
    FileFieldData,
    GenericFieldData,
    LinkFieldData,
    TextFieldData,
)
from nucliadb_models.search import CatalogFacetsResponse
from pydantic import BaseModel, Field

from hyperforge import PROMPT_ENVIRONMENT, logger


class Metadata:
    pass


class KnowledgeGraph:
    pass


class Reason:
    pass


class NucliaDBMemoryConfig(BaseModel):
    key: Optional[str] = None
    url: str
    kbid: str
    internal: bool = True


class MemoryConfig(BaseModel):
    nucliadb: Optional[NucliaDBMemoryConfig] = None


class Rule(BaseModel):
    prompt: Optional[str] = None


class Rules(BaseModel):
    rules: List[Union[Rule, str]] = Field(
        default_factory=list,
        description="List of rules that the workflow should follow. Each rule can be a string or a Rule object with a prompt.",
    )


class Facets(BaseModel):
    chunks: Dict[str, int]
    fields: Dict[str, int]


class Source(BaseModel):
    id: str
    description: str
    labels: Dict[str, List[str]]
    facets_native: CatalogFacetsResponse
    paragraph_facets: Dict[str, int]
    learning_configuration: StoredLearningConfiguration


class CitationMetadata(BaseModel):
    context_id: str = Field(
        description="ID of the context this citation refers to",
    )
    origin_urls: list[str] = Field(
        default_factory=list,
        description="List of origin URLs that this citation refers to",
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index of the chunk in the context's chunks list. This is only set for chunk-level citations.",
    )


class AnswerCitations(BaseModel):
    metadata: dict[str, CitationMetadata] = Field(
        default_factory=dict,
        description="Map of citation_id to citation metadata. block-AA",
    )


class VegaLiteVisualization(BaseModel):
    type: Literal["vega_lite"] = "vega_lite"
    vega_lite_obj: Dict[str, Any] = Field(
        default_factory=dict,
        description="The Vega-Lite Object defining the visualization. Previously validated against the Vega-Lite schema.",
    )

    # If we do server-side rendering in the future, we can add fields like:
    # svg: Optional[str] = ...


# For once we add more visualization types, we can use a Discriminator
# Visualization = Annotated[Union[VegaLiteVisualization,NewType], Discriminator("type")]
# For now, we only have one type.
Visualization = Union[VegaLiteVisualization]


class ExternalUsageOperation(str, Enum):
    INTERNET_SEARCH = "internet_search"


class ExternalUsage(BaseModel):
    operation: ExternalUsageOperation = Field(
        description="The external operation that generated this usage.",
    )
    provider: str = Field(
        description="The external provider that generated this usage.",
        examples=["perplexity", "google", "brave"],
    )
    model: str = Field(
        description="The model identifier that was used. Might also refer to specific api or request type.",
        examples=["sonar", "gemini-3.5-flash", "search"],
    )
    input_tokens: int = Field(
        default=0,
        description="Number of raw input tokens used for this request. This number is forwarded from the external provider.",
    )
    output_tokens: int = Field(
        default=0,
        description="Number of raw output tokens generated for this request. This number is forwarded from the external provider.",
    )
    image: int = Field(
        default=0,
        description="Usage specific to images. Might refer to image tokens, images generated, or images processed.",
    )
    requests: int = Field(
        default=1,
        description="Number of requests made to the external provider.",
    )


class Step(BaseModel):
    original_question_uuid: Optional[str]
    actual_question_uuid: Optional[str]
    module: str
    title: str
    value: Optional[str] = None
    agent_path: str
    reason: Optional[str] = None
    timeit: float
    input_nuclia_tokens: Optional[float]
    output_nuclia_tokens: Optional[float]
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    external_usage: Optional[list[ExternalUsage]] = None

    def __str__(self):
        return f"({self.timeit:.2f}s) {self.module}: {self.title} \n {self.value} \n {self.reason} \n NT:({self.input_nuclia_tokens}:{self.output_nuclia_tokens})"

    def markdown(self):
        return f"""
## {self.title}

{self.value}

- reason: {self.reason}
- timeit: {self.timeit}
- input_tokens: {self.input_nuclia_tokens}
- output_tokens: {self.output_nuclia_tokens}
"""


class ChunkImages(BaseModel):
    table: Optional[str]
    chunk: Optional[str]
    page: Optional[str]


FieldTypes = Union[
    TextFieldData,
    ConversationFieldData,
    FileFieldData,
    LinkFieldData,
    GenericFieldData,
]


class Chunk(BaseModel):
    chunk_id: str
    title: Optional[str] = None
    source: Optional[str] = None
    text: str
    labels: List[str] = Field(default_factory=list)
    url: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    action: Optional[str] = Field(
        default=None,
        description="agent and function called to get this chunk.",
    )
    origin_url: Optional[str] = Field(
        default=None,
        description="URL at the origin of the resource from which this chunk was extracted.",
    )
    origin_agent: Optional[str] = Field(
        default=None,
        description="Agent that originated this chunk. This is useful to keep track of the provenance of the information ",
    )

    def render(
        self,
        citations_id: Optional[str] = None,
    ) -> str:
        if citations_id:
            lines = [f"#### Chunk: [{citations_id}] {self.title or self.chunk_id}"]
        else:
            lines = [f"#### Chunk: {self.title or self.chunk_id}"]
        if self.action:
            lines.append(f"Result of running: {self.action}")
        if self.labels:
            lines.append(f"Tags: {', '.join(self.labels)}")
        if self.url:
            lines.append(f"URLs: {', '.join(self.url)}")
        lines.append(f"``` {self.text} ```\n")
        return "\n".join(lines)


class Prompt(BaseModel):
    prompt: str
    resources: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list)
    description: Optional[str] = None

    def render(self) -> str:
        lines = ["## Prompt"]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.resources:
            lines.append(f"Resources: {', '.join(self.resources)}")
        if self.links:
            lines.append(f"Links: {', '.join(self.links)}")
        lines.append(f"```PROMPT\n{self.prompt}\n```\n")
        return "\n".join(lines)


class Answer(BaseModel):
    answer: str
    original_question_uuid: Optional[str]
    actual_question_uuid: Optional[str]
    module: str
    agent_path: str
    data_visualizations: Optional[list[Visualization]] = None
    citations: Optional[AnswerCitations] = None
    chunks: Optional[list[Chunk]] = None
    structured: Optional[list[str]] = None
    images: Optional[Dict[str, Image]] = None
    image_urls: Optional[list[str]] = None


CONTEXT_TEMPLATE = """

{% if con.citations_id is not none -%}
{% for chunk in con.chunks %}
{{chunk.render(citations_id=con.citations_id ~ "-" ~ loop.index0)}}
{% endfor -%}
{% else -%}
{% for chunk in con.chunks %}
{{chunk.render()}}
{% endfor -%}
{% endif -%}

{% if con.structured | length > 0 -%}
## Extra structured info:
{% for structured in con.structured %}
{{structured}}
{% endfor -%}
{% endif -%}
"""

CONTEXT_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(CONTEXT_TEMPLATE)


class JSONObject(BaseModel):
    json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema that defines the structure of the JSON object.",
    )
    json_object: Dict[str, Any] = Field(
        default_factory=dict,
        description="The actual JSON object that conforms to the provided JSON schema.",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata associated with the JSON object.",
    )
    id: Optional[str] = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this JSON object instance.",
    )


class Context(BaseModel):
    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this context instance",
    )
    original_question_uuid: Optional[str]
    actual_question_uuid: Optional[str]
    question: str
    chunks: List[Chunk] = Field(default_factory=list)
    images: Dict[str, Image] = Field(default_factory=dict)
    prompts: List[Prompt] = Field(default_factory=list)
    structured: List[str] = Field(default_factory=list)
    json_objects: List[JSONObject] = Field(default_factory=list)
    source: str
    agent: str
    # XXX: This is not actually a summary, but an answer attempt for now!
    summary: str = Field(
        default="",
        description="Partial or full answer to the question, generated by the context validation step inside a context agent.",
    )
    agent_id: str = ""
    title: Optional[str] = None
    missing: Optional[str] = None
    citations: list[str] | None = Field(
        default=None,
        description="List of chunk IDs that were considered relevant in the context validation step.",
    )
    citations_id: Optional[str] = Field(
        default=None,
        description="Block ID used for citations in this context.",
    )
    image_urls: List[str] = Field(
        default_factory=list,
        description="List of image URLs associated with this context.",
    )

    def answer_summary_markdown(self) -> str:
        return "# {question}\n\n {summary}".format(
            question=self.question, summary=self.summary
        )

    def context_markdown(self) -> str:
        return CONTEXT_PROMPT_TEMPLATE.render(con=self)

    def stats(self) -> Dict[str, int | str | None]:
        return {
            "chunks": len(self.chunks),
            "images": len(self.images),
            "structured": len(self.structured),
            "source": self.source,
            "question": self.question,
            "agent": self.agent,
            "summary": self.summary,
            "title": self.title,
            "missing": self.missing,
        }

    def prune_to_citations(self) -> None:
        if self.citations is None:
            logger.warning(
                "Cannot prune context as no citations are available.",
                extra={
                    "agent": self.agent,
                    "source": self.source,
                    "agent_id": self.agent_id,
                },
            )
            return
        cited_chunk_ids = {
            citation_id
            for citation_id in self.citations
            if not citation_id.startswith("structured-")
        }
        self.chunks = [
            chunk for chunk in self.chunks if chunk.chunk_id in cited_chunk_ids
        ]
        self.structured = [
            s
            for i, s in enumerate(self.structured)
            if f"structured-{i}" in self.citations
        ]


class HistoryQuestionAnswer(BaseModel):
    question: str
    answer: str


class TrackingInfo(BaseModel):
    rao_id: str
    session: str
    message: str
