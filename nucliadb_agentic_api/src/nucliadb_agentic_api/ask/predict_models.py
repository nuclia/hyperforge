from enum import Enum

from nucliadb_models.search import Image
from pydantic import BaseModel, Field


class FieldInfo(BaseModel):
    """
    Model to represent the field information required
    """

    text: str = Field(..., title="The text of the field")
    metadata: str = Field(
        title="The metadata of the field as a base64 string serialized nucliadb_protos.resources.FieldMetadata protobuf",
    )
    field_id: str = Field(
        ...,
        title="The field ID of the field (rid/field_type/field[/split]) or any unique identifier",
    )


class OperationType(str, Enum):
    graph = "graph"
    label = "label"
    ask = "ask"
    qa = "qa"
    extract = "extract"
    prompt_guard = "prompt_guard"
    llama_guard = "llama_guard"


class NameOperationFilter(BaseModel):
    operation_type: OperationType = Field(..., description="Type of the operation")
    task_names: list[str] = Field(
        default_factory=list,
        description="list of task names. If None or empty, all tasks for that operation are applied.",
    )


class QueryModel(BaseModel):
    """
    Model to represent a query request
    """

    text: str | None = Field(default=None, description="The query text to be processed")
    query_image: Image | None = Field(
        default=None,
        description="Image to be considered as part of the query.  Even if the `rephrase` parameter is set to `false`, the rephrasing process will occur, combining the provided text with the image's visual features in the rephrased query.",
    )
    rephrase: bool = Field(
        default=False,
        description="If true, the model will rephrase the input text before processing",
    )
    rephrase_prompt: str | None = Field(
        default=None,
        description="Custom prompt for rephrasing the input text",
        examples=[
            """Rephrase this question so its better for retrieval, and keep the rephrased question in the same language as the original.
QUESTION: {question}
Please return ONLY the question without any explanation. Just the rephrased question.""",
            """Rephrase this question so its better for retrieval, if in the image there are any machinery components with numeric identifiers, append them to the end of the question separated by a commas.
QUESTION: {question}
Please return ONLY the question without any explanation.""",
        ],
    )
    generative_model: str | None = Field(
        default=None,
        description="The generative model to use for rephrasing",
    )
    semantic_models: list[str] | None = Field(
        default=None,
        description="Semantic models to compute the sentence vector for, if not provided, it will only compute the sentence vector for default semantic model in the Knowledge box's configuration.",
    )
    agentic_entities: bool = Field(
        default=False,
        description="If true, the model will return the entities detected in the sentence guided by an already defined Graph Extraction Agent in the Knowledge Box.",
    )
