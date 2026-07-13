import ast
import asyncio
import json
from time import time
from typing import Any, ClassVar, Dict, List, Literal, Optional, cast

from hyperforge import PROMPT_ENVIRONMENT, logger
from hyperforge.agent import Agent
from hyperforge.configure import agent
from hyperforge.context.agent import ContextAgent
from hyperforge.definition import FunctionDefinition
from hyperforge.manager import Manager
from hyperforge.memory import Chunk, Context, QuestionMemory, Source
from nucliadb_models.filters import (
    And,
    CatalogFilterExpression,
    FieldFilterExpression,
    FilterExpression,
    Keyword,
    Or,
    Resource,
    ResourceFilterExpression,
)
from nucliadb_models.resource import Resource as ResourceResponse
from nucliadb_models.search import (
    AskRequest,
    CatalogRequest,
    CitationsType,
    FieldExtensionStrategy,
    Filter,
    FindRequest,
    FullResourceStrategy,
    KnowledgeboxFindResults,
    MetadataExtensionStrategy,
    NeighbouringParagraphsStrategy,
    ResourceProperties,
    SyncAskResponse,
)

from hyperforge_nucliadb.ask.multi import choose_source
from hyperforge_nucliadb.ask_utils import (
    combine_catalog_filter_expressions,
    combine_filter_expressions,
    get_chunk_text,
    to_field_filter_expression,
    to_resource_filter_expression,
)
from hyperforge_nucliadb.basic_ask_config import (
    BasicAskAgentConfig,
)
from hyperforge_nucliadb.driver import (
    NucliaDBDriver,
    format_ndb_catalog,
    format_ndb_labels,
)

# Example filter expressions for catalog search
EXAMPLE_FILTER_EXP1 = [
    {
        "any": [
            "/l/topic/technology",
            "/l/topic/health",
        ]
    }
]

EXAMPLE_FILTER_EXP2 = [
    {
        "all": [
            "/l/category/research",
            "/icon/application/pdf",
        ]
    }
]

EXAMPLE_FILTER_EXP3 = [
    {
        "none": ["/l/topic"],
    },
    {
        "any": [
            "/icon/application/pdf",
            "/icon/application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ],
    },
]


FACETS_LABEL_SEARCH_TEMPLATE = """
Given the question: '{{ question }}', select the most relevant label sets from the following list to list all options and choose the labels to filter:
{% for labelset, values in labels.items() %}
- {{ labelset }}:
    {% for value in values %}
    - {{ value }}
    {% endfor %}
{% endfor %}

Example 1:
Question: 'Which companies compete with PepsiCo in the food sector?'
Available Labels:
hq_state:\n    \n    - IL\n    \n    - NY\n    \n    - TX\n    \n    - GA\n    \n    - PA\n    \n\n- year:\n    \n    - 2024\n    \n    - 2015\n    \n    - 2020\n    \n    - 2022\n    \n    - 2018\n    \n    - 2016\n    \n    - 2017\n    \n    - 2023\n    \n    - 2019\n    \n    - 2021\n    \n\n- type_of_doc:\n    \n    - Metadata\n    \n    - None\n    \n    - 10Q\n    \n    - 10K File\n    \n\n- sector:\n    \n    - Foods\n    \n    - Snacks\n    \n    - Beverages\n    \n    - Breakfast\n    \n\n- company:\n    \n    - Coca Cola\n    \n    - Kellanova\n    \n    - Keurig\n    \n    - UTZ Brands\n    \n    - Mondelez\n    \n    - PepsiCo\n    \n
Answer:
{"labels_sets": ["company"], "labels": ["sector/Foods"], "reasoning": "The question is about companies in sector Foods, so we filter by the sector Foods and want all the companies."}
"""
FACETS_LABEL_SEARCH_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    FACETS_LABEL_SEARCH_TEMPLATE
)


FACETS_SEARCH_ANSWER_GENERATION_TEMPLATE = """
Your task is to respond to the users question by leveraging the faceted search results obtained from the Knowledge Box. The faceted search has provided all the labels available with a filter extracted from the question as specific labels within the Knowledge Box.
Use this information to construct a comprehensive answer to the user's question.
Here is the user's question: '{{ question }}'.
Here are the details of the faceted search results. Please note that a single resource may have multiple labels and thus may be counted in multiple categories:
    {{ facets }}
"""

FACETS_SEARCH_ANSWER_GENERATION_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    FACETS_SEARCH_ANSWER_GENERATION_TEMPLATE
)

FACETS_LABEL_SELECTION_TEMPLATE = """
Given the question: '{{ question }}', select the most relevant label sets from the following list to perform a bucket aggregation. Each label set shows some label examples:
{{ labels_str }}
You can select multiple categories if needed. Provide your answer as a function call. If no category is relevant, leave the list empty.

Example 1:
Question: 'Show me the distribution of articles about sports and health.'
Available Labels:
{"topic": ["sports", "health", "technology"], "category": ["news", "blog", "research"]}
Answer:
{"labels_sets": ["topic"], "reasoning": "The question is about topics, so we select the 'topic' label category."}
"""

FACETS_LABEL_SELECTION_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    FACETS_LABEL_SELECTION_TEMPLATE
)


FACETS_ANSWER_GENERATION_TEMPLATE = """
Your task is to respond to the users question by leveraging the faceted search results obtained from the Knowledge Box. The faceted search has provided you with counts of documents for specific labels within the Knowledge Box.
Use this information to construct a comprehensive answer to the user's question.
Here is the user's question: '{{ question }}'.
Here are the details of the faceted search results. Please note that a single document may have multiple labels and thus may be counted in multiple categories:
    Document counts for labels {{ labels_selected }} in {{ source_id }} Knowledge Box for a total of {{ total }} documents: "
    {{ facets }}
"""

FACETS_ANSWER_GENERATION_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    FACETS_ANSWER_GENERATION_TEMPLATE
)


CATALOG_FILTER_SELECTION_TEMPLATE = """
The user wants to perform a catalog search in the Knowledge Box based on a question.

You need to identify the most relevant labels and provide a filter expression to perform the catalog search. The filter expression should be in the format used by NucliaDB.

The nucliadb format for filter expressions is as follows:
- Use "any" to indicate that any of the properties can match (logical OR).
- Use "all" to indicate that all of the listed labels must match (logical AND).
- Use "none" to indicate that the listed labels should not match (logical NOT).

Sequential filters are treated as AND conditions.

For filtering by labels, use the format "/l/{labelset}/{label}".

For filtering by label sets without specifying a label, use the format "/l/{labelset}/".

For filtering by file types, use the media type prefixed by "/icon/".

**Example 1:**
Question: 'Find all documents related to technology and health.'
Available Labels:
{"topic": ["technology", "health", "sports"], "category": ["research", "news", "blog"]}
Answer:
{"filters": {{ example_filter_exp1 }}, "reasoning": "The question is about documents related to technology and health, so we create a filter expresion to match any of the labels technology or health in the topic label set."}

**Example 2:**
Question: 'Get all PDFs that are research articles.'
Available Labels:
{"topic": ["technology", "health", "sports"], "category": ["research", "news", "blog"]}
Answer:
{"filters": {{ example_filter_exp2 }}, "reasoning": "The question specifies research articles in PDF format, so we create a filter expression that requires both the category to be research and the media type to be PDF."}

**Example 3:**
Question: 'Show me all documents without a set topic, that are either PDFs or Word format.'
Available Labels:
{"topic": ["technology", "health", "sports"], "category": ["research", "news", "blog"]}
Answer:
{"filters": {{ example_filter_exp3 }}, "reasoning": "The question asks for documents excluding sports, so we use a 'not' filter for the topic sports, and an 'any' filter for the media types PDF or Word."}

Your Task:
Based on the question and the available labels, provide a filter expression to perform the catalog search. The question is: '{{ question }}'.
Here are the available labels in the Knowledge Box:
{{ labels_str }}
"""

CATALOG_FILTER_SELECTION_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    CATALOG_FILTER_SELECTION_TEMPLATE
)


def get_catalog_filter_prompt(question: str, labels_str: str) -> str:
    """Helper function to render catalog filter selection prompt with pre-filled examples."""
    return CATALOG_FILTER_SELECTION_AGENT_TEMPLATE.render(
        question=question,
        labels_str=labels_str,
        example_filter_exp1=EXAMPLE_FILTER_EXP1,
        example_filter_exp2=EXAMPLE_FILTER_EXP2,
        example_filter_exp3=EXAMPLE_FILTER_EXP3,
    )


CATALOG_ANSWER_GENERATION_TEMPLATE = """
Based on the following catalog search results, please answer the following user question: '{{ question }}'

{{ catalog_txt }}
"""

CATALOG_ANSWER_GENERATION_AGENT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(
    CATALOG_ANSWER_GENERATION_TEMPLATE
)


@agent(
    id="basic_ask",
    agent_type="context",
    title="Knowledge Box Basic Ask",
    description="Ask a question to the knowledge box and retrieve relevant information",
    config_schema=BasicAskAgentConfig,
)
class BasicAskAgent(ContextAgent, Agent[BasicAskAgentConfig]):
    labelsets: Dict[str, List[str]]
    synonyms: Dict[str, Dict[str, List[str]]]
    agent_description: str = "Agent that queries a NucliaDB Knowledge Box to get context to answer questions. It is a retrieval agent, so questions should be in a format that makes sense for retrieval."

    __published_functions__: ClassVar[Dict[str, FunctionDefinition]] = {
        "search_by_title": FunctionDefinition(
            name="search_by_title",
            description="Search for context in the Knowledge Box by title. Useful for specific queries where the title is known.",
            parameters={
                "title": {
                    "type": "string",
                    "description": "The title to search for in the Knowledge Box.",
                },
                "filters": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filters to apply when searching in the Knowledge Box.",
                },
            },
        ),
        "ask_labels_list": FunctionDefinition(
            name="ask_labels_list",
            description="Get the labels available in the Knowledge Box as a list of strings. Useful to filter searches by labels.",
            parameters={
                "labelset": {
                    "type": "string",
                    "description": "The label set to get the labels from the Knowledge Box.",
                },
            },
        ),
        "ask_by_title": FunctionDefinition(
            name="ask_by_title",
            description="Ask a question in the Knowledge Box filtered by a specific title. Useful when the user wants information about a known document.",
            parameters={
                "title": {
                    "type": "string",
                    "description": "The title to filter the ask operation in the Knowledge Box.",
                },
                "question": {
                    "type": "string",
                    "description": "The question to ask in the Knowledge Box.",
                },
            },
        ),
        "ask_agent": FunctionDefinition(
            name="ask_agent",
            description="Search for context in the Knowledge Box.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to search for in the Knowledge Box.",
                },
            },
        ),
        "ask_labels": FunctionDefinition(
            name="ask_labels",
            description="Get the labels available in the Knowledge Box. Useful to filter searches by labels.",
            parameters={},
        ),
        "facets_count": FunctionDefinition(
            name="facets_count",
            description="Perform a faceted count in the Knowledge Box. Useful to get counts of different categories or facets within the data.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to perform the faceted count for.",
                },
            },
        ),
        "facets_search": FunctionDefinition(
            name="facets_search",
            description="Perform a faceted search in the Knowledge Box. Useful to get counts of different categories or facets within the data.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to perform the faceted search for.",
                },
            },
        ),
        "catalog_search": FunctionDefinition(
            name="catalog_search",
            description="Perform a catalog search in the Knowledge Box. Useful to find items based on specific criteria or attributes, including labels, file types, or other metadata.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "The question to perform the catalog search for.",
                },
            },
        ),
        "all_images_by_title": FunctionDefinition(
            name="all_images_by_title",
            description="Get all image URLs from a resource in the Knowledge Box filtered by a specific title. Useful when the user wants to retrieve images from known documents.",
            parameters={
                "title": {
                    "type": "string",
                    "description": "The title to filter the image extraction operation in the Knowledge Box.",
                },
            },
        ),
        "search_images": FunctionDefinition(
            name="search_images",
            description="Search for images in a resource in the Knowledge Box filtered by a specific title and relevant image information. Useful when the user wants to retrieve selected images from known documents.",
            parameters={
                "title": {
                    "type": "string",
                    "description": "The title to filter the image search operation in the Knowledge Box.",
                },
                "image_info": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Relevant info to query images from the document.",
                },
            },
        ),
    }

    def __init__(
        self, config: BasicAskAgentConfig, agent_id: Optional[str] = None
    ) -> None:
        super().__init__(config, agent_id)
        self.labelsets: Dict[str, List[str]] = {}
        self.synonyms: Dict[str, Dict[str, List[str]]] = {}

    async def ask_labels_list(
        self,
        labelset: str,
        memory: QuestionMemory,
        manager: Manager,
        **kwargs,
    ) -> List[str]:
        await self.ask_labels(memory=memory, manager=manager, **kwargs)
        if labelset not in self.labelsets:
            raise Exception(f"Label set {labelset} not found in Knowledge Box")
        return self.labelsets[labelset]

    async def ask_labels(
        self,
        memory: QuestionMemory,
        manager: Manager,
        **kwargs,
    ) -> Dict[str, List[str]]:
        sources = self.config.sources
        if len(sources) > 1:
            raise Exception("ask_labels can only be used with one source")
        source = sources[0]
        if source not in self.labelsets:
            nucliadb_driver = get_ndb_driver(manager, source)
            self.labelsets = await nucliadb_driver.labels()
        return self.labelsets

    async def retrieve(
        self,
        manager: Manager,
        source_id: str,
        question: str,
        keyword_filters: Optional[List[str]] = None,
        and_filters: Optional[List[str]] = None,
        or_filters: Optional[List[str]] = None,
        catalog_filter: Optional[FilterExpression] = None,
    ) -> List[str]:
        nucliadb_driver = get_ndb_driver(manager, source_id)

        filter_expression = await self.build_filter_expression(
            nucliadb_driver,
            source_id,
            keyword_filters=keyword_filters,
            and_filters=and_filters,
            or_filters=or_filters,
            filter_expression=catalog_filter,
        )

        find_result = await nucliadb_driver.find_raw(
            FindRequest(
                query=question,
                filter_expression=filter_expression,
            )
        )
        return list(find_result.resources.keys())

    async def search_by_title(
        self,
        memory: QuestionMemory,
        manager: Manager,
        title: str,
        filters: Optional[List[str]] = None,
        catalog_filter: Optional[CatalogFilterExpression] = None,
        **kwargs,
    ) -> Dict[str, List[str]]:
        sources = self.config.sources
        resources = {}
        for source in sources:
            nucliadb_driver = get_ndb_driver(manager, source)
            filter_expression = await self.build_catalog_filter_expression(
                nucliadb_driver,
                filters=filters,
                filter_expression=catalog_filter,
            )
            response = await nucliadb_driver.catalog_search_raw(
                CatalogRequest(
                    query=title,
                    page_size=25,
                    page_number=0,
                    filter_expression=filter_expression,
                )
            )
            resources[source] = list(response.resources.keys())
        return resources

    async def ask_by_title(
        self,
        memory: QuestionMemory,
        manager: Manager,
        title: str,
        question: str,
        **kwargs,
    ) -> List[Context]:
        # First, search by title to get specific matching resource IDs
        resources_by_source = await self.search_by_title(
            memory=memory,
            manager=manager,
            title=title,
        )
        # Then perform ask operation filtered by those resource IDs
        contexts: list[Context] = await asyncio.gather(
            *[
                self.inner_ask_by_title(
                    source_id=source_id,
                    resource_ids=resource_ids,
                    question=question,
                    memory=memory,
                    manager=manager,
                )
                for source_id, resource_ids in resources_by_source.items()
                if len(resource_ids) > 0
            ]
        )
        return contexts

    async def get_all_images(
        self,
        resource: ResourceResponse,
    ) -> List[str]:
        image_urls: List[str] = []
        if resource.data is None or resource.data.files is None:
            logger.info("No files found in resource")
            return image_urls
        for _, field in resource.data.files.items():
            if (
                field.extracted is None
                or field.extracted.file is None
                or field.extracted.file.nested_list_position is None
            ):
                continue
            image_names = field.extracted.file.nested_list_position.keys()
            for image_name in image_names:
                if (
                    field.extracted.file.file_generated is None
                    or image_name not in field.extracted.file.file_generated
                ):
                    logger.info("No generated file found for image: " + image_name)
                    continue
                image_url = field.extracted.file.file_generated[image_name].uri
                if image_url is not None:
                    image_urls.append(image_url)

        return image_urls

    async def get_search_image_urls(
        self,
        response: KnowledgeboxFindResults,
        nucliadb_driver: NucliaDBDriver,
    ) -> List[str]:
        image_urls: List[str] = []
        if response.resources is None:
            logger.info("No resources found in search response")
            return image_urls
        for resource_id, resource_results in response.resources.items():
            if resource_results is None:
                logger.info(f"No results found for resource {resource_id}")
                continue
            if resource_results.fields is None:
                logger.info(f"No fields found for resource {resource_id}")
                continue
            for field_id, field_results in resource_results.fields.items():
                clean_field_id = field_id.replace("/f/", "")
                base_url = f"/kb/{nucliadb_driver.config.kbid}/resource/{resource_id}/file/{clean_field_id}/download/extracted/generated/"
                # Collect unique image references for this field
                images = set(
                    paragraph.reference
                    for paragraph in field_results.paragraphs.values()
                    if paragraph.reference is not None
                )
                if not images:
                    continue
                # Prepare all image api paths
                image_api_paths = [f"api/v1{base_url}{image}" for image in images]
                # Fetch all ephemeral tokens in parallel
                tokens = await asyncio.gather(
                    *[
                        nucliadb_driver.get_ephemeral_token(path=api_path)
                        for api_path in image_api_paths
                    ]
                )
                # Build URLs with their corresponding tokens
                for image, ephemeral_token in zip(images, tokens):
                    image_url = f"{nucliadb_driver.config.url}/v1{base_url}{image}?eph-token={ephemeral_token}"
                    image_urls.append(image_url)
        return image_urls

    async def search_images(
        self,
        memory: QuestionMemory,
        manager: Manager,
        title: str,
        image_info: List[str],
        **kwargs,
    ) -> List[Context]:
        # First, search by title to get specific matching resource IDs )
        resources_by_source = await self.search_by_title(
            memory=memory,
            manager=manager,
            title=title,
        )
        contexts: list[Context] = []
        # Then do a search on the resource to find only paragraphs with images that match the image_info
        for source_id, resource_ids in resources_by_source.items():
            nucliadb_driver = get_ndb_driver(manager, source_id)

            for resource_id in resource_ids:
                filter_expression = {
                    "field": {"prop": "resource", "id": resource_id},
                    "operator": "and",
                    "paragraph": {
                        "or": [
                            {"prop": "kind", "kind": "OCR"},
                            {"prop": "kind", "kind": "INCEPTION"},
                        ]
                    },
                }
                images_urls: List[str] = []
                for info in image_info:
                    logger.info("Searching images with info: " + info)
                    find_request = FindRequest(
                        query=info,
                        filter_expression=filter_expression,
                    )
                    response = await nucliadb_driver.find_raw(find_request)
                    images_urls.extend(
                        await self.get_search_image_urls(response, nucliadb_driver)
                    )

                if len(images_urls) > 0:
                    context = Context(
                        agent_id=self.config.id,
                        original_question_uuid=memory.original_question_uuid,
                        actual_question_uuid=None,
                        question="Search images by title. Title: "
                        + title
                        + " with image info: "
                        + ", ".join(image_info),
                        source=source_id,
                        agent="search_images",
                        title=self.config.title
                        if self.config.title
                        else f"Search images on {source_id} Knowledge Box",
                        image_urls=set(images_urls),
                    )
                    contexts.append(context)
                else:
                    logger.info("No images found for resource " + resource_id)

        return contexts

    async def all_images_by_title(
        self,
        memory: QuestionMemory,
        manager: Manager,
        title: str,
        **kwargs,
    ) -> List[Context]:
        # First, search by title to get specific matching resource IDs
        resources_by_source = await self.search_by_title(
            memory=memory,
            manager=manager,
            title=title,
        )
        contexts: list[Context] = []

        # Then get all the image links from those resource IDs
        for source_id, resource_ids in resources_by_source.items():
            nucliadb_driver = get_ndb_driver(manager, source_id)

            for resource_id in resource_ids:
                resource = await nucliadb_driver.get_resource_by_id(
                    query_params={
                        "show": ["basic", "extracted"],  # type: ignore
                        "extracted": ["metadata", "file"],  # type: ignore
                    },
                    rid=resource_id,
                )
                if resource is None:
                    logger.info("Resource not found: " + resource_id)
                    continue
                images_urls = await self.get_all_images(resource=resource)
                if len(images_urls) > 0:
                    context = Context(
                        agent_id=self.config.id,
                        original_question_uuid=memory.original_question_uuid,
                        actual_question_uuid=None,
                        question="All images by title. Title: " + title,
                        source=source_id,
                        agent="all_images_by_title",
                        title=self.config.title
                        if self.config.title
                        else f"All images by title on {source_id} Knowledge Box",
                        image_urls=images_urls,
                    )
                    contexts.append(context)
                else:
                    logger.info("No images found for resource " + resource_id)

        return contexts

    async def search_images_by_title(
        self,
        memory: QuestionMemory,
        manager: Manager,
        title: str,
        question: str,
        **kwargs,
    ) -> List[Context]:
        # First, search by title to get specific matching resource IDs
        resources_by_source = await self.search_by_title(
            memory=memory,
            manager=manager,
            title=title,
        )

        # Then perform ask operation filtered by those resource IDs
        contexts: list[Context] = await asyncio.gather(
            *[
                self.inner_ask_by_title(
                    source_id=source_id,
                    resource_ids=resource_ids,
                    question=question,
                    memory=memory,
                    manager=manager,
                )
                for source_id, resource_ids in resources_by_source.items()
                if len(resource_ids) > 0
            ]
        )
        return contexts

    async def do_nucliadb_query(
        self,
        memory: QuestionMemory,
        manager: Manager,
        source_id: str,
        question: str,
        resource_id: Optional[str] = None,
    ) -> SyncAskResponse:
        t0 = time()
        nucliadb_driver = get_ndb_driver(manager, source_id)

        if resource_id:
            filter_expression = FilterExpression(field=Resource(id=resource_id))
        else:
            filter_expression = None
        filter_expression = await self.build_filter_expression(
            nucliadb_driver,
            source_id,
            filter_expression=filter_expression,
        )

        # Use full resource strategy by requesting title and summary fields
        ask_request = AskRequest(
            query=question,
            generative_model=self.config.generative_model,
            # TODO: Consider what to do when multiple resources match
            filter_expression=filter_expression,
            rag_strategies=[
                FullResourceStrategy(count=1),
            ],
        )

        response = await nucliadb_driver.ask(
            ask_request, headers={"X-SHOW-CONSUMPTION": "true"}
        )

        await memory.add_step(
            step_module="basic_ask_title",
            step_title=self.step_title("Preparing RAG by title"),
            step_reason="",
            step_value=ask_request.model_dump_json(
                exclude_none=True, exclude_unset=True
            ),
            timeit=time() - t0,
            input_nuclia_tokens=0.0,
            output_nuclia_tokens=0.0,
            step_agent_path=f"/context/{self.agent_id}",
        )
        return response

    async def create_context_from_nucliadb_response(
        self,
        response: SyncAskResponse,
        source_id: str,
        memory: QuestionMemory,
        question: str,
    ) -> Context:
        t0 = time()
        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=None,
            question=question,
            source=source_id,
            agent="basic_ask_title",
            title=self.config.title
            if self.config.title
            else f"Title search on {source_id} Knowledge Box",
        )
        answer = None
        input_tokens = (
            response.metadata.tokens.input_nuclia
            if response.metadata and response.metadata.tokens
            else 0
        )
        output_tokens = (
            response.metadata.tokens.output_nuclia
            if response.metadata and response.metadata.tokens
            else 0
        )

        input_tokens = (
            response.metadata.tokens.input_nuclia
            if response.metadata and response.metadata.tokens
            else 0
        )
        output_tokens = (
            response.metadata.tokens.output_nuclia
            if response.metadata and response.metadata.tokens
            else 0
        )
        context.chunks = []
        answer = response.answer if response.status == "success" else ""
        if response.citations != {}:
            result_chunks = list(response.citations.keys())
        else:
            result_chunks = response.retrieval_results.best_matches
        for chunk_id in result_chunks:
            ids = chunk_id.split("/")
            resource_id = ids[0]
            field_id = f"/{ids[1]}/{ids[2]}" if len(ids) > 2 else ""
            resource = response.retrieval_results.resources[resource_id]
            resource_title = resource.title
            try:
                text = (
                    resource.fields[field_id].paragraphs[chunk_id].text
                    if field_id in resource.fields
                    else ""
                )
            except Exception:
                if (
                    response.augmented_context is not None
                    and chunk_id in response.augmented_context.paragraphs
                ):
                    text = response.augmented_context.paragraphs[chunk_id].text
                else:
                    text = ""
            context.chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    title=resource_title,
                    text=text,
                    source=source_id,
                    origin_agent=self.config.module,
                )
            )

        if answer:
            context.summary = answer
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Ask by title"),
            step_reason="Got answer" if answer else "No answer",
            step_value=answer if answer else "No answer",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens,
            output_nuclia_tokens=output_tokens,
            step_agent_path=f"/context/{self.agent_id}",
        )
        return context

    async def inner_ask_by_title(
        self,
        source_id: str,
        manager: Manager,
        memory: QuestionMemory,
        question: str,
        resource_ids: List[str],
    ) -> Context:
        response = await self.do_nucliadb_query(
            memory=memory,
            manager=manager,
            source_id=source_id,
            question=question,
            resource_id=resource_ids[0] if len(resource_ids) == 1 else None,
        )

        context = await self.create_context_from_nucliadb_response(
            response=response,
            source_id=source_id,
            memory=memory,
            question=question,
        )

        return context

    async def ask_agent(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        **kwargs,
    ) -> List[Context]:
        sources = self.config.sources
        keyword_filters = kwargs.get("keyword_filters", [])
        full_resource = kwargs.get("full_resource", False)

        and_filters = []
        or_filters = []

        labels_dict: Dict[str, List[str]] = {}

        for label in kwargs.get("labels", []):
            try:
                labelset, label_value = label.split("/", 1)
            except ValueError:
                logger.warning(f"Invalid label format: {label}")
            labels_dict.setdefault(labelset, []).append(label_value)

        for labelset, label_values in labels_dict.items():
            if len(label_values) == 1:
                and_filters.append(f"/l/{labelset}/{label_values[0]}")
            else:
                or_filters = [f"/l/{labelset}/{lv}" for lv in label_values]

        # Choose sources based on the question/s and the in
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.agent_id,
            step_title=self.step_title("Choose sources"),
        )

        contexts: list[Context] = await asyncio.gather(
            *[
                self.inner_rag(
                    source_obj=source,
                    question=question,
                    memory=memory,
                    manager=manager,
                    keyword_filters=keyword_filters,
                    and_filters=and_filters,
                    or_filters=or_filters,
                    full_resource=full_resource,
                )
                for source in chosen_sources
            ]
        )
        return contexts

    async def _get_question_context(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question_uuid: str,
        question: str,
        flow_id: str,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[str, str]]:
        sources = self.config.sources
        # Choose sources based on the question/s and the in
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.agent_id,
            step_title=self.step_title("Choose sources"),
        )

        missing: list[tuple[str, str] | None] = await asyncio.gather(
            *[
                self.rag(source, question_uuid, question, memory, manager, flow_id)
                for source in chosen_sources
            ]
        )
        # We only want to fallback if all sources failed
        if any(m is None for m in missing):
            return []
        else:
            # XXX: We might be duplicating some results here if multiple sources return the same missing question
            return [r for r in missing if r is not None]

    async def build_filter_expression(
        self,
        nucliadb_driver: NucliaDBDriver,
        source: str,
        keyword_filters: Optional[List[str]] = None,
        and_filters: Optional[List[str]] = None,
        or_filters: Optional[List[str]] = None,
        resource_filters: Optional[List[str]] = None,
        filter_expression: Optional[FilterExpression] = None,
    ) -> FilterExpression | None:
        """
        Apply all the different filters to create a single filter expression that can be used in a NucliaDB query.
        In particular, it will merge whatever filters were passed by the user with the ones configured at the driver level.
        """
        and_operands: list[FieldFilterExpression] = []

        # Parse keyword filters first
        keyword_filter_result = []
        if len(keyword_filters or []) > 0 and source not in self.synonyms:
            self.synonyms[source] = await nucliadb_driver.synonyms_raw()

        for keyword_filter in keyword_filters or []:
            keyword_filter_result.append(keyword_filter)
            if keyword_filter.lower() in self.synonyms[source]:
                for synonym in self.synonyms[source][keyword_filter.lower()]:
                    keyword_filter_result.append(synonym)

        if len(keyword_filter_result) == 1:
            and_operands.append(Keyword(word=keyword_filter_result[0]))
        elif len(keyword_filter_result) > 1:
            and_operands.append(
                Or(operands=[Keyword(word=word) for word in keyword_filter_result])
            )

        # Add resource filters if exists
        if resource_filters is not None:
            and_operands.append(
                Or(operands=[Resource(id=rid) for rid in resource_filters])
            )

        # Add old format filters if exists (for bw compatibility)
        if nucliadb_driver.config.filters is not None and len(
            nucliadb_driver.config.filters
        ):
            operands = _to_field_filter_expression(nucliadb_driver.config.filters)
            if len(operands) == 1:
                and_operands.append(operands[0])
            elif len(operands) > 1:
                and_operands.append(And(operands=operands))

        # Now add the and/or filters from the function call
        if and_filters is not None:
            operands = _to_field_filter_expression(and_filters)
            if len(operands) == 1:
                and_operands.append(operands[0])
            elif len(operands) > 1:
                and_operands.append(And(operands=operands))

        if or_filters is not None:
            operands = _to_field_filter_expression(or_filters)
            if len(operands) == 1:
                and_operands.append(operands[0])
            elif len(operands) > 1:
                and_operands.append(Or(operands=operands))

        # Now combine all filter expressions
        expressions_to_combine = []
        if len(and_operands) == 1:
            expressions_to_combine.append(FilterExpression(field=and_operands[0]))
        elif len(and_operands) > 1:
            expressions_to_combine.append(
                FilterExpression(field=And(operands=and_operands))
            )
        if filter_expression is not None:
            expressions_to_combine.append(filter_expression)
        if nucliadb_driver.config.filter_expression is not None:
            expressions_to_combine.append(nucliadb_driver.config.filter_expression)

        if len(expressions_to_combine) == 0:
            # Nothing to filter from
            return None
        if len(expressions_to_combine) == 1:
            # Nothing to combine, return the only expression
            return expressions_to_combine[0]
        else:
            # Combine all filter expressions with AND operator
            return combine_filter_expressions(expressions_to_combine, operator="and")

    async def build_catalog_filter_expression(
        self,
        nucliadb_driver: NucliaDBDriver,
        filters: Optional[List[str]] = None,
        classification_labels: Optional[List[str]] = None,
        classification_labels_operand: Literal["and", "or"] = "and",
        filter_expression: Optional[CatalogFilterExpression] = None,
    ) -> CatalogFilterExpression | None:
        """
        Apply all the different filters to create a single catalog filter expression that can be used in a NucliaDB catalog query.
        In particular, it will merge whatever filters were passed by the user with the ones configured at the driver level.
        """

        # First off, possibly create a filter expression from the user-provided filters and classification labels
        and_operands: list[ResourceFilterExpression] = []
        if classification_labels is not None and len(classification_labels) > 0:
            operands = _to_resource_filter_expression(classification_labels)
            if len(operands) == 1:
                and_operands.append(operands[0])
            elif len(operands) > 1:
                if classification_labels_operand == "and":
                    and_operands.append(And(operands=operands))
                else:
                    and_operands.append(Or(operands=operands))
        if filters is not None and len(filters) > 0:
            operands = _to_resource_filter_expression(filters)
            if len(operands) == 1:
                and_operands.append(operands[0])
            elif len(operands) > 1:
                and_operands.append(And(operands=operands))

        to_combine: list[CatalogFilterExpression] = []
        if len(and_operands) == 1:
            to_combine.append(CatalogFilterExpression(resource=and_operands[0]))
        elif len(and_operands) > 1:
            to_combine.append(
                CatalogFilterExpression(resource=And(operands=and_operands))
            )
        if filter_expression is not None:
            to_combine.append(filter_expression)
        if nucliadb_driver.config.catalog_filter_expression is not None:
            to_combine.append(nucliadb_driver.config.catalog_filter_expression)
        if len(to_combine) == 0:
            return None
        elif len(to_combine) == 1:
            return to_combine[0]
        else:
            return combine_catalog_filter_expressions(to_combine, operator="and")

    async def inner_rag(
        self,
        source_obj: Source,
        manager: Manager,
        memory: QuestionMemory,
        question: str,
        question_uuid: Optional[str] = None,
        keyword_filters: List[str] = [],
        and_filters: Optional[List[str]] = None,
        or_filters: Optional[List[str]] = None,
        full_resource: bool = False,
        resource_filters: Optional[List[str]] = None,
    ) -> Context:
        source = source_obj.id

        nucliadb_driver = get_ndb_driver(manager, source)

        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source=source,
            agent="basic_ask",
            title=self.config.title
            if self.config.title
            else f"Retrieval on {source} Knowledge Box",
        )

        if full_resource:
            rag_strategies = [
                FullResourceStrategy(count=1),
                MetadataExtensionStrategy(types=["classification_labels", "origin"]),
            ]
        else:
            rag_strategies = [
                FieldExtensionStrategy(fields=["a/title", "a/summary"]),
                NeighbouringParagraphsStrategy(before=5, after=5),
                MetadataExtensionStrategy(types=["classification_labels", "origin"]),
            ]

        filter_expression = await self.build_filter_expression(
            nucliadb_driver,
            source,
            keyword_filters=keyword_filters,
            and_filters=and_filters,
            or_filters=or_filters,
            resource_filters=resource_filters,
        )
        t0 = time()

        ask_request = AskRequest(
            query=question,
            show=[ResourceProperties.BASIC, ResourceProperties.ORIGIN],
            citations=CitationsType.LLM_FOOTNOTES,
            generative_model=self.config.generative_model,
            filter_expression=filter_expression,
            rag_strategies=rag_strategies,
        )
        await memory.add_step(
            step_module="basic_ask",
            step_title=self.step_title("Preparing RAG"),
            step_reason="",
            step_value=ask_request.model_dump_json(
                exclude_none=True, exclude_unset=True
            ),
            timeit=0.0,
            input_nuclia_tokens=0.0,
            output_nuclia_tokens=0.0,
            step_agent_path=f"/context/{self.agent_id}",
        )
        paragraphs = await nucliadb_driver.ask(
            ask_request, headers={"X-SHOW-CONSUMPTION": "true"}
        )
        answer = None
        input_tokens = (
            paragraphs.consumption.normalized_tokens.input
            if paragraphs.consumption and paragraphs.consumption.normalized_tokens
            else 0
        )
        output_tokens = (
            paragraphs.consumption.normalized_tokens.output
            if paragraphs.consumption and paragraphs.consumption.normalized_tokens
            else 0
        )
        context.chunks = []
        answer = paragraphs.answer if paragraphs.status == "success" else ""
        if paragraphs.citation_footnote_to_context != {}:
            result_chunks = list(paragraphs.citation_footnote_to_context.values())
            answer = clean_citation_footnotes_from_answer(
                answer, paragraphs.citation_footnote_to_context
            )
        else:
            result_chunks = paragraphs.retrieval_results.best_matches
        for chunk_id in result_chunks:
            resource_id = chunk_id.split("/")[0]
            resource = paragraphs.retrieval_results.resources[resource_id]
            text = get_chunk_text(paragraphs, chunk_id)
            context.chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    title=resource.title,
                    text=text,
                    source=source,
                    origin_url=resource.origin.url if resource.origin else None,
                    origin_agent=self.config.module,
                )
            )
            # TODO: Save citations properly

        # XXX: This answer will be overriden by any call to save_ctx_and_return_missing below
        if answer and paragraphs.status == "success":
            context.summary = answer
        elif paragraphs.status in ["no_context", "no_retrieval_data"]:
            context.missing = question

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("RAG retrieval"),
            step_reason="Got answer" if answer else "No answer",
            step_value=answer if answer else "No answer",
            timeit=time() - t0,
            input_nuclia_tokens=input_tokens if input_tokens else 0,
            output_nuclia_tokens=output_tokens if output_tokens else 0,
            step_agent_path=f"/context/{self.agent_id}",
        )
        return context

    async def rag(
        self,
        source_obj: Source,
        question_uuid: str,
        question: str,
        memory: QuestionMemory,
        manager: Manager,
        flow_id: str,
    ) -> tuple[str, str] | None:
        # TODO: Use _inner_rag to avoid code duplication once we validate that the changes compared to this legacy method are correct
        # For now, we keep it separate to avoid changing the behavior
        # context = await self.inner_rag(
        #     source_obj=source_obj,
        #     manager=manager,
        #     memory=memory,
        #     question=question,
        #     question_uuid=question_uuid,
        # )

        # START OF LEGACY BLOCK
        source = source_obj.id

        nucliadb_driver = get_ndb_driver(manager, source)

        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=question_uuid,
            question=question,
            source=source,
            agent="basic_ask",
            title=self.config.title
            if self.config.title
            else f"Retrieval on {source} Knowledge Box",
        )
        t0 = time()
        ask_request = AskRequest(
            query=question,
            citations=True,
            generative_model=self.config.generative_model,
            filters=nucliadb_driver.config.filters,
        )
        paragraphs = await nucliadb_driver.ask(
            ask_request,
        )
        answer = None
        if paragraphs is not None:
            input_tokens = (
                paragraphs.metadata.tokens.input_nuclia
                if paragraphs.metadata and paragraphs.metadata.tokens
                else 0
            )
            output_tokens = (
                paragraphs.metadata.tokens.output_nuclia
                if paragraphs.metadata and paragraphs.metadata.tokens
                else 0
            )
            context.chunks = []
            answer = paragraphs.answer if paragraphs.status == "success" else ""
            if paragraphs.citations != {}:
                result_chunks = list(paragraphs.citations.keys())
            else:
                result_chunks = paragraphs.retrieval_results.best_matches
            for chunk_id in result_chunks:
                resource_id = chunk_id.split("/")[0]
                resource = paragraphs.retrieval_results.resources[resource_id]
                text = get_chunk_text(paragraphs, chunk_id)
                context.chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        title=resource.title,
                        text=text,
                        source=source,
                        origin_agent=self.config.module,
                    )
                )
                # TODO: Save citations properly

        # XXX: This answer will be overriden by any call to save_ctx_and_return_missing below
        if answer:
            context.summary = answer
        # END OF LEGACY BLOCK
        if self.fallback is None:
            if context.summary is not None and context.summary != "":
                missing = await self.save_ctx_and_return_missing(
                    context=context,
                    question=question,
                    memory=memory,
                    manager=manager,
                    flow_id=flow_id,
                )
            else:
                missing = (question_uuid, question)
                logger.info(
                    f"No context found for question {question} in source {source_obj.id}, skipping"
                )
            # START OF LEGACY BLOCK (remove this if we switch to inner_rag)
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("RAG retrieval"),
                step_reason="Got answer" if answer else "No answer",
                step_value=answer if answer else "No answer",
                timeit=time() - t0,
                input_nuclia_tokens=input_tokens if input_tokens else 0,
                output_nuclia_tokens=output_tokens if output_tokens else 0,
                step_agent_path=f"/context/{self.agent_id}",
            )
            # END OF LEGACY BLOCK
            return missing
        missing = await self.save_ctx_and_return_missing(
            context=context,
            question=question,
            memory=memory,
            manager=manager,
            flow_id=flow_id,
        )
        return missing

    async def facets_search(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        **kwargs,
    ) -> List[Context]:
        sources = self.config.sources
        # Perform catalog faceted search
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.agent_id,
            step_title=self.step_title("Choose sources"),
        )
        contexts: list[Context] = await asyncio.gather(
            *[
                self.inner_facets_search(memory, manager, question, source, i)
                for i, source in enumerate(chosen_sources)
            ]
        )

        return contexts

    async def inner_facets_search(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        source: Source,
        idx: int,
    ) -> Context:
        t0 = time()
        ndb = get_ndb_driver(manager, source.id)
        labels = await ndb.labels()

        # Select any or all relevant labels
        # XXX: Could be expanded to select not only labelset but also specific labels within the labelset since faceted search works with prefix
        prompt = FACETS_LABEL_SEARCH_AGENT_TEMPLATE.render(
            question=question, labels=labels
        )
        selected_labels, input_nt, output_nt = await manager.execute_json(
            prompt,
            user_id="rao-facets-search-label-selection",
            schema={
                "type": "object",
                "title": "Relevant Label Set Selection",
                "description": "Response schema for selected label sets.",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning behind the selection of label sets.",
                    },
                    "label_sets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of selected label sets to show all possible values.",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of selected labels to filter data in format labelset/label",
                    },
                },
                "required": ["label_sets", "labels"],
                "additionalProperties": False,
            },
            model=self.config.generative_model,
            tracking=memory.get_tracking_info(),
        )
        labelsets_selected = selected_labels.get("label_sets", [])
        labels_selected: List[str] = selected_labels.get("labels", [])

        if len(labelsets_selected) == 0:
            logger.info(
                f"No labels selected for faceted search on question {question}, skipping."
            )
            await memory.add_step(
                step_module=self.config.module,
                step_title=self.step_title("Faceted search"),
                step_reason="Faceted search skipped",
                step_value=selected_labels.get("reasoning", ""),
                timeit=time() - t0,
                input_nuclia_tokens=input_nt,
                output_nuclia_tokens=output_nt,
                step_agent_path=f"/context/{self.agent_id}/facets_search",
            )
            return Context(
                agent_id=self.agent_id,
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=None,
                missing=question,
                question=question,
                source=source.id,
                chunks=[],
                agent="basic_ask_facets",
                title=f"Faceted search on {source.id} Knowledge Box",
            )
        catalog_request = CatalogRequest(
            faceted=[f"/l/{labelset}" for labelset in labelsets_selected],
        )
        filter_expression = await self.build_catalog_filter_expression(
            ndb,
            classification_labels=labels_selected,
            classification_labels_operand="or",
        )
        catalog_request.filter_expression = filter_expression

        facets_result = await ndb.catalog_search_raw(catalog_request)

        if (
            facets_result.fulltext is not None
            and facets_result.fulltext.facets is not None
        ):
            prompt = FACETS_SEARCH_ANSWER_GENERATION_AGENT_TEMPLATE.render(
                question=question,
                facets=json.dumps(facets_result.fulltext.facets),
            )
            answer, input_nt2, output_nt2, code = await manager.execute(
                prompt,
                user_id="rao-facets-answer-generation",
                model=self.config.generative_model,
                tracking=memory.get_tracking_info(),
            )
            # TODO: Error if code is not success
            input_nt += input_nt2
            output_nt += output_nt2
            context = Context(
                agent_id=self.agent_id,
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=None,
                question=question,
                source=source.id,
                summary=answer,
                chunks=[
                    Chunk(
                        chunk_id=f"facets_search_result-{idx}",
                        text=answer,
                        origin_agent=self.config.module,
                    )
                ],
                agent="basic_ask_facets",
                title=f"Faceted search on {source.id} Knowledge Box",
            )

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Faceted search"),
            step_reason="Faceted search performed",
            step_value=selected_labels.get("reasoning", ""),
            timeit=time() - t0,
            input_nuclia_tokens=input_nt,
            output_nuclia_tokens=output_nt,
            step_agent_path=f"/context/{self.agent_id}/facets_search",
        )
        return context

    async def facets_count(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        **kwargs,
    ) -> List[Context]:
        sources = self.config.sources
        # Perform catalog faceted search
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.agent_id,
            step_title=self.step_title("Choose sources"),
        )
        contexts: list[Context] = await asyncio.gather(
            *[
                self.inner_facets(memory, manager, question, source, i)
                for i, source in enumerate(chosen_sources)
            ]
        )
        return contexts

    async def inner_facets(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        source: Source,
        idx: int,
    ) -> Context:
        t0 = time()
        ndb = get_ndb_driver(manager, source.id)
        labels = await ndb.labels()
        labels_str = format_ndb_labels(labels, max_examples=5)

        # Select any or all relevant labels
        # XXX: Could be expanded to select not only labelset but also specific labels within the labelset since faceted search works with prefix
        prompt = FACETS_LABEL_SELECTION_AGENT_TEMPLATE.render(
            question=question, labels_str=labels_str
        )
        selected_labels, input_nt, output_nt = await manager.execute_json(
            prompt,
            user_id="rao-facets-label-selection",
            schema={
                "type": "object",
                "title": "Relevant Label Set Selection",
                "description": "Response schema for selected label sets.",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning behind the selection of label sets.",
                    },
                    "label_sets": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of selected label sets.",
                    },
                },
                "required": ["label_sets"],
                "additionalProperties": False,
            },
            model=self.config.generative_model,
            tracking=memory.get_tracking_info(),
        )
        labels_selected = selected_labels.get("label_sets", [])
        if len(labels_selected) == 0:
            logger.info(
                f"No labels selected for faceted search on question {question}, skipping."
            )
        else:
            # We are assuming resources equals fields here
            facets, total = await ndb.field_labels(labelsets=labels_selected)
            prompt = FACETS_ANSWER_GENERATION_AGENT_TEMPLATE.render(
                question=question,
                labels_selected=labels_selected,
                source_id=source.id,
                total=total,
                facets=json.dumps(facets),
            )
            answer, input_nt2, output_nt2, code = await manager.execute(
                prompt,
                user_id="rao-facets-answer-generation",
                model=self.config.generative_model,
                tracking=memory.get_tracking_info(),
            )
            # TODO: Error if code is not success
            input_nt += input_nt2
            output_nt += output_nt2
            context = Context(
                agent_id=self.agent_id,
                original_question_uuid=memory.original_question_uuid,
                actual_question_uuid=None,
                question=question,
                source=source.id,
                chunks=[
                    Chunk(
                        chunk_id=f"facets_count_result-{idx}",
                        text=answer,
                        origin_agent=self.config.module,
                    )
                ],
                agent="basic_ask_facets",
                title=f"Faceted search on {source.id} Knowledge Box",
            )

        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Faceted count"),
            step_reason="Faceted search performed",
            step_value=selected_labels.get("reasoning", ""),
            timeit=time() - t0,
            input_nuclia_tokens=input_nt,
            output_nuclia_tokens=output_nt,
            step_agent_path=f"/context/{self.agent_id}/facets_count",
        )
        return context

    async def catalog_search(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        **kwargs,
    ) -> List[Context]:
        sources = self.config.sources
        # Perform catalog search
        chosen_sources = await choose_source(
            memory,
            manager,
            sources,
            question,
            ident=self.agent_id,
            step_title=self.step_title("Choose sources"),
        )
        contexts: list[Context] = await asyncio.gather(
            *[
                self.inner_catalog_search(memory, manager, question, source, i)
                for i, source in enumerate(chosen_sources)
            ]
        )

        return contexts

    async def inner_catalog_search(
        self,
        memory: QuestionMemory,
        manager: Manager,
        question: str,
        source: Source,
        idx: int,
    ) -> Context:
        t0 = time()
        ndb = get_ndb_driver(manager, source.id)
        labels = await ndb.labels()
        labels_str = format_ndb_labels(labels, max_examples=100)

        prompt = get_catalog_filter_prompt(question=question, labels_str=labels_str)
        selected_filters, input_nt, output_nt = await manager.execute_json(
            prompt,
            user_id="rao-catalog-label-selection",
            schema={
                "type": "object",
                "title": "Catalog Filter Expression Selection",
                "description": "Response schema for selected filter expression.",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning behind the selection of the filter expression.",
                    },
                    # I don't want to constrain to the schema because filter expressions can be complex, but also the providers don't let me set a freeform object, so we set a string and then parse it
                    "filters": {
                        "type": "string",
                        "description": "The filter expression to perform the catalog search as a JSON array.",
                    },
                },
                "required": ["filters"],
            },
            model=self.config.generative_model,
            tracking=memory.get_tracking_info(),
        )
        filters = self.parse_selected_filters(question, source, selected_filters)

        # TODO: Consider pagination if we want to be able to show more results
        resp = await ndb.catalog_search_raw(
            q=CatalogRequest(
                query="",
                page_number=0,
                page_size=100,
                filters=filters,
            )
        )
        catalog_str = format_ndb_catalog(resp)
        catalog_txt = (
            f"Catalog search results for filters {[f.model_dump(exclude_none=True) for f in filters]} in {source.id} Knowledge Box. "
            + "The results contain titles, formats, languages, and labels assigned to the resources for each label set. "
            + "If a result does not have a label for a specific label set, it means no label was assigned for that label set. "
            + f"The Knowledge Box contains the following label sets (with some example labels): {format_ndb_labels(labels, max_examples=5)}.\n"
        )
        if resp.fulltext and resp.fulltext.total > len(resp.resources):
            catalog_txt += f"The search matched {resp.fulltext.total}, of which, only the first {len(resp.resources)} are shown"
        catalog_txt += ":\n\nCatalog Results\n\n" + catalog_str
        prompt = CATALOG_ANSWER_GENERATION_AGENT_TEMPLATE.render(
            question=question, catalog_txt=catalog_txt
        )
        answer, answer_nt_input, answer_nt_output, code = await manager.execute(
            prompt,
            user_id="rao-catalog-search-answer",
            model=self.config.generative_model,
            tracking=memory.get_tracking_info(),
        )
        # TODO: Error if code is not success
        context = Context(
            agent_id=self.agent_id,
            original_question_uuid=memory.original_question_uuid,
            actual_question_uuid=None,
            question=question,
            source=source.id,
            chunks=[
                Chunk(
                    chunk_id=f"catalog_search_result-{idx}",
                    text=answer,
                    origin_agent=self.config.module,
                )
            ],
            agent="basic_ask_catalog",
            title=f"Catalog search on {source.id} Knowledge Box",
        )
        await memory.add_step(
            step_module=self.config.module,
            step_title=self.step_title("Catalog search"),
            step_reason="Catalog search performed",
            step_value=selected_filters.get("reasoning", ""),
            timeit=time() - t0,
            input_nuclia_tokens=input_nt + answer_nt_input,
            output_nuclia_tokens=output_nt + answer_nt_output,
            step_agent_path=f"/context/{self.agent_id}/catalog_search",
        )
        return context

    def parse_selected_filters(self, question, source_id, selected_filters):
        filters_str = selected_filters.get("filters", "[]")
        error = None
        try:
            filters_dict = json.loads(filters_str)
        except json.JSONDecodeError as e:
            if "Expecting property name enclosed in double quotes" in str(e):
                try:
                    filters_dict = ast.literal_eval(filters_str)
                except Exception as ast_error:
                    error = f"Error parsing filter expression {filters_str} from catalog search for question {question} in source {source_id}: {ast_error}"

            else:
                error = f"Error parsing filter expression {filters_str} from catalog search for question {question} in source {source_id}"
        except Exception as e:
            error = f"Error parsing filter expression {filters_str} from catalog search for question {question} in source {source_id}: {e}"
        try:
            filters = (
                [Filter.model_validate(f) for f in filters_dict] if not error else []
            )
        except Exception as e:
            error = f"Error validating filters {filters_str} from catalog search for question {question} in source {source_id}: {e}"
        if error:
            logger.error(error)
        return filters if not error else []


def get_ndb_driver(manager: Manager, source: str) -> NucliaDBDriver:
    driver = manager.drivers.get(source)
    if driver is None:
        raise Exception("No NDB available")
    return cast(NucliaDBDriver, driver)


def clean_citation_footnotes_from_answer(
    answer: str, citation_footnote_to_context: dict[str, str]
) -> str:
    """
    Remove the llm footnote citation markers from the answer text.

    Example:

    >>> answer = "Joseph earns a total of 2,466.67 per month[1], which includes both salary and other compensations[2].\n\n[1]: block-AA\n[2]: block-AC"
    >>> citation_footnote_to_context = {
        "block-AA": "resource1/field1/chunk1",
        "block-AC": "resource2/field2/chunk2",
    }
    >>> clean_citation_footnotes_from_answer(answer, citation_footnote_to_context)
    'Joseph earns a total of 2,466.67 per month, which includes both salary and other compensations.'
    """
    # First, remove the footnote definitions at the end of the answer text
    answer = answer.split("\n\n[1]: ")[0]

    # First remove the '[n]' markers from the answer text, which can be in any part of the text.
    # Adding some extra range to cover possible missing footnotes
    for i in range(1, len(citation_footnote_to_context) + 5):
        answer = answer.replace(f"[{i}]", "")
    return answer


def _to_field_filter_expression(
    legacy_filters: list[str],
) -> list[FieldFilterExpression]:
    """
    Converts a list of legacy filter strings to FieldFilterExpression objects.
    """
    field_filter_expressions = []
    for f in legacy_filters:
        fe = to_field_filter_expression(f)
        if fe is not None:
            field_filter_expressions.append(fe)
    return field_filter_expressions


def _to_resource_filter_expression(
    legacy_filters: list[str],
) -> list[ResourceFilterExpression]:
    """
    Converts a list of legacy filter strings to ResourceFilterExpression objects.
    """
    resource_filter_expressions = []
    for f in legacy_filters:
        fe = to_resource_filter_expression(f)
        if fe is not None:
            resource_filter_expressions.append(fe)
    return resource_filter_expressions
