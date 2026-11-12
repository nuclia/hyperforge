from asyncio import gather
from copy import deepcopy
from time import time
from typing import Any, Dict, List

from hyperforge import PROMPT_ENVIRONMENT
from hyperforge.manager import Manager
from hyperforge.memory.memory import QuestionMemory, Source
from nuclia_models.predict.run_agents import RunTextAgentsRequest
from nuclia_models.worker.triggers import Relation
from nucliadb_models import RelationNodeType
from nucliadb_models.search import Filter, KnowledgeGraphEntity

from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_nucliadb.ask.kb_analysis import get_knowledge_base_analysis
from hyperforge_nucliadb.ask.models import Analysis
from hyperforge_nucliadb.ask.prompt_analysis import configure_prompts
from hyperforge_nucliadb.ask.query_analysis import pre_query_analysis
from hyperforge_nucliadb.driver import NucliaDBDriver

KEYWORD_FILTER_PROMPT = "Extract if any the keywords that should appear on the retrieved results and its a must match, make sure that are keywords that are not common words, and that are not the same as the question or answer. Please define ONLY the keywords without any explanation. Only one or two words maximum. JUST A LIST OF KEYWORDS. DO NOT ADD ANY EXTRA NOTES AT THE END, JUST A LIST OF KEYWORDS"

REPHRASE_SEMANTIC_PROMPT = "Rephrase this question so its better for semantic retrieval, and keep the rephrased question in the same language as the original. Please define ONLY the question without any explanation. JUST A SENTENCE. DO NOT ADD ANY EXTRA NOTES AT THE END, JUST ONE SENTENCE"

REPHRASE_LEXICAL_PROMPT = "Rephrase this question so its better for lexical retrieval, and keep the rephrased question in the same language as the original. Please define ONLY the question without any explanation. JUST A SENTENCE. DO NOT ADD ANY EXTRA NOTES AT THE END, JUST ONE SENTENCE"


ASK_JSON_SCHEMA = {
    "title": "ask_configuration",
    "description": "Configuration extracted from reasoning engine",
    "parameters": {
        "type": "object",
        "properties": {
            "link": {
                "type": "boolean",
                "description": "The user wants link reference to the answer?",
            },
            "knowledge_scan": {
                "type": "string",
                "description": "If the query requires a knowledge aggregation or scan search to answer define the entities, labels and relations to query in the KB. Example queries: How many ...",
            },
            "semantic_query": {
                "type": "string",
                "description": REPHRASE_SEMANTIC_PROMPT,
            },
            "lexical_query": {"type": "string", "description": REPHRASE_LEXICAL_PROMPT},
            "visual": {
                "type": "boolean",
                "description": "Is required an analysis of an image to answer this question, answer with false or true",
            },
            "keywords_filter": {
                "type": "array",
                "items": {"type": "string"},
                "description": KEYWORD_FILTER_PROMPT,
            },
            "reason": {"type": "string"},
            "entities": {
                "type": "array",
                "description": "Entities related to the user question to query in the KB",
                "items": {
                    "type": "string",
                },
            },
            "relations": {
                "type": "array",
                "description": "Relations related to the user question to query in the KB",
                "items": {
                    "type": "string",
                },
            },
            "pre_queries": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "Pre queries to run before the main query to gather more information",
            },
        },
        "required": [
            "semantic_query",
            "lexical_query",
            "visual",
            "keywords_filter",
            "reason",
            "pre_queries",
        ],
    },
}


ASK_PROMPT = """

Information about the KB:

# {{sid}}

description: {{sdescription}}

And given the question: {{question}}

## labels: {{slabels}}

## Facets:

{{facets}}

# Important rules to follow

{% for rule in rules %}
{{rule}}
{% endfor -%}


"""

ASK_PROMPT_TEMPLATE = PROMPT_ENVIRONMENT.from_string(ASK_PROMPT)


async def question_analysis(
    memory: QuestionMemory,
    manager: Manager,
    nucliadb_driver: NucliaDBDriver,
    source: Source,
    config: AskAgentConfig,
    question: str,
    step_title: str = "Knowledge Box Ask: Choose parameters",
):
    if question is None:
        raise Exception("Question is None")
    question = question.strip()

    if source is None:
        raise Exception("Source is None")

    kb_analysis = get_knowledge_base_analysis(source)

    query_analysis_result = pre_query_analysis(config, question, kb_analysis)

    ask_json_schema_copy: Dict[str, Any] = deepcopy(ASK_JSON_SCHEMA)
    configure_prompts(config, ask_json_schema_copy, kb_analysis, query_analysis_result)

    facets = ""
    facets += "## Content Types\n"
    facets += "The following content types are available in the KB:\n\n"
    for key, count in kb_analysis.content_types.items():
        facets += f"- {key}: {count}\n"

    facets += "The following languages are available in the KB:\n\n"
    for key, count in kb_analysis.languages.items():
        facets += f"- {key}: {count}\n"

    prompt = ASK_PROMPT_TEMPLATE.render(
        source=source,
        rules=memory.get_rules(),
        question=question,
        facets=facets,
        sid=source.id,
        sdescription=source.description,
        slabels=source.labels,
    )

    t0 = time()
    input_tokens = 0.0
    output_tokens = 0.0

    task1 = manager.execute_json(
        user_id="arag-ask",
        prompt=prompt,
        model=config.configuration_model,
        schema=ask_json_schema_copy,
        tracking=memory.get_tracking_info(),
    )

    task2 = nucliadb_driver.driver.run_text_agents(
        kbid=nucliadb_driver.config.kbid,
        content=RunTextAgentsRequest(user_id="arag-ask", texts=[question]),
    )

    task3 = manager.tokens_predict(
        text=question,
        model=kb_analysis.entity_model,
    )

    # Extract configuration from reasoning engine
    (configuration_json, input, output), das_answer, tokens = await gather(
        task1, task2, task3
    )

    input_tokens += input
    output_tokens += output

    entities_to_filter = deepcopy(config.query_entities)
    relations_to_filter: List[Relation] = []
    labels_to_filter: List[Filter] = []
    if len(config.filters) > 0:
        labels_to_filter.append(Filter(all=config.filters))
    new_text_fields: List[str] = []
    new_json_fields: Dict[str, Any] = {}

    for detected_token in tokens.tokens:
        entities_to_filter.append(
            KnowledgeGraphEntity(
                name=detected_token.text,
                type=RelationNodeType.ENTITY,
                subtype=detected_token.ner,
            )
        )

    for result in das_answer.results:
        input_tokens += result.input_nuclia_tokens
        output_tokens += result.output_nuclia_tokens
        for payload in result.payloads:
            # we could implement for guards
            if payload.entities is not None:
                for entity in payload.entities:
                    for entity_class, entity_value in entity.labels.items():
                        entities_to_filter.append(
                            KnowledgeGraphEntity(
                                name=entity_value,
                                type=RelationNodeType.ENTITY,
                                subtype=entity_class,
                            )
                        )

            if payload.relations is not None:
                for relations in payload.relations:
                    relations_to_filter.extend(relations.relations)

            if payload.labels is not None:
                das_filters: list[str] = []
                for label in payload.labels:
                    if isinstance(label.labels, list):
                        for label_obj_str in label.labels:
                            das_filters.append(label_obj_str)
                    elif isinstance(label.labels, dict):
                        for key, value in label.labels.items():
                            das_filters.append(f"{key}:{value}")
                    else:
                        raise Exception(f"Unknown label type: {label.labels}")
                labels_to_filter.append(Filter(all=das_filters))

            if payload.asks is not None:
                for ask in payload.asks:
                    if ask.empty is False and ask.text is not None:
                        new_text_fields.append(ask.text)
                    elif ask.empty is False and ask.json_output is not None:
                        new_json_fields.update(ask.json_output)

    for entity_llm in configuration_json.get("entities", []):
        entities_to_filter.append(
            KnowledgeGraphEntity(
                name=entity_llm,
                type=RelationNodeType.ENTITY,
            )
        )

    if kb_analysis.default_semantic_model is None and (
        kb_analysis.semantic_models is None or len(kb_analysis.semantic_models) == 0
    ):
        raise Exception("No semantic model found in the knowledge base")

    semantic_model = (
        kb_analysis.default_semantic_model
        if kb_analysis.default_semantic_model
        else kb_analysis.semantic_models[0]
    )

    semantic_query = configuration_json["semantic_query"]
    lexical_query = configuration_json["lexical_query"]
    visual = configuration_json.get("visual", False)
    link = configuration_json.get("link", False)
    knowledge_graph = configuration_json.get("knowledge_graph", None)
    keywords_filter: List[str] = configuration_json.get("keywords_filter", [])
    if isinstance(keywords_filter, str):
        keywords_filter = [word.strip() for word in keywords_filter.split(",")]

    reason = configuration_json.get("reason")
    pre_queries = configuration_json.get("pre_queries", [])
    relations = configuration_json.get("relations", [])
    await memory.add_step(
        step_module="ask",
        step_title=step_title,
        step_agent_path=f"/context/{config.id if config.id else 'default'}",
        step_value=str(configuration_json),
        step_reason=reason if reason is not None else "",
        timeit=time() - t0,
        input_nuclia_tokens=input_tokens,
        output_nuclia_tokens=output_tokens,
    )

    return Analysis(
        semantic_query=semantic_query,
        lexical_query=lexical_query,
        visual=visual,
        keywords_filter=keywords_filter,
        reason=reason,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        labels=labels_to_filter,
        semantic_model=semantic_model,
        entities=entities_to_filter,
        pre_queries=pre_queries,
        relations=relations_to_filter,
        new_text_fields=new_text_fields,
        new_json_fields=new_json_fields,
        semantic_config=kb_analysis.semantic_model_configs[semantic_model],
        top_k=config.top_k,
        link=link,
        knowledge_graph=knowledge_graph,
    )
