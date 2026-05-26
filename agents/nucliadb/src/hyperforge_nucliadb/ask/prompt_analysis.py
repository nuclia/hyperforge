from typing import Any, Dict

from hyperforge_nucliadb.ask.config import AskAgentConfig
from hyperforge_nucliadb.ask.kb_analysis import KnowledgeBoxInfo
from hyperforge_nucliadb.ask.query_analysis import QueryAnalisys

# https://huggingface.co/collections/ModelSpace/gemmax2-673714f5049bfa3a90bee6b6
TRANSLATE_QUERY_PROMPT = "Rephrase this question so its better for lexical retrieval, translate the lexical rephrased question to {language}. Please define ONLY the question without any explanation. JUST A SENTENCE. DO NOT ADD ANY EXTRA NOTES AT THE END, JUST ONE SENTENCE"


def configure_prompts(
    config: AskAgentConfig,
    ask_json_schema_copy: Dict[str, Dict[str, Any]],
    kb_analysis: KnowledgeBoxInfo,
    query_analysis: QueryAnalisys,
) -> None:
    """
    Configure the prompts based on the provided configuration.
    This function modifies the ask_json_schema_copy in place to set the
    appropriate prompts for semantic, lexical, and visual queries.
    """
    if config.visual_enable_prompt is not None:
        ask_json_schema_copy["parameters"]["properties"]["visual"]["description"] = (
            config.visual_enable_prompt
        )

    if config.rephrase_semantic_custom_prompt is not None:
        ask_json_schema_copy["parameters"]["properties"]["semantic_query"][
            "description"
        ] = config.rephrase_semantic_custom_prompt

    # Rephrase as lexical to query lexical engine
    if config.rephrase_lexical_custom_prompt is not None:
        ask_json_schema_copy["parameters"]["properties"]["lexical_query"][
            "description"
        ] = config.rephrase_lexical_custom_prompt
    elif query_analysis.translate_need is not None:
        ask_json_schema_copy["parameters"]["properties"]["lexical_query"][
            "description"
        ] = TRANSLATE_QUERY_PROMPT.format(language=query_analysis.translate_need)

    # Rephrase as keywords that must appear only if the language of the query matches the language of the KB

    if (
        query_analysis.translate_need is None
        and config.keywords_custom_prompt is not None
    ):
        ask_json_schema_copy["parameters"]["properties"]["keywords_filter"][
            "description"
        ] = config.keywords_custom_prompt
