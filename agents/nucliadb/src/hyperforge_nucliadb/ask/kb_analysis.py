from dataclasses import dataclass, field
from typing import Dict, Optional

from hyperforge.models import Source
from nuclia.lib.nua_responses import SemanticConfig


@dataclass
class KnowledgeBoxInfo:
    content_types: Dict[str, int]
    languages: Dict[str, int]
    paragraph_facets: Dict[str, int]
    entity_model: str
    generative_model: str
    default_semantic_model: Optional[str] = None
    semantic_models: Optional[list[str]] = None
    semantic_model_configs: Dict[str, SemanticConfig] = field(default_factory=dict)


def get_knowledge_base_analysis(source: Source):
    content_types = {}
    languages: Dict[str, int] = {}

    for key, count in source.facets_native.facets.items():
        if key.startswith("/n/i"):
            content_type = key.replace("/n/i/", "")
            if content_type.count("/") >= 1:
                content_types[content_type] = count
        elif key.startswith("/s/p/"):
            language = key.replace("/s/p/", "")
            languages[language] = count

    return KnowledgeBoxInfo(
        paragraph_facets=source.paragraph_facets,
        content_types=content_types,
        languages=languages,
        semantic_models=source.learning_configuration.semantic_models,
        entity_model=source.learning_configuration.ner_model,
        generative_model=source.learning_configuration.generative_model,
        default_semantic_model=source.learning_configuration.default_semantic_model,
        semantic_model_configs=source.learning_configuration.semantic_model_configs,
    )
