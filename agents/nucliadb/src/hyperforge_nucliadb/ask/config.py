from typing import List, Literal, Optional

from hyperforge.context.config import ContextAgentConfig
from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults
from hyperforge.utils import WidgetType
from nucliadb_models.search import KnowledgeGraphEntity
from pydantic import Field
from pydantic.config import ConfigDict


class AskAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Knowledge Box Ask")
    module: Literal["ask"] = "ask"
    published_functions: Optional[tuple[str, ...]] = Field(
        default=("search_by_title", "ask_analysis_query", "ask_agent"),
        title="Published functions",
        description="List of functions published by this agent to be used by other agents in the chain",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    pre_queries: List[str] = Field(default_factory=list)
    filters: List[str] = Field(default_factory=list)
    security_groups: List[str] = Field(default_factory=list)
    rephrase_semantic_custom_prompt: Optional[str] = None
    rephrase_lexical_custom_prompt: Optional[str] = None
    keywords_custom_prompt: Optional[str] = None
    visual_enable_prompt: Optional[str] = None
    date_range_enabled: bool = False
    before: int = 2
    after: int = 2
    top_k: int = 20
    extra_fields: List[str] = Field(default_factory=list)
    full_resource: bool = False
    vllm: bool = True
    query_entities: List[KnowledgeGraphEntity] = Field(default_factory=list)
    retrieve_related: Optional[str] = None
    sources: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "show_in_node": True,
        },
    )
    configuration_model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.fast),
        title="Generative model",
        description="Model used to generate the configuration",
    )
    fast_answer: bool = True
    # Setting this to True so that this PR will not break the current behavior
    # of the agent
    ai_parameter_search: bool = True
