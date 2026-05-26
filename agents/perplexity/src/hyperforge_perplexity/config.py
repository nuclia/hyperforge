from typing import ClassVar, List, Literal, Optional, Tuple

from hyperforge.context.config import ContextAgentConfig
from hyperforge.driver import DriverConfig, EncryptedPayload
from hyperforge.utils import WidgetType
from pydantic import Field
from pydantic.config import ConfigDict


class PerplexityInnerConfig(EncryptedPayload):
    encrypted_fields: ClassVar[list[str]] = ["key"]
    key: str


class PerplexityDriverConfig(DriverConfig[PerplexityInnerConfig]):
    model_config = ConfigDict(title="Perplexity")
    provider: Literal["perplexity"]
    config: PerplexityInnerConfig


class PerplexityAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Perplexity Answer")
    module: Literal["perplexity"] = "perplexity"
    published_functions: Optional[Tuple[str, ...]] = Field(
        default=("internet_search",),
        title="Published functions",
        description="List of functions published by this agent to be used by other agents in the chain",
        json_schema_extra={
            "widget": WidgetType.NOT_SHOWN,
        },
    )
    domain: List[str] = Field(
        default_factory=list,
        title="Domain Filter",
        description="Domains to restrict the Perplexity search to.",
    )
    search_context_size: Literal["low", "medium", "high"] = Field(
        default="low",
        title="Search Context Size",
        description="Determines how much search context Perplexity retrieves for the model. "
        "Options are: `low` (minimizes context for cost savings but less comprehensive answers), "
        "`medium` (balanced approach suitable for most queries), "
        "and `high` (maximizes context for comprehensive answers but at higher cost).",
    )
    related_questions: bool = Field(
        default=False,
        title="Generate Related Questions",
        description="Determines if Perplexity should return related questions to the original. "
        "They will be stored in the Agentic Memory as future questions",
    )
    images: bool = Field(
        default=False,
        title="Return Images",
        description="Determines whether Perplexity search results should include images.",
    )
    prompt: Optional[str] = Field(
        None,
        title="Custom Prompt",
        description="Custom prompt to use for the Perplexity agent.",
        json_schema_extra={
            "widget": WidgetType.EXPANDABLE_TEXTAREA,
        },
    )
    source: str = "perplexity"
