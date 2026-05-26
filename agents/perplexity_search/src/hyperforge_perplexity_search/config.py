from typing import List, Literal

from hyperforge.context.config import ContextAgentConfig
from pydantic import Field
from pydantic.config import ConfigDict


class PerplexitySearchAgentConfig(ContextAgentConfig):
    model_config = ConfigDict(title="Perplexity Search")
    module: Literal["perplexity_search"] = "perplexity_search"
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
    max_results: int = Field(
        default=10,
        title="Max Results",
        description="Maximum number of search results to return.",
    )
    max_tokens_per_page: int = Field(
        default=4096,
        title="Max Tokens per Page",
        description="Maximum number of tokens to return per search result page.",
    )

    source: str = "perplexity"
