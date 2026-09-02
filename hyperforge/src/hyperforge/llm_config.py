"""Structured generative model configuration.

This module defines the `LLMConfig` model and the `LLMField` annotated type
that provides backwards-compatible coercion from plain model ID strings to
structured configuration objects.

Usage in agent configs:

    from hyperforge.llm_config import LLMConfig, LLMField, llm_defaults

    class MyAgentConfig(AgentConfig):
        planner_model: LLMField = Field(
            default=LLMConfig(model_id=llm_defaults.smart)
        )
"""

from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, BeforeValidator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from hyperforge.utils import WidgetType

# ---------------------------------------------------------------------------
# Reasoning configuration
# ---------------------------------------------------------------------------


class ReasoningEffort(str, Enum):
    """Reasoning effort levels supported by model providers."""

    NONE = "none"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"


class ReasoningConfig(BaseModel):
    """Advanced reasoning configuration for models that support it."""

    effort: Optional[ReasoningEffort] = None
    budget_tokens: Optional[int] = None


class SimpleReasoning(str, Enum):
    """Simplified reasoning toggle.

    When set, this is internally mapped to a ReasoningConfig:
    - DISABLED -> ReasoningConfig(effort=ReasoningEffort.NONE)
    - ENABLED -> ReasoningConfig(effort=ReasoningEffort.HIGH)
    """

    DISABLED = "disabled"
    ENABLED = "enabled"


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------

# Discriminator value used for locating LLMConfig instances in stored JSONB.
# Migration scripts can locate all LLMConfig instances by querying for
# fields containing `{"_type": "llm_config"}`.
LLM_CONFIG_TYPE = "llm_config"


class LLMConfig(BaseModel):
    """Structured generative model configuration.

    The `_type` field serves as a discriminator/marker that makes instances
    of this model identifiable when stored as JSONB in the database.
    Migration scripts can locate all LLMConfig instances by querying for
    fields containing `{"_type": "llm_config"}`.
    """

    model_config = {"populate_by_name": True, "serialize_by_alias": True}

    type: str = Field(
        default=LLM_CONFIG_TYPE,
        alias="_type",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )
    model_id: str = Field(
        title="Model",
        description="The model identifier (e.g. 'chatgpt-azure-4o-mini', 'chatgpt-4.1')",
        json_schema_extra={"widget": WidgetType.MODEL_SELECT},
    )
    reasoning: Optional[SimpleReasoning] = Field(
        default=None,
        title="Reasoning",
        description="Simplified reasoning toggle. Set to 'enabled' for reasoning models.",
        json_schema_extra={"widget": WidgetType.ENUM_SELECT},
    )
    advanced_reasoning: Optional[ReasoningConfig] = Field(
        default=None,
        title="Advanced reasoning",
        description="Fine-grained reasoning configuration (effort level, budget tokens). Takes precedence over 'reasoning' if both are set.",
        json_schema_extra={"widget": WidgetType.NOT_SHOWN},
    )

    def get_effective_reasoning(self) -> Optional[ReasoningConfig]:
        """Resolve the effective reasoning configuration.

        `advanced_reasoning` takes precedence over `reasoning` if both are set.
        """
        if self.advanced_reasoning is not None:
            return self.advanced_reasoning
        if self.reasoning is not None:
            if self.reasoning == SimpleReasoning.ENABLED:
                return ReasoningConfig(effort=ReasoningEffort.HIGH)
            else:
                return ReasoningConfig(effort=ReasoningEffort.NONE)
        return None


# ---------------------------------------------------------------------------
# LLMField: Annotated type with backwards-compatible coercion
# ---------------------------------------------------------------------------


def _coerce_llm_config(value: Any) -> Any:
    """BeforeValidator that accepts str, dict, or LLMConfig and normalizes to LLMConfig.

    - str: Treated as a bare model ID (legacy format). Converted to LLMConfig(model_id=value).
    - dict: Passed through to pydantic for normal LLMConfig validation.
    - LLMConfig: Returned as-is.
    """
    if isinstance(value, str):
        return LLMConfig(model_id=value)
    if isinstance(value, LLMConfig):
        return value
    # dict or other mapping - let pydantic handle it
    return value


LLMField = Annotated[LLMConfig, BeforeValidator(_coerce_llm_config)]
"""Annotated type for agent config fields that hold an LLMConfig.

Accepts:
- A plain model ID string (backwards compatible with legacy configs)
- A dict with LLMConfig fields (structured format)
- An LLMConfig instance

Always validates/serializes as a full LLMConfig object.
"""


# ---------------------------------------------------------------------------
# Centralized defaults (environment-configurable)
# ---------------------------------------------------------------------------


class LLMDefaults(BaseSettings):
    """Central registry for model defaults.

    All values are overridable via environment variables with the
    HYPERFORGE_LLM_ prefix. For example:
        HYPERFORGE_LLM_DEFAULT=chatgpt-4.5
        HYPERFORGE_LLM_SMART=chatgpt-4.5
    """

    model_config = SettingsConfigDict(env_prefix="HYPERFORGE_LLM_")

    # General purpose (low cost, fast)
    default: str = "chatgpt-azure-4o-mini"
    # Smart tier (complex planning, tool use)
    smart: str = "chatgpt-4.1"
    # Fast tier (high throughput, configuration)
    fast: str = "gemini-2.5-flash"
    # Reasoning tier (decision making, conditions)
    reasoning: str = "chatgpt-o3-mini"


llm_defaults = LLMDefaults()
