"""Tests for hyperforge.llm_config: backwards-compatible coercion and reasoning resolution."""

from hyperforge.llm_config import (
    LLMConfig,
    LLMField,
    ReasoningConfig,
    ReasoningEffort,
    SimpleReasoning,
    llm_defaults,
)
from pydantic import BaseModel, Field


class SampleConfig(BaseModel):
    model: LLMField = Field(
        default_factory=lambda: LLMConfig(model_id=llm_defaults.default)
    )


class TestLLMFieldBackwardsCompat:
    """Tests that the LLMField validator correctly coerces legacy plain strings."""

    def test_coerce_from_plain_string(self):
        config = SampleConfig.model_validate({"model": "gpt-5"})
        assert config.model.model_id == "gpt-5"

    def test_coerce_from_structured_dict(self):
        config = SampleConfig.model_validate(
            {"model": {"_type": "llm_config", "model_id": "gpt-5"}}
        )
        assert config.model.model_id == "gpt-5"

    def test_round_trip_from_legacy_string(self):
        """String in -> serialize -> deserialize preserves the value."""
        config = SampleConfig.model_validate({"model": "gpt-5"})
        data = config.model_dump()
        config2 = SampleConfig.model_validate(data)
        assert config2.model.model_id == "gpt-5"


class TestSimpleReasoningResolution:
    """Tests that SimpleReasoning maps correctly to ReasoningConfig."""

    def test_enabled_maps_to_high_effort(self):
        cfg = LLMConfig(model_id="chatgpt-o3", reasoning=SimpleReasoning.ENABLED)
        effective = cfg.get_effective_reasoning()
        assert effective.effort == ReasoningEffort.HIGH

    def test_disabled_maps_to_none_effort(self):
        cfg = LLMConfig(model_id="chatgpt-o3", reasoning=SimpleReasoning.DISABLED)
        effective = cfg.get_effective_reasoning()
        assert effective.effort == ReasoningEffort.NONE

    def test_advanced_takes_precedence_over_simple(self):
        cfg = LLMConfig(
            model_id="chatgpt-o3",
            reasoning=SimpleReasoning.DISABLED,
            advanced_reasoning=ReasoningConfig(
                effort=ReasoningEffort.MEDIUM, budget_tokens=10000
            ),
        )
        effective = cfg.get_effective_reasoning()
        assert effective.effort == ReasoningEffort.MEDIUM
        assert effective.budget_tokens == 10000

    def test_no_reasoning_returns_none(self):
        cfg = LLMConfig(model_id="chatgpt-4.1")
        assert cfg.get_effective_reasoning() is None
