"""Bound external connector results before they can be sent to an LLM."""

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping, Sequence

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BYTES_PER_KB = 1024


def migrate_legacy_byte_limits(value: Any) -> Any:
    """Convert stored byte-based agent limits to the current KB fields."""
    if not isinstance(value, Mapping):
        return value

    migrated = dict(value)
    legacy_fields = (
        ("max_tool_result_bytes", "max_tool_result_kb"),
        ("max_tool_result_item_bytes", "max_tool_result_item_kb"),
    )
    for legacy_field, kb_field in legacy_fields:
        legacy_value = migrated.pop(legacy_field, None)
        if kb_field not in migrated and legacy_value is not None:
            migrated[kb_field] = ceil(legacy_value / BYTES_PER_KB)
    return migrated


@dataclass(frozen=True)
class ResultPayloadBudget:
    """Limits for data returned by a connector in a single tool call."""

    max_bytes: int
    max_item_bytes: int


class ResultPayloadSettings(BaseSettings):
    """Deployment defaults for connector results that may enter LLM context."""

    model_config = SettingsConfigDict(
        env_prefix="HYPERFORGE_TOOL_RESULT_",
        populate_by_name=True,
    )

    max_bytes: int = Field(
        default=64 * 1024,
        ge=1,
    )
    max_item_bytes: int = Field(
        default=16 * 1024,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_item_limit(self) -> "ResultPayloadSettings":
        if self.max_item_bytes > self.max_bytes:
            raise ValueError("max_item_bytes cannot exceed max_bytes")
        return self


@dataclass(frozen=True)
class OversizedResult:
    """A safe, bounded description of a rejected connector result."""

    kind: str
    observed_bytes: int
    max_bytes: int

    def trace_value(self) -> str:
        """Return non-sensitive metadata suitable for an execution step."""
        value = (
            f"kind={self.kind}; observed_bytes={self.observed_bytes}; "
            f"byte_limit={self.max_bytes}"
        )
        return value

    def render(self) -> str:
        details = [
            "Tool result was not included because it exceeds the configured safety budget.",
            f"Observed size: {self.observed_bytes} bytes; limit: {self.max_bytes} bytes.",
        ]
        details.append(
            "Retry with narrower filters, pagination, aggregation, or a summary-oriented request."
        )
        return "\n".join(details)


def budget_from_config(config: Any) -> ResultPayloadBudget:
    """Resolve optional agent overrides against deployment-level defaults."""
    settings = ResultPayloadSettings()
    legacy_max_bytes = getattr(config, "max_tool_result_bytes", None)
    legacy_max_item_bytes = getattr(config, "max_tool_result_item_bytes", None)
    max_kb = getattr(config, "max_tool_result_kb", None)
    max_item_kb = getattr(config, "max_tool_result_item_kb", None)
    max_bytes = legacy_max_bytes or (
        max_kb * BYTES_PER_KB if max_kb is not None else settings.max_bytes
    )
    max_item_bytes = legacy_max_item_bytes or (
        max_item_kb * BYTES_PER_KB
        if max_item_kb is not None
        else settings.max_item_bytes
    )
    if max_item_bytes > max_bytes:
        raise ValueError(
            "The tool result item limit cannot exceed the total result limit"
        )
    return ResultPayloadBudget(max_bytes=max_bytes, max_item_bytes=max_item_bytes)


def inspect_text_blocks(
    texts: Sequence[str], budget: ResultPayloadBudget
) -> OversizedResult | None:
    """Inspect several text blocks as one connector result."""
    for text in texts:
        overflow = _inspect_text(text, budget)
        if overflow is not None:
            return overflow

    observed_bytes = 0
    for idx, text in enumerate(texts):
        if idx:
            observed_bytes += 1  # "\n" separator
        observed_bytes += len(text.encode("utf-8"))

    if observed_bytes <= budget.max_bytes:
        return None
    return OversizedResult(
        kind="text",
        observed_bytes=observed_bytes,
        max_bytes=budget.max_bytes,
    )


def _inspect_text(text: str, budget: ResultPayloadBudget) -> OversizedResult | None:
    observed_bytes = len(text.encode("utf-8"))
    if observed_bytes <= budget.max_bytes and observed_bytes <= budget.max_item_bytes:
        return None
    return OversizedResult(
        kind="text",
        observed_bytes=observed_bytes,
        max_bytes=min(budget.max_bytes, budget.max_item_bytes),
    )
