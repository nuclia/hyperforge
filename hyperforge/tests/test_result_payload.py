from types import SimpleNamespace

import pytest

from hyperforge.result_payload import (
    ResultPayloadBudget,
    ResultPayloadSettings,
    budget_from_config,
    inspect_text_blocks,
)


def test_accepts_a_small_text_result():
    assert (
        inspect_text_blocks(
            ["small result"], ResultPayloadBudget(max_bytes=32, max_item_bytes=16)
        )
        is None
    )


def test_rejects_large_text_without_including_payload_content():
    result = inspect_text_blocks(
        ["x" * 100], ResultPayloadBudget(max_bytes=32, max_item_bytes=16)
    )

    assert result is not None
    assert result.kind == "text"
    assert "100 bytes" in result.render()
    assert "x" * 100 not in result.render()


def test_accepts_multiple_small_blocks_within_the_total_budget():
    assert (
        inspect_text_blocks(
            ["x" * 10, "y" * 10],
            ResultPayloadBudget(max_bytes=32, max_item_bytes=16),
        )
        is None
    )


def test_rejects_multiple_small_blocks_that_exceed_the_total_budget():
    result = inspect_text_blocks(
        ["x" * 10, "y" * 10],
        ResultPayloadBudget(max_bytes=16, max_item_bytes=16),
    )

    assert result is not None
    assert result.observed_bytes == 21
    assert result.max_bytes == 16


def test_uses_deployment_defaults_when_agent_has_no_overrides(monkeypatch):
    monkeypatch.setenv("ARAG_TOOL_RESULT_MAX_BYTES", "100")
    monkeypatch.setenv("ARAG_TOOL_RESULT_MAX_ITEM_BYTES", "50")

    budget = budget_from_config(
        SimpleNamespace(max_tool_result_bytes=None, max_tool_result_item_bytes=None)
    )

    assert budget == ResultPayloadBudget(max_bytes=100, max_item_bytes=50)


def test_uses_arag_tool_result_environment_prefix(monkeypatch):
    monkeypatch.setenv("ARAG_TOOL_RESULT_MAX_BYTES", "100")
    monkeypatch.setenv("ARAG_TOOL_RESULT_MAX_ITEM_BYTES", "50")

    settings = ResultPayloadSettings()

    assert settings.max_bytes == 100
    assert settings.max_item_bytes == 50


def test_agent_overrides_take_precedence_over_deployment_defaults(monkeypatch):
    monkeypatch.setenv("ARAG_TOOL_RESULT_MAX_BYTES", "100")
    monkeypatch.setenv("ARAG_TOOL_RESULT_MAX_ITEM_BYTES", "50")

    budget = budget_from_config(
        SimpleNamespace(max_tool_result_bytes=80, max_tool_result_item_bytes=40)
    )

    assert budget == ResultPayloadBudget(max_bytes=80, max_item_bytes=40)


def test_rejects_deployment_item_limit_above_total_limit():
    with pytest.raises(ValueError, match="max_item_bytes cannot exceed max_bytes"):
        ResultPayloadSettings(max_bytes=32, max_item_bytes=64)
