from hyperforge.result_payload import (
    ResultPayloadBudget,
    inspect_text_blocks,
)


def test_accepts_a_small_text_result():
    assert inspect_text_blocks(["small result"], ResultPayloadBudget(max_bytes=32)) is None


def test_rejects_large_text_without_including_payload_content():
    result = inspect_text_blocks(["x" * 100], ResultPayloadBudget(max_bytes=32))

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