import asyncio

import pytest
from pydantic import ValidationError

from hyperforge.harness_sdk import (
    AgentHarness,
    InMemoryHarnessStorage,
    ModelDelta,
    ToolCallContext,
)
from hyperforge.harness_sdk.tools import core


class NoModel:
    async def stream(self, **kwargs):
        yield ModelDelta(text="unused")


def context(storage=None, **identity):
    harness = AgentHarness(
        model="test",
        model_client=NoModel(),
        storage=storage,
        execution_context={
            "account_id": "account",
            "user_id": "user",
            "project_id": "project",
            **identity,
        },
    )
    return ToolCallContext(harness=harness, name="memory")


@pytest.mark.asyncio
async def test_named_memories_upsert_across_sessions_and_identical_writes_are_noops():
    storage = InMemoryHarnessStorage()
    first = context(storage)
    created = await core.remember(
        first, core.RememberInput(name="maintenance-backlog", text="Use an inner join.")
    )
    original = await storage.get_memory(created.value["id"])
    second = context(storage)
    updated = await core.remember(
        second, core.RememberInput(name="maintenance-backlog", text="Use a LEFT JOIN.")
    )
    recalled = await core.recall(second, core.RecallInput(name="maintenance-backlog"))
    unchanged = await core.remember(
        second, core.RememberInput(name="maintenance-backlog", text="Use a LEFT JOIN.")
    )

    assert created.value["operation"] == "created"
    assert updated.value["operation"] == "updated"
    assert unchanged.value["operation"] == "unchanged"
    assert created.value["id"] == updated.value["id"] == unchanged.value["id"]
    assert len(storage.memories) == 1
    assert original is not None
    current = await storage.get_memory(created.value["id"])
    assert current is not None and current.created_datetime == original.created_datetime
    assert current.updated_datetime is not None
    assert recalled.items[0]["text"] == "Use a LEFT JOIN."
    assert len(storage.events[second.harness.conversation_id]) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "other", "shared"),
    [
        ("user", {"project_id": "other-project"}, True),
        ("project", {"user_id": "other-user"}, True),
        ("user_project", {"project_id": "other-project"}, False),
        ("user_project", {"user_id": "other-user"}, False),
        ("user_project", {"account_id": "other-account"}, False),
        ("user_project", {"memory_namespace": "other-app"}, False),
    ],
)
async def test_short_names_respect_scope_and_namespace(scope, other, shared):
    storage = InMemoryHarnessStorage()
    original = await core.remember(
        context(storage),
        core.RememberInput(name="route", scope=scope, text="Original route"),
    )
    changed = await core.remember(
        context(storage, **other),
        core.RememberInput(name="route", scope=scope, text="New route"),
    )
    assert (original.value["id"] == changed.value["id"]) is shared
    assert len(storage.memories) == (1 if shared else 2)


@pytest.mark.asyncio
async def test_memory_search_is_bounded_ranked_and_matches_names_with_different_wording():
    ctx = context()
    await core.remember(
        ctx, core.RememberInput(name="maintenance", text="Inspect a schema.")
    )
    best = await core.remember(
        ctx, core.RememberInput(name="maintenance-overdue", text="Group by priority.")
    )
    for index in range(10):
        await core.remember(
            ctx, core.RememberInput(name=f"misc-{index}", text="Unrelated procedure.")
        )
    results = await core.recall(
        ctx,
        core.RecallInput(query="maintenance overdue priority assigned work", limit=1),
    )
    assert results.items[0]["id"] == best.value["id"]
    assert len((await core.recall(ctx, core.RecallInput())).items) == 5
    assert not (await core.recall(ctx, core.RecallInput(query="and the"))).items
    assert not (await core.recall(ctx, core.RecallInput(query="unmatchedxyz"))).items
    assert not (await core.recall(ctx, core.RecallInput(name="missing"))).items


@pytest.mark.asyncio
async def test_search_does_not_return_another_namespace_or_let_it_hide_own_results():
    storage = InMemoryHarnessStorage()
    own = context(storage)
    memory = await core.remember(
        own, core.RememberInput(name="route", text="maintenance")
    )
    other = context(storage, account_id="another-account")
    for index in range(10):
        await core.remember(
            other,
            core.RememberInput(
                name=f"other-{index}", text="maintenance overdue priority"
            ),
        )
    results = await core.recall(
        own, core.RecallInput(query="maintenance overdue priority")
    )
    assert [item["id"] for item in results.items] == [memory.value["id"]]


@pytest.mark.asyncio
async def test_forget_by_name_does_not_delete_another_namespace():
    storage = InMemoryHarnessStorage()
    own = context(storage)
    await core.remember(own, core.RememberInput(name="route", text="Useful route"))
    other = context(storage, user_id="another-user")
    await core.forget(other, core.ForgetInput(name="route"))
    assert len(storage.memories) == 1
    await core.forget(own, core.ForgetInput(name="route"))
    assert not storage.memories


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["recall", "remember", "forget"])
async def test_point_operations_reject_a_store_returning_the_wrong_namespace(action):
    storage = InMemoryHarnessStorage()
    ctx = context(storage)
    saved = await core.remember(ctx, core.RememberInput(name="route", text="Route"))
    storage.memories[saved.value["id"]].metadata["user_id"] = "different-user"
    values = {
        "recall": core.RecallInput(name="route"),
        "remember": core.RememberInput(name="route", text="Correction"),
        "forget": core.ForgetInput(name="route"),
    }
    with pytest.raises(ValueError, match="current scope"):
        await getattr(core, action)(ctx, values[action])


@pytest.mark.asyncio
async def test_concurrent_named_writes_do_not_append_duplicates():
    storage = InMemoryHarnessStorage()
    await asyncio.gather(
        *[
            core.remember(
                context(storage),
                core.RememberInput(name="route", text=f"Version {index}"),
            )
            for index in range(10)
        ]
    )
    assert len(storage.memories) == 1


@pytest.mark.asyncio
async def test_cold_recall_blocks_search_and_point_reads_but_preserves_upsert_identity():
    ctx = context()
    original = await core.remember(
        ctx, core.RememberInput(name="route", text="Original route")
    )
    ctx.harness.execution_context["memory_recall_enabled"] = False
    assert not (await core.recall(ctx, core.RecallInput(query="route"))).items
    assert not (await core.recall(ctx, core.RecallInput(name="route"))).items
    updated = await core.remember(
        ctx, core.RememberInput(name="route", text="Corrected route")
    )
    assert updated.value["id"] == original.value["id"]
    assert updated.value["operation"] == "updated"


@pytest.mark.parametrize("name", ["", "../escape", "Mixed Case", "x" * 65])
def test_names_are_short_non_path_keys(name):
    with pytest.raises(ValidationError):
        core.RememberInput(name=name, text="Procedure")


def test_memory_inputs_require_names_and_reject_empty_content_and_unbounded_reads():
    for value in [
        {"id": "id", "text": "ID-based writes are not supported"},
        {"name": "route", "text": "   "},
        {"name": "route", "text": "x" * 8001},
        {"text": "Anonymous append is not allowed"},
    ]:
        with pytest.raises(ValidationError):
            core.RememberInput(**value)
    for value in [{}, {"id": "id"}]:
        with pytest.raises(ValidationError):
            core.ForgetInput(**value)
    for limit in [0, 51]:
        with pytest.raises(ValidationError):
            core.RecallInput(limit=limit)
