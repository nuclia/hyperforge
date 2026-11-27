from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, Field

from hyperforge.harness_sdk import (
    HarnessContextReference,
    HarnessContextType,
    ToolCallContext,
)
from hyperforge.manager import Manager


class RetrievalOutput(BaseModel):
    """Canonical structured output returned by retrieval and connector tools."""

    contexts: list[dict[str, Any]] = Field(default_factory=list)


def retrieval_context_factory(output: RetrievalOutput) -> HarnessContextReference:
    return HarnessContextReference(
        type=HarnessContextType.RETRIEVAL,
        content={"items": output.contexts},
    )


def resolve_manager(context: ToolCallContext) -> Manager:
    """Build or retrieve a `Manager` from the executing `AgentHarness`."""
    harness = context.harness
    existing = harness.execution_context.get("manager")
    if isinstance(existing, Manager):
        return existing

    manager = Manager()
    nua = getattr(harness.model_client, "nua", None)
    if nua is None:
        nua = harness.execution_context.get("nua_client")
    if nua is not None:
        manager.nua = cast(Any, nua)

    drivers = harness.execution_context.get("drivers")
    if isinstance(drivers, dict):
        manager.drivers.update(drivers)
    return manager


def resolve_workflow_rules(context: ToolCallContext) -> list[str]:
    wf_ctx = context.harness.execution_context.get("workflow_context")
    if isinstance(wf_ctx, dict) and isinstance(wf_ctx.get("rules"), list):
        return [str(r) for r in wf_ctx["rules"] if r]
    return []


def resolve_workflow_history(context: ToolCallContext) -> list[str]:
    wf_ctx = context.harness.execution_context.get("workflow_context")
    if not isinstance(wf_ctx, dict) or not isinstance(wf_ctx.get("history"), list):
        return []
    lines: list[str] = []
    for entry in wf_ctx["history"]:
        if isinstance(entry, dict) and entry.get("content"):
            role = "User" if entry.get("role") == "user" else "Assistant"
            lines.append(f"{role}: {entry['content']}")
    return lines


__all__ = [
    "RetrievalOutput",
    "resolve_manager",
    "resolve_workflow_history",
    "resolve_workflow_rules",
    "retrieval_context_factory",
]
