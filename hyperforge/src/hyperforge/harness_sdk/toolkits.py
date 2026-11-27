from __future__ import annotations

import abc
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Self, TypeVar

from pydantic import BaseModel, Field, field_validator

from .skills import HarnessSkill, namespace_skill
from .tools import HarnessTool

if TYPE_CHECKING:
    from hyperforge.driver import Driver

    from .clients import ModelClient


def _validate_namespace(value: str | None) -> str | None:
    if value is None or value == "":
        return None
    if not value.isascii() or not value.isidentifier() or value.startswith("_"):
        raise ValueError(
            f"Toolkit namespace {value!r} must be an ASCII public Python identifier"
        )
    if "__" in value:
        raise ValueError(f"Toolkit namespace {value!r} must not contain '__'")
    return value


class ToolkitConfig(BaseModel):
    """Base configuration shared by all connector and pipeline toolkits."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    toolkit: str = Field(
        ...,
        title="Toolkit provider",
        description="Registered toolkit provider identifier",
    )
    namespace: str | None = Field(
        default=None,
        title="Tool namespace",
        description="Optional ASCII Python identifier prefix applied to tools and skills",
    )
    title: str | None = Field(
        default=None,
        title="Toolkit title",
        description="Human-readable label for this toolkit instance",
    )
    lazy_load: bool = Field(
        default=False,
        title="Lazy-load tools",
        description="Defer loading tool schemas until discovered via search_tools/load_skill",
    )
    ingest_skills: bool = Field(
        default=True,
        title="Ingest toolkit skills",
        description="Automatically ingest skills bundled with this toolkit when attached",
    )

    @field_validator("namespace")
    @classmethod
    def check_namespace(cls, value: str | None) -> str | None:
        return _validate_namespace(value)


T_ToolkitConfig = TypeVar("T_ToolkitConfig", bound=ToolkitConfig)


@dataclass(frozen=True)
class ToolkitRuntime:
    """Shared runtime dependencies supplied to toolkits during initialization/preload."""

    drivers: Mapping[str, Driver] = field(default_factory=dict)
    model_client: ModelClient | None = None
    nua_client: Any = None
    execution_context: Mapping[str, Any] = field(default_factory=dict)


class BaseToolkit(abc.ABC, Generic[T_ToolkitConfig]):
    """Base class for per-connector and core pipeline toolkits.

    Each toolkit encapsulates a set of `HarnessTool`s and optional `HarnessSkill`s
    without owning an opinionated agent loop.
    """

    toolkit_id: ClassVar[str] = ""
    config: T_ToolkitConfig
    namespace: str | None

    def __init__(
        self, config: T_ToolkitConfig, *, namespace: str | None = None
    ) -> None:
        resolved_namespace = (
            _validate_namespace(namespace)
            if namespace is not None
            else config.namespace
        )
        self.config = config
        self.namespace = resolved_namespace

    @classmethod
    async def from_config(
        cls,
        config: T_ToolkitConfig,
        *,
        namespace: str | None = None,
        runtime: ToolkitRuntime | None = None,
    ) -> Self:
        instance = cls(config, namespace=namespace)
        if runtime is not None:
            await instance.preload(runtime)
        return instance

    async def preload(self, runtime: ToolkitRuntime) -> None:
        """Optional async discovery hook (e.g., remote MCP tools/prompts)."""
        del runtime

    @abc.abstractmethod
    def build_tools(self) -> Sequence[HarnessTool[Any, Any]]:
        """Return the raw `HarnessTool` definitions for this toolkit."""

    def build_skills(self) -> Sequence[HarnessSkill]:
        """Return optional `HarnessSkill` playbooks bundled with this toolkit."""
        return ()

    def get_tools(self) -> list[HarnessTool[Any, Any]]:
        """Return tools with `namespace` prefixing and `lazy_load` applied."""
        prefix = f"{self.namespace}__" if self.namespace else ""
        resolved: list[HarnessTool[Any, Any]] = []
        for item in self.build_tools():
            target_name = (
                item.name
                if not prefix or item.name.startswith(prefix)
                else f"{prefix}{item.name}"
            )
            target_lazy = item.lazy_load or self.config.lazy_load
            if target_name != item.name or target_lazy != item.lazy_load:
                item = replace(item, name=target_name, lazy_load=target_lazy)
            resolved.append(item)
        return resolved

    def get_skills(self) -> list[HarnessSkill]:
        """Return bundled skills scoped to `namespace` when `ingest_skills` is True."""
        if not self.config.ingest_skills:
            return []
        return [namespace_skill(skill, self.namespace) for skill in self.build_skills()]


async def assemble_toolkits(
    toolkits: Sequence[BaseToolkit[Any]],
    runtime: ToolkitRuntime | None = None,
) -> tuple[list[HarnessTool[Any, Any]], list[HarnessSkill]]:
    """Preload toolkits and collect their namespaced tools and ingested skills."""
    effective_runtime = runtime or ToolkitRuntime()
    tools: list[HarnessTool[Any, Any]] = []
    skills: list[HarnessSkill] = []
    seen_tools: set[str] = set()
    seen_skills: set[str] = set()

    for toolkit in toolkits:
        await toolkit.preload(effective_runtime)
        for item in toolkit.get_tools():
            if item.name in seen_tools:
                raise ValueError(f"Duplicate toolkit tool name: {item.name}")
            seen_tools.add(item.name)
            tools.append(item)
        for skill in toolkit.get_skills():
            if skill.id in seen_skills:
                raise ValueError(f"Duplicate toolkit skill id: {skill.id}")
            seen_skills.add(skill.id)
            skills.append(skill)

    return tools, skills


__all__ = [
    "BaseToolkit",
    "ToolkitConfig",
    "ToolkitRuntime",
    "assemble_toolkits",
]
