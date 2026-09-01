from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CODEMODE_RUNTIME_SECONDS = 30.0
DEFAULT_CODEMODE_MEMORY_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class UsageLimits:
    max_tool_calls: int | None = None
    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_time: float | None = None
    max_spawn_depth: int = 1
    max_concurrent_agents: int = 4
    max_codemode_runtime_seconds: float = DEFAULT_CODEMODE_RUNTIME_SECONDS
    max_codemode_memory_bytes: int = DEFAULT_CODEMODE_MEMORY_BYTES

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if value is not None and value <= 0 and name != "max_spawn_depth":
                raise ValueError(f"{name} must be greater than zero")
        if self.max_spawn_depth < 0:
            raise ValueError("max_spawn_depth must be non-negative")


@dataclass
class HarnessUsage:
    tool_calls: int = 0
    turns: int = 0
    input_tokens: float = 0
    output_tokens: float = 0


class UsageLimitExceeded(RuntimeError):
    pass
