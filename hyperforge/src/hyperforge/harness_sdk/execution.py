from contextvars import ContextVar, Token

_current_tool_call_id: ContextVar[str | None] = ContextVar(
    "hyperforge_current_tool_call_id", default=None
)


def current_tool_call_id() -> str | None:
    return _current_tool_call_id.get()


def set_current_tool_call_id(call_id: str) -> Token[str | None]:
    return _current_tool_call_id.set(call_id)


def reset_current_tool_call_id(token: Token[str | None]) -> None:
    _current_tool_call_id.reset(token)


__all__ = [
    "current_tool_call_id",
    "reset_current_tool_call_id",
    "set_current_tool_call_id",
]
