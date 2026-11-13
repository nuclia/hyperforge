from __future__ import annotations

from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Protocol

from .models import HarnessConversation, HarnessEvent, HarnessMemory


class HarnessStorageProtocol(Protocol):
    async def create_conversation(self, conversation: HarnessConversation) -> None: ...

    async def get_conversation(
        self, conversation_id: str
    ) -> HarnessConversation | None: ...

    async def update_conversation(self, conversation: HarnessConversation) -> None: ...

    async def append_event(self, event: HarnessEvent) -> None: ...

    def iter_events(self, conversation_id: str) -> AsyncIterator[HarnessEvent]: ...

    async def remember(self, memory: HarnessMemory) -> None: ...

    async def recall(
        self,
        *,
        scope: str,
        query: str,
        limit: int = 20,
    ) -> list[HarnessMemory]: ...

    async def forget(self, memory_id: str) -> None: ...


class InMemoryHarnessStorage:
    """Process-local storage intended for tests and ephemeral harnesses."""

    def __init__(self) -> None:
        self.conversations: dict[str, HarnessConversation] = {}
        self.events: dict[str, list[HarnessEvent]] = defaultdict(list)
        self.memories: dict[str, HarnessMemory] = {}

    async def create_conversation(self, conversation: HarnessConversation) -> None:
        if conversation.id in self.conversations:
            raise ValueError(f"Conversation already exists: {conversation.id}")
        self.conversations[conversation.id] = conversation.model_copy(deep=True)

    async def get_conversation(
        self, conversation_id: str
    ) -> HarnessConversation | None:
        conversation = self.conversations.get(conversation_id)
        return conversation.model_copy(deep=True) if conversation is not None else None

    async def update_conversation(self, conversation: HarnessConversation) -> None:
        if conversation.id not in self.conversations:
            raise ValueError(f"Conversation not found: {conversation.id}")
        self.conversations[conversation.id] = conversation.model_copy(deep=True)

    async def append_event(self, event: HarnessEvent) -> None:
        self.events[event.conversation_id].append(event.model_copy(deep=True))
        conversation = self.conversations.get(event.conversation_id)
        if (
            conversation is not None
            and event.created_datetime > conversation.updated_datetime
        ):
            self.conversations[event.conversation_id] = conversation.model_copy(
                update={"updated_datetime": event.created_datetime}
            )

    async def iter_events(self, conversation_id: str) -> AsyncIterator[HarnessEvent]:
        for event in self.events.get(conversation_id, ()):
            yield event.model_copy(deep=True)

    async def remember(self, memory: HarnessMemory) -> None:
        self.memories[memory.id] = memory.model_copy(deep=True)

    async def recall(
        self,
        *,
        scope: str,
        query: str,
        limit: int = 20,
    ) -> list[HarnessMemory]:
        needle = query.casefold().strip()
        values = [
            memory
            for memory in self.memories.values()
            if memory.scope == scope
            and (not needle or needle in memory.text.casefold())
        ]
        return [memory.model_copy(deep=True) for memory in values[-limit:]]

    async def forget(self, memory_id: str) -> None:
        self.memories.pop(memory_id, None)
