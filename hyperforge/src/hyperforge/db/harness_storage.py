from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import databases
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from hyperforge.database import metadata
from hyperforge.harness_sdk.models import (
    HarnessConversation,
    HarnessEvent,
    HarnessEventType,
    HarnessMemory,
)

harness_conversations = sa.Table(
    "harness_conversations",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False, index=True),
    sa.Column("id", sa.String, primary_key=True, nullable=False),
    sa.Column("title", sa.String, nullable=False),
    sa.Column("category", sa.String, nullable=False),
    sa.Column("tags", JSONB, nullable=False),
    sa.Column("metadata", JSONB, nullable=False),
    sa.Column("created_datetime", sa.DateTime(timezone=True), nullable=False),
    sa.Column("updated_datetime", sa.DateTime(timezone=True), nullable=False),
)

harness_events = sa.Table(
    "harness_events",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False),
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False),
    sa.Column("id", sa.String, primary_key=True, nullable=False),
    sa.Column("seq", sa.BigInteger, sa.Identity(), nullable=False, index=True),
    sa.Column("conversation_id", sa.String, nullable=False, index=True),
    sa.Column("turn_id", sa.String, nullable=True),
    sa.Column("harness_agent_id", sa.String, nullable=True),
    sa.Column("parent_agent_id", sa.String, nullable=True),
    sa.Column("parent_call_id", sa.String, nullable=True),
    sa.Column("category", sa.String, nullable=False),
    sa.Column("tags", JSONB, nullable=False),
    sa.Column("metadata", JSONB, nullable=False),
    sa.Column("type", sa.String, nullable=False),
    sa.Column("created_datetime", sa.DateTime(timezone=True), nullable=False),
    sa.Column("payload", JSONB, nullable=False),
    sa.ForeignKeyConstraint(
        ["account", "agent_id", "conversation_id"],
        [
            "harness_conversations.account",
            "harness_conversations.agent_id",
            "harness_conversations.id",
        ],
        onupdate="CASCADE",
        ondelete="CASCADE",
    ),
)

harness_memories = sa.Table(
    "harness_memories",
    metadata,
    sa.Column("account", sa.String, primary_key=True, nullable=False),
    sa.Column("agent_id", sa.String, primary_key=True, nullable=False),
    sa.Column("id", sa.String, primary_key=True, nullable=False),
    sa.Column("seq", sa.BigInteger, sa.Identity(), nullable=False, index=True),
    sa.Column("scope", sa.String, nullable=False, index=True),
    sa.Column("text", sa.Text, nullable=False),
    sa.Column("metadata", JSONB, nullable=False),
    sa.Column("created_datetime", sa.DateTime(timezone=True), nullable=False),
)


def _decode_json_column(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


class PostgresHarnessStorage:
    """Durable, tenant-scoped `HarnessStorageProtocol` backed by PostgreSQL."""

    def __init__(
        self,
        database: databases.Database,
        *,
        account: str,
        agent_id: str,
    ) -> None:
        if not account.strip():
            raise ValueError("account must not be blank")
        if not agent_id.strip():
            raise ValueError("agent_id must not be blank")
        self.database = database
        self.account = account
        self.agent_id = agent_id

    async def create_conversation(self, conversation: HarnessConversation) -> None:
        existing = await self.get_conversation(conversation.id)
        if existing is not None:
            raise ValueError(f"Conversation already exists: {conversation.id}")
        statement = harness_conversations.insert().values(
            account=self.account,
            agent_id=self.agent_id,
            id=conversation.id,
            title=conversation.title,
            category=conversation.category,
            tags=list(conversation.tags),
            metadata=dict(conversation.metadata),
            created_datetime=conversation.created_datetime,
            updated_datetime=conversation.updated_datetime,
        )
        await self.database.execute(statement)

    async def get_conversation(
        self, conversation_id: str
    ) -> HarnessConversation | None:
        statement = harness_conversations.select().where(
            harness_conversations.c.account == self.account,
            harness_conversations.c.agent_id == self.agent_id,
            harness_conversations.c.id == conversation_id,
        )
        row = await self.database.fetch_one(statement)
        if row is None:
            return None
        return HarnessConversation(
            id=row["id"],
            title=row["title"],
            category=row["category"],
            tags=_decode_json_column(row["tags"], []),
            metadata=_decode_json_column(row["metadata"], {}),
            created_datetime=row["created_datetime"],
            updated_datetime=row["updated_datetime"],
        )

    async def update_conversation(self, conversation: HarnessConversation) -> None:
        existing = await self.get_conversation(conversation.id)
        if existing is None:
            raise ValueError(f"Conversation not found: {conversation.id}")
        statement = (
            harness_conversations.update()
            .where(
                harness_conversations.c.account == self.account,
                harness_conversations.c.agent_id == self.agent_id,
                harness_conversations.c.id == conversation.id,
            )
            .values(
                title=conversation.title,
                category=conversation.category,
                tags=list(conversation.tags),
                metadata=dict(conversation.metadata),
                updated_datetime=conversation.updated_datetime,
            )
        )
        await self.database.execute(statement)

    async def append_event(self, event: HarnessEvent) -> None:
        insert_stmt = harness_events.insert().values(
            account=self.account,
            agent_id=self.agent_id,
            id=event.id,
            conversation_id=event.conversation_id,
            turn_id=event.turn_id,
            harness_agent_id=event.agent_id,
            parent_agent_id=event.parent_agent_id,
            parent_call_id=event.parent_call_id,
            category=event.category,
            tags=list(event.tags),
            metadata=dict(event.metadata),
            type=event.type.value,
            created_datetime=event.created_datetime,
            payload=event.model_dump(mode="json")["payload"],
        )
        await self.database.execute(insert_stmt)
        update_stmt = (
            harness_conversations.update()
            .where(
                harness_conversations.c.account == self.account,
                harness_conversations.c.agent_id == self.agent_id,
                harness_conversations.c.id == event.conversation_id,
                harness_conversations.c.updated_datetime < event.created_datetime,
            )
            .values(updated_datetime=event.created_datetime)
        )
        await self.database.execute(update_stmt)

    async def iter_events(self, conversation_id: str) -> AsyncIterator[HarnessEvent]:
        statement = (
            harness_events.select()
            .where(
                harness_events.c.account == self.account,
                harness_events.c.agent_id == self.agent_id,
                harness_events.c.conversation_id == conversation_id,
            )
            .order_by(harness_events.c.seq.asc())
        )
        rows = await self.database.fetch_all(statement)
        for row in rows:
            yield HarnessEvent(
                id=row["id"],
                conversation_id=row["conversation_id"],
                turn_id=row["turn_id"],
                agent_id=row["harness_agent_id"],
                parent_agent_id=row["parent_agent_id"],
                parent_call_id=row["parent_call_id"],
                category=row["category"],
                tags=_decode_json_column(row["tags"], []),
                metadata=_decode_json_column(row["metadata"], {}),
                type=HarnessEventType(row["type"]),
                created_datetime=row["created_datetime"],
                payload=_decode_json_column(row["payload"], {}),
            )

    async def remember(self, memory: HarnessMemory) -> None:
        statement = harness_memories.insert().values(
            account=self.account,
            agent_id=self.agent_id,
            id=memory.id,
            scope=memory.scope,
            text=memory.text,
            metadata=dict(memory.metadata),
            created_datetime=memory.created_datetime,
        )
        await self.database.execute(statement)

    async def recall(
        self,
        *,
        scope: str,
        query: str,
        limit: int = 20,
    ) -> list[HarnessMemory]:
        effective_limit = max(1, limit)
        needle = query.strip()
        statement = harness_memories.select().where(
            harness_memories.c.account == self.account,
            harness_memories.c.agent_id == self.agent_id,
            harness_memories.c.scope == scope,
        )
        if needle:
            escaped = (
                needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            statement = statement.where(
                harness_memories.c.text.ilike(f"%{escaped}%", escape="\\")
            )
        statement = statement.order_by(harness_memories.c.seq.desc()).limit(
            effective_limit
        )
        rows = await self.database.fetch_all(statement)
        return [
            HarnessMemory(
                id=row["id"],
                scope=row["scope"],
                text=row["text"],
                metadata=_decode_json_column(row["metadata"], {}),
                created_datetime=row["created_datetime"],
            )
            for row in reversed(rows)
        ]

    async def forget(self, memory_id: str) -> None:
        statement = harness_memories.delete().where(
            harness_memories.c.account == self.account,
            harness_memories.c.agent_id == self.agent_id,
            harness_memories.c.id == memory_id,
        )
        await self.database.execute(statement)


__all__ = [
    "PostgresHarnessStorage",
    "harness_conversations",
    "harness_events",
    "harness_memories",
]
