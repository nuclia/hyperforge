from abc import ABC, abstractmethod
from typing import AsyncIterator

from hyperforge.pubsub import AgentMessage, StartInteraction


class AgentTimeoutError(Exception):
    pass


class Broker(ABC):
    # Activation
    @abstractmethod
    async def publish_activation(
        self, msg: StartInteraction, trace: dict[str, str]
    ) -> None: ...

    @abstractmethod
    def subscribe_activations(
        self,
    ) -> AsyncIterator[tuple[StartInteraction, dict[str, str]]]:
        """Yields (StartInteraction, trace_headers) pairs. Called only by the server."""
        ...

    # Answer stream
    @abstractmethod
    async def publish(self, topic: str, message: AgentMessage) -> None: ...

    @abstractmethod
    def subscribe(
        self, topic: str, from_cursor: str = "0"
    ) -> AsyncIterator[tuple[str, AgentMessage]]:
        """Yields (cursor, message) pairs. Raises AgentTimeoutError on keepalive timeout."""
        ...

    # Reply channel — used for feedback and OAuth callbacks
    @abstractmethod
    async def send_reply(self, key: str, payload: str) -> None: ...

    @abstractmethod
    async def receive_reply(self, key: str, timeout_ms: int) -> str | None: ...

    # Lifecycle
    @property
    @abstractmethod
    def keepalive_seconds(self) -> float: ...

    @abstractmethod
    async def initialize(self) -> None: ...

    @abstractmethod
    async def finalize(self) -> None: ...
