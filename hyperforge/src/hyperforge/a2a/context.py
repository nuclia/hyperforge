from typing import Any

from hyperforge.a2a.settings import A2ASettings
from hyperforge.a2a.task_store import A2ATaskStore
from hyperforge.broker import Broker


class A2AServerContext:
    """Lightweight application context passed to the interaction pipeline.

    ``hyperforge.api.v1.interaction.stream_response`` is typed against the
    ``HTTPApplication`` but only accesses ``settings``, ``agent_manager`` and
    ``broker`` at runtime. This holder duck-types those three attributes so the
    A2A gRPC server can reuse the exact same broker-driven interaction flow as
    the HTTP/WS API without spinning up a FastAPI app.
    """

    def __init__(
        self,
        settings: A2ASettings,
        agent_manager: Any,
        broker: Broker,
        task_store: A2ATaskStore,
    ):
        self.settings = settings
        self.agent_manager = agent_manager
        self.broker = broker
        self.task_store = task_store
