import functools
import os

import nucliadb_telemetry
import nucliadb_telemetry.metrics
from nucliadb_telemetry.utils import get_telemetry
from opentelemetry import trace

SERVICE_NAME = os.environ.get("RAO_SERVICE_NAME", "hyperforge")


def tracer(service_name: str = SERVICE_NAME):
    provider = get_telemetry(service_name)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()


agent_observer = nucliadb_telemetry.metrics.Observer(
    "hyperforge_agent", labels={"agent": ""}
)


# Decorator to trace agent executions
def trace_agent(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        self = args[0]
        agent_name = f"{self.__class__.__name__}.{func.__name__}"
        try:
            attributes = {"agent_id": self.config.id}
        except:
            attributes = {}

        with (
            tracer().start_as_current_span(agent_name, attributes=attributes),
            agent_observer({"agent": agent_name}),
        ):
            return await func(*args, **kwargs)

    return wrapper
