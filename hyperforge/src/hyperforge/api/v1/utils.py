from nucliadb_telemetry.utils import get_telemetry
from opentelemetry import trace

from hyperforge.api import SERVICE_NAME


def tracer():
    provider = get_telemetry(SERVICE_NAME)
    if provider:
        return provider.get_tracer(__name__)
    else:
        return trace.NoOpTracer()
