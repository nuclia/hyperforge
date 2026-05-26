import contextlib
import time
from typing import Any

from nucliadb_telemetry import metrics

buckets = [
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    30.0,
    60.0,
    metrics.INF,
]

generative_first_chunk_histogram = metrics.Histogram(
    name="generative_reasoning_first_chunk",
    buckets=buckets,
)
reasoning_first_chunk_histogram = metrics.Histogram(
    name="generative_first_chunk",
    buckets=buckets,
)
rag_histogram = metrics.Histogram(
    name="rag",
    labels={"step": ""},
    buckets=buckets,
)

MetricsData = dict[str, int | float]


class Metrics:
    def __init__(self: "Metrics", id: str):
        self.id = id
        self.child_spans: list[Metrics] = []
        self._metrics: MetricsData = {}

    @contextlib.contextmanager
    def time(self, step: str):
        start_time = time.monotonic()
        try:
            yield
        finally:
            elapsed = time.monotonic() - start_time
            self._metrics[step] = elapsed
            rag_histogram.observe(elapsed, labels={"step": step})

    def child_span(self, id: str) -> "Metrics":
        child_span = Metrics(id)
        self.child_spans.append(child_span)
        return child_span

    def set(self, key: str, value: int | float):
        self._metrics[key] = value

    def get(self, key: str) -> int | float | None:
        return self._metrics.get(key)

    def to_dict(self) -> MetricsData:
        return self._metrics

    def dump(self) -> dict[str, Any]:
        result = {}
        for child in self.child_spans:
            result.update(child.dump())
        result[self.id] = self.to_dict()
        return result

    def __getitem__(self, key: str) -> int | float:
        return self._metrics[key]


class AskMetrics(Metrics):
    def __init__(self: "AskMetrics"):
        super().__init__(id="ask")
        self.global_start = time.monotonic()
        self.first_chunk_yielded_at: float | None = None
        self.first_reasoning_chunk_yielded_at: float | None = None

    def record_first_chunk_yielded(self):
        self.first_chunk_yielded_at = time.monotonic()
        generative_first_chunk_histogram.observe(
            self.first_chunk_yielded_at - self.global_start
        )

    def record_first_reasoning_chunk_yielded(self):
        self.first_reasoning_chunk_yielded_at = time.monotonic()
        reasoning_first_chunk_histogram.observe(
            self.first_reasoning_chunk_yielded_at - self.global_start
        )

    def get_first_chunk_time(self) -> float | None:
        if self.first_chunk_yielded_at is None:
            return None
        return self.first_chunk_yielded_at - self.global_start

    def get_first_reasoning_chunk_time(self) -> float | None:
        if self.first_reasoning_chunk_yielded_at is None:
            return None
        return self.first_reasoning_chunk_yielded_at - self.global_start
