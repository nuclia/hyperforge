import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to with".split()
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(_CAMEL_RE.sub(" ", text)):
        token = raw.lower()
        if token in _STOPWORDS:
            continue
        tokens.append(token)
        # Crude stemming so "files"/"file", "searching"/"search" match.
        for suffix in ("ing", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                tokens.append(token[: -len(suffix)])
                break
    return tokens


@dataclass
class IndexedTool:
    name: str  # fully qualified name exposed by the proxy
    server: str
    tool: str  # name on the upstream server
    description: str
    input_schema: dict[str, Any]
    server_description: str = ""
    tags: list[str] = field(default_factory=list)

    def document(self) -> dict[str, list[str]]:
        params = self.input_schema.get("properties", {}) or {}
        param_text = " ".join(
            f"{k} {v.get('description', '')}"
            for k, v in params.items()
            if isinstance(v, dict)
        )
        return {
            "name": tokenize(f"{self.tool} {self.tool.replace('_', ' ')}"),
            "server": tokenize(f"{self.server} {' '.join(self.tags)}"),
            "description": tokenize(self.description),
            "params": tokenize(param_text),
            "server_description": tokenize(self.server_description),
        }


FIELD_WEIGHTS = {
    "name": 3.0,
    "server": 2.0,
    "description": 1.5,
    "params": 0.7,
    "server_description": 0.5,
}


class ToolIndex:
    """In-memory BM25F-style full text index over tool metadata."""

    k1 = 1.2
    b = 0.75

    def __init__(self, tools: list[IndexedTool] | None = None):
        self._tools: list[IndexedTool] = []
        self._tfs: list[Counter[str]] = []
        self._lengths: list[float] = []
        self._df: Counter[str] = Counter()
        self._avg_len = 0.0
        if tools:
            self.build(tools)

    def build(self, tools: list[IndexedTool]) -> None:
        self._tools = list(tools)
        self._tfs = []
        self._lengths = []
        self._df = Counter()
        for tool in self._tools:
            tf: Counter[str] = Counter()
            for fname, tokens in tool.document().items():
                weight = FIELD_WEIGHTS[fname]
                for tok in tokens:
                    tf[tok] += weight
            self._tfs.append(tf)
            self._lengths.append(sum(tf.values()))
            self._df.update(tf.keys())
        self._avg_len = (sum(self._lengths) / len(self._lengths)) if self._tools else 0

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def tools(self) -> list[IndexedTool]:
        return list(self._tools)

    def search(self, query: str, limit: int = 10) -> list[tuple[IndexedTool, float]]:
        terms = set(tokenize(query))
        n = len(self._tools)
        if not terms:
            # Empty query: list tools so the agent can browse.
            return [(t, 0.0) for t in self._tools[:limit]]
        scored: list[tuple[IndexedTool, float]] = []
        for tool, tf, length in zip(self._tools, self._tfs, self._lengths):
            score = 0.0
            for term in terms:
                freq = tf.get(term)
                if not freq:
                    continue
                idf = math.log(1 + (n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                norm = 1 - self.b + self.b * length / (self._avg_len or 1)
                score += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
            if score > 0:
                scored.append((tool, score))
        scored.sort(key=lambda x: (-x[1], x[0].name))
        return scored[:limit]
