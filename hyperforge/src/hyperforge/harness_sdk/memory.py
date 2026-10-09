"""Stable memory keys and small lexical retrieval helpers.

Storage adapters must still enforce their own tenant authorization. These keys
separate names by scope and stable application identity, never conversation ID.
"""

import json
import re
import uuid
from typing import Any

_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "using",
        "with",
    }
)


def memory_namespace(scope: str, metadata: dict[str, Any]) -> str:
    identity = {
        "namespace": metadata.get("memory_namespace"),
        "account_id": metadata.get("account_id"),
    }
    if scope != "project":
        identity["user_id"] = metadata.get("user_id")
    if scope != "user":
        identity["project_id"] = metadata.get("project_id")
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def named_memory_id(name: str, scope: str, metadata: dict[str, Any]) -> str:
    key = json.dumps(["harness-memory", memory_namespace(scope, metadata), scope, name])
    return uuid.uuid5(uuid.NAMESPACE_URL, key).hex


def memory_search_terms(query: str) -> set[str]:
    return {
        term
        for term in re.findall(r"\w+", query.casefold().replace("_", " "))
        if term not in _STOP_WORDS
    }
