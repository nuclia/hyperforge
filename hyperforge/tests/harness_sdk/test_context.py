import json

from pydantic import BaseModel

from hyperforge.harness_sdk import context
from hyperforge.harness_sdk.context import (
    format_context,
    make_context,
    register_context,
)
from hyperforge.harness_sdk.models import HarnessContextReference, HarnessContextType


class RetrievalOutput(BaseModel):
    query: str
    classification_context: str
    items: list[dict]


def test_retrieval_context_preserves_top_level_metadata():
    reference = HarnessContextReference(
        type=HarnessContextType.RETRIEVAL,
        content={
            "query": "Kestrel Unit",
            "classification_context": "Kestrel Unit and VFD are the same concept.",
            "items": [{"text": "ELEC-02", "images": {"unused": "image"}}],
        },
    )

    assert json.loads(format_context(reference)) == {
        "query": "Kestrel Unit",
        "classification_context": "Kestrel Unit and VFD are the same concept.",
        "items": [{"text": "ELEC-02"}],
    }


def test_registered_formatter_overrides_default_retrieval_formatter(monkeypatch):
    monkeypatch.setattr(context, "_DEFINITIONS", dict(context._DEFINITIONS))

    @register_context(HarnessContextType.RETRIEVAL, RetrievalOutput)
    def format_retrieval_output(output: RetrievalOutput) -> str:
        return f"{output.classification_context}\n{output.items[0]['text']}"

    output = RetrievalOutput(
        query="Kestrel Unit",
        classification_context="Kestrel Unit and VFD are the same concept.",
        items=[{"text": "ELEC-02"}],
    )

    reference = make_context(HarnessContextType.RETRIEVAL, output)

    assert (
        format_context(reference)
        == "Kestrel Unit and VFD are the same concept.\nELEC-02"
    )
