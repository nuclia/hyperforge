import json
import os
import random
from collections.abc import AsyncGenerator, AsyncIterator
from unittest.mock import Mock, patch

import pytest
from nuclia_models.predict.generative_responses import GenerativeChunk
from nucliadb_models.internal.predict import (
    Ner,
    QueryInfo,
    RerankModel,
    RerankResponse,
    SentenceSearch,
    TokenSearch,
)
from nucliadb_models.search import ChatModel, RephraseModel
from nucliadb_protos.utils_pb2 import RelationNode
from nucliadb_utils.utilities import Utility

from nucliadb_agentic_api.src.nucliadb_agentic_api.ask.predict import (
    NUCLIA_LEARNING_ID_HEADER,
    PredictEngine,
    RephraseResponse,
    convert_relations,
)
from nucliadb_agentic_api.src.nucliadb_agentic_api.ask.predict_models import QueryModel
from nucliadb_agentic_api.tests.fixtures.utils import global_utility

DUMMY_RELATION_NODE = [
    RelationNode(value="Ferran", ntype=RelationNode.NodeType.ENTITY, subtype="PERSON"),
    RelationNode(
        value="Joan Antoni", ntype=RelationNode.NodeType.ENTITY, subtype="PERSON"
    ),
]

DUMMY_REPHRASE_QUERY = "This is a rephrased query"
DUMMY_LEARNING_ID = "00"
DUMMY_LEARNING_MODEL = "chatgpt"


class DummyPredictEngine(PredictEngine):
    default_semantic_threshold = 0.7

    def __init__(self):
        self.onprem = True
        self.cluster_url = "http://localhost:8000"
        self.public_url = "http://localhost:8000"
        self.calls = []
        self.ndjson_reasoning = [
            b'{"chunk": {"type": "reasoning", "text": "dummy "}}\n',
            b'{"chunk": {"type": "reasoning", "text": "reasoning"}}\n',
        ]
        self.ndjson_answer = [
            b'{"chunk": {"type": "text", "text": "valid "}}\n',
            b'{"chunk": {"type": "text", "text": "answer "}}\n',
            b'{"chunk": {"type": "text", "text": "to"}}\n',
            b'{"chunk": {"type": "status", "code": "0"}}\n',
        ]
        self.max_context = 1000

    async def initialize(self):
        pass

    async def finalize(self):
        pass

    def get_predict_headers(self, kbid: str) -> dict[str, str]:
        return {}

    async def make_request(self, method: str, **request_args):
        json_data = {"foo": "bar"}
        response = Mock(status_code=200)
        response.json = Mock(return_value=json_data)
        response.content = json.dumps(json_data).encode("utf-8")
        response.headers = {NUCLIA_LEARNING_ID_HEADER: DUMMY_LEARNING_ID}
        response.is_stream_consumed = True
        return response

    async def rephrase_query(self, kbid: str, item: RephraseModel) -> RephraseResponse:
        self.calls.append(("rephrase_query", item))
        return RephraseResponse(
            rephrased_query=DUMMY_REPHRASE_QUERY, use_chat_history=None
        )

    async def chat_query_ndjson(
        self, kbid: str, item: ChatModel, extra_headers: dict[str, str] | None = None
    ) -> tuple[str, str, AsyncGenerator[GenerativeChunk, None]]:
        self.calls.append(("chat_query_ndjson", item))

        async def generate():
            if item.reasoning is not False:
                for chunk in self.ndjson_reasoning:
                    yield GenerativeChunk.model_validate_json(chunk)
            for chunk in self.ndjson_answer:
                yield GenerativeChunk.model_validate_json(chunk)

        return (DUMMY_LEARNING_ID, DUMMY_LEARNING_MODEL, generate())

    async def query(self, kbid: str, item: QueryModel) -> QueryInfo:
        assert item.text is not None

        self.calls.append(
            (
                "query",
                item,
            )
        )

        response = QueryInfo(
            query=item.text,
            rephrased_query=f"Rephrased: {item.text}"
            if item.rephrase or item.query_image
            else None,
            language="en",
            visual_llm=True,
            max_context=self.max_context,
            sentence=SentenceSearch(),
            entities=TokenSearch(
                tokens=[Ner(text="text", ner="PERSON", start=0, end=2)], time=0.0
            ),
        )
        assert response.sentence is not None  # for mypy
        response.sentence.vectors["en-2024-04-24"] = [
            random.random() for _ in range(768)
        ]
        response.sentence.timings["en-2024-04-24"] = 0.5
        response.sentence.vectors["multilingual-2024-05-06"] = [
            random.random() for _ in range(1024)
        ]
        response.sentence.timings["multilingual-2024-05-06"] = 0.7

        return response

    async def detect_entities(self, kbid: str, sentence: str) -> list[RelationNode]:
        self.calls.append(("detect_entities", sentence))
        dummy_data = os.environ.get("TEST_RELATIONS", None)
        if dummy_data is not None:  # pragma: no cover
            return convert_relations(json.loads(dummy_data))
        else:
            return DUMMY_RELATION_NODE

    async def rerank(self, kbid: str, item: RerankModel) -> RerankResponse:
        self.calls.append(("rerank", (kbid, item)))
        # as we don't have information about the retrieval scores, return a
        # random score given by the dict iteration
        response = RerankResponse(
            context_scores={
                paragraph_id: i for i, paragraph_id in enumerate(item.context.keys())
            }
        )
        return response


@pytest.fixture(scope="function")
async def dummy_predict() -> AsyncIterator[DummyPredictEngine]:
    """Mock start_ and stop_ predict engine functions so we don't overwrite the
    utility. Then, set our own dummy predict utility, accessible from tests

    """
    with (
        patch("nuclia_arag_api.app.start_predict_engine"),
        patch("nuclia_arag_api.app.stop_predict_engine"),
    ):
        predict_util = DummyPredictEngine()
        await predict_util.initialize()

        with global_utility(Utility.PREDICT, predict_util):
            yield predict_util

        await predict_util.finalize()
