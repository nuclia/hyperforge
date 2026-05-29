from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

from nucliadb_models.internal.predict import RerankModel
from nucliadb_models.search import (
    SCORE_TYPE,
)
from nucliadb_telemetry.metrics import Observer

from nucliadb_agentic_api.ask.predict import (
    ProxiedPredictAPIError,
    SendToPredictError,
    get_predict,
)

reranker_observer = Observer("reranker", labels={"type": ""})


@dataclass
class RerankableItem:
    id: str
    score: float
    score_type: SCORE_TYPE
    content: str


@dataclass
class RankedItem:
    id: str
    score: float
    score_type: SCORE_TYPE


@dataclass
class RerankingOptions:
    kbid: str

    # Query used to retrieve the results to be reranked. Smart rerankers will use it
    query: str


class Reranker(ABC):
    @abstractproperty
    def window(self) -> int | None:
        """Number of elements the reranker requests. `None` means no specific
        window is enforced."""
        ...

    @property
    def needs_extra_results(self) -> bool:
        return self.window is not None

    async def rerank(
        self, items: list[RerankableItem], options: RerankingOptions
    ) -> list[RankedItem]:
        """Given a query and a set of resources, rerank elements and return the
        list of reranked items sorted by decreasing score. The list will contain
        at most, `window` elements.

        NOTE: Other search engines allow a mix of reranked and not reranked
        results, there's no technical reason we can't do it

        """
        # Enforce reranker window and drop the rest
        items = items[: self.window]
        if len(items) == 0:
            return []
        reranked = await self._rerank(items, options)
        return reranked

    @abstractmethod
    async def _rerank(
        self, items: list[RerankableItem], options: RerankingOptions
    ) -> list[RankedItem]: ...


class NoopReranker(Reranker):
    """No-operation reranker. Given a list of items to rerank, it does nothing
    with them and return the items in the same order. It can be use to not alter
    the previous ordering.

    """

    @property
    def window(self) -> int | None:
        return None

    @reranker_observer.wrap({"type": "noop"})
    async def _rerank(
        self, items: list[RerankableItem], options: RerankingOptions
    ) -> list[RankedItem]:
        return [
            RankedItem(
                id=item.id,
                score=item.score,
                score_type=item.score_type,
            )
            for item in items
        ]


class PredictReranker(Reranker):
    """Rerank using a reranking model.

    It uses Predict API to rerank elements using a model trained for this

    """

    def __init__(self, window: int):
        self._window = window

    @property
    def window(self) -> int:
        return self._window

    @reranker_observer.wrap({"type": "predict"})
    async def _rerank(
        self, items: list[RerankableItem], options: RerankingOptions
    ) -> list[RankedItem]:
        if len(items) == 0:
            return []

        predict = get_predict()

        # Conversion to format expected by predict. At the same time,
        # deduplicates paragraphs found in different indices
        context = {item.id: item.content for item in items}
        request = RerankModel(
            question=options.query,
            user_id="",  # TODO
            context=context,
        )
        try:
            response = await predict.rerank(options.kbid, request)
        except (SendToPredictError, ProxiedPredictAPIError, TimeoutError):
            # predict failed, we can't rerank
            reranked = [
                RankedItem(
                    id=item.id,
                    score=item.score,
                    score_type=item.score_type,
                )
                for item in items
            ]
        else:
            reranked = [
                RankedItem(
                    id=id,
                    score=score,
                    score_type=SCORE_TYPE.RERANKER,
                )
                for id, score in response.context_scores.items()
            ]
        sort_by_score(reranked)
        best = reranked
        return best


def sort_by_score(items: list[RankedItem]):
    """Sort `items` in place by decreasing score"""
    items.sort(key=lambda item: item.score, reverse=True)
