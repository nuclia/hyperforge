from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import TypeVar

from nucliadb_models.retrieval import Score, WeightedCombSumScore
from nucliadb_models.search import SCORE_TYPE
from nucliadb_telemetry.metrics import Observer

from nucliadb_agentic_api.ask.utils.ids import ParagraphId
from nucliadb_agentic_api.ask.utils.text_blocks import (
    ScoredTextBlock,
)

rank_fusion_observer = Observer(
    "rank_fusion",
    labels={"type": ""},
    buckets=[
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
    ],
)

ScoredItem = TypeVar("ScoredItem", bound=ScoredTextBlock)


class IndexSource(str, Enum):
    KEYWORD = auto()
    SEMANTIC = auto()
    GRAPH = auto()


class RankFusionAlgorithm(ABC):
    def __init__(self, window: int):
        self._window = window

    @property
    def window(self) -> int:
        """Phony number used to compute the number of elements to retrieve and
        feed the rank fusion algorithm.

        This is here for convinience, but a query plan should be the way to go.

        """
        return self._window

    def fuse(self, sources: dict[str, list[ScoredItem]]) -> list[ScoredItem]:
        """Fuse elements from multiple sources and return a list of merged
        results.

        If only one source is provided, rank fusion will be skipped.

        """
        sources_with_results = [x for x in sources.values() if len(x) > 0]
        if len(sources_with_results) == 1:
            # skip rank fusion, we only have a source
            merged = sources_with_results[0]
        else:
            merged = self._fuse(sources)

        # sort and return the unordered results from the implementation
        merged.sort(key=lambda r: r.score, reverse=True)
        return merged

    @abstractmethod
    def _fuse(self, sources: dict[str, list[ScoredItem]]) -> list[ScoredItem]:
        """Rank fusion implementation.

        Each concrete subclass must provide an implementation that merges
        `sources`, a group of unordered matches, into a list of unordered
        results with the new rank fusion score.

        Results can be deduplicated or changed by the rank fusion algorithm.

        """
        ...


class WeightedCombSum(RankFusionAlgorithm):
    """Score-based rank fusion algorithm. Multiply each score by a list-specific
    weight (boost). Then adds the retrieval score of documents contained in more
    than one list and sort by score.

    wCombSUM = Σ(r ∈ R) (w(r) · S(r, d))

    where:
    - d is a document
    - R is the set of retrievers
    - w(r) weight (boost) for retriever r
    - S(r, d) is the score of document d given by retriever r

    wCombSUM boosts matches from multiple retrievers and deduplicate them. As a
    score ranking algorithm, comparison of different scores may lead to bad
    results.

    """

    def __init__(
        self,
        *,
        window: int,
        weights: dict[str, float] | None = None,
        default_weight: float = 1.0,
    ):
        super().__init__(window)
        self._weights = weights or {}
        self._default_weight = default_weight

    @rank_fusion_observer.wrap({"type": "weighted_comb_sum"})
    def _fuse(self, sources: dict[str, list[ScoredItem]]) -> list[ScoredItem]:
        # accumulated scores per paragraph
        scores: dict[ParagraphId, tuple[float, SCORE_TYPE, list[Score]]] = {}
        # pointers from paragraph to the original source
        match_positions: dict[ParagraphId, list[tuple[int, int]]] = {}

        rankings = [
            (values, self._weights.get(source, self._default_weight))
            for source, values in sources.items()
        ]
        for i, (ranking, weight) in enumerate(rankings):
            for j, item in enumerate(ranking):
                id = item.paragraph_id
                score, score_type, history = scores.setdefault(
                    id, (0, item.score_type, [])
                )
                score += item.score * weight
                history.append(item.current_score)
                if {score_type, item.score_type} == {
                    SCORE_TYPE.BM25,
                    SCORE_TYPE.VECTOR,
                }:
                    score_type = SCORE_TYPE.BOTH
                scores[id] = (score, score_type, history)

                position = (i, j)
                match_positions.setdefault(item.paragraph_id, []).append(position)

        merged = []
        for paragraph_id, positions in match_positions.items():
            # we are getting only one position, effectively deduplicating
            # multiple matches for the same text block
            i, j = match_positions[paragraph_id][0]
            score, score_type, history = scores[paragraph_id]
            item = rankings[i][0][j]
            history.append(WeightedCombSumScore(score=score))
            item.scores = history
            item.score_type = score_type
            merged.append(item)

        return merged
