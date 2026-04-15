"""Optional cross-encoder reranker for improved retrieval precision."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docvault.config import Config
    from docvault.retrieval.search import SearchResult

logger = logging.getLogger(__name__)


class Reranker:
    """Re-scores retrieved chunks using a cross-encoder model.

    A cross-encoder jointly processes the query and each candidate chunk,
    producing a more accurate relevance score than the bi-encoder used for
    initial retrieval — at the cost of O(k) inference calls.

    The reranker is disabled by default (``config.rerank = False``).  Enable
    it by setting ``RERANK=true`` in your ``.env`` or passing ``rerank=True``
    to :class:`~docvault.pipeline.Pipeline`.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2`` (fast, ~22 MB).

    Args:
        config: :class:`~docvault.config.Config` instance supplying
            ``rerank_model``, ``rerank_top_n``, and device preference.

    Attributes:
        model_name: Name of the loaded cross-encoder model.
    """

    def __init__(self, config: "Config") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading cross-encoder '%s' on device '%s'.",
            config.rerank_model,
            device,
        )

        self._model = CrossEncoder(config.rerank_model, device=device)
        self._top_n = config.rerank_top_n
        self.model_name: str = config.rerank_model

        logger.info("Reranker ready — model: '%s'.", self.model_name)

    def rerank(
        self,
        query: str,
        results: "list[SearchResult]",
        top_n: int | None = None,
    ) -> "list[SearchResult]":
        """Re-score and optionally trim a list of retrieval results.

        Pairs *query* with every candidate chunk text and runs a single
        batched forward pass through the cross-encoder.  The bi-encoder
        scores on :class:`~docvault.retrieval.search.SearchResult` are
        **replaced** in-place by the cross-encoder scores so that downstream
        code (citation display, confidence indicators) always sees the
        best available score.

        Args:
            query: The original user query.
            results: Candidate results from
                :class:`~docvault.retrieval.search.Searcher`.
            top_n: Number of results to keep after reranking.  Defaults to
                ``config.rerank_top_n``.  Pass ``None`` to return all results
                reranked but untruncated.

        Returns:
            Results sorted by cross-encoder score descending, trimmed to
            *top_n* if specified.

        Raises:
            ValueError: If *results* is empty.
        """
        if not results:
            raise ValueError("rerank() received an empty results list.")

        n = top_n if top_n is not None else self._top_n

        # Build (query, passage) pairs for the cross-encoder
        pairs = [[query, r.chunk.text] for r in results]

        logger.debug(
            "Reranking %d candidate(s) with cross-encoder '%s'.",
            len(pairs),
            self.model_name,
        )

        cross_scores: list[float] = self._model.predict(pairs).tolist()

        # Replace bi-encoder scores with cross-encoder scores
        for result, score in zip(results, cross_scores):
            result.score = float(score)

        # Sort descending
        results.sort(key=lambda r: r.score, reverse=True)

        if n is not None and n > 0:
            results = results[:n]

        logger.info(
            "Reranking complete: returning %d result(s); "
            "top cross-encoder score=%.4f.",
            len(results),
            results[0].score if results else 0.0,
        )
        return results
