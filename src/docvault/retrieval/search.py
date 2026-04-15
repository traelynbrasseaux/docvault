"""Query embedding and FAISS top-k retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from docvault.ingest.metadata import Chunk

if TYPE_CHECKING:
    from docvault.config import Config
    from docvault.embeddings.encoder import Encoder
    from docvault.retrieval.index import VectorIndex

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single retrieval result returned by :class:`Searcher`.

    Attributes:
        chunk: The retrieved :class:`~docvault.ingest.metadata.Chunk`.
        score: Cosine similarity score in the range ``[-1, 1]``, higher is
            more relevant.
    """

    chunk: Chunk
    score: float

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"SearchResult(score={self.score:.4f}, "
            f"doc='{self.chunk.metadata.doc_name}', "
            f"page={self.chunk.metadata.page_number})"
        )


class Searcher:
    """Retrieves the most relevant chunks for a natural-language query.

    Embeds the query with the same encoder used during ingest, then performs
    an exact inner-product search on the FAISS index.  Because embeddings are
    unit-normalized, the inner product equals cosine similarity.

    Args:
        index: A loaded :class:`~docvault.retrieval.index.VectorIndex`.
        encoder: The :class:`~docvault.embeddings.encoder.Encoder` used to
            embed the query at search time.
        config: :class:`~docvault.config.Config` supplying the default *top_k*.
    """

    def __init__(
        self,
        index: "VectorIndex",
        encoder: "Encoder",
        config: "Config",
    ) -> None:
        self._index = index
        self._encoder = encoder
        self._config = config

    def search(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve the top-k most relevant chunks for *query*.

        Args:
            query: Natural-language question or search string.
            top_k: Number of results to return.  Defaults to
                ``config.top_k`` if ``None``.

        Returns:
            List of :class:`SearchResult` objects sorted by score descending
            (most relevant first).

        Raises:
            RuntimeError: If the index is empty or not loaded.
            ValueError: If *top_k* is less than 1.
        """
        if not self._index.is_loaded or self._index.ntotal == 0:
            raise RuntimeError(
                "The vector index is empty. Ingest documents before querying."
            )

        k = top_k if top_k is not None else self._config.top_k
        if k < 1:
            raise ValueError(f"top_k must be >= 1, got {k}.")

        # Cap k at the index size to avoid FAISS errors
        k = min(k, self._index.ntotal)

        logger.debug("Embedding query: %r", query[:120])
        query_vec: np.ndarray = self._encoder.encode_query(query)  # (1, D)

        logger.debug("Searching index for top-%d results.", k)
        scores_arr, indices_arr = self._index._index.search(query_vec, k)  # type: ignore[union-attr]

        scores: list[float] = scores_arr[0].tolist()
        indices: list[int] = indices_arr[0].tolist()

        results: list[SearchResult] = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                # FAISS returns -1 when fewer than k results exist
                continue
            results.append(
                SearchResult(chunk=self._index.chunks[idx], score=float(score))
            )

        # Sort descending by score (FAISS already returns them sorted, but
        # be defensive in case the index type is swapped in the future)
        results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            "Query returned %d result(s); top score=%.4f.",
            len(results),
            results[0].score if results else 0.0,
        )
        return results
