"""End-to-end RAG pipeline orchestrator."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from docvault.config import Config
from docvault.embeddings.cache import EmbeddingCache
from docvault.embeddings.encoder import Encoder
from docvault.generation.client import LLMClient
from docvault.generation.prompt import build_prompt
from docvault.generation.response import Response, parse_response
from docvault.ingest.chunking import ChunkingStrategy, chunk_pages
from docvault.ingest.extract import extract_pdfs
from docvault.ingest.metadata import Chunk
from docvault.retrieval.index import CACHE_FILENAME, VectorIndex
from docvault.retrieval.reranker import Reranker
from docvault.retrieval.search import SearchResult, Searcher

logger = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates the full docvault RAG pipeline.

    Wraps every sub-system behind a clean two-method interface:

    - :meth:`ingest` — extract → chunk → embed → index PDFs.
    - :meth:`query`  — embed → retrieve → (rerank) → generate → parse.

    All heavy objects (encoder, index, LLM client) are instantiated lazily
    and cached on the instance so repeated calls within the same session
    pay the load cost only once.

    Args:
        config: Optional :class:`~docvault.config.Config`.  A default
            instance (reading from environment / ``.env``) is created if
            ``None`` is passed.

    Example::

        pipeline = Pipeline()
        pipeline.ingest([Path("data/sample_docs/report.pdf")])
        response = pipeline.query("What are the key findings?")
        print(response.answer)
        print(response.format_citations())
    """

    def __init__(self, config: Config | None = None) -> None:
        self._config: Config = config or Config()
        self._encoder: Encoder | None = None
        self._index: VectorIndex | None = None
        self._cache: EmbeddingCache | None = None
        self._searcher: Searcher | None = None
        self._reranker: Reranker | None = None
        self._client: LLMClient | None = None

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    def _get_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self._config)
        return self._encoder

    def _get_index(self, load: bool = True) -> VectorIndex:
        if self._index is None:
            self._index = VectorIndex(self._config)
            if load:
                self._index.load()
        return self._index

    def _get_cache(self) -> EmbeddingCache:
        if self._cache is None:
            cache_path = self._config.index_dir / CACHE_FILENAME
            self._cache = EmbeddingCache(cache_path)
        return self._cache

    def _get_searcher(self) -> Searcher:
        if self._searcher is None:
            self._searcher = Searcher(
                index=self._get_index(load=True),
                encoder=self._get_encoder(),
                config=self._config,
            )
        return self._searcher

    def _get_reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker(self._config)
        return self._reranker

    def _get_client(self) -> LLMClient:
        if self._client is None:
            self._client = LLMClient(self._config)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        pdf_paths: list[Path],
        strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
        show_progress: bool = False,
    ) -> dict:
        """Extract, chunk, embed, and index a list of PDF files.

        If an index already exists on disk the new documents are appended
        (incremental update) — existing chunks are not re-embedded thanks to
        the embedding cache.

        Args:
            pdf_paths: Paths to PDF files to ingest.
            strategy: Chunking strategy to apply.  One of ``"fixed"``,
                ``"recursive"``, or ``"semantic"``.
            show_progress: Show a tqdm progress bar during embedding.

        Returns:
            Summary dictionary with keys:
            ``docs_processed``, ``total_chunks``, ``new_chunks``,
            ``cache_hits``, ``cache_misses``, ``index_total``.

        Raises:
            ValueError: If *pdf_paths* is empty.
        """
        if not pdf_paths:
            raise ValueError("ingest() requires at least one PDF path.")

        strategy = ChunkingStrategy(strategy)
        t_start = time.perf_counter()
        logger.info(
            "Ingest started: %d file(s), strategy='%s'.",
            len(pdf_paths),
            strategy.value,
        )

        # --- Stage 1: Extract ---
        t0 = time.perf_counter()
        pages = extract_pdfs(pdf_paths)
        logger.info(
            "Extraction: %d page(s) from %d file(s) in %.2fs.",
            len(pages),
            len(pdf_paths),
            time.perf_counter() - t0,
        )

        if not pages:
            logger.warning("No text extracted from any of the supplied PDFs.")
            return {
                "docs_processed": 0,
                "total_chunks": 0,
                "new_chunks": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "index_total": self._get_index().ntotal,
            }

        # --- Stage 2: Chunk ---
        t0 = time.perf_counter()
        encoder_for_semantic = (
            self._get_encoder() if strategy == ChunkingStrategy.SEMANTIC else None
        )
        all_chunks: list[Chunk] = []
        for pdf_path in pdf_paths:
            doc_pages = [p for p in pages if p["doc_name"] == Path(pdf_path).name]
            if doc_pages:
                chunks = chunk_pages(
                    doc_pages,
                    strategy=strategy,
                    chunk_size=self._config.chunk_size,
                    chunk_overlap=self._config.chunk_overlap,
                    encoder=encoder_for_semantic,
                )
                all_chunks.extend(chunks)
        logger.info(
            "Chunking: %d chunk(s) in %.2fs.",
            len(all_chunks),
            time.perf_counter() - t0,
        )

        # --- Stage 3: Embed (with cache) ---
        t0 = time.perf_counter()
        cache = self._get_cache()
        encoder = self._get_encoder()
        embeddings = cache.encode_with_cache(
            all_chunks, encoder, show_progress=show_progress
        )
        cache.save()
        logger.info(
            "Embedding: %d chunk(s) in %.2fs (hits=%d, misses=%d).",
            len(all_chunks),
            time.perf_counter() - t0,
            cache.hits,
            cache.misses,
        )

        # --- Stage 4: Index ---
        t0 = time.perf_counter()
        index = self._get_index(load=True)
        index.add(all_chunks, embeddings)
        index.save()
        # Invalidate cached searcher so it picks up the updated index
        self._searcher = None
        logger.info(
            "Indexing: %d total vector(s) after update in %.2fs.",
            index.ntotal,
            time.perf_counter() - t0,
        )

        elapsed = time.perf_counter() - t_start
        docs_processed = len({p["doc_name"] for p in pages})
        summary = {
            "docs_processed": docs_processed,
            "total_chunks": len(all_chunks),
            "new_chunks": cache.misses,
            "cache_hits": cache.hits,
            "cache_misses": cache.misses,
            "index_total": index.ntotal,
        }
        logger.info(
            "Ingest complete in %.2fs: %s", elapsed, summary
        )
        return summary

    def query(
        self,
        question: str,
        top_k: int | None = None,
        rerank: bool | None = None,
    ) -> Response:
        """Retrieve relevant chunks and generate a grounded answer.

        Args:
            question: Natural-language question to answer.
            top_k: Number of chunks to retrieve.  Defaults to
                ``config.top_k``.
            rerank: Whether to apply cross-encoder reranking.  Defaults to
                ``config.rerank``.

        Returns:
            A :class:`~docvault.generation.response.Response` with the
            answer text and resolved :class:`~docvault.generation.response.CitedSource`
            objects.

        Raises:
            RuntimeError: If the index is empty (no documents ingested yet).
            ValueError: If *question* is blank.
        """
        if not question.strip():
            raise ValueError("question must not be empty.")

        use_rerank = rerank if rerank is not None else self._config.rerank
        t_start = time.perf_counter()
        logger.info("Query started: %r", question[:120])

        # --- Stage 1: Retrieve ---
        t0 = time.perf_counter()
        searcher = self._get_searcher()
        results: list[SearchResult] = searcher.search(question, top_k=top_k)
        logger.info(
            "Retrieval: %d result(s) in %.2fs.", len(results), time.perf_counter() - t0
        )

        # --- Stage 2: (Optional) Rerank ---
        if use_rerank and results:
            t0 = time.perf_counter()
            reranker = self._get_reranker()
            results = reranker.rerank(question, results)
            logger.info(
                "Reranking: %d result(s) kept in %.2fs.",
                len(results),
                time.perf_counter() - t0,
            )

        # --- Stage 3: Build prompt ---
        system_msg, user_msg, included = build_prompt(
            query=question,
            results=results,
            max_context_tokens=self._config.max_context_tokens,
        )

        # --- Stage 4: Generate ---
        t0 = time.perf_counter()
        client = self._get_client()
        raw_text = client.generate(system_msg, user_msg)
        logger.info(
            "Generation: %d chars in %.2fs.", len(raw_text), time.perf_counter() - t0
        )

        # --- Stage 5: Parse ---
        response = parse_response(question, raw_text, included)

        elapsed = time.perf_counter() - t_start
        logger.info(
            "Query complete in %.2fs: %d citation(s) resolved.",
            elapsed,
            len(response.cited_sources),
        )
        return response

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Return index statistics without loading the encoder or LLM client.

        Returns:
            Dictionary from :meth:`~docvault.retrieval.index.VectorIndex.stats`.
        """
        return self._get_index(load=True).stats()

    def remove_document(self, doc_name: str) -> int:
        """Remove all chunks for *doc_name* from the index and save.

        Args:
            doc_name: Filename of the PDF to remove (e.g. ``"report.pdf"``).

        Returns:
            Number of chunks removed.
        """
        index = self._get_index(load=True)
        removed = index.remove_doc(doc_name)
        if removed:
            index.save()
            self._searcher = None  # invalidate
        return removed
