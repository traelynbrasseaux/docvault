"""Hash-based embedding cache backed by a pickle file."""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from docvault.ingest.metadata import Chunk

if TYPE_CHECKING:
    from docvault.embeddings.encoder import Encoder

logger = logging.getLogger(__name__)


def _cache_key(doc_name: str, text: str) -> str:
    """Compute a stable SHA-256 key for a chunk.

    Args:
        doc_name: Source document filename.
        text: Chunk text.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    payload = f"{doc_name}::{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class EmbeddingCache:
    """Persist and retrieve pre-computed embeddings keyed by chunk content.

    Avoids re-encoding chunks that have not changed since the last ingest run.
    The cache is a plain Python ``dict`` serialized to a ``.pkl`` file so it
    survives across processes.

    Args:
        cache_path: Path to the ``.pkl`` cache file.  The file is created on
            the first :meth:`save` call if it does not exist.

    Attributes:
        cache_path: Resolved path to the backing pickle file.
        hits: Number of cache hits since the object was created.
        misses: Number of cache misses since the object was created.
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = Path(cache_path)
        self._store: dict[str, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the cache from disk if the file exists."""
        if not self.cache_path.exists():
            logger.debug("No existing cache at '%s' — starting fresh.", self.cache_path)
            return
        try:
            with self.cache_path.open("rb") as fh:
                self._store = pickle.load(fh)
            logger.info(
                "Loaded embedding cache from '%s' (%d entries).",
                self.cache_path,
                len(self._store),
            )
        except (OSError, pickle.UnpicklingError) as exc:
            logger.warning(
                "Could not load cache from '%s': %s — starting fresh.",
                self.cache_path,
                exc,
            )
            self._store = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, doc_name: str, text: str) -> np.ndarray | None:
        """Retrieve a cached embedding vector.

        Args:
            doc_name: Source document filename.
            text: Chunk text.

        Returns:
            The cached float32 embedding array, or ``None`` on a cache miss.
        """
        key = _cache_key(doc_name, text)
        result = self._store.get(key)
        if result is not None:
            self.hits += 1
        else:
            self.misses += 1
        return result

    def set(self, doc_name: str, text: str, embedding: np.ndarray) -> None:
        """Store an embedding vector in the cache.

        Args:
            doc_name: Source document filename.
            text: Chunk text.
            embedding: The embedding vector to cache.
        """
        key = _cache_key(doc_name, text)
        self._store[key] = embedding

    def save(self) -> None:
        """Persist the in-memory cache to the backing pickle file.

        Creates parent directories if needed.

        Raises:
            OSError: If the file cannot be written.
        """
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("wb") as fh:
                pickle.dump(self._store, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(
                "Saved embedding cache to '%s' (%d entries).",
                self.cache_path,
                len(self._store),
            )
        except OSError as exc:
            raise OSError(
                f"Failed to save embedding cache to {self.cache_path}: {exc}"
            ) from exc

    def encode_with_cache(
        self,
        chunks: list[Chunk],
        encoder: "Encoder",
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode chunks, skipping those already in the cache.

        Chunks with a cache hit are returned directly; uncached chunks are
        batched together, encoded, stored, and merged back into the result
        array in original order.

        Args:
            chunks: Chunks to encode.
            encoder: Encoder used to compute missing embeddings.
            show_progress: Show a tqdm progress bar during encoding.

        Returns:
            Float32 array of shape ``(len(chunks), embedding_dim)`` with one
            row per chunk, in the same order as *chunks*.
        """
        if not chunks:
            return np.empty((0, encoder.embedding_dim), dtype=np.float32)

        results: list[np.ndarray | None] = [None] * len(chunks)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, chunk in enumerate(chunks):
            cached = self.get(chunk.metadata.doc_name, chunk.text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(chunk.text)

        if uncached_texts:
            logger.info(
                "Encoding %d new chunk(s) (%d cache hit(s)).",
                len(uncached_texts),
                len(chunks) - len(uncached_texts),
            )
            new_embeddings = encoder.encode(
                uncached_texts, show_progress=show_progress
            )
            for local_i, global_i in enumerate(uncached_indices):
                emb = new_embeddings[local_i]
                results[global_i] = emb
                self.set(
                    chunks[global_i].metadata.doc_name,
                    chunks[global_i].text,
                    emb,
                )
        else:
            logger.info("All %d chunk(s) served from cache.", len(chunks))

        return np.vstack(results).astype(np.float32)
