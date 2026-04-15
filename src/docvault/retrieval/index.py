"""FAISS index management: build, save, load, and incremental update."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from docvault.ingest.metadata import Chunk

if TYPE_CHECKING:
    from docvault.config import Config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# File-name constants (all files land in config.index_dir)
# ---------------------------------------------------------------------------

INDEX_FILENAME = "docvault.faiss"
METADATA_FILENAME = "docvault_chunks.json"
CACHE_FILENAME = "embeddings_cache.pkl"


class VectorIndex:
    """Manages a FAISS ``IndexFlatIP`` index and its parallel chunk metadata.

    The index uses inner-product (dot product) similarity, which is equivalent
    to cosine similarity when embeddings are L2-normalized — as guaranteed by
    :class:`~docvault.embeddings.encoder.Encoder`.

    Two files are persisted to *config.index_dir*:

    - ``docvault.faiss`` — the serialized FAISS index.
    - ``docvault_chunks.json`` — a JSON array of chunk dicts in index order.

    These files are always kept in sync: every :meth:`save` call writes both.

    Args:
        config: :class:`~docvault.config.Config` instance supplying
            ``index_dir`` and ``embedding_model`` (for dimension validation).

    Attributes:
        index_path: Path to the ``.faiss`` file.
        metadata_path: Path to the ``.json`` metadata file.
        cache_path: Path to the optional embedding cache ``.pkl`` file.
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self.index_path: Path = config.index_dir / INDEX_FILENAME
        self.metadata_path: Path = config.index_dir / METADATA_FILENAME
        self.cache_path: Path = config.index_dir / CACHE_FILENAME

        self._index = None  # faiss.Index, lazy-loaded
        self._chunks: list[Chunk] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index."""
        return int(self._index.ntotal) if self._index is not None else 0

    @property
    def chunks(self) -> list[Chunk]:
        """Ordered list of :class:`~docvault.ingest.metadata.Chunk` objects
        corresponding to each row in the FAISS index."""
        return self._chunks

    @property
    def is_loaded(self) -> bool:
        """``True`` if the index has been built or loaded from disk."""
        return self._index is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _require_faiss(self):
        """Import faiss, raising a clear error if not installed."""
        try:
            import faiss
            return faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required for the vector index. "
                "Install it with: pip install faiss-cpu"
            ) from exc

    def _init_index(self, dim: int):
        """Create a fresh IndexFlatIP for the given embedding dimension."""
        faiss = self._require_faiss()
        self._index = faiss.IndexFlatIP(dim)
        logger.debug("Created new IndexFlatIP with dim=%d.", dim)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load an existing index and metadata from *config.index_dir*.

        Does nothing and returns ``False`` if no index file is found — the
        caller should then build the index from scratch.

        Returns:
            ``True`` if the index was loaded successfully, ``False`` otherwise.

        Raises:
            OSError: If the files exist but cannot be read or are corrupted.
        """
        faiss = self._require_faiss()

        if not self.index_path.exists():
            logger.info("No existing index at '%s'.", self.index_path)
            return False

        if not self.metadata_path.exists():
            logger.warning(
                "Index file found but metadata file '%s' is missing — "
                "rebuilding from scratch is recommended.",
                self.metadata_path,
            )
            return False

        try:
            self._index = faiss.read_index(str(self.index_path))
            logger.info(
                "Loaded FAISS index from '%s' (%d vectors).",
                self.index_path,
                self._index.ntotal,
            )
        except Exception as exc:
            raise OSError(
                f"Failed to read FAISS index from {self.index_path}: {exc}"
            ) from exc

        try:
            with self.metadata_path.open("r", encoding="utf-8") as fh:
                raw: list[dict] = json.load(fh)
            self._chunks = [Chunk.from_dict(item) for item in raw]
            logger.info(
                "Loaded %d chunk metadata record(s) from '%s'.",
                len(self._chunks),
                self.metadata_path,
            )
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            raise OSError(
                f"Failed to read chunk metadata from {self.metadata_path}: {exc}"
            ) from exc

        if self._index.ntotal != len(self._chunks):
            raise OSError(
                f"Index/metadata mismatch: FAISS has {self._index.ntotal} vectors "
                f"but metadata has {len(self._chunks)} records. "
                f"Delete '{self._config.index_dir}' and re-ingest."
            )

        return True

    def save(self) -> None:
        """Write the FAISS index and chunk metadata to *config.index_dir*.

        Creates the directory if it does not exist.

        Raises:
            RuntimeError: If no index has been built yet.
            OSError: If either file cannot be written.
        """
        if self._index is None:
            raise RuntimeError(
                "Cannot save: no index has been built. Call add() first."
            )

        faiss = self._require_faiss()

        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            logger.info(
                "Saved FAISS index to '%s' (%d vectors).",
                self.index_path,
                self._index.ntotal,
            )
        except Exception as exc:
            raise OSError(
                f"Failed to write FAISS index to {self.index_path}: {exc}"
            ) from exc

        try:
            with self.metadata_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    [c.to_dict() for c in self._chunks],
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info(
                "Saved chunk metadata to '%s' (%d records).",
                self.metadata_path,
                len(self._chunks),
            )
        except OSError as exc:
            raise OSError(
                f"Failed to write chunk metadata to {self.metadata_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """Add chunks and their pre-computed embeddings to the index.

        Supports incremental updates: if an index already exists in memory
        (because :meth:`load` was called first), the new vectors are appended
        rather than replacing the existing ones.

        Args:
            chunks: Chunk objects whose texts were encoded into *embeddings*.
            embeddings: Float32 array of shape ``(len(chunks), dim)``.
                Must be L2-normalized (guaranteed by
                :class:`~docvault.embeddings.encoder.Encoder`).

        Raises:
            ValueError: If *chunks* and *embeddings* have mismatched lengths,
                or if the embedding dimension differs from the existing index.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "must have the same length."
            )
        if len(chunks) == 0:
            logger.warning("add() called with zero chunks — nothing to do.")
            return

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        dim = embeddings.shape[1]

        if self._index is None:
            self._init_index(dim)
        elif self._index.d != dim:
            raise ValueError(
                f"Embedding dimension mismatch: index expects {self._index.d}, "
                f"got {dim}. Re-create the index with a consistent encoder."
            )

        self._index.add(embeddings)  # type: ignore[union-attr]
        self._chunks.extend(chunks)

        logger.info(
            "Added %d vector(s); index now contains %d total.",
            len(chunks),
            self._index.ntotal,
        )

    def remove_doc(self, doc_name: str) -> int:
        """Remove all chunks belonging to *doc_name* and rebuild the index.

        This is an O(n) operation that rebuilds the FAISS index from scratch
        after filtering out the target document.  Suitable for small-to-medium
        indexes; for very large indexes consider a full re-ingest.

        Args:
            doc_name: The ``doc_name`` field to filter out.

        Returns:
            Number of chunks that were removed.
        """
        if self._index is None:
            return 0

        kept_chunks = [c for c in self._chunks if c.metadata.doc_name != doc_name]
        removed = len(self._chunks) - len(kept_chunks)

        if removed == 0:
            logger.info("No chunks found for doc '%s'.", doc_name)
            return 0

        # Identify kept indices and extract their vectors
        faiss = self._require_faiss()
        kept_indices = [
            i for i, c in enumerate(self._chunks) if c.metadata.doc_name != doc_name
        ]

        if kept_indices:
            all_vectors = np.zeros(
                (self._index.ntotal, self._index.d), dtype=np.float32
            )
            self._index.reconstruct_n(0, self._index.ntotal, all_vectors)
            kept_vectors = all_vectors[kept_indices]
        else:
            kept_vectors = np.empty((0, self._index.d), dtype=np.float32)

        # Rebuild
        self._index = faiss.IndexFlatIP(self._index.d)
        if len(kept_vectors):
            self._index.add(kept_vectors)
        self._chunks = kept_chunks

        logger.info(
            "Removed %d chunk(s) for doc '%s'. Index now has %d vector(s).",
            removed,
            doc_name,
            self._index.ntotal,
        )
        return removed

    def doc_names(self) -> list[str]:
        """Return the sorted unique list of document names in the index.

        Returns:
            Sorted list of ``doc_name`` strings.
        """
        return sorted({c.metadata.doc_name for c in self._chunks})

    def stats(self) -> dict:
        """Return a summary dict for display/logging.

        Returns:
            Dictionary with ``total_chunks``, ``total_docs``, ``doc_names``,
            and ``index_path`` keys.
        """
        return {
            "total_chunks": self.ntotal,
            "total_docs": len(self.doc_names()),
            "doc_names": self.doc_names(),
            "index_path": str(self.index_path),
        }
