"""Sentence-transformer encoder wrapper with batched encoding and normalization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from docvault.config import Config

logger = logging.getLogger(__name__)


class Encoder:
    """Wraps a ``sentence-transformers`` model for chunk and query encoding.

    Embeddings are L2-normalized to unit vectors so that inner product equals
    cosine similarity.  This pairs directly with ``faiss.IndexFlatIP``.

    Args:
        config: :class:`~docvault.config.Config` instance supplying the model
            name, batch size, and device preference.

    Attributes:
        model_name: Name of the loaded sentence-transformer model.
        embedding_dim: Dimensionality of the output embedding vectors.
    """

    def __init__(self, config: "Config") -> None:
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers and torch are required. "
                "Install them with: pip install sentence-transformers torch"
            ) from exc

        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Loading sentence-transformer '%s' on device '%s'.",
            config.embedding_model,
            device,
        )

        self._model = SentenceTransformer(config.embedding_model, device=device)
        self._batch_size = config.embedding_batch_size
        self.model_name: str = config.embedding_model
        self.embedding_dim: int = self._model.get_sentence_embedding_dimension()
        self._device = device

        logger.info(
            "Encoder ready — model: '%s', dim: %d, device: %s.",
            self.model_name,
            self.embedding_dim,
            device,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode a list of texts into L2-normalized embedding vectors.

        Args:
            texts: Strings to encode.
            batch_size: Override the default batch size from config.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Float32 NumPy array of shape ``(len(texts), embedding_dim)``.
            Each row is a unit vector.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError("encode() received an empty list of texts.")

        bs = batch_size if batch_size is not None else self._batch_size
        logger.debug("Encoding %d text(s) in batches of %d.", len(texts), bs)

        embeddings: np.ndarray = self._model.encode(
            texts,
            batch_size=bs,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a unit-normalized embedding vector.

        This is a convenience wrapper around :meth:`encode` for the common
        retrieval use-case where only one string needs to be embedded.

        Args:
            query: The query string.

        Returns:
            Float32 NumPy array of shape ``(1, embedding_dim)``.
        """
        return self.encode([query])
