"""Embeddings sub-package: sentence-transformer encoder and embedding cache."""

from docvault.embeddings.cache import EmbeddingCache
from docvault.embeddings.encoder import Encoder

__all__ = ["Encoder", "EmbeddingCache"]
