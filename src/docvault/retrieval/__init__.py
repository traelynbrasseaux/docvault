"""Retrieval sub-package: FAISS index, similarity search, and reranking."""

from docvault.retrieval.index import CACHE_FILENAME, INDEX_FILENAME, METADATA_FILENAME, VectorIndex
from docvault.retrieval.reranker import Reranker
from docvault.retrieval.search import SearchResult, Searcher

__all__ = [
    "CACHE_FILENAME",
    "INDEX_FILENAME",
    "METADATA_FILENAME",
    "Reranker",
    "SearchResult",
    "Searcher",
    "VectorIndex",
]
