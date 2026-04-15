"""Dataclass-based configuration loaded from environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Central configuration for docvault.

    All fields are populated from environment variables with sensible defaults.
    Copy ``.env.example`` to ``.env`` and edit before running.

    Attributes:
        anthropic_api_key: Anthropic API key. Required for generation.
        embedding_model: Sentence-transformer model name used for encoding.
        chunk_size: Target character count per chunk.
        chunk_overlap: Approximate character overlap between adjacent chunks.
        top_k: Number of chunks to retrieve from the FAISS index per query.
        llm_model: Claude model ID to use for answer generation.
        index_dir: Directory where FAISS index and metadata files are stored.
        data_dir: Root data directory.
        rerank: Whether to run the cross-encoder reranker after retrieval.
        rerank_model: Cross-encoder model name used for reranking.
        rerank_top_n: Number of results to keep after reranking.
        embedding_batch_size: Batch size for sentence-transformer encoding.
        max_context_tokens: Approximate token budget for retrieved context.
        llm_max_tokens: Max tokens for the LLM response.
        llm_temperature: Sampling temperature for the LLM.
    """

    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "64"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
    )
    index_dir: Path = field(
        default_factory=lambda: Path(os.getenv("INDEX_DIR", "data/indexes"))
    )
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("DATA_DIR", "data"))
    )
    rerank: bool = field(
        default_factory=lambda: os.getenv("RERANK", "false").lower() == "true"
    )
    rerank_model: str = field(
        default_factory=lambda: os.getenv(
            "RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
    )
    rerank_top_n: int = field(
        default_factory=lambda: int(os.getenv("RERANK_TOP_N", "3"))
    )
    embedding_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    )
    max_context_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_TOKENS", "6000"))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "1024"))
    )
    llm_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2"))
    )

    def __post_init__(self) -> None:
        if isinstance(self.index_dir, str):
            self.index_dir = Path(self.index_dir)
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

    def require_api_key(self) -> str:
        """Return the Anthropic API key, raising an error if it is not set.

        Returns:
            The Anthropic API key string.

        Raises:
            ValueError: If ``ANTHROPIC_API_KEY`` is not set in the environment.
        """
        if not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Copy .env.example to .env and fill in your Anthropic API key, "
                "or export the variable in your shell before running docvault."
            )
        return self.anthropic_api_key
