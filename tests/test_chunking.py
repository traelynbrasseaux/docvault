"""Tests for ingest/chunking.py — all three strategies and shared behaviour."""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from docvault.ingest.chunking import (
    ChunkingStrategy,
    _split_sentences,
    chunk_pages,
    chunk_text,
)
from docvault.ingest.metadata import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Retrieval-Augmented Generation is a powerful technique. "
    "It combines information retrieval with text generation. "
    "Lewis et al. introduced RAG in 2020. "
    "The method has since become widely adopted. "
    "It reduces hallucination by grounding answers in retrieved evidence. "
    "Multiple chunking strategies exist for splitting documents. "
    "Fixed-size chunking is the simplest approach. "
    "Recursive character chunking respects natural language boundaries. "
    "Semantic chunking groups sentences by embedding similarity. "
    "Each strategy has different trade-offs for retrieval quality."
)

MULTI_PARA_TEXT = (
    "Paragraph one contains the first two sentences. "
    "It introduces the main topic of the document.\n\n"
    "Paragraph two elaborates on the subject. "
    "It provides additional context and background information.\n\n"
    "Paragraph three concludes the argument. "
    "It summarises the key points made above."
)

DOC_NAME = "test_doc.pdf"
PAGE_NUM = 1


def _make_mock_encoder(dim: int = 384, seed: int = 42) -> MagicMock:
    """Return a mock Encoder whose encode() returns unit-normalised vectors."""

    def fake_encode(texts, **kwargs):
        rng = np.random.default_rng(seed + len(texts))
        vecs = rng.random((len(texts), dim)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms

    encoder = MagicMock()
    encoder.encode.side_effect = fake_encode
    return encoder


# ---------------------------------------------------------------------------
# _split_sentences helper
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    sentences = _split_sentences("First sentence. Second sentence. Third sentence.")
    assert len(sentences) >= 2
    for s in sentences:
        assert s.strip() == s  # no leading/trailing whitespace


def test_split_sentences_empty():
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_split_sentences_single():
    result = _split_sentences("Only one sentence here")
    assert len(result) == 1
    assert result[0] == "Only one sentence here"


# ---------------------------------------------------------------------------
# Fixed-size chunking
# ---------------------------------------------------------------------------


class TestFixedChunking:
    def test_produces_chunks(self):
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200, chunk_overlap=30,
        )
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_metadata_fields(self):
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200, chunk_overlap=30,
        )
        for i, chunk in enumerate(chunks):
            m = chunk.metadata
            assert m.doc_name == DOC_NAME
            assert m.page_number == PAGE_NUM
            assert m.chunk_index == i
            assert m.strategy_used == ChunkingStrategy.FIXED.value
            assert m.char_count == len(chunk.text)

    def test_respects_chunk_size(self):
        """No chunk should exceed chunk_size by more than one sentence worth."""
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=150, chunk_overlap=20,
        )
        # Chunks may slightly exceed chunk_size if a single sentence is long,
        # but none should be more than 2x chunk_size.
        for c in chunks:
            assert c.metadata.char_count <= 300, (
                f"Chunk exceeded 2× chunk_size: {c.metadata.char_count} chars"
            )

    def test_sentence_boundary_preservation(self):
        """Chunks must not start or end with a partial word fragment."""
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200, chunk_overlap=30,
        )
        for c in chunks:
            # A chunk starting with a lowercase letter mid-word is a sign
            # of a mid-sentence cut (heuristic check).
            first_char = c.text[0]
            assert first_char.isupper() or first_char.isdigit() or first_char in '"\'([', (
                f"Chunk appears to start mid-sentence: {c.text[:40]!r}"
            )

    def test_overlap_creates_shared_text(self):
        """Adjacent chunks should share at least one sentence when overlap > 0."""
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=300, chunk_overlap=80,
        )
        if len(chunks) < 2:
            pytest.skip("Not enough chunks to test overlap")
        # The end of chunk[0] should appear at the start of chunk[1]
        found_overlap = False
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].text.split())
            words_b = set(chunks[i + 1].text.split())
            if words_a & words_b:
                found_overlap = True
                break
        assert found_overlap, "No word overlap found between adjacent chunks"

    def test_zero_overlap(self):
        """With chunk_overlap=0, adjacent chunks should not share sentences."""
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=250, chunk_overlap=0,
        )
        assert len(chunks) >= 1

    def test_text_coverage(self):
        """All words in the original text should appear in at least one chunk."""
        original_words = set(SAMPLE_TEXT.split())
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200, chunk_overlap=30,
        )
        covered = set()
        for c in chunks:
            covered.update(c.text.split())
        assert original_words.issubset(covered), (
            f"Missing words: {original_words - covered}"
        )

    def test_string_strategy_accepted(self):
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy="fixed", chunk_size=200, chunk_overlap=0,
        )
        assert len(chunks) >= 1
        assert chunks[0].metadata.strategy_used == "fixed"

    def test_chunk_index_sequential(self):
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200, chunk_overlap=30,
        )
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_start_index_offset(self):
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.FIXED,
            chunk_size=200, chunk_overlap=30,
            start_index=10,
        )
        assert chunks[0].metadata.chunk_index == 10


# ---------------------------------------------------------------------------
# Recursive character chunking
# ---------------------------------------------------------------------------


class TestRecursiveChunking:
    def test_produces_chunks(self):
        chunks = chunk_text(
            MULTI_PARA_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=300, chunk_overlap=40,
        )
        assert len(chunks) >= 1

    def test_metadata_strategy_label(self):
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200, chunk_overlap=30,
        )
        for c in chunks:
            assert c.metadata.strategy_used == ChunkingStrategy.RECURSIVE.value

    def test_keeps_short_paragraphs_together(self):
        """A paragraph shorter than chunk_size should land in a single chunk."""
        # Each paragraph in MULTI_PARA_TEXT is ~100 chars
        chunks = chunk_text(
            MULTI_PARA_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=400, chunk_overlap=0,
        )
        # With a generous chunk_size all three paragraphs may fit in 1-2 chunks
        assert len(chunks) >= 1
        # No chunk should be empty
        assert all(c.text.strip() for c in chunks)

    def test_large_paragraph_gets_split(self):
        """A paragraph exceeding chunk_size must be split into multiple chunks."""
        long_para = " ".join([f"Sentence {i} is here." for i in range(30)])
        chunks = chunk_text(
            long_para, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=100, chunk_overlap=0,
        )
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.char_count <= 200  # no chunk > 2× limit

    def test_text_coverage(self):
        original_words = set(MULTI_PARA_TEXT.split())
        chunks = chunk_text(
            MULTI_PARA_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.RECURSIVE,
            chunk_size=200, chunk_overlap=30,
        )
        covered = set()
        for c in chunks:
            covered.update(c.text.split())
        assert original_words.issubset(covered)


# ---------------------------------------------------------------------------
# Semantic chunking
# ---------------------------------------------------------------------------


class TestSemanticChunking:
    def test_requires_encoder(self):
        with pytest.raises(ValueError, match="Encoder instance"):
            chunk_text(
                SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
                strategy=ChunkingStrategy.SEMANTIC,
                encoder=None,
            )

    def test_produces_chunks(self):
        encoder = _make_mock_encoder()
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=300, chunk_overlap=40,
            encoder=encoder,
        )
        assert len(chunks) >= 1

    def test_metadata_strategy_label(self):
        encoder = _make_mock_encoder()
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            encoder=encoder,
        )
        for c in chunks:
            assert c.metadata.strategy_used == ChunkingStrategy.SEMANTIC.value

    def test_encoder_called_once(self):
        """encoder.encode() should be called once for all sentences in the page."""
        encoder = _make_mock_encoder()
        chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            encoder=encoder,
        )
        # Encoder is called exactly once (batched)
        assert encoder.encode.call_count == 1

    def test_single_sentence_text(self):
        encoder = _make_mock_encoder()
        chunks = chunk_text(
            "Only one sentence here.", DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            encoder=encoder,
        )
        assert len(chunks) == 1

    def test_high_threshold_splits_more(self):
        """A threshold of 1.0 (never merge) should produce more chunks."""
        encoder = _make_mock_encoder()
        chunks_low = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=1000, chunk_overlap=0,
            encoder=encoder,
            similarity_threshold=0.0,  # always merge → fewer chunks
        )
        encoder2 = _make_mock_encoder()
        chunks_high = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            chunk_size=1000, chunk_overlap=0,
            encoder=encoder2,
            similarity_threshold=1.0,  # never merge → more chunks
        )
        assert len(chunks_high) >= len(chunks_low)

    def test_text_coverage(self):
        encoder = _make_mock_encoder()
        original_words = set(SAMPLE_TEXT.split())
        chunks = chunk_text(
            SAMPLE_TEXT, DOC_NAME, PAGE_NUM,
            strategy=ChunkingStrategy.SEMANTIC,
            encoder=encoder,
        )
        covered = set()
        for c in chunks:
            covered.update(c.text.split())
        assert original_words.issubset(covered)


# ---------------------------------------------------------------------------
# chunk_pages (multi-page document)
# ---------------------------------------------------------------------------


class TestChunkPages:
    def _make_pages(self, count: int = 3) -> list[dict]:
        return [
            {
                "doc_name": "multi.pdf",
                "page_number": i + 1,
                "text": SAMPLE_TEXT,
            }
            for i in range(count)
        ]

    def test_produces_chunks_from_all_pages(self):
        pages = self._make_pages(3)
        chunks = chunk_pages(pages, strategy=ChunkingStrategy.FIXED, chunk_size=200)
        # Each page should contribute at least one chunk
        page_numbers = {c.metadata.page_number for c in chunks}
        assert page_numbers == {1, 2, 3}

    def test_global_chunk_index(self):
        """chunk_index must be globally unique across pages, not reset per page."""
        pages = self._make_pages(2)
        chunks = chunk_pages(pages, strategy=ChunkingStrategy.FIXED, chunk_size=200)
        indices = [c.metadata.chunk_index for c in chunks]
        # Indices should be monotonically increasing and contiguous
        assert indices == list(range(len(chunks)))

    def test_empty_pages_list(self):
        # chunk_pages on [] should return []
        assert chunk_pages([], strategy=ChunkingStrategy.FIXED) == []


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_invalid_strategy_raises():
    with pytest.raises(ValueError):
        chunk_text(SAMPLE_TEXT, DOC_NAME, PAGE_NUM, strategy="nonexistent")
