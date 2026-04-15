"""Tests for retrieval/index.py, retrieval/search.py, and retrieval/reranker.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docvault.config import Config
from docvault.ingest.metadata import Chunk, ChunkMetadata
from docvault.retrieval.index import INDEX_FILENAME, METADATA_FILENAME, VectorIndex
from docvault.retrieval.search import SearchResult, Searcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIM = 64  # small dimension for fast tests


def _unit_vectors(n: int, dim: int = DIM, seed: int = 0) -> np.ndarray:
    """Return n random unit-normalised float32 vectors."""
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunk(
    doc_name: str = "test.pdf",
    page: int = 1,
    idx: int = 0,
    text: str = "sample text",
) -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            doc_name=doc_name,
            page_number=page,
            chunk_index=idx,
            strategy_used="fixed",
            char_count=len(text),
        ),
    )


def _make_chunks(n: int, doc_name: str = "doc.pdf") -> list[Chunk]:
    return [_make_chunk(doc_name=doc_name, idx=i, text=f"chunk {i} text") for i in range(n)]


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.index_dir = tmp_path / "indexes"
    cfg.top_k = 3
    return cfg


@pytest.fixture()
def empty_index(tmp_config: Config) -> VectorIndex:
    return VectorIndex(tmp_config)


@pytest.fixture()
def loaded_index(tmp_config: Config) -> VectorIndex:
    """VectorIndex with 10 chunks already added."""
    idx = VectorIndex(tmp_config)
    chunks = _make_chunks(10)
    vecs = _unit_vectors(10)
    idx.add(chunks, vecs)
    return idx


# ---------------------------------------------------------------------------
# VectorIndex — building
# ---------------------------------------------------------------------------


class TestVectorIndexAdd:
    def test_ntotal_after_add(self, empty_index: VectorIndex):
        empty_index.add(_make_chunks(5), _unit_vectors(5))
        assert empty_index.ntotal == 5

    def test_chunks_property_length(self, empty_index: VectorIndex):
        chunks = _make_chunks(7)
        empty_index.add(chunks, _unit_vectors(7))
        assert len(empty_index.chunks) == 7

    def test_chunks_property_content(self, empty_index: VectorIndex):
        chunks = _make_chunks(3)
        empty_index.add(chunks, _unit_vectors(3))
        for original, stored in zip(chunks, empty_index.chunks):
            assert original.text == stored.text
            assert original.metadata.doc_name == stored.metadata.doc_name

    def test_incremental_add(self, empty_index: VectorIndex):
        empty_index.add(_make_chunks(4), _unit_vectors(4, seed=0))
        empty_index.add(_make_chunks(3, doc_name="b.pdf"), _unit_vectors(3, seed=1))
        assert empty_index.ntotal == 7
        assert len(empty_index.chunks) == 7

    def test_empty_add_is_noop(self, empty_index: VectorIndex):
        empty_index.add([], np.empty((0, DIM), dtype=np.float32))
        assert empty_index.ntotal == 0

    def test_mismatched_lengths_raises(self, empty_index: VectorIndex):
        with pytest.raises(ValueError, match="same length"):
            empty_index.add(_make_chunks(3), _unit_vectors(5))

    def test_dimension_mismatch_raises(self, empty_index: VectorIndex):
        empty_index.add(_make_chunks(2), _unit_vectors(2, dim=DIM))
        with pytest.raises(ValueError, match="dimension"):
            empty_index.add(_make_chunks(2), _unit_vectors(2, dim=DIM * 2))

    def test_is_loaded_true_after_add(self, empty_index: VectorIndex):
        assert not empty_index.is_loaded
        empty_index.add(_make_chunks(1), _unit_vectors(1))
        assert empty_index.is_loaded


# ---------------------------------------------------------------------------
# VectorIndex — persistence
# ---------------------------------------------------------------------------


class TestVectorIndexPersistence:
    def test_save_creates_files(self, tmp_config: Config):
        idx = VectorIndex(tmp_config)
        idx.add(_make_chunks(3), _unit_vectors(3))
        idx.save()

        assert (tmp_config.index_dir / INDEX_FILENAME).exists()
        assert (tmp_config.index_dir / METADATA_FILENAME).exists()

    def test_save_without_data_raises(self, empty_index: VectorIndex):
        with pytest.raises(RuntimeError, match="no index"):
            empty_index.save()

    def test_load_roundtrip_ntotal(self, tmp_config: Config):
        idx_a = VectorIndex(tmp_config)
        idx_a.add(_make_chunks(6), _unit_vectors(6))
        idx_a.save()

        idx_b = VectorIndex(tmp_config)
        loaded = idx_b.load()
        assert loaded is True
        assert idx_b.ntotal == 6

    def test_load_roundtrip_metadata(self, tmp_config: Config):
        original_chunks = _make_chunks(4, doc_name="persist.pdf")
        idx_a = VectorIndex(tmp_config)
        idx_a.add(original_chunks, _unit_vectors(4))
        idx_a.save()

        idx_b = VectorIndex(tmp_config)
        idx_b.load()
        for orig, loaded in zip(original_chunks, idx_b.chunks):
            assert orig.text == loaded.text
            assert orig.metadata.doc_name == loaded.metadata.doc_name
            assert orig.metadata.page_number == loaded.metadata.page_number

    def test_load_returns_false_when_no_file(self, tmp_config: Config):
        idx = VectorIndex(tmp_config)
        assert idx.load() is False

    def test_incremental_load_and_add(self, tmp_config: Config):
        """Load existing index, add more vectors, save again."""
        idx_a = VectorIndex(tmp_config)
        idx_a.add(_make_chunks(5), _unit_vectors(5, seed=0))
        idx_a.save()

        idx_b = VectorIndex(tmp_config)
        idx_b.load()
        idx_b.add(_make_chunks(3, doc_name="extra.pdf"), _unit_vectors(3, seed=1))
        idx_b.save()

        idx_c = VectorIndex(tmp_config)
        idx_c.load()
        assert idx_c.ntotal == 8


# ---------------------------------------------------------------------------
# VectorIndex — utility methods
# ---------------------------------------------------------------------------


class TestVectorIndexUtils:
    def test_doc_names(self, tmp_config: Config):
        idx = VectorIndex(tmp_config)
        idx.add(_make_chunks(3, "a.pdf"), _unit_vectors(3, seed=0))
        idx.add(_make_chunks(2, "b.pdf"), _unit_vectors(2, seed=1))
        assert idx.doc_names() == ["a.pdf", "b.pdf"]

    def test_stats_keys(self, loaded_index: VectorIndex):
        s = loaded_index.stats()
        assert "total_chunks" in s
        assert "total_docs" in s
        assert "doc_names" in s
        assert "index_path" in s

    def test_stats_values(self, tmp_config: Config):
        idx = VectorIndex(tmp_config)
        idx.add(_make_chunks(4, "x.pdf"), _unit_vectors(4))
        s = idx.stats()
        assert s["total_chunks"] == 4
        assert s["total_docs"] == 1
        assert s["doc_names"] == ["x.pdf"]

    def test_remove_doc(self, tmp_config: Config):
        idx = VectorIndex(tmp_config)
        idx.add(_make_chunks(5, "keep.pdf"), _unit_vectors(5, seed=0))
        idx.add(_make_chunks(3, "remove.pdf"), _unit_vectors(3, seed=1))
        removed = idx.remove_doc("remove.pdf")
        assert removed == 3
        assert idx.ntotal == 5
        assert "remove.pdf" not in idx.doc_names()
        assert "keep.pdf" in idx.doc_names()

    def test_remove_nonexistent_doc_returns_zero(self, loaded_index: VectorIndex):
        removed = loaded_index.remove_doc("nonexistent.pdf")
        assert removed == 0
        assert loaded_index.ntotal == 10


# ---------------------------------------------------------------------------
# Searcher
# ---------------------------------------------------------------------------


def _make_searcher(tmp_config: Config, n_chunks: int = 10) -> tuple[VectorIndex, Searcher]:
    """Build a loaded index + Searcher with n_chunks entries."""
    idx = VectorIndex(tmp_config)
    chunks = _make_chunks(n_chunks)
    vecs = _unit_vectors(n_chunks)
    idx.add(chunks, vecs)

    mock_encoder = MagicMock()
    mock_encoder.encode_query.return_value = _unit_vectors(1, seed=99)
    mock_encoder.embedding_dim = DIM

    searcher = Searcher(index=idx, encoder=mock_encoder, config=tmp_config)
    return idx, searcher


class TestSearcher:
    def test_search_returns_correct_count(self, tmp_config: Config):
        _, searcher = _make_searcher(tmp_config, n_chunks=10)
        results = searcher.search("test query", top_k=3)
        assert len(results) == 3

    def test_search_returns_search_results(self, tmp_config: Config):
        _, searcher = _make_searcher(tmp_config, n_chunks=5)
        results = searcher.search("query", top_k=2)
        assert all(isinstance(r, SearchResult) for r in results)

    def test_results_sorted_descending(self, tmp_config: Config):
        _, searcher = _make_searcher(tmp_config, n_chunks=10)
        results = searcher.search("query", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_uses_default_top_k(self, tmp_config: Config):
        tmp_config.top_k = 4
        _, searcher = _make_searcher(tmp_config, n_chunks=10)
        results = searcher.search("query")  # no top_k argument
        assert len(results) == 4

    def test_search_caps_k_at_ntotal(self, tmp_config: Config):
        """top_k > ntotal should not error; returns ntotal results instead."""
        _, searcher = _make_searcher(tmp_config, n_chunks=3)
        results = searcher.search("query", top_k=100)
        assert len(results) == 3

    def test_search_empty_index_raises(self, tmp_config: Config):
        idx = VectorIndex(tmp_config)
        mock_encoder = MagicMock()
        searcher = Searcher(index=idx, encoder=mock_encoder, config=tmp_config)
        with pytest.raises(RuntimeError, match="empty"):
            searcher.search("anything")

    def test_search_invalid_top_k_raises(self, tmp_config: Config):
        _, searcher = _make_searcher(tmp_config, n_chunks=5)
        with pytest.raises(ValueError):
            searcher.search("query", top_k=0)

    def test_result_chunks_are_from_index(self, tmp_config: Config):
        idx, searcher = _make_searcher(tmp_config, n_chunks=5)
        results = searcher.search("query", top_k=3)
        index_texts = {c.text for c in idx.chunks}
        for r in results:
            assert r.chunk.text in index_texts

    def test_encoder_called_once(self, tmp_config: Config):
        _, searcher = _make_searcher(tmp_config, n_chunks=5)
        searcher.search("my query", top_k=2)
        searcher._encoder.encode_query.assert_called_once_with("my query")


# ---------------------------------------------------------------------------
# SearchResult dataclass
# ---------------------------------------------------------------------------


def test_search_result_repr():
    chunk = _make_chunk()
    r = SearchResult(chunk=chunk, score=0.85)
    assert "0.8500" in repr(r)
    assert "test.pdf" in repr(r)
