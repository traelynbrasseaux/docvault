"""Tests for embeddings/encoder.py and embeddings/cache.py."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from docvault.config import Config
from docvault.embeddings.cache import EmbeddingCache, _cache_key
from docvault.ingest.metadata import Chunk, ChunkMetadata

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

DIM = 384  # all-MiniLM-L6-v2 output dimension


def _unit_vectors(n: int, seed: int = 42) -> np.ndarray:
    """Return n random unit-normalised float32 vectors of dimension DIM."""
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _make_chunk(doc_name: str = "doc.pdf", text: str = "hello world") -> Chunk:
    return Chunk(
        text=text,
        metadata=ChunkMetadata(
            doc_name=doc_name,
            page_number=1,
            chunk_index=0,
            strategy_used="fixed",
            char_count=len(text),
        ),
    )


class FakeSentenceTransformer:
    """Lightweight SentenceTransformer stand-in for unit tests."""

    def __init__(self, model_name_or_path: str, device: str = "cpu") -> None:
        self._dim = DIM

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        sentences,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        n = len(sentences)
        vecs = _unit_vectors(n, seed=n)
        return vecs


# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def encoder(monkeypatch):
    """Encoder instance backed by FakeSentenceTransformer."""
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        FakeSentenceTransformer,
        raising=False,
    )
    # Patch torch.cuda.is_available so tests run without a GPU
    with patch("torch.cuda.is_available", return_value=False):
        from docvault.embeddings.encoder import Encoder
        cfg = Config()
        cfg.embedding_model = "all-MiniLM-L6-v2"
        cfg.embedding_batch_size = 32
        return Encoder(cfg)


class TestEncoder:
    def test_embedding_dim_attribute(self, encoder):
        assert encoder.embedding_dim == DIM

    def test_encode_shape(self, encoder):
        texts = ["hello", "world", "foo bar baz"]
        vecs = encoder.encode(texts)
        assert vecs.shape == (3, DIM)

    def test_encode_dtype(self, encoder):
        vecs = encoder.encode(["test"])
        assert vecs.dtype == np.float32

    def test_encode_unit_normalized(self, encoder):
        """All output vectors should have L2 norm ≈ 1.0."""
        vecs = encoder.encode(["sentence one", "sentence two", "sentence three"])
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_query_shape(self, encoder):
        vec = encoder.encode_query("what is RAG?")
        assert vec.shape == (1, DIM)

    def test_encode_query_normalized(self, encoder):
        vec = encoder.encode_query("what is RAG?")
        norm = float(np.linalg.norm(vec[0]))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_raises(self, encoder):
        with pytest.raises(ValueError, match="empty"):
            encoder.encode([])

    def test_encode_single_text(self, encoder):
        vecs = encoder.encode(["just one text"])
        assert vecs.shape == (1, DIM)

    def test_batch_size_override(self, encoder):
        texts = [f"sentence {i}" for i in range(20)]
        vecs = encoder.encode(texts, batch_size=5)
        assert vecs.shape == (20, DIM)

    def test_model_name_attribute(self, encoder):
        assert isinstance(encoder.model_name, str)
        assert len(encoder.model_name) > 0


# ---------------------------------------------------------------------------
# _cache_key helper
# ---------------------------------------------------------------------------


def test_cache_key_deterministic():
    k1 = _cache_key("doc.pdf", "hello world")
    k2 = _cache_key("doc.pdf", "hello world")
    assert k1 == k2


def test_cache_key_different_for_different_inputs():
    k1 = _cache_key("doc.pdf", "text A")
    k2 = _cache_key("doc.pdf", "text B")
    k3 = _cache_key("other.pdf", "text A")
    assert k1 != k2
    assert k1 != k3
    assert k2 != k3


def test_cache_key_is_hex_string():
    k = _cache_key("doc.pdf", "some text")
    assert isinstance(k, str)
    # SHA-256 hex is 64 characters
    assert len(k) == 64
    assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# EmbeddingCache tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_cache(tmp_path: Path) -> EmbeddingCache:
    """Fresh EmbeddingCache backed by a temp file."""
    return EmbeddingCache(tmp_path / "test_cache.pkl")


class TestEmbeddingCache:
    def test_initial_miss(self, tmp_cache: EmbeddingCache):
        result = tmp_cache.get("doc.pdf", "some text")
        assert result is None
        assert tmp_cache.misses == 1
        assert tmp_cache.hits == 0

    def test_set_then_get(self, tmp_cache: EmbeddingCache):
        emb = _unit_vectors(1)[0]
        tmp_cache.set("doc.pdf", "hello", emb)
        retrieved = tmp_cache.get("doc.pdf", "hello")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, emb)
        assert tmp_cache.hits == 1

    def test_different_doc_same_text(self, tmp_cache: EmbeddingCache):
        emb_a = _unit_vectors(1, seed=1)[0]
        emb_b = _unit_vectors(1, seed=2)[0]
        tmp_cache.set("a.pdf", "text", emb_a)
        tmp_cache.set("b.pdf", "text", emb_b)
        ra = tmp_cache.get("a.pdf", "text")
        rb = tmp_cache.get("b.pdf", "text")
        assert ra is not None and rb is not None
        assert not np.array_equal(ra, rb)

    def test_save_and_reload(self, tmp_path: Path):
        cache_path = tmp_path / "cache.pkl"
        emb = _unit_vectors(1)[0]

        c1 = EmbeddingCache(cache_path)
        c1.set("doc.pdf", "persistent text", emb)
        c1.save()

        c2 = EmbeddingCache(cache_path)  # reload from disk
        retrieved = c2.get("doc.pdf", "persistent text")
        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, emb)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        deep_path = tmp_path / "a" / "b" / "c" / "cache.pkl"
        cache = EmbeddingCache(deep_path)
        cache.set("doc.pdf", "text", _unit_vectors(1)[0])
        cache.save()
        assert deep_path.exists()

    def test_load_nonexistent_is_silent(self, tmp_path: Path):
        cache = EmbeddingCache(tmp_path / "no_such_file.pkl")
        # Should not raise; cache is simply empty
        assert cache.get("x", "y") is None

    def test_hits_and_misses_counters(self, tmp_cache: EmbeddingCache):
        tmp_cache.get("doc.pdf", "miss1")   # miss
        tmp_cache.get("doc.pdf", "miss2")   # miss
        emb = _unit_vectors(1)[0]
        tmp_cache.set("doc.pdf", "hit_key", emb)
        tmp_cache.get("doc.pdf", "hit_key")  # hit
        tmp_cache.get("doc.pdf", "hit_key")  # hit
        assert tmp_cache.misses == 2
        assert tmp_cache.hits == 2

    def test_encode_with_cache_skips_cached(self, tmp_path: Path):
        """encode_with_cache() should only call encoder for uncached chunks."""
        cache = EmbeddingCache(tmp_path / "cache.pkl")

        # Pre-populate cache with the first chunk
        chunk_a = _make_chunk("doc.pdf", "already cached text")
        chunk_b = _make_chunk("doc.pdf", "needs to be encoded")
        pre_emb = _unit_vectors(1)[0]
        cache.set(chunk_a.metadata.doc_name, chunk_a.text, pre_emb)

        mock_encoder = MagicMock()
        mock_encoder.embedding_dim = DIM
        mock_encoder.encode.return_value = _unit_vectors(1)

        result = cache.encode_with_cache([chunk_a, chunk_b], mock_encoder)

        # Encoder should only have been called for the uncached chunk
        assert mock_encoder.encode.call_count == 1
        called_texts = mock_encoder.encode.call_args[0][0]
        assert "needs to be encoded" in called_texts
        assert "already cached text" not in called_texts

        assert result.shape == (2, DIM)
        assert result.dtype == np.float32

    def test_encode_with_cache_all_cached(self, tmp_path: Path):
        """encoder.encode() should not be called at all when everything is cached."""
        cache = EmbeddingCache(tmp_path / "cache.pkl")
        chunk = _make_chunk("doc.pdf", "cached")
        emb = _unit_vectors(1)[0]
        cache.set(chunk.metadata.doc_name, chunk.text, emb)

        mock_encoder = MagicMock()
        mock_encoder.embedding_dim = DIM

        result = cache.encode_with_cache([chunk], mock_encoder)
        mock_encoder.encode.assert_not_called()
        assert result.shape == (1, DIM)

    def test_encode_with_cache_empty_chunks(self, tmp_path: Path):
        cache = EmbeddingCache(tmp_path / "cache.pkl")
        mock_encoder = MagicMock()
        mock_encoder.embedding_dim = DIM
        result = cache.encode_with_cache([], mock_encoder)
        assert result.shape == (0, DIM)
