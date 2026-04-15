"""Integration tests for pipeline.py using mock sub-systems.

The real Anthropic API and sentence-transformer model are never called.
PDF extraction is patched to return controlled page dicts so no real files
are needed on disk.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from docvault.config import Config
from docvault.generation.response import CitedSource, Response
from docvault.ingest.metadata import Chunk, ChunkMetadata
from docvault.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIM = 384  # must match the mock encoder's output dimension

FAKE_ANSWER = (
    "The document describes a retrieval-augmented approach [1]. "
    "It also mentions fine-tuning strategies [2]."
)

# Text used for each fake page — long enough to produce multiple chunks
PAGE_TEXT = (
    "Retrieval-Augmented Generation combines retrieval with generation. "
    "The encoder embeds both documents and queries into a shared space. "
    "At query time the top-k most similar chunks are retrieved. "
    "These chunks are concatenated into a prompt for the language model. "
    "The model generates a grounded answer that cites its sources. "
    "This approach significantly reduces hallucination in LLM outputs."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vecs = rng.random((n, DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def _fake_pages(doc_name: str = "fake.pdf", num_pages: int = 2) -> list[dict]:
    return [
        {"doc_name": doc_name, "page_number": i + 1, "text": PAGE_TEXT}
        for i in range(num_pages)
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_config(tmp_path: Path) -> Config:
    cfg = Config()
    cfg.index_dir = tmp_path / "indexes"
    cfg.data_dir = tmp_path / "data"
    cfg.anthropic_api_key = "sk-test-fake-key-for-unit-tests"
    cfg.top_k = 3
    cfg.chunk_size = 200
    cfg.chunk_overlap = 30
    cfg.rerank = False
    return cfg


@pytest.fixture()
def mock_encoder():
    """Mock Encoder that returns deterministic unit vectors."""
    enc = MagicMock()
    enc.embedding_dim = DIM

    def _encode(texts, **kwargs):
        return _unit_vectors(len(texts), seed=len(texts))

    def _encode_query(text):
        return _unit_vectors(1, seed=999)

    enc.encode.side_effect = _encode
    enc.encode_query.side_effect = _encode_query
    return enc


@pytest.fixture()
def pipeline_with_mocks(tmp_config: Config, mock_encoder) -> Pipeline:
    """Pipeline with encoder and LLM client pre-patched."""
    p = Pipeline(config=tmp_config)
    p._encoder = mock_encoder  # inject mock encoder directly
    return p


# ---------------------------------------------------------------------------
# Ingest tests
# ---------------------------------------------------------------------------


class TestPipelineIngest:
    def test_ingest_returns_summary_keys(self, pipeline_with_mocks: Pipeline):
        with patch(
            "docvault.pipeline.extract_pdfs",
            return_value=_fake_pages("a.pdf", num_pages=1),
        ):
            summary = pipeline_with_mocks.ingest([Path("a.pdf")])

        expected_keys = {
            "docs_processed", "total_chunks", "new_chunks",
            "cache_hits", "cache_misses", "index_total",
        }
        assert expected_keys.issubset(summary.keys())

    def test_ingest_produces_chunks(self, pipeline_with_mocks: Pipeline):
        with patch(
            "docvault.pipeline.extract_pdfs",
            return_value=_fake_pages("doc.pdf", num_pages=2),
        ):
            summary = pipeline_with_mocks.ingest([Path("doc.pdf")])

        assert summary["total_chunks"] > 0

    def test_ingest_builds_index(self, pipeline_with_mocks: Pipeline):
        with patch(
            "docvault.pipeline.extract_pdfs",
            return_value=_fake_pages("doc.pdf", num_pages=1),
        ):
            summary = pipeline_with_mocks.ingest([Path("doc.pdf")])

        assert summary["index_total"] > 0
        assert pipeline_with_mocks._index is not None
        assert pipeline_with_mocks._index.ntotal == summary["index_total"]

    def test_ingest_saves_index_files(
        self, tmp_config: Config, pipeline_with_mocks: Pipeline
    ):
        with patch(
            "docvault.pipeline.extract_pdfs",
            return_value=_fake_pages("doc.pdf", num_pages=1),
        ):
            pipeline_with_mocks.ingest([Path("doc.pdf")])

        assert (tmp_config.index_dir / "docvault.faiss").exists()
        assert (tmp_config.index_dir / "docvault_chunks.json").exists()

    def test_ingest_incremental(self, pipeline_with_mocks: Pipeline):
        """Two separate ingest calls should accumulate vectors."""
        with patch(
            "docvault.pipeline.extract_pdfs",
            return_value=_fake_pages("first.pdf", num_pages=1),
        ):
            s1 = pipeline_with_mocks.ingest([Path("first.pdf")])

        first_total = s1["index_total"]

        with patch(
            "docvault.pipeline.extract_pdfs",
            return_value=_fake_pages("second.pdf", num_pages=1),
        ):
            s2 = pipeline_with_mocks.ingest([Path("second.pdf")])

        assert s2["index_total"] > first_total

    def test_ingest_empty_extraction_returns_zero(self, pipeline_with_mocks: Pipeline):
        with patch("docvault.pipeline.extract_pdfs", return_value=[]):
            summary = pipeline_with_mocks.ingest([Path("empty.pdf")])
        assert summary["total_chunks"] == 0

    def test_ingest_empty_list_raises(self, pipeline_with_mocks: Pipeline):
        with pytest.raises(ValueError, match="at least one"):
            pipeline_with_mocks.ingest([])

    def test_ingest_cache_hit_on_repeat(self, pipeline_with_mocks: Pipeline):
        """Re-ingesting the same document should produce cache hits."""
        pages = _fake_pages("repeat.pdf", num_pages=1)
        with patch("docvault.pipeline.extract_pdfs", return_value=pages):
            pipeline_with_mocks.ingest([Path("repeat.pdf")])

        # Reset the searcher and re-ingest the same doc
        pipeline_with_mocks._searcher = None
        with patch("docvault.pipeline.extract_pdfs", return_value=pages):
            s2 = pipeline_with_mocks.ingest([Path("repeat.pdf")])

        assert s2["cache_hits"] > 0


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


def _ingest_and_get_pipeline(
    pipeline: Pipeline, doc_name: str = "doc.pdf"
) -> Pipeline:
    """Helper: ingest fake pages and return the pipeline."""
    with patch(
        "docvault.pipeline.extract_pdfs",
        return_value=_fake_pages(doc_name, num_pages=2),
    ):
        pipeline.ingest([Path(doc_name)])
    return pipeline


class TestPipelineQuery:
    def test_query_returns_response(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        with patch.object(
            pipeline_with_mocks._get_client().__class__,
            "generate",
            return_value=FAKE_ANSWER,
        ):
            # Patch the LLM client at instance level
            pipeline_with_mocks._client = MagicMock()
            pipeline_with_mocks._client.generate.return_value = FAKE_ANSWER

            response = pipeline_with_mocks.query("What does the document say?")

        assert isinstance(response, Response)

    def test_query_answer_text(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = FAKE_ANSWER

        response = pipeline_with_mocks.query("Tell me about retrieval.")
        assert response.answer == FAKE_ANSWER

    def test_query_question_stored(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = "Some answer [1]."

        question = "What is RAG?"
        response = pipeline_with_mocks.query(question)
        assert response.query == question

    def test_query_citations_resolved(self, pipeline_with_mocks: Pipeline):
        """Citations [1] and [2] in the answer should map to CitedSource objects."""
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = FAKE_ANSWER  # cites [1] and [2]

        response = pipeline_with_mocks.query("What does this say?")
        assert response.has_citations
        citation_numbers = {src.citation_number for src in response.cited_sources}
        assert 1 in citation_numbers
        assert 2 in citation_numbers

    def test_query_cited_source_fields(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = "Answer here [1]."

        response = pipeline_with_mocks.query("Question?")
        if response.cited_sources:
            src = response.cited_sources[0]
            assert isinstance(src.doc_name, str) and src.doc_name
            assert isinstance(src.page_number, int) and src.page_number >= 1
            assert isinstance(src.chunk_text, str) and src.chunk_text
            assert isinstance(src.score, float)

    def test_query_empty_question_raises(self, pipeline_with_mocks: Pipeline):
        with pytest.raises(ValueError, match="empty"):
            pipeline_with_mocks.query("")

    def test_query_whitespace_question_raises(self, pipeline_with_mocks: Pipeline):
        with pytest.raises(ValueError, match="empty"):
            pipeline_with_mocks.query("   ")

    def test_query_empty_index_raises(self, pipeline_with_mocks: Pipeline):
        # No ingest performed — index is empty
        with pytest.raises(RuntimeError, match="empty"):
            pipeline_with_mocks.query("Will this fail?")

    def test_query_top_k_override(self, pipeline_with_mocks: Pipeline):
        """The search should be called with the overridden top_k."""
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = "Answer [1]."

        # Patch the searcher to track calls
        original_search = pipeline_with_mocks._get_searcher().search
        with patch.object(
            pipeline_with_mocks._get_searcher(),
            "search",
            wraps=original_search,
        ) as mock_search:
            pipeline_with_mocks.query("Question?", top_k=2)
            mock_search.assert_called_once()
            _, kwargs = mock_search.call_args
            assert kwargs.get("top_k") == 2 or mock_search.call_args[0][1] == 2

    def test_query_calls_llm_once(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = "Answer."

        pipeline_with_mocks.query("Question?")
        assert pipeline_with_mocks._client.generate.call_count == 1

    def test_query_format_citations_output(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        pipeline_with_mocks._client = MagicMock()
        pipeline_with_mocks._client.generate.return_value = "The answer is [1]."

        response = pipeline_with_mocks.query("Question?")
        citation_str = response.format_citations()
        if response.cited_sources:
            assert "Sources:" in citation_str
            assert "[1]" in citation_str


# ---------------------------------------------------------------------------
# Pipeline stats and remove_document helpers
# ---------------------------------------------------------------------------


class TestPipelineHelpers:
    def test_stats_returns_dict(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        stats = pipeline_with_mocks.stats()
        assert "total_chunks" in stats
        assert "total_docs" in stats

    def test_remove_document(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks, doc_name="to_remove.pdf")
        before = pipeline_with_mocks._index.ntotal

        removed = pipeline_with_mocks.remove_document("to_remove.pdf")
        assert removed > 0
        assert pipeline_with_mocks._index.ntotal == before - removed
        assert "to_remove.pdf" not in pipeline_with_mocks._index.doc_names()

    def test_remove_nonexistent_document(self, pipeline_with_mocks: Pipeline):
        _ingest_and_get_pipeline(pipeline_with_mocks)
        removed = pipeline_with_mocks.remove_document("ghost.pdf")
        assert removed == 0
