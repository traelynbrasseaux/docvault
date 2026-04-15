"""Text chunking strategies: fixed-size, recursive character, and semantic."""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

from docvault.ingest.metadata import Chunk, ChunkMetadata

if TYPE_CHECKING:
    from docvault.embeddings.encoder import Encoder

logger = logging.getLogger(__name__)

# Sentence boundary: punctuation followed by whitespace and an uppercase letter
# or digit.  Not perfect (misses "Mr. Smith" less often than simpler patterns)
# but avoids hard dependencies on NLTK/spaCy.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d\"\'])")
_PARA_SEP = re.compile(r"\n{2,}")
_WORD_SEP = re.compile(r"\s+")


class ChunkingStrategy(str, Enum):
    """Supported chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a punctuation-aware regex.

    Args:
        text: Input text.

    Returns:
        List of non-empty sentence strings.
    """
    parts = _SENT_BOUNDARY.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_words(text: str) -> list[str]:
    """Split text into whitespace-delimited words.

    Args:
        text: Input text.

    Returns:
        List of words.
    """
    return [w for w in _WORD_SEP.split(text) if w]


def _overlap_start(sentences: list[str], end: int, chunk_overlap: int) -> int:
    """Find the sentence index that gives approximately *chunk_overlap* chars
    of overlap with the previous chunk.

    Walks backwards from *end - 1* accumulating character counts until the
    total exceeds *chunk_overlap*, then returns that index.  Always returns
    a value >= 0.

    Args:
        sentences: Full list of sentences for the document section.
        end: Exclusive end index of the previous chunk.
        chunk_overlap: Target overlap in characters.

    Returns:
        Start index for the next chunk.
    """
    acc = 0
    for j in range(end - 1, -1, -1):
        acc += len(sentences[j]) + 1  # +1 for joining space
        if acc >= chunk_overlap:
            return j
    return 0


def _chunks_from_units(
    units: list[str],
    doc_name: str,
    page_number: int,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int = 0,
) -> list[Chunk]:
    """Greedily pack *units* (sentences or words) into chunks.

    Accumulates units until adding the next would exceed *chunk_size*.
    Applies backward-overlap so adjacent chunks share approximately
    *chunk_overlap* characters.

    Args:
        units: Atomic text units (sentences or words) to pack.
        doc_name: Source document filename.
        page_number: 1-based source page number.
        strategy: Strategy name stored in metadata.
        chunk_size: Target maximum character count per chunk.
        chunk_overlap: Target overlap in characters between adjacent chunks.
        start_index: Initial chunk_index value (used when merging across pages).

    Returns:
        List of :class:`~docvault.ingest.metadata.Chunk` objects.
    """
    chunks: list[Chunk] = []
    chunk_index = start_index
    pos = 0
    n = len(units)

    while pos < n:
        end = pos
        total = 0

        # Greedily accumulate units
        while end < n:
            unit_len = len(units[end]) + (1 if end > pos else 0)
            if total + unit_len > chunk_size and end > pos:
                break
            total += unit_len
            end += 1

        text = " ".join(units[pos:end])
        chunks.append(
            Chunk(
                text=text,
                metadata=ChunkMetadata(
                    doc_name=doc_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    strategy_used=strategy,
                    char_count=len(text),
                ),
            )
        )
        chunk_index += 1

        if end >= n:
            break

        # Determine next starting position with overlap
        next_pos = _overlap_start(units, end, chunk_overlap)
        # Guard: always make forward progress
        pos = next_pos if next_pos > pos else pos + 1

    return chunks


# ---------------------------------------------------------------------------
# Public strategy implementations
# ---------------------------------------------------------------------------


def _fixed_chunks(
    text: str,
    doc_name: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int = 0,
) -> list[Chunk]:
    """Split *text* into fixed-size chunks using sentence boundaries.

    Sentences are the atomic unit; the chunker fills each chunk greedily up to
    *chunk_size* characters without cutting mid-sentence.

    Args:
        text: Input text.
        doc_name: Source document filename.
        page_number: 1-based source page number.
        chunk_size: Target maximum character count per chunk.
        chunk_overlap: Target overlap in characters between adjacent chunks.
        start_index: Initial chunk_index offset.

    Returns:
        List of :class:`~docvault.ingest.metadata.Chunk` objects.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []
    return _chunks_from_units(
        sentences,
        doc_name,
        page_number,
        ChunkingStrategy.FIXED.value,
        chunk_size,
        chunk_overlap,
        start_index,
    )


def _recursive_chunks(
    text: str,
    doc_name: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
    start_index: int = 0,
) -> list[Chunk]:
    """Split *text* recursively: paragraphs → sentences → words.

    Tries to keep whole paragraphs together.  If a paragraph exceeds
    *chunk_size* it is further split into sentences; if a sentence still
    exceeds the limit it is split at word boundaries.

    Args:
        text: Input text.
        doc_name: Source document filename.
        page_number: 1-based source page number.
        chunk_size: Target maximum character count per chunk.
        chunk_overlap: Target overlap in characters between adjacent chunks.
        start_index: Initial chunk_index offset.

    Returns:
        List of :class:`~docvault.ingest.metadata.Chunk` objects.
    """
    # Level 1 — split by paragraph separators
    paragraphs = [p.strip() for p in _PARA_SEP.split(text) if p.strip()]

    # Level 2 — if any paragraph is too large, explode to sentences
    units: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            units.append(para)
        else:
            sentences = _split_sentences(para)
            for sent in sentences:
                if len(sent) <= chunk_size:
                    units.append(sent)
                else:
                    # Level 3 — sentence is still too large: split at words
                    words = _split_words(sent)
                    # Re-join words into word-level "sentences" fitting chunk_size
                    buf: list[str] = []
                    buf_len = 0
                    for word in words:
                        wlen = len(word) + (1 if buf else 0)
                        if buf_len + wlen > chunk_size and buf:
                            units.append(" ".join(buf))
                            buf = [word]
                            buf_len = len(word)
                        else:
                            buf.append(word)
                            buf_len += wlen
                    if buf:
                        units.append(" ".join(buf))

    if not units:
        return []

    return _chunks_from_units(
        units,
        doc_name,
        page_number,
        ChunkingStrategy.RECURSIVE.value,
        chunk_size,
        chunk_overlap,
        start_index,
    )


def _semantic_chunks(
    text: str,
    doc_name: str,
    page_number: int,
    chunk_size: int,
    chunk_overlap: int,
    encoder: "Encoder",
    similarity_threshold: float = 0.5,
    start_index: int = 0,
) -> list[Chunk]:
    """Split *text* by embedding-based sentence similarity.

    Embeds every sentence, then groups adjacent sentences whose cosine
    similarity exceeds *similarity_threshold* into the same chunk.  A new
    chunk is also started whenever the accumulated length would exceed
    *chunk_size*.

    Args:
        text: Input text.
        doc_name: Source document filename.
        page_number: 1-based source page number.
        chunk_size: Maximum character count per chunk before forcing a split.
        chunk_overlap: Target overlap in characters between adjacent chunks.
        encoder: Pre-loaded :class:`~docvault.embeddings.encoder.Encoder`
            instance used to embed sentences.
        similarity_threshold: Cosine similarity threshold below which a new
            chunk is started.  Range [0, 1]; higher → more splits.
        start_index: Initial chunk_index offset.

    Returns:
        List of :class:`~docvault.ingest.metadata.Chunk` objects.

    Raises:
        ValueError: If *encoder* is ``None``.
    """
    if encoder is None:
        raise ValueError(
            "An Encoder instance is required for semantic chunking. "
            "Pass encoder= when calling chunk_text() with strategy='semantic'."
        )

    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        chunk_text_val = sentences[0]
        return [
            Chunk(
                text=chunk_text_val,
                metadata=ChunkMetadata(
                    doc_name=doc_name,
                    page_number=page_number,
                    chunk_index=start_index,
                    strategy_used=ChunkingStrategy.SEMANTIC.value,
                    char_count=len(chunk_text_val),
                ),
            )
        ]

    # Embed all sentences at once (batched)
    embeddings: np.ndarray = encoder.encode(sentences)  # (N, D)

    # Compute cosine similarities between consecutive sentences.
    # Embeddings are already unit-normalized → dot product == cosine similarity.
    sims: np.ndarray = np.einsum("ij,ij->i", embeddings[:-1], embeddings[1:])

    chunks: list[Chunk] = []
    chunk_index = start_index
    group_start = 0
    acc_len = len(sentences[0])

    for i in range(1, len(sentences)):
        new_len = acc_len + 1 + len(sentences[i])
        low_sim = float(sims[i - 1]) < similarity_threshold
        too_long = new_len > chunk_size

        if low_sim or too_long:
            chunk_text_val = " ".join(sentences[group_start:i])
            chunks.append(
                Chunk(
                    text=chunk_text_val,
                    metadata=ChunkMetadata(
                        doc_name=doc_name,
                        page_number=page_number,
                        chunk_index=chunk_index,
                        strategy_used=ChunkingStrategy.SEMANTIC.value,
                        char_count=len(chunk_text_val),
                    ),
                )
            )
            chunk_index += 1

            # Overlap: include trailing sentences from previous group
            if chunk_overlap > 0:
                prev_sents = sentences[group_start:i]
                overlap_sents: list[str] = []
                olen = 0
                for s in reversed(prev_sents):
                    olen += len(s) + 1
                    overlap_sents.insert(0, s)
                    if olen >= chunk_overlap:
                        break
                group_start = i - len(overlap_sents)
            else:
                group_start = i

            acc_len = sum(len(sentences[k]) + 1 for k in range(group_start, i + 1))
        else:
            acc_len = new_len

    # Flush remaining sentences
    if group_start < len(sentences):
        chunk_text_val = " ".join(sentences[group_start:])
        chunks.append(
            Chunk(
                text=chunk_text_val,
                metadata=ChunkMetadata(
                    doc_name=doc_name,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    strategy_used=ChunkingStrategy.SEMANTIC.value,
                    char_count=len(chunk_text_val),
                ),
            )
        )

    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    doc_name: str,
    page_number: int,
    strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoder: "Encoder | None" = None,
    similarity_threshold: float = 0.5,
    start_index: int = 0,
) -> list[Chunk]:
    """Split *text* into chunks using the specified strategy.

    Args:
        text: Input text to chunk.
        doc_name: Filename of the source PDF stored in chunk metadata.
        page_number: 1-based page number stored in chunk metadata.
        strategy: One of ``"fixed"``, ``"recursive"``, or ``"semantic"``.
        chunk_size: Target maximum character count per chunk.
        chunk_overlap: Approximate character overlap between adjacent chunks.
        encoder: Required only for ``"semantic"`` strategy.
        similarity_threshold: Cosine similarity split threshold (semantic only).
        start_index: Starting chunk_index for emitted chunks.

    Returns:
        List of :class:`~docvault.ingest.metadata.Chunk` objects.

    Raises:
        ValueError: If an unknown strategy is given, or if ``encoder`` is
            missing when strategy is ``"semantic"``.
    """
    strategy = ChunkingStrategy(strategy)

    if strategy == ChunkingStrategy.FIXED:
        return _fixed_chunks(
            text, doc_name, page_number, chunk_size, chunk_overlap, start_index
        )
    elif strategy == ChunkingStrategy.RECURSIVE:
        return _recursive_chunks(
            text, doc_name, page_number, chunk_size, chunk_overlap, start_index
        )
    elif strategy == ChunkingStrategy.SEMANTIC:
        return _semantic_chunks(
            text,
            doc_name,
            page_number,
            chunk_size,
            chunk_overlap,
            encoder=encoder,  # type: ignore[arg-type]
            similarity_threshold=similarity_threshold,
            start_index=start_index,
        )
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy!r}")


def chunk_pages(
    pages: list[dict[str, Any]],
    strategy: ChunkingStrategy | str = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    encoder: "Encoder | None" = None,
    similarity_threshold: float = 0.5,
) -> list[Chunk]:
    """Chunk all pages extracted from a document.

    Passes a global ``chunk_index`` counter across pages so each chunk has a
    unique index within the document.

    Args:
        pages: Page-level dicts as returned by
            :func:`~docvault.ingest.extract.extract_pdf`.
        strategy: Chunking strategy to apply.
        chunk_size: Target maximum character count per chunk.
        chunk_overlap: Approximate character overlap between adjacent chunks.
        encoder: Required for semantic strategy.
        similarity_threshold: Cosine similarity split threshold (semantic only).

    Returns:
        All chunks from the document, in page order.
    """
    all_chunks: list[Chunk] = []
    for page in pages:
        page_chunks = chunk_text(
            text=page["text"],
            doc_name=page["doc_name"],
            page_number=page["page_number"],
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            encoder=encoder,
            similarity_threshold=similarity_threshold,
            start_index=len(all_chunks),
        )
        all_chunks.extend(page_chunks)
    logger.debug(
        "Chunked '%s' into %d chunk(s) using '%s' strategy.",
        pages[0]["doc_name"] if pages else "unknown",
        len(all_chunks),
        str(strategy),
    )
    return all_chunks
